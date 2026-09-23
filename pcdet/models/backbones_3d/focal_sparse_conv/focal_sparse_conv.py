import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from pcdet.utils.spconv_utils import spconv
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu
from pcdet.models.backbones_3d.focal_sparse_conv.focal_sparse_utils import split_voxels, check_repeat, FocalLoss
from pcdet.utils import common_utils


class Mish(nn.Module):
    """Mish激活函数：平滑非单调，增强梯度流"""

    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class DPWS(nn.Module):
    def __init__(self, in_channel, num_heads=2, reduction=4):
        super().__init__()
        self.in_channel = in_channel
        self.reduction = reduction

        self.downsampling = nn.AvgPool2d(kernel_size=2, stride=2)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_pool = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, stride=2, groups=in_channel,
                                   bias=False)

        self.conv_for_q = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1)
        self.conv_for_k = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, stride=2)
        self.conv_for_v = nn.Conv2d(in_channel, in_channel, kernel_size=5, padding=2, stride=2)

        self.attention = nn.MultiheadAttention(embed_dim=in_channel, num_heads=num_heads)
        self.silu = nn.SiLU()

    def forward(self, inputs):
        b, c, h, w = inputs.shape

        avg_pool = self.global_avg_pool(inputs)
        max_pool = self.global_max_pool(inputs)
        conv_pool_0 = self.conv_pool(inputs)
        conv_pool = conv_pool_0.mean([2, 3], keepdim=True)

        q = self.conv_for_q(inputs)
        k = self.conv_for_k(inputs)
        v = self.conv_for_v(inputs)

        q = q.mean(-2).permute(2, 0, 1)
        k = k.mean(-2).permute(2, 0, 1)
        v = v.mean(-2).permute(2, 0, 1)

        attn_output, _ = self.attention(q, k, v)
        attn_output = attn_output.permute(1, 2, 0).mean(dim=-1)
        scale = torch.sigmoid(attn_output.view(b, c, 1, 1))
        return inputs * scale * 0.1 + inputs * (torch.sigmoid(avg_pool) + torch.sigmoid(max_pool) + torch.sigmoid(conv_pool)) * 0.05 + inputs


class CDFM(nn.Module):
    def __init__(self, feature_dim=16, latent_dim=64):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        self.joint_encoder = nn.Sequential(
            nn.Conv1d(2 * feature_dim, latent_dim // 2, 1),
            Mish(),
            nn.Conv1d(latent_dim // 2, 2 * latent_dim, 1)
        )

        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim + feature_dim, 2 * feature_dim, 1),
            Mish(),
            nn.Conv1d(2 * feature_dim, feature_dim, 1)
        )

        self.enhance = nn.Sequential(
            nn.Conv1d(feature_dim, feature_dim, 3, padding=1, groups=8),
            Mish(),
            nn.Conv1d(feature_dim, feature_dim, 1)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, img_feat, vox_feat):
        _, feat_dim, _ = img_feat.shape

        joint_feat = torch.cat([img_feat, vox_feat], dim=1)
        dist_params = self.joint_encoder(joint_feat)
        mu, logvar = dist_params.chunk(2, dim=1)

        z = self.reparameterize(mu, logvar)

        img_cond = torch.cat([mu, img_feat], dim=1)
        vox_cond = torch.cat([logvar, vox_feat], dim=1)

        img_res = self.decoder(img_cond)
        vox_res = self.decoder(vox_cond)

        fused_img = F.normalize(self.enhance(img_res), dim=-1) + img_feat
        fused_vox = F.normalize(self.enhance(vox_res), dim=-1) + vox_feat

        alpha = torch.sigmoid((fused_img + fused_vox).mean(dim=1, keepdim=True))
        final_img = 0.1 * alpha * img_feat + (1 - alpha) * 0.01 * fused_img + img_feat
        final_vox = (1 - alpha) * 0.01 * fused_vox + 0.1 * alpha * vox_feat + vox_feat

        return final_img, final_vox


class DMM(nn.Module):
    def __init__(self, feature_dim=16, latent_dim=64):
        super().__init__()
        self.fusion_core = CDFM(feature_dim, latent_dim)
        self.proj = nn.Conv1d(feature_dim, feature_dim, 1)

    def forward(self, img_feat, vox_feat):
        final_img, final_vox = self.fusion_core(img_feat, vox_feat)

        img_out = self.proj(img_feat + final_img)
        vox_out = self.proj(vox_feat + final_vox)

        batch_size, feat_dim, seq_len = img_out.shape
        img_out_flat = img_out.transpose(1, 2).reshape(batch_size * seq_len, feat_dim)
        vox_out_flat = vox_out.transpose(1, 2).reshape(batch_size * seq_len, feat_dim)

        return img_out_flat, vox_out_flat, final_img, final_vox


class FocalSparseConv(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, voxel_stride, norm_fn=None, indice_key=None,
                 image_channel=3, kernel_size=3, padding=1, mask_multi=False, use_img=False,
                 topk=False, threshold=0.5, skip_mask_kernel=False, enlarge_voxel_channels=-1,
                 point_cloud_range=None, voxel_size=None,
                 visualize_heatmap=True, vis_frame_ids=None):
        if point_cloud_range is None:
            point_cloud_range = [-3, -40, 0, 1, 40, 70.4]
        if voxel_size is None:
            voxel_size = [0.1, 0.05, 0.05]
        super(FocalSparseConv, self).__init__()

        self.conv = spconv.SubMConv3d(inplanes, planes, kernel_size=kernel_size, stride=1, bias=False,
                                      indice_key=indice_key)
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU(True)
        offset_channels = kernel_size ** 3

        self.topk = topk
        self.threshold = threshold
        self.voxel_stride = voxel_stride
        self.focal_loss = FocalLoss()
        self.mask_multi = mask_multi
        self.skip_mask_kernel = skip_mask_kernel
        self.use_img = use_img

        # 缓存：存储融合过程的中间特征用于分阶段BEV可视化
        self._vis_cache = None

        # 可视化开关
        self.visualize_heatmap = visualize_heatmap
        self.vis_frame_ids = vis_frame_ids  # None=全部帧, 或指定列表如 ['000008']
        self.vis_step = 0
        # 层标识（防止不同层的文件互相覆盖）
        self.vis_name = indice_key if indice_key else f'focal_{id(self)}'
        # 热力图保存目录（使用绝对路径）
        self.vis_dir = os.path.abspath('./heatmap_output')
        if self.visualize_heatmap and not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)
            print(f"[BEV vis] heatmap output dir: {self.vis_dir}")

        voxel_channel = enlarge_voxel_channels if enlarge_voxel_channels > 0 else inplanes
        in_channels = image_channel + voxel_channel if use_img else voxel_channel

        self.conv_enlarge = spconv.SparseSequential(spconv.SubMConv3d(inplanes, enlarge_voxel_channels,
                                                                      kernel_size=3, stride=1, padding=1, bias=False,
                                                                      indice_key=indice_key + '_enlarge'),
                                                    norm_fn(enlarge_voxel_channels),
                                                    nn.ReLU(True)) if enlarge_voxel_channels > 0 else None

        self.conv_imp = spconv.SubMConv3d(in_channels, offset_channels, kernel_size=3, stride=1, padding=1, bias=False,
                                          indice_key=indice_key + '_imp')

        _step = int(kernel_size // 2)
        kernel_offsets = [[i, j, k] for i in range(-_step, _step + 1) for j in range(-_step, _step + 1) for k in
                          range(-_step, _step + 1)]
        kernel_offsets.remove([0, 0, 0])
        self.kernel_offsets = torch.Tensor(kernel_offsets).cuda()
        self.inv_idx = torch.Tensor([2, 1, 0]).long().cuda()
        self.point_cloud_range = torch.Tensor(point_cloud_range).cuda()
        self.voxel_size = torch.Tensor(voxel_size).cuda()

        self.Mish = Mish()
        self.DPWS = DPWS(in_channel=16, reduction=8)
        self.CDFM = CDFM(feature_dim=16, latent_dim=64)
        self.DMM = DMM(feature_dim=16, latent_dim=64)

    # ------------------------------------------------------------------
    # 将gt_boxes 3D框转为BEV网格像素坐标，用于画在热力图上
    # ------------------------------------------------------------------
    def _get_bev_gt_boxes(self, batch_dict, batch_id, down):
        gt_boxes = batch_dict['gt_boxes'][batch_id].detach().cpu().numpy()
        pc = self.point_cloud_range.cpu().numpy()
        vs = self.voxel_size.cpu().numpy()
        valid = gt_boxes[:, -1] >= 0
        gt_boxes = gt_boxes[valid]
        if len(gt_boxes) == 0:
            return []
        cx = (gt_boxes[:, 0] - pc[2]) / vs[2] / down
        cy = (gt_boxes[:, 1] - pc[1]) / vs[1] / down
        lp = gt_boxes[:, 4] / vs[2] / down
        wp = gt_boxes[:, 3] / vs[1] / down
        heading = gt_boxes[:, 6]
        return list(zip(cx, cy, lp, wp, heading))

    def _draw_bev_boxes(self, ax, gt_boxes_bev):
        """在BEV热力图上绘制gt_boxes矩形框"""
        for box in gt_boxes_bev:
            cx, cy, l, w, heading = box
            cos_h, sin_h = np.cos(heading), np.sin(heading)
            corners = []
            for dx, dy in [(-l/2, -w/2), (l/2, -w/2), (l/2, w/2), (-l/2, w/2)]:
                rx = cx + dx * cos_h - dy * sin_h
                ry = cy + dx * sin_h + dy * cos_h
                corners.append([rx, ry])
            polygon = patches.Polygon(corners, fill=False, edgecolor='lime', linewidth=1.0)
            ax.add_patch(polygon)

    # ------------------------------------------------------------------
    # BEV视角下原始图像特征 + 原始点云特征的逐通道(16)热力图
    # 图像特征：体素投影到2D图像→采样特征→按体素BEV坐标填回BEV网格
    # 点云特征：稀疏体素特征→按体素BEV坐标转换为密集BEV网格
    # 两者使用相同体素的BEV坐标，保证对齐
    # ------------------------------------------------------------------
    def visualize_fusion_stages(self, x, batch_dict, target_dim=1600):
        """可视化融合过程的5个阶段：
        1-2. CDFM的图像/LiDAR BEV输入
        3-4. CDFM输出的两种模态
        5. DMM输出的融合特征
        所有图叠加黑底白点的点云BEV视图
        """
        if not self.visualize_heatmap or self._vis_cache is None:
            return

        cache = self._vis_cache.get(0, None)
        if cache is None:
            return

        voxel_indices = cache['indices']  # (N, 4) = [batch, z, y, x]
        pc = self.point_cloud_range.cpu().numpy()
        vs = self.voxel_size.cpu().numpy()
        stride = self.voxel_stride

        # BEV网格大小
        bev_rows = int((pc[5] - pc[2]) / vs[2])  # X(forward) → rows
        bev_cols = int((pc[4] - pc[1]) / vs[1])  # Y(left-right) → cols
        down = max(1, max(bev_rows, bev_cols) // target_dim)
        rows_small, cols_small = bev_rows // down, bev_cols // down

        # 体素的BEV像素坐标
        x_full = voxel_indices[:, 3].astype(np.int64) * stride
        y_full = voxel_indices[:, 2].astype(np.int64) * stride
        row_bin = (x_full // down).astype(int)
        col_bin = (y_full // down).astype(int)
        valid = (row_bin >= 0) & (row_bin < rows_small) & (col_bin >= 0) & (col_bin < cols_small)
        row_bin_v = row_bin[valid]
        col_bin_v = col_bin[valid]

        # 黑底白点的占据网格
        occupancy = np.zeros((rows_small, cols_small), dtype=np.float32)
        occupancy[row_bin_v, col_bin_v] = 1.0

        frame_tag = f"frame{batch_dict['frame_id'][0]}" if 'frame_id' in batch_dict else f'step{self.vis_step}'

        stages = [
            ('input_img', 'Input_Image'),
            ('input_lidar', 'Input_LiDAR'),
            ('output_img', 'Output_Image'),
            ('output_lidar', 'Output_LiDAR'),
            ('fused', 'Fused'),
        ]

        for key, name_tag in stages:
            if key not in cache:
                continue

            feats = cache[key]          # (N, 16)
            feats_v = feats[valid]      # 只取有效BEV位置的体素
            n_valid = feats_v.shape[0]
            if n_valid == 0:
                continue

            for ch in range(feats_v.shape[1]):
                feat_ch = feats_v[:, ch]

                # 投影到BEV
                bev_vals = np.zeros((rows_small, cols_small), dtype=np.float32)
                cnt_map = np.zeros((rows_small, cols_small), dtype=np.int32)
                for i in range(n_valid):
                    r, c = row_bin_v[i], col_bin_v[i]
                    bev_vals[r, c] += feat_ch[i]
                    cnt_map[r, c] += 1
                data_mask = cnt_map > 0
                bev_vals[data_mask] /= cnt_map[data_mask]

                # 归一化
                fm = bev_vals.copy()
                dv = fm[data_mask]
                if len(dv) > 0:
                    lo, hi = np.percentile(dv, [2, 98])
                    fm = np.clip((fm - lo) / (hi - lo), 0, 1) if hi > lo else fm
                    fm[data_mask] = np.where(hi > lo, fm[data_mask], 0.5)

                # 原始 colormap（jet）
                heat_rgb = plt.cm.jet(fm)[:, :, :3]
                canvas = heat_rgb.copy()
                # 白色点云叠加
                occ_mask = occupancy > 0
                canvas[occ_mask] = canvas[occ_mask] * 0.65 + np.array([1.0, 1.0, 1.0]) * 0.35

                fig, ax = plt.subplots(figsize=(8, 8))
                ax.imshow(canvas, origin='upper')
                ax.axis('off')
                fig.savefig(os.path.join(self.vis_dir, f'bev_fusion_{name_tag}_ch{ch}_{self.vis_name}_{frame_tag}.png'),
                            bbox_inches='tight', pad_inches=0, dpi=150)
                plt.close(fig)
            print(f"[VIS] Saved fusion stage: {name_tag}")

    # ------------------------------------------------------------------
    # 原有的方法（construct_multimodal_features, _gen_sparse_features等）
    # ------------------------------------------------------------------
    def construct_multimodal_features(self, x, x_rgb, batch_dict, fuse_sum=False):
        batch_index = x.indices[:, 0]
        spatial_indices = x.indices[:, 1:] * self.voxel_stride
        voxels_3d = spatial_indices * self.voxel_size + self.point_cloud_range[:3]
        calibs = batch_dict['calib']
        batch_size = batch_dict['batch_size']
        h, w = batch_dict['images'].shape[2:]

        if not x_rgb.shape == batch_dict['images'].shape:
            x_rgb = nn.functional.interpolate(x_rgb, (h, w), mode='bilinear')

        x_rgb = self.DPWS(x_rgb)

        image_with_voxelfeatures = []
        for b in range(batch_size):
            x_rgb_batch = x_rgb[b]
            calib = calibs[b]
            voxels_3d_batch = voxels_3d[batch_index == b]
            voxel_features_sparse = x.features[batch_index == b]

            if 'noise_scale' in batch_dict:
                voxels_3d_batch[:, :3] /= batch_dict['noise_scale'][b]
            if 'noise_rot' in batch_dict:
                voxels_3d_batch = common_utils.rotate_points_along_z(voxels_3d_batch[:, self.inv_idx].unsqueeze(0),
                                                                     -batch_dict['noise_rot'][b].unsqueeze(0))[0, :,
                                  self.inv_idx]
            if 'flip_x' in batch_dict:
                voxels_3d_batch[:, 1] *= -1 if batch_dict['flip_x'][b] else 1
            if 'flip_y' in batch_dict:
                voxels_3d_batch[:, 2] *= -1 if batch_dict['flip_y'][b] else 1

            voxels_2d, _ = calib.lidar_to_img(voxels_3d_batch[:, self.inv_idx].cpu().numpy())
            voxels_2d_int = torch.Tensor(voxels_2d).to(x_rgb_batch.device).long()
            filter_idx = (0 <= voxels_2d_int[:, 1]) * (voxels_2d_int[:, 1] < h) * (0 <= voxels_2d_int[:, 0]) * (
                        voxels_2d_int[:, 0] < w)

            voxels_2d_int = voxels_2d_int[filter_idx]
            image_features_batch = torch.zeros((voxel_features_sparse.shape[0], x_rgb_batch.shape[0]),
                                               device=x_rgb_batch.device)
            image_features_batch[filter_idx] = x_rgb_batch[:, voxels_2d_int[:, 1], voxels_2d_int[:, 0]].permute(1, 0)

            image_features_batch = image_features_batch.permute(1, 0).reshape(1, 16, -1)
            voxel_features_sparse = voxel_features_sparse.permute(1, 0).reshape(1, 16, -1)

            img_out_flat, vox_out_flat, final_img, final_vox = self.DMM(image_features_batch, voxel_features_sparse)

            # 缓存中间特征用于分阶段BEV可视化
            if self._vis_cache is not None:
                vox_indices_b = x.indices[batch_index == b].detach().cpu().numpy()
                self._vis_cache[int(b)] = {
                    'indices': vox_indices_b,
                    'input_img': image_features_batch.squeeze(0).detach().cpu().numpy().T,
                    'input_lidar': voxel_features_sparse.squeeze(0).detach().cpu().numpy().T,
                    'output_img': final_img.squeeze(0).detach().cpu().numpy().T,
                    'output_lidar': final_vox.squeeze(0).detach().cpu().numpy().T,
                    'fused': (img_out_flat + vox_out_flat).detach().cpu().numpy(),
                }

            if fuse_sum:
                image_with_voxelfeature = img_out_flat + vox_out_flat
            else:
                image_with_voxelfeature = torch.cat([img_out_flat, vox_out_flat], dim=1)

            image_with_voxelfeatures.append(image_with_voxelfeature)

        image_with_voxelfeatures = torch.cat(image_with_voxelfeatures, dim=0)
        return image_with_voxelfeatures

    def _gen_sparse_features(self, x, imps_3d, batch_dict, voxels_3d):
        batch_size = x.batch_size
        voxel_features_fore = []
        voxel_indices_fore = []
        voxel_features_back = []
        voxel_indices_back = []

        box_of_pts_cls_targets = []
        mask_voxels = []
        mask_kernel_list = []

        for b in range(batch_size):
            if self.training:
                index = x.indices[:, 0]
                batch_index = index == b
                mask_voxel = imps_3d[batch_index, -1].sigmoid()
                voxels_3d_batch = voxels_3d[batch_index].unsqueeze(0)
                mask_voxels.append(mask_voxel)
                gt_boxes = batch_dict['gt_boxes'][b, :, :-1].unsqueeze(0)
                box_of_pts_batch = points_in_boxes_gpu(voxels_3d_batch[:, :, self.inv_idx], gt_boxes).squeeze(0)
                box_of_pts_cls_targets.append(box_of_pts_batch >= 0)

            features_fore, indices_fore, features_back, indices_back, mask_kernel = split_voxels(
                x, b, imps_3d, voxels_3d, self.kernel_offsets,
                mask_multi=self.mask_multi, topk=self.topk, threshold=self.threshold
            )
            mask_kernel_list.append(mask_kernel)
            voxel_features_fore.append(features_fore)
            voxel_indices_fore.append(indices_fore)
            voxel_features_back.append(features_back)
            voxel_indices_back.append(indices_back)

        voxel_features_fore = torch.cat(voxel_features_fore, dim=0)
        voxel_indices_fore = torch.cat(voxel_indices_fore, dim=0)
        voxel_features_back = torch.cat(voxel_features_back, dim=0)
        voxel_indices_back = torch.cat(voxel_indices_back, dim=0)
        mask_kernel = torch.cat(mask_kernel_list, dim=0)

        x_fore = spconv.SparseConvTensor(voxel_features_fore, voxel_indices_fore, x.spatial_shape, x.batch_size)
        x_back = spconv.SparseConvTensor(voxel_features_back, voxel_indices_back, x.spatial_shape, x.batch_size)

        loss_box_of_pts = 0
        if self.training:
            mask_voxels = torch.cat(mask_voxels)
            box_of_pts_cls_targets = torch.cat(box_of_pts_cls_targets)
            mask_voxels_two_classes = torch.cat([1 - mask_voxels.unsqueeze(-1), mask_voxels.unsqueeze(-1)], dim=1)
            loss_box_of_pts = self.focal_loss(mask_voxels_two_classes, box_of_pts_cls_targets.long())

        return x_fore, x_back, loss_box_of_pts, mask_kernel

    def combine_out(self, x_fore, x_back, remove_repeat=False):
        x_fore_features = torch.cat([x_fore.features, x_back.features], dim=0)
        x_fore_indices = torch.cat([x_fore.indices, x_back.indices], dim=0)

        if remove_repeat:
            index = x_fore_indices[:, 0]
            features_out_list = []
            indices_coords_out_list = []
            for b in range(x_fore.batch_size):
                batch_index = index == b
                features_out, indices_coords_out, _ = check_repeat(
                    x_fore_features[batch_index], x_fore_indices[batch_index], flip_first=False
                )
                features_out_list.append(features_out)
                indices_coords_out_list.append(indices_coords_out)
            x_fore_features = torch.cat(features_out_list, dim=0)
            x_fore_indices = torch.cat(indices_coords_out_list, dim=0)

        x_fore = x_fore.replace_feature(x_fore_features)
        x_fore.indices = x_fore_indices
        return x_fore

    def forward(self, x, batch_dict, x_rgb=None):
        # 可视化缓存：仅当指定帧或无限制时初始化
        _do_vis = self.visualize_heatmap and self.use_img
        if _do_vis and self.vis_frame_ids is not None:
            _fid = batch_dict['frame_id'][0] if 'frame_id' in batch_dict else None
            _do_vis = _fid in self.vis_frame_ids
        self._vis_cache = {} if _do_vis else None

        spatial_indices = x.indices[:, 1:] * self.voxel_stride
        voxels_3d = spatial_indices * self.voxel_size + self.point_cloud_range[:3]

        if self.use_img:
            features_multimodal = self.construct_multimodal_features(x, x_rgb, batch_dict, fuse_sum=False)
            x_predict = spconv.SparseConvTensor(features_multimodal, x.indices, x.spatial_shape, x.batch_size)

            # 可视化融合过程的5个阶段
            self.visualize_fusion_stages(x, batch_dict)
        else:
            x_predict = self.conv_enlarge(x) if self.conv_enlarge else x

        imps_3d = self.conv_imp(x_predict).features
        x_fore, x_back, loss_box_of_pts, mask_kernel = self._gen_sparse_features(x, imps_3d, batch_dict, voxels_3d)

        if not self.skip_mask_kernel:
            x_fore = x_fore.replace_feature(x_fore.features * mask_kernel.unsqueeze(-1))
        out = self.combine_out(x_fore, x_back, remove_repeat=True)
        out = self.conv(out)

        if self.use_img:
            out = out.replace_feature(self.construct_multimodal_features(out, x_rgb, batch_dict, fuse_sum=True))

        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        # 更新全局步数（用于文件名）
        if self.visualize_heatmap:
            self.vis_step += 1

        return out, batch_dict, loss_box_of_pts