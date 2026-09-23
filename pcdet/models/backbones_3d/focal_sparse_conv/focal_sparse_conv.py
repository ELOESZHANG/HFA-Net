import torch
import torch.nn.functional as F
import torch.nn as nn

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

        # 下采样层，使用平均池化进行下采样，这里设置核大小为2，步长为2
        self.downsampling = nn.AvgPool2d(kernel_size=2, stride=2)

        # 多尺度池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv_pool = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, stride=2, groups=in_channel,
                                   bias=False)

        # 用于得到不同尺度特征的卷积层
        self.conv_for_q = nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1)
        self.conv_for_k = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1, stride=2)
        self.conv_for_v = nn.Conv2d(in_channel, in_channel, kernel_size=5, padding=2, stride=2)

        # Transformer 的多头自注意力
        self.attention = nn.MultiheadAttention(embed_dim=in_channel, num_heads=num_heads)

        # 激活函数
        self.silu = nn.SiLU()

    def forward(self, inputs):
        # 下采样操作
        b, c, h, w = inputs.shape

        # 全局池化操作
        avg_pool = self.global_avg_pool(inputs)  # 输出形状为 (b, c, 1, 1)
        max_pool = self.global_max_pool(inputs)  # 输出形状为 (b, c, 1, 1)
        conv_pool_0 = self.conv_pool(inputs)  # 输出形状为 (b, c, h/2, w/2)

        # 对 conv_pool 进行平均池化处理，调整其大小为 (b, c, 1, 1)
        conv_pool = conv_pool_0.mean([2, 3], keepdim=True)

        # 得到不同尺度的特征并调整形状
        q = self.conv_for_q(inputs) # 输出形状为 (b, c, h, w)
        k = self.conv_for_k(inputs) # 输出形状为 (b, c, h//2, w//2)
        v = self.conv_for_v(inputs)# 输出形状为 (b, c, h//2, w//2)

        # 调整q, k, v的形状以适应MultiheadAttention的输入要求 (seq_len, batch, embed_dim)
        q = q.mean(-2).permute(2, 0, 1)  # 形状变为 (w, b, c)
        k = k.mean(-2).permute(2, 0, 1)  # 形状变为 (w//2), b, c)
        v = v.mean(-2).permute(2, 0, 1)  # 确保k和v的seq_len相同，例如调整为与k相同的尺寸

        # 自注意力计算
        attn_output, _ = self.attention(q, k, v)
        attn_output = attn_output.permute(1, 2, 0).mean(dim=-1)  # 输出形状为 (b, c)

        # 通道加权
        scale = torch.sigmoid(attn_output.view(b, c, 1, 1))  # 输出形状为 (b, c, 1, 1)
        # 最终输出
        # return inputs * scale * 0.1 + inputs   ##left
        # return inputs * (torch.sigmoid(avg_pool) + torch.sigmoid(max_pool) + torch.sigmoid(conv_pool)) * 0.5 + inputs    ##right
        # return inputs * scale * 0.1 + inputs * (torch.sigmoid(avg_pool) + torch.sigmoid(max_pool) + torch.sigmoid(conv_pool)) * 0.5 + inputs
        return inputs * scale*0.1 + inputs * (torch.sigmoid(avg_pool) + torch.sigmoid(max_pool) + torch.sigmoid(conv_pool))*0.05+inputs

class CDFM(nn.Module):
    def __init__(self, feature_dim=16, latent_dim=64):
        super().__init__()
        self.feature_dim = feature_dim
        self.latent_dim = latent_dim

        # 联合分布编码器
        self.joint_encoder = nn.Sequential(
            nn.Conv1d(2 * feature_dim, latent_dim // 2, 1),
            Mish(),
            nn.Conv1d(latent_dim // 2, 2 * latent_dim, 1)  # 输出均值和对数方差
        )

        # 条件解码器
        self.decoder = nn.Sequential(
            nn.Conv1d(latent_dim + feature_dim, 2 * feature_dim, 1),
            Mish(),
            nn.Conv1d(2 * feature_dim, feature_dim, 1)
        )

        # 特征增强层
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
        _, feat_dim, _  = img_feat.shape

        # 拼接特征建立联合分布
        joint_feat = torch.cat([img_feat, vox_feat], dim=1)

        # 编码联合分布参数
        dist_params = self.joint_encoder(joint_feat)
        mu, logvar = dist_params.chunk(2, dim=1)

        # 重参数化采样
        z = self.reparameterize(mu, logvar)

        # 双重条件解码（图像条件+点云条件）
        img_cond = torch.cat([mu, img_feat], dim=1)
        vox_cond = torch.cat([logvar, vox_feat], dim=1)

        # 解码生成残差特征
        img_res = self.decoder(img_cond)
        vox_res = self.decoder(vox_cond)

        # 增强特征交互
        fused_img = F.normalize(self.enhance(img_res),dim=-1) + img_feat
        fused_vox = F.normalize(self.enhance(vox_res),dim=-1) + vox_feat

        # 动态混合
        alpha = torch.sigmoid((fused_img + fused_vox).mean(dim=1, keepdim=True))
        final_img = 0.1 * alpha * img_feat + (1 - alpha) * 0.01 * fused_img + img_feat
        final_vox = (1 - alpha) * 0.01 * fused_vox + 0.1 * alpha * vox_feat + vox_feat

        # return fused_img, fused_vox     #/dy
        return final_img, final_vox

class DMM(nn.Module):
    def __init__(self, feature_dim=16, latent_dim=64):
        super().__init__()
        self.fusion_core = CDFM(feature_dim, latent_dim)

        # 特征投影层
        self.proj = nn.Conv1d(feature_dim, feature_dim, 1)

    def forward(self, img_feat, vox_feat):
        # 分布感知融合
        fused_img, fused_vox = self.fusion_core(img_feat, vox_feat)  # 只接收两个返回值

        # 残差连接与投影
        img_out = self.proj(img_feat + fused_img)
        vox_out = self.proj(vox_feat + fused_vox)

        # 显式使用所有维度参数
        batch_size, feat_dim, seq_len = img_out.shape
        return (
            img_out.transpose(1, 2).reshape(batch_size * seq_len, feat_dim),
            vox_out.transpose(1, 2).reshape(batch_size * seq_len, feat_dim)
        )


class FocalSparseConv(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, voxel_stride, norm_fn=None, indice_key=None,
                 image_channel=3, kernel_size=3, padding=1, mask_multi=False, use_img=False,
                 topk=False, threshold=0.5, skip_mask_kernel=False, enlarge_voxel_channels=-1,
                 point_cloud_range = None,
                 voxel_size = None):
        if point_cloud_range is None:
            point_cloud_range = [-3, -40, 0, 1, 40, 70.4]
        if voxel_size is None:
                 voxel_size=[0.1, 0.05, 0.05]
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

    def construct_multimodal_features(self, x, x_rgb, batch_dict, fuse_sum=False):
        """
            Construct the multimodal features with both lidar sparse features and image features.
            Args:
                x: [N, C] lidar sparse features
                x_rgb: [b, c, h, w] image features
                batch_dict: input and output information during forward
                fuse_sum: bool, manner for fusion, True - sum, False - concat

            Return:
                image_with_voxelfeatures: [N, C] fused multimodal features
        """
        batch_index = x.indices[:, 0]
        spatial_indices = x.indices[:, 1:] * self.voxel_stride
        voxels_3d = spatial_indices * self.voxel_size + self.point_cloud_range[:3]
        calibs = batch_dict['calib']
        batch_size = batch_dict['batch_size']
        h, w = batch_dict['images'].shape[2:]

        if not x_rgb.shape == batch_dict['images'].shape:
            x_rgb = nn.functional.interpolate(x_rgb, (h, w), mode='bilinear')

        image_with_voxelfeatures = []
        voxels_2d_int_list = []
        filter_idx_list = []

        x_rgb = self.DPWS(x_rgb)

        for b in range(batch_size):
            x_rgb_batch = x_rgb[b]

            calib = calibs[b]
            voxels_3d_batch = voxels_3d[batch_index == b]
            voxel_features_sparse = x.features[batch_index == b]

            # Reverse the point cloud transformations to the original coords.
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

            filter_idx_list.append(filter_idx)
            voxels_2d_int = voxels_2d_int[filter_idx]
            voxels_2d_int_list.append(voxels_2d_int)

            image_features_batch = torch.zeros((voxel_features_sparse.shape[0], x_rgb_batch.shape[0]),
                                               device=x_rgb_batch.device)
            image_features_batch[filter_idx] = x_rgb_batch[:, voxels_2d_int[:, 1], voxels_2d_int[:, 0]].permute(1, 0)
        ##fusion
            image_features_batch = image_features_batch.permute(1, 0).reshape( 1, 16,-1)
            voxel_features_sparse = voxel_features_sparse.permute(1, 0).reshape( 1, 16,-1)
            image_features_batch, voxel_features_sparse = self.DMM(image_features_batch,voxel_features_sparse)

###
            if fuse_sum:
                image_with_voxelfeature = image_features_batch + voxel_features_sparse
            else:
                image_with_voxelfeature = torch.cat([image_features_batch, voxel_features_sparse], dim=1)

            image_with_voxelfeatures.append(image_with_voxelfeature)

        image_with_voxelfeatures = torch.cat(image_with_voxelfeatures)
        return image_with_voxelfeatures

    def _gen_sparse_features(self, x, imps_3d, batch_dict, voxels_3d):
        """
            Generate the output sparse features from the focal sparse conv.
            Args:
                x: [N, C], lidar sparse features
                imps_3d: [N, kernelsize**3], the predicted importance values
                batch_dict: input and output information during forward
                voxels_3d: [N, 3], the 3d positions of voxel centers
        """
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

            features_fore, indices_fore, features_back, indices_back, mask_kernel = split_voxels(x, b, imps_3d,
                                                                                                 voxels_3d,
                                                                                                 self.kernel_offsets,
                                                                                                 mask_multi=self.mask_multi,
                                                                                                 topk=self.topk,
                                                                                                 threshold=self.threshold)

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
        """
            Combine the foreground and background sparse features together.
            Args:
                x_fore: [N1, C], foreground sparse features
                x_back: [N2, C], background sparse features
                remove_repeat: bool, whether to remove the spatial replicate features.
        """
        x_fore_features = torch.cat([x_fore.features, x_back.features], dim=0)
        x_fore_indices = torch.cat([x_fore.indices, x_back.indices], dim=0)

        if remove_repeat:
            index = x_fore_indices[:, 0]
            features_out_list = []
            indices_coords_out_list = []
            for b in range(x_fore.batch_size):
                batch_index = index == b
                features_out, indices_coords_out, _ = check_repeat(x_fore_features[batch_index],
                                                                   x_fore_indices[batch_index], flip_first=False)
                features_out_list.append(features_out)
                indices_coords_out_list.append(indices_coords_out)
            x_fore_features = torch.cat(features_out_list, dim=0)
            x_fore_indices = torch.cat(indices_coords_out_list, dim=0)

        x_fore = x_fore.replace_feature(x_fore_features)
        x_fore.indices = x_fore_indices

        return x_fore

    def forward(self, x, batch_dict, x_rgb=None):
        spatial_indices = x.indices[:, 1:] * self.voxel_stride
        voxels_3d = spatial_indices * self.voxel_size + self.point_cloud_range[:3]

        if self.use_img:
            features_multimodal = self.construct_multimodal_features(x, x_rgb, batch_dict)
            x_predict = spconv.SparseConvTensor(features_multimodal, x.indices, x.spatial_shape, x.batch_size)
        else:
            x_predict = self.conv_enlarge(x) if self.conv_enlarge else x

        imps_3d = self.conv_imp(x_predict).features

        x_fore, x_back, loss_box_of_pts, mask_kernel = self._gen_sparse_features(x, imps_3d, batch_dict, voxels_3d)

        if not self.skip_mask_kernel:
            x_fore = x_fore.replace_feature(x_fore.features * mask_kernel.unsqueeze(-1))
        out = self.combine_out(x_fore, x_back, remove_repeat=True)
        out = self.conv(out)

        if self.use_img:
            out = out.replace_feature(self.construct_multimodal_features(out, x_rgb, batch_dict, True))

        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        return out, batch_dict, loss_box_of_pts
