"""Feature Extraction and Parameter Prediction networks
"""
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.tools import sample_and_group_multi

# import MinkowskiEngine as ME
# import MinkowskiEngine.MinkowskiFunctional as MEF

# from network.common import get_norm
# from network.residual_block import conv, conv_tr, get_block

_raw_features_sizes = {'xyz': 3, 'dxyz': 3, 'ppf': 4}
_raw_features_order = {'xyz': 0, 'dxyz': 1, 'ppf': 2}


class DGR_ResUNet2(ME.MinkowskiNetwork):
    BLOCK_NORM_TYPE = 'BN'
    REGION_TYPE = ME.RegionType.HYPERCUBE
    NORM_TYPE = 'BN'
    CHANNELS = [None, 32, 64, 128, 256]
    TR_CHANNELS = [None, 64, 64, 64, 128]

    # To use the model, must call initialize_coords before forward pass.
    # Once data is processed, call clear to reset the model before calling initialize_coords
    def __init__(self,
                in_channels=1,
                out_channels=1,
                bn_momentum=0.05,
                conv1_kernel_size=3,
                normalize_feature=False,
                D=6):
        ME.MinkowskiNetwork.__init__(self, D)
        NORM_TYPE = self.NORM_TYPE
        BLOCK_NORM_TYPE = self.BLOCK_NORM_TYPE
        CHANNELS = self.CHANNELS
        TR_CHANNELS = self.TR_CHANNELS
        REGION_TYPE = self.REGION_TYPE

        self.conv1 = conv(
            in_channels=in_channels,
            out_channels=CHANNELS[1],
            kernel_size=conv1_kernel_size,
            stride=1,
            dilation=1,
            has_bias=False,
            region_type=REGION_TYPE,
            dimension=D)
        self.norm1 = get_norm(NORM_TYPE, CHANNELS[1], bn_momentum=bn_momentum, dimension=D)

        self.block1 = get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[1],
            CHANNELS[1],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D)

        self.conv2 = conv(
            in_channels=CHANNELS[1],
            out_channels=CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            has_bias=False,
            region_type=REGION_TYPE,
            dimension=D)
        self.norm2 = get_norm(NORM_TYPE, CHANNELS[2], bn_momentum=bn_momentum, dimension=D)

        self.block2 = get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[2],
            CHANNELS[2],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D)

        self.conv3 = conv(
            in_channels=CHANNELS[2],
            out_channels=CHANNELS[3],
            kernel_size=3,
            stride=2,
            dilation=1,
            has_bias=False,
            region_type=REGION_TYPE,
            dimension=D)
        self.norm3 = get_norm(NORM_TYPE, CHANNELS[3], bn_momentum=bn_momentum, dimension=D)

        self.block3 = get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[3],
            CHANNELS[3],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D)

        self.conv4 = conv(
            in_channels=CHANNELS[3],
            out_channels=CHANNELS[4],
            kernel_size=3,
            stride=2,
            dilation=1,
            has_bias=False,
            region_type=REGION_TYPE,
            dimension=D)
        self.norm4 = get_norm(NORM_TYPE, CHANNELS[4], bn_momentum=bn_momentum, dimension=D)

        self.block4 = get_block(
            BLOCK_NORM_TYPE,
            CHANNELS[4],
            CHANNELS[4],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D)

        ##################################
        self.conv4_tr = conv_tr(
            in_channels=CHANNELS[4],
            out_channels=TR_CHANNELS[4],
            kernel_size=3,
            stride=2,
            dilation=1,
            has_bias=False,
            region_type=REGION_TYPE,
            dimension=D)
        self.norm4_tr = get_norm(
            NORM_TYPE, TR_CHANNELS[4], bn_momentum=bn_momentum, dimension=D)

        self.block4_tr = get_block(
            BLOCK_NORM_TYPE,
            TR_CHANNELS[4],
            TR_CHANNELS[4],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D)

        self.conv3_tr = conv_tr(
            in_channels=CHANNELS[3] + TR_CHANNELS[4],
            out_channels=TR_CHANNELS[3],
            kernel_size=3,
            stride=2,
            dilation=1,
            has_bias=False,
            region_type=REGION_TYPE,
            dimension=D)
        self.norm3_tr = get_norm(
            NORM_TYPE, TR_CHANNELS[3], bn_momentum=bn_momentum, dimension=D)

        self.block3_tr = get_block(
            BLOCK_NORM_TYPE,
            TR_CHANNELS[3],
            TR_CHANNELS[3],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D)

        self.conv2_tr = conv_tr(
            in_channels=CHANNELS[2] + TR_CHANNELS[3],
            out_channels=TR_CHANNELS[2],
            kernel_size=3,
            stride=2,
            dilation=1,
            has_bias=False,
            region_type=REGION_TYPE,
            dimension=D)
        self.norm2_tr = get_norm(
            NORM_TYPE, TR_CHANNELS[2], bn_momentum=bn_momentum, dimension=D)

        self.block2_tr = get_block(
            BLOCK_NORM_TYPE,
            TR_CHANNELS[2],
            TR_CHANNELS[2],
            bn_momentum=bn_momentum,
            region_type=REGION_TYPE,
            dimension=D)

        self.conv1_tr = conv(
            in_channels=CHANNELS[1] + TR_CHANNELS[2],
            out_channels=TR_CHANNELS[1],
            kernel_size=1,
            stride=1,
            dilation=1,
            has_bias=False,
            dimension=D)

        self.final = ME.MinkowskiConvolution(
            in_channels=TR_CHANNELS[1],
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            dilation=1,
            has_bias=True,
            dimension=D)

    def forward(self, x):
        out_s1 = self.conv1(x)
        out_s1 = self.norm1(out_s1)
        out_s1 = self.block1(out_s1)
        out = MEF.relu(out_s1)

        out_s2 = self.conv2(out)
        out_s2 = self.norm2(out_s2)
        out_s2 = self.block2(out_s2)
        out = MEF.relu(out_s2)

        out_s4 = self.conv3(out)
        out_s4 = self.norm3(out_s4)
        out_s4 = self.block3(out_s4)
        out = MEF.relu(out_s4)

        out_s8 = self.conv4(out)
        out_s8 = self.norm4(out_s8)
        out_s8 = self.block4(out_s8)
        out = MEF.relu(out_s8)

        out = self.conv4_tr(out)
        out = self.norm4_tr(out)
        out = self.block4_tr(out)
        out_s4_tr = MEF.relu(out)

        out = ME.cat(out_s4_tr, out_s4)

        out = self.conv3_tr(out)
        out = self.norm3_tr(out)
        out = self.block3_tr(out)
        out_s2_tr = MEF.relu(out)

        out = ME.cat(out_s2_tr, out_s2)

        out = self.conv2_tr(out)
        out = self.norm2_tr(out)
        out = self.block2_tr(out)
        out_s1_tr = MEF.relu(out)

        out = ME.cat(out_s1_tr, out_s1)
        out = self.conv1_tr(out)
        out = MEF.relu(out)
        out = self.final(out)

        return out



class ParameterPredictionNet(nn.Module):
    def __init__(self, weights_dim):
        """PointNet based Parameter prediction network

        Args:
            weights_dim: Number of weights to predict (excluding beta), should be something like
                         [3], or [64, 3], for 3 types of features
        """
        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)
        self.weights_dim = weights_dim

        # Pointnet
        self.prepool = nn.Sequential(
            nn.Conv1d(4, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Conv1d(64, 128, 1),
            nn.GroupNorm(8, 128),
            nn.ReLU(),

            nn.Conv1d(128, 1024, 1),
            nn.GroupNorm(16, 1024),
            nn.ReLU(),
        )

        self.pooling = nn.AdaptiveMaxPool1d(1)

        self.postpool = nn.Sequential(
            nn.Linear(1024, 512),
            nn.GroupNorm(16, 512),
            nn.ReLU(),

            nn.Linear(512, 256),
            nn.GroupNorm(16, 256),
            nn.ReLU(),

            nn.Linear(256, 2 + np.prod(weights_dim)),
        )

        self._logger.info('Predicting weights with dim {}.'.format(self.weights_dim))

    def forward(self, src, ref):
        """ Returns alpha, beta, and gating_weights (if needed)

        Args:
            src: (B, 3, J)
            ref: (B, 3, K)

        Returns:
            beta (B, ), alpha (B, ), weightings (B, 2)
        """
        src_padded = F.pad(src, (0, 0, 0, 1), mode='constant', value=0)  # TODO (x, y, z, 0)  [B, 4, J]
        ref_padded = F.pad(ref, (0, 0, 0, 1), mode='constant', value=1)  # (x, y, z, 1)       [B, 4, K]

        concatenated = torch.cat([src_padded, ref_padded], dim=-1)   # TODO  [B, 4, J+K]

        # x = get_graph_feature(x)
        # x = F.relu(self.bn1(self.conv1(x)))
        # x1 = x.max(dim=-1, keepdim=True)[0]

        # src_embedding_p, tgt_embedding_p = self.pointer(src_embedding, tgt_embedding)

        # PointRNN

        prepool_feat = self.prepool(concatenated)        # [B, 1024, J+K]
        pooled = torch.flatten(self.pooling(prepool_feat), start_dim=-2)  # [B, 1024, 1] => [B, 1024]
        raw_weights = self.postpool(pooled)    # [B, 2]

        beta  = F.softplus(raw_weights[:, 0])
        alpha = F.softplus(raw_weights[:, 1])
        return beta, alpha

class ParameterPredictionNetConstant(nn.Module):
    def __init__(self, weights_dim):
        """ Parameter Prediction Network with single alpha/beta as parameter.

        See: Ablation study (Table 4) in paper
        """
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)

        self.anneal_weights = nn.Parameter(torch.zeros(2 + np.prod(weights_dim)))
        self.weights_dim = weights_dim
        self._logger.info('Predicting weights with dim {}.'.format(self.weights_dim))

    def forward(self, x):
        """ Returns beta, gating_weights
        """
        batch_size = x[0].shape[0]
        raw_weights = self.anneal_weights
        beta = F.softplus(raw_weights[0].expand(batch_size))
        alpha = F.softplus(raw_weights[1].expand(batch_size))

        return beta, alpha

def get_prepool(in_dim, out_dim):
    """Shared FC part in PointNet before max pooling
    """
    net = nn.Sequential(
        nn.Conv2d(in_dim, out_dim // 2, 1),
        nn.GroupNorm(8, out_dim // 2),
        nn.ReLU(),

        nn.Conv2d(out_dim // 2, out_dim // 2, 1),
        nn.GroupNorm(8, out_dim // 2),
        nn.ReLU(),

        nn.Conv2d(out_dim // 2, out_dim, 1),
        nn.GroupNorm(8, out_dim),
        nn.ReLU(),
    )
    return net

def get_postpool(in_dim, out_dim):
    """Linear layers in PointNet after max pooling

    Args:
        in_dim: Number of input channels
        out_dim: Number of output channels. Typically smaller than in_dim
    """
    net = nn.Sequential(
        nn.Conv1d(in_dim, in_dim, 1),
        nn.GroupNorm(8, in_dim),
        nn.ReLU(),

        nn.Conv1d(in_dim, out_dim, 1),
        nn.GroupNorm(8, out_dim),
        nn.ReLU(),

        nn.Conv1d(out_dim, out_dim, 1),
    )

    return net

class FeatExtractionEarlyFusion(nn.Module):
    """Feature extraction Module that extracts hybrid features"""
    def __init__(self, feature_opt, feature_dim, radius, num_neighbors):
        super().__init__()

        self._logger = logging.getLogger(self.__class__.__name__)
        self._logger.info('Using early fusion, feature dim = {}'.format(feature_dim))
        self.npoint = -1
        self.radius = radius
        self.n_sample = num_neighbors

        self.feature_opt = sorted(feature_opt, key=lambda f: _raw_features_order[f])
        self._logger.info('Feature extraction using features {}'.format(', '.join(self.feature_opt)))

        # Layers
        raw_dim = np.sum([_raw_features_sizes[f] for f in self.feature_opt])  # number of channels after concat
        self.prepool = get_prepool(raw_dim, feature_dim * 2)
        self.postpool = get_postpool(feature_dim * 2, feature_dim)

    def forward(self, xyz, normals):
        """Forward pass of the feature extraction network

        Args:
            xyz: (B, N, 3)
            normals: (B, N, 3)

        Returns:
            cluster features (B, N, C)
        """
        # TODO DGCNN
        # compute xyz, dxyz and ppf features
        features = sample_and_group_multi(self.npoint, self.radius, self.n_sample, xyz, normals)
        features['xyz'] = features['xyz'][:, :, None, :]

        # Gate and concat xyz, dxyz and ppf
        concat = []
        for i in range(len(self.feature_opt)):
            f = self.feature_opt[i]
            expanded = (features[f]).expand(-1, -1, self.n_sample, -1)
            concat.append(expanded)
        fused_input_feat = torch.cat(concat, -1)         # [B, N, n_sample, 10]

        # Prepool_FC, pool, postpool-FC
        new_feat = fused_input_feat.permute(0, 3, 2, 1)  # [B, 10, n_sample, N]
        new_feat = self.prepool(new_feat)

        pooled_feat = torch.max(new_feat, 2)[0]          # Max pooling [B, C, N]

        post_feat = self.postpool(pooled_feat)           # Post pooling dense layers
        cluster_feat = post_feat.permute(0, 2, 1)
        cluster_feat = cluster_feat / torch.norm(cluster_feat, dim=-1, keepdim=True)

        return cluster_feat   # [B, N, C]

### ***************************************************************************

class ResContextBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(ResContextBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=1)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(shortcut)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        output = shortcut + resA2
        return output

class ResBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, kernel_size=(3, 3), stride=1,
                 pooling=True, drop_out=True):
        super(ResBlock, self).__init__()
        self.pooling = pooling
        self.drop_out = drop_out
        self.conv1 = nn.Conv2d(in_filters, out_filters, kernel_size=(1, 1), stride=stride)
        self.act1 = nn.LeakyReLU()

        self.conv2 = nn.Conv2d(in_filters, out_filters, kernel_size=(3, 3), padding=1)
        self.act2 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, kernel_size=(3, 3),dilation=2, padding=2)
        self.act3 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv4 = nn.Conv2d(out_filters, out_filters, kernel_size=(2, 2), dilation=2, padding=1)
        self.act4 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)

        self.conv5 = nn.Conv2d(out_filters*3, out_filters, kernel_size=(1, 1))
        self.act5 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        if pooling:
            self.dropout = nn.Dropout2d(p=dropout_rate)
            self.pool = nn.AvgPool2d(kernel_size=kernel_size, stride=2, padding=1)
        else:
            self.dropout = nn.Dropout2d(p=dropout_rate)

    def forward(self, x):
        shortcut = self.conv1(x)
        shortcut = self.act1(shortcut)

        resA = self.conv2(x)
        resA = self.act2(resA)
        resA1 = self.bn1(resA)

        resA = self.conv3(resA1)
        resA = self.act3(resA)
        resA2 = self.bn2(resA)

        resA = self.conv4(resA2)
        resA = self.act4(resA)
        resA3 = self.bn3(resA)

        concat = torch.cat((resA1, resA2, resA3),dim=1)
        resA = self.conv5(concat)
        resA = self.act5(resA)
        resA = self.bn4(resA)
        resA = shortcut + resA

        if self.pooling:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            resB = self.pool(resB)
            return resB, resA
        else:
            if self.drop_out:
                resB = self.dropout(resA)
            else:
                resB = resA
            return resB

class UpBlock(nn.Module):
    def __init__(self, in_filters, out_filters, dropout_rate, drop_out=True):
        super(UpBlock, self).__init__()
        self.drop_out = drop_out
        self.in_filters = in_filters
        self.out_filters = out_filters

        self.dropout1 = nn.Dropout2d(p=dropout_rate)

        self.dropout2 = nn.Dropout2d(p=dropout_rate)

        self.conv1 = nn.Conv2d(in_filters//4 + 2*out_filters, out_filters, (3,3), padding=1)
        self.act1 = nn.LeakyReLU()
        self.bn1 = nn.BatchNorm2d(out_filters)

        self.conv2 = nn.Conv2d(out_filters, out_filters, (3,3),dilation=2, padding=2)
        self.act2 = nn.LeakyReLU()
        self.bn2 = nn.BatchNorm2d(out_filters)

        self.conv3 = nn.Conv2d(out_filters, out_filters, (2,2), dilation=2,padding=1)
        self.act3 = nn.LeakyReLU()
        self.bn3 = nn.BatchNorm2d(out_filters)


        self.conv4 = nn.Conv2d(out_filters*3,out_filters,kernel_size=(1,1))
        self.act4 = nn.LeakyReLU()
        self.bn4 = nn.BatchNorm2d(out_filters)

        self.dropout3 = nn.Dropout2d(p=dropout_rate)

    def forward(self, x, skip):
        upA = nn.PixelShuffle(2)(x)
        if self.drop_out:
            upA = self.dropout1(upA)

        upB = torch.cat((upA,skip),dim=1)
        if self.drop_out:
            upB = self.dropout2(upB)

        upE = self.conv1(upB)
        upE = self.act1(upE)
        upE1 = self.bn1(upE)

        upE = self.conv2(upE1)
        upE = self.act2(upE)
        upE2 = self.bn2(upE)

        upE = self.conv3(upE2)
        upE = self.act3(upE)
        upE3 = self.bn3(upE)

        concat = torch.cat((upE1,upE2,upE3),dim=1)
        upE = self.conv4(concat)
        upE = self.act4(upE)
        upE = self.bn4(upE)
        if self.drop_out:
            upE = self.dropout3(upE)

        return upE

class SalsaNext(nn.Module):
    def __init__(self, dim_out, n_height):
        super(SalsaNext, self).__init__()
        self.dim_out = dim_out
        self.n_height = n_height
        feat_out_dim = n_height * dim_out
        score_out_dim = n_height

        self.downCntx1 = ResContextBlock(n_height, 32)
        self.downCntx2 = ResContextBlock(32, 32)
        self.downCntx3 = ResContextBlock(32, 32)

        self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=True, drop_out=False)
        self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=True)
        self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock4 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=True)
        self.resBlock5 = ResBlock(2 * 4 * 32, 2 * 4 * 32, 0.2, pooling=False)

        self.upBlock1 = UpBlock(2 * 4 * 32, 4 * 32, 0.2)
        self.upBlock2 = UpBlock(4 * 32, 4 * 32, 0.2)
        self.upBlock3 = UpBlock(4 * 32, 2 * 32, 0.2)
        self.upBlock4 = UpBlock(2 * 32, 32, 0.2, drop_out=False)

        self.out_feat = nn.Conv2d(32, feat_out_dim, 1)
        self.out_score = nn.Conv2d(32, score_out_dim, 1)

    def forward(self, x):
        B, C, H, W = x.size()

        downCntx = self.downCntx1(x)
        downCntx = self.downCntx2(downCntx)
        downCntx = self.downCntx3(downCntx)

        down0c, down0b = self.resBlock1(downCntx)
        down1c, down1b = self.resBlock2(down0c)
        down2c, down2b = self.resBlock3(down1c)
        down3c, down3b = self.resBlock4(down2c)
        down5c = self.resBlock5(down3c)

        up4e = self.upBlock1(down5c, down3b)
        up3e = self.upBlock2(up4e, down2b)
        up2e = self.upBlock3(up3e, down1b)
        up1e = self.upBlock4(up2e, down0b)

        feat  = self.out_feat(up1e)
        score = self.out_score(up1e)

        feat  = feat.permute(0, 2, 3, 1)      # BHWC1
        score = score.permute(0, 2, 3, 1)    # BHWC2
        feat  = feat.view([B, H, W, self.n_height, self.dim_out]).permute(0, 4, 1, 2, 3)    # [N, dim_out, H, W, n_height]
        score = score.view([B, H, W, self.n_height, 1]).permute(0, 4, 1, 2, 3)  # [N, 1, H, W, n_height]

        return feat, down5c, score

class ResNet(nn.Module):
    def __init__(self, dim_out, n_height):
        super().__init__()
        self.dim_out = dim_out
        self.n_height = n_height   # grid_size[2]
        feat_out_dim = n_height * dim_out
        score_out_dim = n_height

        self.downCntx1 = ResContextBlock(n_height, 32)

        self.resBlock1 = ResBlock(32, 2 * 32, 0.2, pooling=False, drop_out=False)
        # self.resBlock2 = ResBlock(2 * 32, 2 * 2 * 32, 0.2, pooling=False)
        # self.resBlock3 = ResBlock(2 * 2 * 32, 2 * 32, 0.2, pooling=False)

        self.out_feat = nn.Conv2d(32*2, feat_out_dim, 1)
        self.out_score = nn.Conv2d(32*2, score_out_dim, 1)

    def forward(self, x):
        B, C, H, W = x.size()

        x = self.downCntx1(x)

        x = self.resBlock1(x)
        # x = self.resBlock2(x)
        # x = self.resBlock3(x)

        feat  = self.out_feat(x)
        score = self.out_score(x)

        feat  = feat.permute(0, 2, 3, 1)      # BHWC1
        score = score.permute(0, 2, 3, 1)    # BHWC2
        feat  = feat.view([B, H, W, self.n_height, self.dim_out]).permute(0, 4, 1, 2, 3)    # [N, dim_out, H, W, n_height]
        score = score.view([B, H, W, self.n_height, 1]).permute(0, 4, 1, 2, 3)  # [N, 1, H, W, n_height]

        return feat, x, score


class ParameterPredictionV3(nn.Module):
    def __init__(self, dim_in):
        """ Parameter prediction network
        """
        super().__init__()
        # self._logger = logging.getLogger(self.__class__.__name__)

        self.pre_net = nn.Sequential(
            nn.Conv1d(dim_in+1, 64, 1),
            nn.GroupNorm(8, 64),
            nn.ReLU() )

        self.pooling = nn.AdaptiveMaxPool1d(1)

        self.post_net = nn.Sequential(
            nn.Linear(64, 16),
            nn.GroupNorm(4, 16),
            nn.ReLU(),

            nn.Linear(16, 2) )

    def forward(self, x1, x2):
        """
        :param x1: [B, C, J]
        :param x2: [B, C, K]
        """
        src_padded = F.pad(x1, (0, 0, 0, 1), mode='constant', value=0)  # TODO (..., 0) channel+1
        ref_padded = F.pad(x2, (0, 0, 0, 1), mode='constant', value=1)  # (..., 1)

        concatenated = torch.cat([src_padded, ref_padded], dim=2)  # TODO (B, C+1, J+K)

        pre_feat = self.pre_net(concatenated)
        pooled = torch.flatten(self.pooling(pre_feat), start_dim=-2)  # [B, 64]
        raw_weights = self.post_net(pooled)    # [B, 2]

        beta  = F.softplus(raw_weights[:, 0])
        alpha = F.softplus(raw_weights[:, 1])
        return beta, alpha

class ParameterPredictionV2(nn.Module):
    def __init__(self):
        """ Parameter prediction network
        """
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)

        self.mid_dim = 128
        self.pre_net = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(),

            nn.Conv2d(256, self.mid_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.mid_dim, affine=False),
            nn.ReLU() )

        self.pooling = nn.AdaptiveMaxPool1d(1)

        self.post_net = nn.Sequential(
            nn.Linear(self.mid_dim, 64),
            nn.GroupNorm(8, 64),
            nn.ReLU(),

            nn.Linear(64, 2) )

    def forward(self, x1, x2):
        """
        :param x1: [B, C, H, W]
        :param x2: [B, C, H, W]
        """
        B = len(x1)
        concatenated = torch.cat([x1, x2], dim=1)  # [B, 2C, H, W]
        # self._logger.info('Middle features for predicting params: [{}]'.format(concatenated.size()))

        pre_feat = self.pre_net(concatenated)
        pooled = torch.flatten(self.pooling(pre_feat.view(B, self.mid_dim, -1)), start_dim=-2)
        raw_weights = self.post_net(pooled)

        beta  = F.softplus(raw_weights[:, 0])
        alpha = F.softplus(raw_weights[:, 1])
        return beta, alpha

class ParameterPrediction(nn.Module):
    def __init__(self):
        """ Parameter prediction network
        """
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)

        self.mid_dim = 256
        self.pre_net = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512, affine=False),
            nn.ReLU(),

            nn.Conv2d(512, self.mid_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.mid_dim, affine=False),
            nn.ReLU() )

        self.pooling = nn.AdaptiveMaxPool1d(1)

        self.post_net = nn.Sequential(
            nn.Linear(self.mid_dim, 128),
            nn.GroupNorm(8, 128),
            nn.ReLU(),

            nn.Linear(128, 2) )

    def forward(self, x1, x2):
        """
        :param x1: [B, C, H, W]
        :param x2: [B, C, H, W]
        """
        B = len(x1)
        concatenated = torch.cat([x1, x2], dim=1)  # [B, 2C, H, W]
        # self._logger.info('Middle features for predicting params: [{}]'.format(concatenated.size()))

        pre_feat = self.pre_net(concatenated)
        pooled = torch.flatten(self.pooling(pre_feat.view(B, self.mid_dim, -1)), start_dim=-2)
        raw_weights = self.post_net(pooled)

        beta  = F.softplus(raw_weights[:, 0])
        alpha = F.softplus(raw_weights[:, 1])
        return beta, alpha

### ***************************************************************************

class UNet(nn.Module):
    def __init__(self, dim_out, n_height, dilation=1, group_conv=False, input_batch_norm=False,
                    dropout=0., circular_padding=False):
        super(UNet, self).__init__()
        self.dim_out = dim_out
        self.n_height = n_height
        feat_out_dim = n_height * dim_out
        score_out_dim = n_height

        self.inc = inconv(n_height, 64, dilation, input_batch_norm, circular_padding)

        self.down1 = down(64, 128, dilation, group_conv, circular_padding)
        self.down2 = down(128, 256, dilation, group_conv, circular_padding)
        self.down3 = down(256, 512, dilation, group_conv, circular_padding)
        self.down4 = down(512, 512, dilation, group_conv, circular_padding)

        self.up1 = up(1024, 256, circular_padding, group_conv=group_conv)
        self.up2 = up(512, 128, circular_padding, group_conv=group_conv)
        self.up3 = up(256, 64, circular_padding, group_conv=group_conv)
        self.up4 = up(128, 64, circular_padding, group_conv=group_conv)

        self.dropout = nn.Dropout(p=dropout)
        self.out_feat = nn.Conv2d(64, feat_out_dim, 1)
        self.out_score = nn.Conv2d(64, score_out_dim, 1)

    def forward(self, x):
        B, C, H, W = x.size()
        x1 = self.inc(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        feat  = self.out_feat(self.dropout(x))
        score = self.out_score(self.dropout(x))

        feat  = feat.permute(0, 2, 3, 1)      # BHWC1
        score = score.permute(0, 2, 3, 1)    # BHWC2
        feat  = feat.view([B, H, W, self.n_height, self.dim_out]).permute(0, 4, 1, 2, 3)    # [N, dim_out, H, W, n_height]
        score = score.view([B, H, W, self.n_height, 1]).permute(0, 4, 1, 2, 3)  # [N, 1, H, W, n_height]

        return feat, x5, score


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, group_conv, dilation=1):
        super(double_conv, self).__init__()
        if group_conv:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, groups=min(out_ch, in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),   # TODO LeakyReLU

                nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),

                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        x = self.conv(x)
        return x

class double_conv_circular(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, group_conv, dilation=1):
        super(double_conv_circular, self).__init__()
        if group_conv:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1, 0), groups=min(out_ch, in_ch)),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1, 0), groups=out_ch),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv1 = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=(1, 0)),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(out_ch, out_ch, 3, padding=(1, 0)),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        # add circular padding
        # (We implement ring convolution by connecting both ends of matrix via circular padding)
        # TODO useful ???
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.conv1(x)
        x = F.pad(x, (1, 1, 0, 0), mode='circular')
        x = self.conv2(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, input_batch_norm, circular_padding):
        super(inconv, self).__init__()
        if input_batch_norm:
            if circular_padding:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv_circular(in_ch, out_ch, group_conv=False, dilation=dilation)
                )
            else:
                self.conv = nn.Sequential(
                    nn.BatchNorm2d(in_ch),
                    double_conv(in_ch, out_ch, group_conv=False, dilation=dilation)
                )
        else:
            if circular_padding:
                self.conv = double_conv_circular(in_ch, out_ch, group_conv=False, dilation=dilation)
            else:
                self.conv = double_conv(in_ch, out_ch, group_conv=False, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x

class down(nn.Module):
    def __init__(self, in_ch, out_ch, dilation, group_conv, circular_padding):
        super(down, self).__init__()
        if circular_padding:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv_circular(in_ch, out_ch, group_conv=group_conv, dilation=dilation)
            )
        else:
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(2),
                double_conv(in_ch, out_ch, group_conv=group_conv, dilation=dilation)
            )

    def forward(self, x):
        x = self.mpconv(x)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, circular_padding, bilinear=True, group_conv=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        elif group_conv:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2, groups=in_ch//2)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        if circular_padding:
            self.conv = double_conv_circular(in_ch, out_ch, group_conv=group_conv)
        else:
            self.conv = double_conv(in_ch, out_ch, group_conv=group_conv)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        # for padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

### ***************************************************************************

def change_default_args(**kwargs):
    import inspect

    def get_pos_to_kw_map(func):
        pos_to_kw = {}
        fsig = inspect.signature(func)
        pos = 0
        for name, info in fsig.parameters.items():
            if info.kind is info.POSITIONAL_OR_KEYWORD:
                pos_to_kw[pos] = name
            pos += 1
        return pos_to_kw

    def layer_wrapper(layer_class):
        class DefaultArgLayer(layer_class):
            def __init__(self, *args, **kw):
                pos_to_kw = get_pos_to_kw_map(layer_class.__init__)
                kw_to_pos = {kw: pos for pos, kw in pos_to_kw.items()}
                for key, val in kwargs.items():
                    if key not in kw and kw_to_pos[key] > len(args):
                        kw[key] = val
                super().__init__(*args, **kw)
        return DefaultArgLayer

    return layer_wrapper

class RPN(nn.Module):
    def __init__(self,
                 use_norm=True,
                 num_class=2,
                 layer_nums=(3, 5, 5),
                 layer_strides=(2, 2, 2),
                 num_filters=(128, 128, 256),
                 upsample_strides=(1, 2, 4),
                 num_upsample_filters=(256, 256, 256),
                 num_input_features=128,
                 num_anchor_per_loc=2,
                 encode_background_as_zeros=True,
                 use_direction_classifier=True,
                 use_groupnorm=False,
                 num_groups=32,
                 box_code_size=7,
                 num_direction_bins=2,
                 name='rpn'):
        """deprecated. exists for checkpoint backward compilability (SECOND v1.0)
        """
        super(RPN, self).__init__()
        self._num_anchor_per_loc = num_anchor_per_loc
        self._use_direction_classifier = use_direction_classifier
        assert len(layer_nums) == 3
        assert len(layer_strides) == len(layer_nums)
        assert len(num_filters) == len(layer_nums)
        assert len(upsample_strides) == len(layer_nums)
        assert len(num_upsample_filters) == len(layer_nums)
        upsample_strides = [
            np.round(u).astype(np.int64) for u in upsample_strides
        ]
        factors = []
        for i in range(len(layer_nums)):
            assert int(np.prod(
                layer_strides[:i + 1])) % upsample_strides[i] == 0
            factors.append(
                np.prod(layer_strides[:i + 1]) // upsample_strides[i])
        assert all([x == factors[0] for x in factors])
        if use_norm:
            if use_groupnorm:
                BatchNorm2d = change_default_args(
                    num_groups=num_groups, eps=1e-3)(GroupNorm)
            else:
                BatchNorm2d = change_default_args(
                    eps=1e-3, momentum=0.01)(nn.BatchNorm2d)
            Conv2d = change_default_args(bias=False)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=False)(
                nn.ConvTranspose2d)
        else:
            BatchNorm2d = Empty
            Conv2d = change_default_args(bias=True)(nn.Conv2d)
            ConvTranspose2d = change_default_args(bias=True)(
                nn.ConvTranspose2d)

        # note that when stride > 1, conv2d with same padding isn't
        # equal to pad-conv2d. we should use pad-conv2d.
        block2_input_filters = num_filters[0]
        self.block1 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                num_input_features, num_filters[0], 3,
                stride=layer_strides[0]),
            BatchNorm2d(num_filters[0]),
            nn.ReLU(),
        )
        for i in range(layer_nums[0]):
            self.block1.add(
                Conv2d(num_filters[0], num_filters[0], 3, padding=1))
            self.block1.add(BatchNorm2d(num_filters[0]))
            self.block1.add(nn.ReLU())
        self.deconv1 = Sequential(
            ConvTranspose2d(
                num_filters[0],
                num_upsample_filters[0],
                upsample_strides[0],
                stride=upsample_strides[0]),
            BatchNorm2d(num_upsample_filters[0]),
            nn.ReLU(),
        )
        self.block2 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(
                block2_input_filters,
                num_filters[1],
                3,
                stride=layer_strides[1]),
            BatchNorm2d(num_filters[1]),
            nn.ReLU(),
        )
        for i in range(layer_nums[1]):
            self.block2.add(
                Conv2d(num_filters[1], num_filters[1], 3, padding=1))
            self.block2.add(BatchNorm2d(num_filters[1]))
            self.block2.add(nn.ReLU())
        self.deconv2 = Sequential(
            ConvTranspose2d(
                num_filters[1],
                num_upsample_filters[1],
                upsample_strides[1],
                stride=upsample_strides[1]),
            BatchNorm2d(num_upsample_filters[1]),
            nn.ReLU(),
        )
        self.block3 = Sequential(
            nn.ZeroPad2d(1),
            Conv2d(num_filters[1], num_filters[2], 3, stride=layer_strides[2]),
            BatchNorm2d(num_filters[2]),
            nn.ReLU(),
        )
        for i in range(layer_nums[2]):
            self.block3.add(
                Conv2d(num_filters[2], num_filters[2], 3, padding=1))
            self.block3.add(BatchNorm2d(num_filters[2]))
            self.block3.add(nn.ReLU())
        self.deconv3 = Sequential(
            ConvTranspose2d(
                num_filters[2],
                num_upsample_filters[2],
                upsample_strides[2],
                stride=upsample_strides[2]),
            BatchNorm2d(num_upsample_filters[2]),
            nn.ReLU(),
        )
        if encode_background_as_zeros:
            num_cls = num_anchor_per_loc * num_class
        else:
            num_cls = num_anchor_per_loc * (num_class + 1)
        self.conv_cls = nn.Conv2d(sum(num_upsample_filters), num_cls, 1)
        self.conv_box = nn.Conv2d(
            sum(num_upsample_filters), num_anchor_per_loc * box_code_size, 1)
        if use_direction_classifier:
            self.conv_dir_cls = nn.Conv2d(
                sum(num_upsample_filters),
                num_anchor_per_loc * num_direction_bins, 1)

        if self._use_rc_net:
            self.conv_rc = nn.Conv2d(
                sum(num_upsample_filters), num_anchor_per_loc * box_code_size,
                1)

    def forward(self, x):
        # t = time.time()
        # torch.cuda.synchronize()

        x = self.block1(x)
        up1 = self.deconv1(x)
        x = self.block2(x)
        up2 = self.deconv2(x)
        x = self.block3(x)
        up3 = self.deconv3(x)
        x = torch.cat([up1, up2, up3], dim=1)
        box_preds = self.conv_box(x)
        cls_preds = self.conv_cls(x)

        # [N, C, y(H), x(W)]
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()
        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()
        ret_dict = {
            "box_preds": box_preds,
            "cls_preds": cls_preds,
        }
        if self._use_direction_classifier:
            dir_cls_preds = self.conv_dir_cls(x)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["dir_cls_preds"] = dir_cls_preds
        if self._use_rc_net:
            rc_preds = self.conv_rc(x)
            rc_preds = rc_preds.permute(0, 2, 3, 1).contiguous()
            ret_dict["rc_preds"] = rc_preds
        # torch.cuda.synchronize()
        # print("rpn forward time", time.time() - t)

        return ret_dict


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(2)
    # pytorch_device = torch.device('cuda:0')

    net = ParameterPredictionNet([0])
    print(net)
    with open('./net_1.txt', 'w') as f:
        f.write('%s' % net)