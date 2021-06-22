import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np

# sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '../'))
from network.matchnet import angle
from network.tools import gather_neighbour, gather_neighbour_V2


def FC(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Linear(channels[i - 1], channels[i], bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
                # layers.append(nn.InstanceNorm1d(channels[i]))

                # if channels[i] < 32: num_groups = 4
                # elif channels[i] < 256: num_groups = 8
                # else: num_groups = 16
                # layers.append(nn.GroupNorm(num_groups, channels[i]))

            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            # layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


def MLP(channels: list, do_bn=True, full=False):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))

        if i < (n-1) or full:
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
                # layers.append(nn.InstanceNorm1d(channels[i]))

                # if channels[i] < 16: num_groups = 1
                # elif channels[i] < 32: num_groups = 4
                # elif channels[i] < 256: num_groups = 8
                # else: num_groups = 16
                # layers.append(nn.GroupNorm(num_groups, channels[i]))

            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            # layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)


class MLP2D(nn.Sequential):
    def __init__(
        self,
        in_size: int,
        out_size: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        conv=nn.Conv2d,
        normalization: str = 'group',
        activation: str = 'leakyrelu',
        name: str = ''
        ):
        super().__init__()
        #####################################
        # bias = bias and (not bn)  # TODO
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        nn.init.kaiming_normal_(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)
        self.add_module(name + 'conv', conv_unit)

        #####################################
        if normalization is not None:
            if normalization == 'bn':
                norm_unit = nn.BatchNorm2d(out_size)
            elif normalization == 'group':
                num_groups = 8 if out_size >= 64 else 4
                norm_unit = nn.GroupNorm(num_groups, out_size)
            elif normalization == 'instance':
                norm_unit = nn.InstanceNorm2d(out_size)

            self.add_module(name + 'norm', norm_unit)

        #####################################
        if activation is not None:
            if activation == 'relu':
                act_unit = nn.ReLU(inplace=True)
            elif activation == 'leakyrelu':
                act_unit = nn.LeakyReLU(negative_slope=0.2, inplace=True)

            self.add_module(name + 'activation', act_unit)


def feat_grouping(xyz, normals, neigh_idx):
    """ group for xyz, xyz_norm and ppf features
        :param xyz: XYZ coordinates of the points [B, N, 3]
        :param normals: Corresponding normals for the points (required for ppf computation) [B, N, 3]
        :param neigh_idx: point index [B, N, nsample]

        :return: concated feature [B, N, nsample, 10]
    """
    nsample = neigh_idx.shape[-1]

    grouped_xyz = gather_neighbour(xyz, neigh_idx)   # [B, N, nsample, 3]
    di = grouped_xyz - xyz.unsqueeze(2)              # [B, N, nsample, 3]
    ni = gather_neighbour(normals, neigh_idx)        # [B, N, nsample, 3]  normal of neighbor points
    nr = normals.unsqueeze(2)                        # [B, N, 1, 3] normal of center points

    # ppf feature
    nr_d = angle(nr, di)
    ni_d = angle(ni, di)
    nr_ni = angle(nr, ni)
    d_norm = torch.norm(di, dim=-1)
    ppf_feat = torch.stack([nr_d, ni_d, nr_ni, d_norm], dim=-1)  # [B, N, nsample, 4]  # TODO  no d_norm

    xyz_norm = di                        # [B, N, nsample, 3]
    concat = [xyz.unsqueeze(2).expand(-1, -1, nsample, -1),
              xyz_norm,
              ppf_feat]
    concat = torch.cat(concat, dim=-1)   # [B, N, nsample, 10]
    return concat


class Att_pooling(nn.Module):
    def __init__(self, d_in, d_out):
        """ Attentive Pooling
        """
        super().__init__()
        self.fc = nn.Conv2d(d_in, d_in, (1, 1), bias=False)
        self.mlp = MLP2D(d_in, d_out)

    def forward(self, feature_set):
        """ Computing Attention Scores and Weighted Summation
            :param feature_set: [B, C, N, nsample]
        """
        att_scores = F.softmax(self.fc(feature_set), dim=3)  # [B, C, N, nsample]

        f_agg = feature_set * att_scores
        f_agg = torch.sum(f_agg, dim=3, keepdim=True)        # [B, C, N, 1]
        f_agg = self.mlp(f_agg)                              # [B, C, N, 1]
        return f_agg


class Building_block(nn.Module):
    def __init__(self, d_out):
        """ Local Spatial Encoding
            d_in -> d_out//2 -> d_out
        """
        super().__init__()
        self.d_in = 10  # 6/10
        self.mlp1 = MLP2D(self.d_in, d_out//2)
        self.att_pooling_1 = Att_pooling(d_out, d_out//2)

        self.mlp2 = MLP2D(d_out//2, d_out//2)
        self.att_pooling_2 = Att_pooling(d_out, d_out)

    def forward(self, xyz, feature, neigh_idx):
        """ Relative Point Position Encoding and Point Feature Augmentation
        :param xyz: [B, 3, N]
        :param feature: [B, C, N, 1]
        :param neigh_idx: [B, N, nsample]
        """
        f_xyz = self.relative_pos_encoding(xyz, neigh_idx)  # [B, 6, N, nsample]
        assert f_xyz.shape[1] == self.d_in

        ########## block 1 ##########
        f_xyz = self.mlp1(f_xyz)                            # [B, d_out//2, N, nsample]
        f_neighbours = gather_neighbour_V2(feature.squeeze(-1), neigh_idx)   # [B, d_out//2, N, nsample]

        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)  # [B, d_out, N, nsample]
        f_pc_agg = self.att_pooling_1(f_concat)             # [B, d_out//2, N, 1]

        ########## block 2 ##########
        f_xyz = self.mlp2(f_xyz)                            # [B, d_out//2, N, nsample]
        f_neighbours = gather_neighbour_V2(f_pc_agg.squeeze(-1), neigh_idx)  # [B, d_out//2, N, nsample]

        f_concat = torch.cat([f_neighbours, f_xyz], dim=1)  # [B, d_out, N, nsample]
        f_pc_agg = self.att_pooling_2(f_concat)             # [B, d_out, N, 1]
        return f_pc_agg

    @staticmethod
    def relative_pos_encoding(xyz, neigh_idx):
        """ Finding KNN neighbouring points using index, then Relative Point Position Encoding
        :param xyz: [B, 3, N]
        :param neigh_idx: [B, N, nsample]

        :return: [B, 6, N, nsample]
        """
        nsample = neigh_idx.shape[-1]
        neighbor_xyz = gather_neighbour_V2(xyz, neigh_idx)                # [B, 3, N, nsample]
        xyz_tile = xyz.unsqueeze(-1).repeat(1, 1, 1, nsample)             # [B, 3, N, nsample]
        relative_xyz = neighbor_xyz - xyz_tile                            # [B, 3, N, nsample]
        # relative_feat = torch.cat([relative_xyz, xyz_tile], dim=1)        # [B, 6, N, nsample]
        relative_dis = torch.sqrt(torch.sum(torch.pow(relative_xyz, 2), dim=1, keepdim=True))
        relative_feat = torch.cat([relative_dis, relative_xyz, xyz_tile, neighbor_xyz], dim=1)
        return relative_feat


class Dilated_res_block(nn.Module):
    def __init__(self, d_in, d_out):
        """ Dilated Residual Block
        """
        super().__init__()
        self.mlp1 = MLP2D(d_in, d_out//2)
        self.lfa = Building_block(d_out)
        self.mlp2 = MLP2D(d_out, d_out*2, activation=None)
        self.mlp_skip = MLP2D(d_in, d_out*2, activation=None)

    def forward(self, feature, xyz, neigh_idx):
        f_pc = self.mlp1(feature)                # [B, d_out//2, N, 1]
        f_pc = self.lfa(xyz, f_pc, neigh_idx)    # [B, d_out, N, 1]
        f_pc = self.mlp2(f_pc)                   # [B, 2*d_out, N, 1]
        shortcut = self.mlp_skip(feature)        # [B, 2*d_out, N, 1]
        return F.leaky_relu(f_pc + shortcut, negative_slope=0.2)  # [B, 2*d_out, N, 1]


class RandLA(nn.Module):
    """ Network architecture
    """
    def __init__(self, args, num_classes=19):
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)

        self.d_feat_in  = args.feat_len
        self.d_mid_out  = args.d_out                       # feature dimension of each layer
        self.d_feat_out = args.out_feat_dim                # output feature dimension
        self.num_points = args.num_points                  # Number of input points
        self.num_knn    = args.num_knn
        self.sub_sampling_ratio = args.sub_sampling_ratio  # sampling ratio of random sampling at each layer
        self.use_ppf    = args.use_ppf
        self.num_layers = len(self.d_mid_out)
        self.num_classes = num_classes                     # Number of valid classes
        dim_temp = 8

        if self.use_ppf:
            self._logger.info('Using PPF features.')
            self.d_feat_in = 10
            dim_temp = 12

        ####### Pre-FC layer #########
        self.mlp_pre = MLP2D(self.d_feat_in, dim_temp)

        #######  Encoder  #########
        self.dilated_res_blocks = nn.ModuleList()
        for i in range(self.num_layers):
            d_out = self.d_mid_out[i]
            self.dilated_res_blocks.append(Dilated_res_block(dim_temp, d_out))
            dim_temp = 2 * d_out

        ####### Mid-layer #########
        d_out = dim_temp
        self.mlp_mid = MLP2D(dim_temp, d_out)

        #######  Decoder  #########
        self.decoder_blocks = nn.ModuleList()
        for j in range(self.num_layers):
            if j < self.num_layers - 1:
                dim_temp = d_out + 2 * self.d_mid_out[-j-2]
                d_out = 2 * self.d_mid_out[-j-2]
            else:
                dim_temp = 4 * self.d_mid_out[0]
                d_out = 2 * self.d_mid_out[0]
            self.decoder_blocks.append(MLP2D(dim_temp, d_out))

        self.mlp_out = nn.Conv2d(d_out, self.d_feat_out, kernel_size=(1,1), bias=False)

        layers = [self.d_feat_out, 64, 32, self.num_classes]
        self.fc_label = MLP(layers, do_bn=True, full=False)
        self.dropout = nn.Dropout(0.5)

    def compute_index(self, num_points):
        # idex number for each sub-sampled points
        idx = [0,]
        index = 0
        div_num = 1
        for i in range(self.num_layers):
            if index == 0:
                index = num_points
            else:
                div_num *= self.sub_sampling_ratio[i - 1]
                index += num_points // div_num
            idx.append(index)
        self.idx = np.array(idx)

        # idex number of random_sample
        sample = [0,]
        index = 0
        div_num = 1
        for i in range(self.num_layers):
            div_num *= self.sub_sampling_ratio[i]
            index += num_points // div_num
            sample.append(index)
        self.sample = np.array(sample)

    def forward(self, features, xyz, neigh_idx, sub_idx, interp_idx):
        """
        :param features:  [B, n, C], including [x, y, z, ...]
        :param xyz:       [B, n + 1/4*n + 1/16*n + 1/64*n + 1/128*n + ..., 3], including [x, y, z]
        :param neigh_idx: [B, n + 1/4*n + 1/16*n + 1/64*n + 1/128*n + ..., nsample]
        :param sub_idx:   [B, 1/4*n + 1/16*n + 1/64*n + 1/128*n + 1/256*n + ..., nsample]
        :param interp_idx:[B, n + 1/4*n + 1/16*n + 1/64*n + 1/128*n + ..., 1]
        """
        num_points = features.shape[1]
        self.compute_index(num_points)

        xyz = xyz.permute(0, 2, 1).contiguous()    # [B, 3, n1+n2+...]

        if self.use_ppf:
            assert features.size()[2] >= 6, "feature dimension error"
            features = feat_grouping(features[:, :, :3], features[:, :, 3:6],
                                    neigh_idx[:, self.idx[0]:self.idx[1], :])  # [B, N, nsample, 10]
            features = features.permute(0, 3, 2, 1).contiguous()     # [B, 10, nsample, N]

            features = self.mlp_pre(features)                        # [B, C, nsample, N]
            features = torch.mean(features, dim=2, keepdims=False)   # [B, C, N]
            features = features.unsqueeze(dim=3)                     # [B, C, N, 1]
        else:
            features = features.permute(0, 2, 1).contiguous()        # [B, C, N]
            features = self.mlp_pre(features.unsqueeze(dim=3))       # [B, C, N, 1]

        ############################ Encoder ############################
        f_encoder_list = []
        for i in range(self.num_layers):
            idx_1, idx_2 = self.idx[i], self.idx[i+1]
            f_encoder_i = self.dilated_res_blocks[i](features, xyz[:, :, idx_1:idx_2],
                                                    neigh_idx[:, idx_1:idx_2, :])  # [B, C, N, 1]

            idx_1, idx_2 = self.sample[i], self.sample[i+1]
            f_sampled_i = self.random_sample(f_encoder_i, sub_idx[:, idx_1:idx_2, :])

            features = f_sampled_i        # [B, C, N, 1]
            if i == 0:
                f_encoder_list.append(f_encoder_i)
            f_encoder_list.append(f_sampled_i)
        ############################ Encoder ############################

        features = self.mlp_mid(f_encoder_list[-1])  # [B, C, N, 1]

        ############################ Decoder ############################
        for j in range(self.num_layers):
            idx_1, idx_2 = self.idx[self.num_layers-j-1], self.idx[self.num_layers-j]
            f_interp_i = self.nearest_interpolation(features, interp_idx[:, idx_1:idx_2, :])
            features = self.decoder_blocks[j](torch.cat([f_encoder_list[-j - 2], f_interp_i], dim=1))
        ############################ Decoder ############################
        del f_encoder_list

        features = self.mlp_out(features)    # [B, C, N, 1]
        features = features.squeeze(3)       # [B, C, N]

        logits = self.dropout(features)
        logits = self.fc_label(logits)       # [B, num_class, N]

        xyz = xyz[:, :, 0:num_points]        # [B, 3, N], all points of the raw input
        assert xyz.shape[-1] == features.shape[-1]

        return features, xyz, logits         # [B, C, N], [B, num_class, N]

    @staticmethod
    def random_sample(feature, pool_idx):
        """
        :param feature: [B, C, N, 1] input features matrix
        :param pool_idx: [B, N', k] (N' is the selected position after pooling, N' < N)

        :return: pool_features = [B, C, N', 1] pooled features matrix
        """
        dims = feature.shape[1]
        feature = feature.squeeze(dim=3)   # [B, C, N]
        batch_size, _, num_neigh = pool_idx.shape

        pool_idx = pool_idx.reshape(batch_size, -1).unsqueeze(1).repeat(1, dims, 1)  # [B, C, N'*k]
        pool_features = torch.gather(feature, 2, pool_idx)                           # [B, C, N'*k]
        pool_features = pool_features.reshape(batch_size, dims, -1, num_neigh)       # [B, C, N', k]

        pool_features = pool_features.max(dim=3, keepdim=True)[0]                    # [B, C, N', 1]
        return pool_features

    @staticmethod
    def nearest_interpolation(feature, interp_idx):
        """
        :param feature: [B, C, N, 1] input features matrix
        :param interp_idx: [B, N', 1] nearest neighbour index (N' is the selected position before pooling, N' > N)

        :return: [B, C, N', 1] interpolated features matrix
        """
        dims = feature.shape[1]
        feature = feature.squeeze(dim=3)  # [B, C, N]
        batch_size, up_num_points, _ = interp_idx.shape

        interp_idx = interp_idx.reshape(batch_size, up_num_points).unsqueeze(1).repeat(1, dims, 1)  # [B, C, N']
        interpolated_features = torch.gather(feature, 2, interp_idx)    # [B, C, N']
        interpolated_features = interpolated_features.unsqueeze(3)      # [B, C, N', 1]
        return interpolated_features


if __name__ == '__main__':
    # os.environ['CUDA_VISIBLE_DEVICES'] = str(2)
    # pytorch_device = torch.device('cuda:0')

    net = RandLA(1024, 6, 64)
    # print(net)
    with open('./net.txt', 'w') as f:
        f.write('%s' % net)

