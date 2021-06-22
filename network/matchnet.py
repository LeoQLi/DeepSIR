import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

_EPS = 1e-12


############################################################ RPMNet

def angle(v1: torch.Tensor, v2: torch.Tensor):
    """Compute angle between 2 vectors

    For robustness, we use the same formulation as in PPFNet, i.e.
        angle(v1, v2) = atan2(cross(v1, v2), dot(v1, v2)).
    This handles the case where one of the vectors is 0.0, since torch.atan2(0.0, 0.0)=0.0

    Args:
        v1: (B, *, 3)
        v2: (B, *, 3)

    Returns:
    """
    cross_prod = torch.stack([v1[..., 1] * v2[..., 2] - v1[..., 2] * v2[..., 1],
                              v1[..., 2] * v2[..., 0] - v1[..., 0] * v2[..., 2],
                              v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]], dim=-1)
    cross_prod_norm = torch.norm(cross_prod, dim=-1)
    dot_prod = torch.sum(v1 * v2, dim=-1)

    return torch.atan2(cross_prod_norm, dot_prod)


def angle_difference(src, dst):
    """Calculate angle between each pair of vectors.
    Assumes points are l2-normalized to unit length.

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    dist = torch.matmul(src, dst.permute(0, 2, 1).contiguous())
    dist = torch.acos(dist)

    return dist


def square_distance(src, dst):
    """Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
             = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Args:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1).contiguous())
    dist += torch.sum(src ** 2, dim=-1)[:, :, None]
    dist += torch.sum(dst ** 2, dim=-1)[:, None, :]
    return dist


def match_features(feat_src, feat_ref, metric='l2'):
    """ Compute pairwise distance between features

    Args:
        feat_src: (B, J, C)
        feat_ref: (B, K, C)
        metric: either 'angle' or 'l2' (squared euclidean)

    Returns:
        Matching matrix (B, J, K). i'th row describes how well the i'th point
            in the src agrees with every point in the ref.
    """
    assert feat_src.shape[-1] == feat_ref.shape[-1]

    if metric == 'l2':
        dist_matrix = square_distance(feat_src, feat_ref)
    elif metric == 'angle':
        feat_src_norm = feat_src / (torch.norm(feat_src, dim=-1, keepdim=True) + _EPS)
        feat_ref_norm = feat_ref / (torch.norm(feat_ref, dim=-1, keepdim=True) + _EPS)

        dist_matrix = angle_difference(feat_src_norm, feat_ref_norm)
    else:
        raise NotImplementedError

    return dist_matrix


def square_distance_V2(src, dst):
    """Calculate Euclid distance between each two points.
        src^T * dst = xn * xm + yn * ym + zn * zm；
        sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
        sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
        dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
             = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Args:
        src: source points, [B, C, N]
        dst: target points, [B, C, M]
    Returns:
        dist: per-point square distance, [B, N, M]
    """
    dist = -2 * torch.matmul(src.permute(0, 2, 1).contiguous(), dst)   # [B, N, M]
    dist += torch.sum(src ** 2, dim=1)[:, :, None]                     # [B, N, 1]
    dist += torch.sum(dst ** 2, dim=1)[:, None, :]                     # [B, 1, M]
    return dist


def match_features_V2(feat_src, feat_ref, metric='l2'):
    """ Compute pairwise distance between features

    Args:
        feat_src: (B, C, J)
        feat_ref: (B, C, K)
        metric: either 'angle' or 'l2' (squared euclidean)

    Returns:
        Matching matrix (B, J, K). i'th row describes how well the i'th point
            in the src agrees with every point in the ref.
    """
    assert feat_src.shape[1] == feat_ref.shape[1]

    if metric == 'l2':
        dist_matrix = square_distance_V2(feat_src, feat_ref)
    elif metric == 'euclidean':
        dist_matrix = square_distance_V2(feat_src, feat_ref)
        dist_matrix = torch.sqrt(dist_matrix + _EPS)
    elif metric == 'angle':
        feat_src = feat_src / (torch.norm(feat_src, dim=1, keepdim=True) + _EPS)
        feat_ref = feat_ref / (torch.norm(feat_ref, dim=1, keepdim=True) + _EPS)

        dist_matrix = torch.matmul(feat_src.permute(0, 2, 1).contiguous(), feat_ref)
        dist_matrix = torch.acos(dist_matrix)
    else:
        raise NotImplementedError

    return dist_matrix


def feat_dist(feat_src, feat_ref, metric='sqeuclidean'):
    """ Compute pairwise distance between features, Similar to scipy.spatial's cdist, but symbolic.

    The currently supported metrics can be listed as `cdist.supported_metrics` and are:
        - 'euclidean', although with a fudge-factor epsilon.
        - 'sqeuclidean', the squared euclidean.
        - 'cityblock', the manhattan or L1 distance.
        - 'angle', arccosine angle distance.

    Args:
        feat_src: (B, C, J)
        feat_ref: (B, C, K)
        metric (string): Which distance metric to use, see notes.

    Returns:
        Matching matrix (B, J, K). i'th row describes how well the i'th point
            in the src agrees with every point in the ref.

    Note:
        When a square root is taken (such as in the Euclidean case), a small
        epsilon is added because the gradient of the square-root at zero is
        undefined. Thus, it will never return exact zero in these cases.
    """
    assert feat_src.shape[1] == feat_ref.shape[1]

    # [B, C, J, K] = [B, C, J, 1] - [B, C, 1, K]
    diffs = feat_src[:, :, :, None] - feat_ref[:, :, None, :]

    if metric == 'sqeuclidean':
        # return torch.dist(feat_src[:, :, :, None], feat_ref[:, :, None, :], 2)
        # return F.pairwise_distance(feat_src[:, :, :, None], feat_ref[:, :, None, :], p=2).pow(2)
        return torch.sum(diffs ** 2, dim=1)
    elif metric == 'euclidean':
        return torch.sqrt(torch.sum(diffs ** 2, dim=1) + _EPS)
    elif metric == 'cityblock':
        return torch.sum(torch.abs(diffs), dim=1)
    elif metric == 'angle':
        feat_src = feat_src / (torch.norm(feat_src, dim=1, keepdim=True) + _EPS)
        feat_ref = feat_ref / (torch.norm(feat_ref, dim=1, keepdim=True) + _EPS)

        dist = torch.matmul(feat_src.permute(0, 2, 1).contiguous(), feat_ref)
        dist = torch.acos(dist)

        return dist
    else:
        raise NotImplementedError('The following metric is not implemented: {}'.format(metric))


def compute_affinity(beta, feat_distance, alpha=0.5):
    """ Compute logarithm of Initial match matrix values, i.e. log(m_jk). Equ.5 of the paper.

    Args:
        beta (B,), alpha (B,), feat_distance (B, J, K)

    Returns:
        hybrid_affinity (B, J, K)
    """
    if isinstance(alpha, float):
        hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha)
    else:
        hybrid_affinity = -beta[:, None, None] * (feat_distance - alpha[:, None, None])
    return hybrid_affinity


def sinkhorn(log_alpha, n_iters=5, slack=True, eps=-1):
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)

    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """
    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                    log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                    log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                    log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return log_alpha


def weighted_procrustes(X, Y, w):
    """
    src (torch.Tensor): (B, M, 3) source points
    tgt (torch.Tensor): (B, N, 3) target points
    weights (torch.Tensor): (B, N)
    """
    # https://ieeexplore.ieee.org/document/88573
    # "Least-squares estimation of transformation parameters between two point patterns"
    assert X.shape == Y.shape

    w_norm = w / (torch.abs(w).sum(dim=1) + _EPS)     # [B, N]
    # centroid
    mux = (w_norm * X).sum(dim=0, keepdim=True)       # [B, N]*[B, M, 3]=[B, 1, 3]
    muy = (w_norm * Y).sum(dim=0, keepdim=True)       # [B, M, 3]

    # Use CPU for small arrays
    Sxy = (Y - muy).t().mm(w_norm * (X - mux)).cpu().double()
    U, D, V = Sxy.svd()

    S = torch.eye(3).double()
    if U.det() * V.det() < 0:
        S[-1, -1] = -1

    R = U.mm(S.mm(V.t())).float()
    t = (muy.cpu().squeeze() - R.mm(mux.cpu().t()).squeeze()).float()
    return R, t


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

        pre_layers = [4, 64, 64, 64, 128, 1024]
        self.prepool = MLP(pre_layers, do_bn=True, full=True)

        self.pooling = nn.AdaptiveMaxPool1d(1)

        post_layers = [1024, 512, 256, 2+np.prod(weights_dim)]
        self.postpool = FC(post_layers)

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
        # pooled = torch.flatten(self.pooling(prepool_feat), start_dim=-2)  # [B, 1024, 1] => [B, 1024]
        pooled = torch.squeeze(self.pooling(prepool_feat), dim=-1)
        raw_weights = self.postpool(pooled)    # [B, 2]

        beta  = F.softplus(raw_weights[:, 0])
        alpha = F.softplus(raw_weights[:, 1])
        return beta, alpha


class ParameterPredictionNetConstant(nn.Module):
    def __init__(self, weights_dim):
        """ Parameter Prediction Network with single alpha/beta as parameter.
        """
        super().__init__()
        self._logger = logging.getLogger(self.__class__.__name__)

        self.anneal_weights = nn.Parameter(torch.zeros(2 + np.prod(weights_dim)))
        self.weights_dim = weights_dim
        self._logger.info('Predicting weights with dim {}.'.format(self.weights_dim))

    def forward(self, x1, x2):
        """ Returns beta, gating_weights
        """
        batch_size = x1.shape[0]
        raw_weights = self.anneal_weights
        beta = F.softplus(raw_weights[0].expand(batch_size))
        alpha = F.softplus(raw_weights[1].expand(batch_size))

        return beta, alpha


############################################################ PRNet

class KeyPointNet(nn.Module):
    def __init__(self, num_keypoints):
        super(KeyPointNet, self).__init__()
        self.num_keypoints = num_keypoints

    def forward(self, *input):
        src = input[0]
        tgt = input[1]
        src_embedding = input[2]
        tgt_embedding = input[3]
        batch_size, num_dims, num_points = src_embedding.size()
        src_norm = torch.norm(src_embedding, dim=1, keepdim=True)
        tgt_norm = torch.norm(tgt_embedding, dim=1, keepdim=True)
        src_topk_idx = torch.topk(src_norm, k=self.num_keypoints, dim=2, sorted=False)[1]
        tgt_topk_idx = torch.topk(tgt_norm, k=self.num_keypoints, dim=2, sorted=False)[1]
        src_keypoints_idx = src_topk_idx.repeat(1, 3, 1)
        tgt_keypoints_idx = tgt_topk_idx.repeat(1, 3, 1)
        src_embedding_idx = src_topk_idx.repeat(1, num_dims, 1)
        tgt_embedding_idx = tgt_topk_idx.repeat(1, num_dims, 1)

        src_keypoints = torch.gather(src, dim=2, index=src_keypoints_idx)
        tgt_keypoints = torch.gather(tgt, dim=2, index=tgt_keypoints_idx)

        src_embedding = torch.gather(src_embedding, dim=2, index=src_embedding_idx)
        tgt_embedding = torch.gather(tgt_embedding, dim=2, index=tgt_embedding_idx)
        return src_keypoints, tgt_keypoints, src_embedding, tgt_embedding


class TemperatureNet(nn.Module):
    def __init__(self, dims, temp_factor):
        super(TemperatureNet, self).__init__()
        self.temp_factor = temp_factor
        self.nn = nn.Sequential(nn.Linear(dims, 128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                nn.Linear(128, 128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                nn.Linear(128, 128),
                                nn.BatchNorm1d(128),
                                nn.ReLU(),
                                nn.Linear(128, 1),
                                nn.ReLU())

    def forward(self, src_embedding, tgt_embedding):
        """
        src_embedding: [B, C, N]
        tgt_embedding: [B, C, N]
        """
        src_embedding = src_embedding.mean(dim=2)
        tgt_embedding = tgt_embedding.mean(dim=2)
        residual = torch.abs(src_embedding - tgt_embedding)
        temperature = torch.clamp(self.nn(residual), 1.0/self.temp_factor, 1.0*self.temp_factor)

        return temperature, residual


class SVDHead(nn.Module):
    def __init__(self,):
        super(SVDHead, self).__init__()
        self.cat_sampler = 'gumbel_softmax'
        self.my_iter = torch.ones(1)

    def forward(self, *input):
        """
        src_embedding: [B, C, N]
        tgt_embedding: [B, C, N]
        src: [B, C, N]
        tgt: [B, C, N]
        temperature: [B, 1]
        """
        src_embedding = input[0]
        tgt_embedding = input[1]
        src = input[2]
        tgt = input[3]
        temperature = input[4].view(batch_size, 1, 1)
        batch_size, num_dims, num_points = src.size()

        if self.cat_sampler == 'softmax':
            d_k = src_embedding.size(1)
            scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
            scores = torch.softmax(temperature*scores, dim=2)
        elif self.cat_sampler == 'gumbel_softmax':
            d_k = src_embedding.size(1)
            scores = torch.matmul(src_embedding.transpose(2, 1).contiguous(), tgt_embedding) / math.sqrt(d_k)
            scores = scores.view(batch_size*num_points, num_points)
            temperature = temperature.repeat(1, num_points, 1).view(-1, 1)
            scores = F.gumbel_softmax(scores, tau=temperature, hard=True)
            scores = scores.view(batch_size, num_points, num_points)
        else:
            raise Exception('not implemented')

        src_corr = torch.matmul(tgt, scores.transpose(2, 1).contiguous())
        src_centered = src - src.mean(dim=2, keepdim=True)
        src_corr_centered = src_corr - src_corr.mean(dim=2, keepdim=True)
        H = torch.matmul(src_centered, src_corr_centered.transpose(2, 1).contiguous()).cpu()

        R = []
        for i in range(src.size(0)):
            u, s, v = torch.svd(H[i])
            r = torch.matmul(v, u.transpose(1, 0)).contiguous()
            r_det = torch.det(r).item()
            diag = torch.from_numpy(np.array([[1.0, 0, 0],
                                              [0, 1.0, 0],
                                              [0, 0, r_det]]).astype('float32')).to(v.device)
            r = torch.matmul(torch.matmul(v, diag), u.transpose(1, 0)).contiguous()
            R.append(r)

        R = torch.stack(R, dim=0).cuda()

        t = torch.matmul(-R, src.mean(dim=2, keepdim=True)) + src_corr.mean(dim=2, keepdim=True)
        if self.training:
            self.my_iter += 1
        return R, t.view(batch_size, 3)


############################################################ DCP


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class EncoderDecoder(nn.Module):
    """ A standard Encoder-Decoder architecture. Base for this and many other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        """ Take in and process masked src and target sequences.
        """
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask))


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x-mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + sublayer(self.norm(x))


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Encoder(nn.Module):
    """ Generic N layer encoder with masking.
    """
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class Decoder(nn.Module):
    """ Generic N layer decoder with masking.
    """
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    # [B, h, N1, N2]=[B, h, N1, d_k]*[B, h, d_k, N2]
    scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask==0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.0):
        """ Take in model size and number of heads.
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """ Implements Figure 2
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2).contiguous()
             for l, x in zip(self.linears, (query, key, value))]     #  [B, h, N, d_k]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    """ Implements FFN equation.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.leaky_relu(self.w_1(x), negative_slope=0.2)))


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.n_heads    = args.n_heads
        self.N          = args.n_blocks
        self.dropout    = args.dropout
        self.n_ff_dims  = args.n_ff_dims
        self.n_emb_dims = args.n_emb_dims
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.n_heads, self.n_emb_dims)
        ff = PositionwiseFeedForward(self.n_emb_dims, self.n_ff_dims, self.dropout)
        self.model = EncoderDecoder(Encoder(EncoderLayer(self.n_emb_dims, c(attn), c(ff), self.dropout), self.N),
                                    Decoder(DecoderLayer(self.n_emb_dims, c(attn), c(attn), c(ff), self.dropout), self.N),
                                    nn.Sequential(),
                                    nn.Sequential(),
                                    nn.Sequential())

    def forward(self, *input):
        """
        input: tuple (src, tgt)
        """
        src = input[0]
        tgt = input[1]
        src = src.transpose(2, 1).contiguous()
        tgt = tgt.transpose(2, 1).contiguous()
        tgt_embedding = self.model(src, tgt, None, None).transpose(2, 1).contiguous()
        src_embedding = self.model(tgt, src, None, None).transpose(2, 1).contiguous()
        return src_embedding, tgt_embedding


############################################################ SuperGlue


def MLP(channels: list, do_bn=True):
    """ Multi-layer perceptron """
    n = len(channels)
    layers = []
    for i in range(1, n):
        layers.append(nn.Conv1d(channels[i - 1], channels[i], kernel_size=1, bias=True))
        if i < (n-1):
            if do_bn:
                layers.append(nn.BatchNorm1d(channels[i]))
                # layers.append(nn.InstanceNorm1d(channels[i]))
            layers.append(nn.LeakyReLU(negative_slope=0.2, inplace=True))
            # layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class NormLayer(nn.Module):
    def __init__(self, size, eps=1e-6):
        super(NormLayer, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(size))   # learnable parameter
        self.b_2 = nn.Parameter(torch.zeros(size))  # learnable parameter
        self.eps = eps
        self.size = size

    def forward(self, x):
        """
        x: [B, C, N]
        """
        assert x.shape[1] == self.size
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2.view(1, -1, 1) * (x - mean) / (std + self.eps) + self.b_2.view(1, -1, 1)


class KeypointEncoder(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs"""
    def __init__(self, layers, feature_dim):
        super().__init__()
        self.encoder = MLP(layers + [feature_dim])
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, scores):
        inputs = [kpts.transpose(1, 2), scores.unsqueeze(1)]
        return self.encoder(torch.cat(inputs, dim=1))


def attention_ein(query, key, value):
    """
    query: [B, C/head, head, N1]
    key:   [B, C/head, head, N2]
    value: [B, C/head, head, N2]
    """
    dim = query.shape[1]
    scores = torch.einsum('bdhn,bdhm->bhnm', query, key) / dim**.5       # [B, head, N1, N2]  bhnd*bhdm=bhnm
    prob = F.softmax(scores, dim=-1)                                     # [B, head, N1, N2]
    return torch.einsum('bhnm,bdhm->bdhn', prob, value), prob            # [B, C/head, head, N1]  bdhnm*bdhm=bdhn


class MultiHeadAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, feature_dim: int, num_layers=3):
        super().__init__()
        assert feature_dim % num_heads == 0
        self.dim = feature_dim // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv1d(feature_dim, feature_dim, kernel_size=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(num_layers)])   # 3 Conv1d

    def forward(self, query, key, value):
        """
        query: [B, C, N1]
        key:   [B, C, N2]
        value: [B, C, N2]
        """
        batch_dim = query.size(0)

        # 1) Do all the linear projections in batch from feature_dim => h x d_k
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]    # [B, C/head, head, N]

        # 2) Apply attention on all the projected vectors in batch.
        x, prob = attention_ein(query, key, value)                               # x: [B, C/head, head, N1]
        self.prob.append(prob)

        # 3) "Concat" using a view and apply a final linear.
        x = x.contiguous().view(batch_dim, self.dim*self.num_heads, -1)
        return self.merge(x)                                                     # [B, C, N1]


class AttentionalPropagation(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads, feature_dim)
        self.mlp = MLP([feature_dim*2, feature_dim*2, feature_dim])
        nn.init.constant_(self.mlp[-1].bias, 0.0)

    def forward(self, x, source):
        message = self.attn(x, source, source)
        return self.mlp(torch.cat([x, message], dim=1))   # TODO  x + mlp(norm(message))


class AttentionalGNN(nn.Module):
    def __init__(self, feature_dim: int, layer_names: list, num_heads=4):
        super().__init__()
        self.layers = nn.ModuleList([AttentionalPropagation(feature_dim, num_heads)
                                    for _ in range(len(layer_names))] )
        self.names = layer_names
        # self.norm = NormLayer(feature_dim)

    def forward(self, desc0, desc1):
        for layer, name in zip(self.layers, self.names):
            layer.attn.prob = []
            if name == 'self':
                src0, src1 = desc0, desc1
            elif name == 'cross':
                src0, src1 = desc1, desc0
            delta0, delta1 = layer(desc0, src0), layer(desc1, src1)

            # delta0, delta1 = self.norm(delta0), self.norm(delta1)      # TODO
            desc0, desc1 = (desc0 + delta0), (desc1 + delta1)
        return desc0, desc1


def log_sinkhorn_iterations(Z, log_mu, log_nu, iters: int):
    """ Perform Sinkhorn Normalization in Log-space for stability"""
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
    return Z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores, alpha, iters: int):
    """ Perform Differentiable Optimal Transport in Log-space for stability"""
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns = (m*one).to(scores), (n*one).to(scores)

    bins0 = alpha.expand(b, m, 1)
    bins1 = alpha.expand(b, 1, n)
    alpha = alpha.expand(b, 1, 1)

    couplings = torch.cat([torch.cat([scores, bins0], -1),
                           torch.cat([bins1, alpha], -1)], 1)

    norm = - (ms + ns).log()
    log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
    log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

    Z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    Z = Z - norm  # multiply probabilities by M+N
    return Z


def arange_like(x, dim: int):
    return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1


class SuperGlue(nn.Module):
    """SuperGlue feature matching middle-end

    Given two sets of keypoints and locations, we determine the
    correspondences by:
      1. Keypoint Encoding (normalization + visual feature and location fusion)
      2. Graph Neural Network with multiple self and cross-attention layers
      3. Final projection layer
      4. Optimal Transport Layer (a differentiable Hungarian matching algorithm)
      5. Thresholding matrix based on mutual exclusivity and a match_threshold

    The correspondence ids use -1 to indicate non-matching points.
    """
    default_config = {
        'descriptor_dim': 64,  # 128
        'keypoint_encoder': [4, 16, 32, 64],  # [32, 64, 128]
        'GNN_layers': ['self', 'cross'] * 3,  # 9
        'sinkhorn_iterations': 5,   # 20
        'match_threshold': 0.2, }

    def __init__(self):
        super().__init__()
        self.config = self.default_config

        self.kenc = KeypointEncoder(self.config['keypoint_encoder'], self.config['descriptor_dim'])

        self.gnn = AttentionalGNN(self.config['descriptor_dim'], self.config['GNN_layers'])

        self.final_proj = nn.Conv1d(self.config['descriptor_dim'], self.config['descriptor_dim'],
                                    kernel_size=1, bias=True)

        bin_score = nn.Parameter(torch.tensor(1.))
        self.register_parameter('bin_score', bin_score)

    def forward(self, data_src, data_ref):
        """ Run SuperGlue on a pair of keypoints and descriptors
        :param desc: [B, C, N]
        :param kpts: [B, N, 3]
        :param score: [B, N]
        """
        desc0, kpts0, score0 = data_src
        desc1, kpts1, score1 = data_ref

        # kpts0 = kpts0 - torch.mean(kpts0, dim=1, keepdim=True)
        # kpts1 = kpts1 - torch.mean(kpts1, dim=1, keepdim=True)

        # # Keypoint MLP encoder.
        # desc0 = desc0 + self.kenc(kpts0, score0)    # [B, C, N]
        # desc1 = desc1 + self.kenc(kpts1, score1)    # [B, C, N]

        # Multi-layer Transformer network.
        desc0, desc1 = self.gnn(desc0, desc1)       # [B, C, N]

        # Final MLP projection.
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)  # [B, C, N]

        mdesc0 = F.normalize(mdesc0, p=2, dim=1)
        mdesc1 = F.normalize(mdesc1, p=2, dim=1)  # TODO ???

        # Compute matching descriptor distance.
        scores = torch.einsum('bdn,bdm->bnm', mdesc0, mdesc1)
        scores = scores / self.config['descriptor_dim']**.5

        # Run the optimal transport.
        scores = log_optimal_transport(scores, self.bin_score,
                                        iters=self.config['sinkhorn_iterations'])

        scores = torch.exp(scores[:, :-1, :-1])


        return scores

        # Get the matches with score above "match_threshold".
        max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0 & (mscores0 > self.config['match_threshold'])
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1)) # use -1 for invalid match
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1)) # use -1 for invalid match

        # first sample of batch
        return {'matches0': indices0[0],
                'matches1': indices1[0],
                'matching_scores0': mscores0[0],
                'matching_scores1': mscores1[0] }



if __name__ == '__main__':
    import time
    device = torch.device('cuda:0')
    # device = torch.device('cpu')

    a = torch.randn(2, 1024, 64).to(device)
    b = torch.randn(2, 1050, 64).to(device)

    time0 = time.time()
    # c = cdist(a, b)
    # diffs = torch.unsqueeze(a, dim=2) - torch.unsqueeze(b, dim=1)
    # c = torch.sum(diffs ** 2, dim=-1)
    c = feat_dist(a.permute(0, 2, 1).contiguous(), b.permute(0, 2, 1).contiguous())
    print(time.time() - time0)

    time0 = time.time()
    d = square_distance(a, b)
    print(time.time() - time0)

    print(c.shape, d.shape)
    print(c[0, 7, 10].item())
    print(d[0, 7, 10].item())
    # print(c[3, 0, 30].item())
    # print(d[3, 0, 30].item())
    # print(torch.eq(c, d))
    print(torch.equal(c, d))
