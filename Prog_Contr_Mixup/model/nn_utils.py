import math
import torch
import numpy as np
from torch import nn
from torch.nn import BatchNorm1d
from torch.autograd import Function
from torch.autograd import Variable
import torch.nn.functional as F

def mysinkdist(M, lam, numItermax=1000, stopThr=1e-1, cuda = True):
    n,m = M.shape
    r = torch.ones(n, requires_grad=True).cuda() / n
    c = torch.ones(m, requires_grad=True).cuda() / m
    kinit = torch.exp(-M/lam)
    k = torch.diag(1./r).mm(kinit)
    u,v = r,c
    vprev = v*2
    while(torch.abs(v-vprev).sum()>stopThr):
        vprev = v
        v = c / (k.T.matmul(u)+0.001)
        u = r / (k.matmul(v)+0.001)
    p = torch.diag(u) @ k @ torch.diag(v)
    return p, torch.sum(p*M)
def batch_pairwise_squared_distances(x, y):
    '''                                                                                              
    Modified from https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/3         
    Input: x is a bxNxd matrix y is an optional bxMxd matirx                                                             
    Output: dist is a bxNxM matrix where dist[b,i,j] is the square norm between x[b,i,:] and y[b,j,:]
    i.e. dist[i,j] = ||x[b,i,:]-y[b,j,:]||^2                                                         
    '''                                                                                              
    x_norm = (x**2).sum(2).view(x.shape[0],x.shape[1],1)
    y_t = y.permute(0,2,1).contiguous()
    y_norm = (y**2).sum(2).view(y.shape[0],1,y.shape[1])
    dist = x_norm + y_norm - 2.0 * torch.bmm(x, y_t)
    dist[dist != dist] = 0 # replace nan values with 0
    return torch.clamp(dist, 0.0, np.inf)

def alignmix(A,Ap):
    # sinkhorn_dist = SinkhornSolver(epsilon=0.01)
    # M=torch.cdist(A,Ap)
    A_tilde=[]
    Ap_tilde=[]
    M = batch_pairwise_squared_distances(A.reshape(A.shape[0],A.shape[1],1),Ap.reshape(A.shape[0],A.shape[1],1))
    for i in range(A.shape[0]):
        R,_=mysinkdist(M[i],0.01,numItermax=100,cuda=True)
        A_tilde.append(torch.matmul(Ap[i],R))
        Ap_tilde.append(torch.matmul(A[i],R))

    return torch.stack(A_tilde), torch.stack(Ap_tilde)
def to_one_hot(inp,num_classes):
    device = inp.device
    y_onehot = torch.FloatTensor(inp.size(0), num_classes)
    y_onehot.zero_()
    y_onehot.scatter_(1, inp.long().unsqueeze(1).data.cpu(), 1)
    return Variable(y_onehot.to(device), requires_grad=False)

class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)


class GBN(torch.nn.Module):
    """
        Ghost Batch Normalization
        https://arxiv.org/abs/1705.08741
    """

    def __init__(self, input_dim, virtual_batch_size=128, momentum=0.01):
        super(GBN, self).__init__()

        self.input_dim = input_dim
        self.virtual_batch_size = virtual_batch_size
        self.bn = BatchNorm1d(self.input_dim, momentum=momentum)

    def forward(self, x):
        chunks = x.chunk(int(np.ceil(x.shape[0] / self.virtual_batch_size)), 0)
        res = [self.bn(x_) for x_ in chunks]

        return torch.cat(res, dim=0)

class EmbeddingGenerator(torch.nn.Module):
    """
        Classical embeddings generator
        adopted from https://github.com/dreamquark-ai/tabnet/
    """

    def __init__(self, input_dim, cat_dims, cat_idxs, cat_emb_dim=[]):
        """ This is an embedding module for an enite set of features
        Parameters
        ----------
        input_dim : int
            Number of features coming as input (number of columns)
        cat_dims : list of int
            Number of modalities for each categorial features
            If the list is empty, no embeddings will be done
        cat_idxs : list of int
            Positional index for each categorical features in inputs
        cat_emb_dim : int or list of int
            Embedding dimension for each categorical features
            If int, the same embdeding dimension will be used for all categorical features
        """
        super(EmbeddingGenerator, self).__init__()
        if cat_dims == [] or cat_idxs == []:
            self.skip_embedding = True
            self.post_embed_dim = input_dim
            return

        # heuristic
        if (len(cat_emb_dim) == 0):
            # use heuristic
            cat_emb_dim = [min(600, round(1.6 * n_cats ** .56)) for n_cats in cat_dims]

        self.skip_embedding = False
        if isinstance(cat_emb_dim, int):
            self.cat_emb_dims = [cat_emb_dim]*len(cat_idxs)
        else:
            self.cat_emb_dims = cat_emb_dim

        # check that all embeddings are provided
        if len(self.cat_emb_dims) != len(cat_dims):
            msg = """ cat_emb_dim and cat_dims must be lists of same length, got {len(self.cat_emb_dims)}
                      and {len(cat_dims)}"""
            raise ValueError(msg)
        self.post_embed_dim = int(input_dim + np.sum(self.cat_emb_dims) - len(self.cat_emb_dims))

        self.embeddings = torch.nn.ModuleList()

        # Sort dims by cat_idx
        sorted_idxs = np.argsort(cat_idxs)
        cat_dims = [cat_dims[i] for i in sorted_idxs]
        self.cat_emb_dims = [self.cat_emb_dims[i] for i in sorted_idxs]

        for cat_dim, emb_dim in zip(cat_dims, self.cat_emb_dims):
            self.embeddings.append(torch.nn.Embedding(cat_dim, int(emb_dim)))
        # record continuous indices
        self.continuous_idx = torch.ones(input_dim, dtype=torch.bool)
        self.continuous_idx[cat_idxs] = 0

    def forward(self, x):
        """
        Apply embdeddings to inputs
        Inputs should be (batch_size, input_dim)
        Outputs will be of size (batch_size, self.post_embed_dim)
        """
        if self.skip_embedding:
            # no embeddings required
            return x
        cols = []
        cat_feat_counter = 0
        for feat_init_idx, is_continuous in enumerate(self.continuous_idx):
            # Enumerate through continuous idx boolean mask to apply embeddings
            if is_continuous:
                cols.append(x[:, feat_init_idx].float().view(-1, 1))
            else:
                cols.append(self.embeddings[cat_feat_counter](x[:, feat_init_idx].long()))
                cat_feat_counter += 1
        # concat
        post_embeddings = torch.cat(cols, dim=1)
        return post_embeddings

def get_lambda(alpha=1.0, dist='alpha', n=1):
    '''Return lambda'''
    if alpha > 0.:
        if dist == 'alpha':
            lam = np.random.beta(alpha, alpha, size=n)
        elif dist == 'uniform':
            lam  = np.random.uniform(0.0, alpha, size=n)
    else:
        lam = 1.0
    return lam


def mixup_process(out, target_reweighted, lam):
    indices = np.random.permutation(out.size(0))
    out = out*lam + out[indices]*(1-lam)
    target_shuffled_onehot = target_reweighted[indices]
    target_reweighted = target_reweighted * lam + target_shuffled_onehot * (1 - lam)
    return out, target_reweighted, indices


def mixup_class(out, labels, lam):
    yk = torch.unique(labels)
    if (len(lam) == len(yk)): # - class wise lamda
        yk = list(zip(lam, yk))

    new_zs = []
    idxs = []
    perms = []
    lam_ = lam
    for y in yk: # TODO: how to parallelize this
        if type(y) == tuple:
            lam_, y = y
        idx = labels == y
        if len(lam) == len(out):
            lam_ = lam[idx].reshape(-1, 1)  # - boradcast along features
        idx = torch.arange(idx.size(0))[idx]
        perm = torch.randperm(idx.size(0))
        idx_perm = idx[perm]
        zns = lam_ * out[idx] + (1 - lam_) * out[idx_perm]
        # zns,_ = alignmix(out[idx], out[idx_perm])
        new_zs.append(zns)
        perms.append(idx_perm)
        idxs.append(idx)

    return torch.cat(new_zs, axis=0), torch.cat(idxs, axis=0), torch.cat(perms, axis=0)


def mixup_process_label_free(out, lam):
    indices = np.random.permutation(out.size(0))
    indices = torch.Tensor(indices).long()
    out = out * lam + out[indices] * (1 - lam)
    return out, indices


def mixup(out, indices, lam):
    if len(out.shape) == 1:  # - 1d vector
        lam = lam.squeeze()  # - avoid broadcast below
    out_idxs = torch.arange(len(indices)) % len(out)
    out = out[out_idxs] * lam + out[indices]  * (1 - lam)
    return out

def mixup_full_indices(out, lambda_indices, one_minus_lambda_indices, lam):
    if len(out.shape) == 1:  # - 1d vector
        lam = lam.squeeze()  # - avoid broadcast below
    return lam * out[lambda_indices] + (1 - lam) * out[one_minus_lambda_indices]

def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight)

