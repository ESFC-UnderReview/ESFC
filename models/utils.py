import torch
import networkx as nx
import numpy as np
from networkx.algorithms import tree
from scipy.sparse import coo_matrix
from numba import jit
import torchvision
import torchvision.transforms as transforms

__all__ = ['extractor', 'padder', 'averager', 'gnn_to_graph', 'cover_and_merge', 'hamilton', 'mask2dist', 'MS_MST', 'permute_2d_to_1d', 'mask2randDist', 'transform_to_tensor']


def permute_2d_to_1d(x, permutation):
    d0, d1, d2, d3 = x.size()
    ret = x[:, :, permutation[0, :], permutation[1, :]].view(d0, d1, d2, d3)
    return ret


def extractor(x, size):
    lst = []
    for i in range(size):
        lst.append(x[:,:,size*i:size*i+size - 1])
    x = torch.cat(lst, dim=2)
    return x


def padder(x, size):
    B,C,_ = x.size()
    padding = torch.zeros([B,C,1],device='cuda',requires_grad=True)
    lst = []
    for i in range(size):
        frag = x[:,:,(size - 1)*i:(size - 1)*(i+1)]
        if i < size - 1:
            frag = torch.cat([frag, padding], dim=2)
        lst.append(frag)
    x = torch.cat(lst, dim=2)
    return x


def averager(x):
    x = torch.sum(x, dim=0, keepdim=False)
    x = torch.sum(x, dim=0, keepdim=False).cpu().detach().numpy()
    return x


def gnn_to_graph(mask, size):
    _row_edge = list(range(size*size-size))
    _col_edge = list(range(size, size*size))

    _row_inside = list(range(size*size-1))
    del _row_inside[size-1:size*size:size]
    _col_inside = list(range(1, size*size))
    del _col_inside[size-1:size*size:size]

    # [mask_upper, mask_mid_upper]
    mask[0] = averager(mask[0])
    mask[1] = averager(mask[1])
    mask[2] = averager(mask[2])
    mask[3] = averager(mask[3])

    coo_upper = coo_matrix((mask[0], (_row_edge, _col_edge)), shape=(size*size, size*size), dtype=float)
    coo_mid_upper = coo_matrix((mask[1], (_row_inside, _col_inside)), shape=(size*size, size*size), dtype=float)
    coo_lower = coo_matrix((mask[2], (_col_edge, _row_edge)), shape=(size*size, size*size), dtype=float)
    coo_mid_lower = coo_matrix((mask[3], (_col_inside, _row_inside)), shape=(size*size, size*size), dtype=float)
    G = nx.from_scipy_sparse_matrix(coo_upper + coo_mid_upper + coo_lower + coo_mid_lower)
        # G.append(nx.from_scipy_sparse_matrix(coo_upper + coo_mid_upper))

    return G


def cover_and_merge(ref, mst, size):
    dual_G_size = size // 2
    for i in mst.edges():
        assert i[1] - i[0] == 1 or i[1] - i[0] == dual_G_size
        ind = 2 * i[0] + 1 + i[0] // dual_G_size * size
        x_cord = ind // size
        y_cord = ind - x_cord * size
        if i[1] - i[0] == 1:
            ref.add_edge((x_cord, y_cord), (x_cord, y_cord + 1))
            ref.add_edge((x_cord + 1, y_cord), (x_cord + 1, y_cord + 1))
            ref.remove_edge((x_cord, y_cord), (x_cord + 1, y_cord))
            ref.remove_edge((x_cord, y_cord + 1), (x_cord + 1, y_cord + 1))
        if i[1] - i[0] == dual_G_size:
            ref.add_edge((x_cord + 1, y_cord), (x_cord + 2, y_cord))
            ref.add_edge((x_cord + 1, y_cord - 1), (x_cord + 2, y_cord - 1))
            ref.remove_edge((x_cord + 1, y_cord - 1), (x_cord + 1, y_cord))
            ref.remove_edge((x_cord + 2, y_cord - 1), (x_cord + 2, y_cord))
    return ref


def hamilton(ref, size):
    for i in range(1, size//2):
        ind_1 = 2*i-1
        ind_2 = 2 * i
        for j in range(size):
            ref.remove_edge((j, ind_1), (j, ind_2))
        for k in range(size):
            ref.remove_edge((ind_1, k), (ind_2, k))
    return ref


def mask2dist(mask, h):
    G = gnn_to_graph(mask, h)
    mst = tree.minimum_spanning_tree(G, algorithm="prim")
    ref = nx.grid_2d_graph(h * 2, h * 2)  # 4x4 grid
    ref = hamilton(ref, h * 2)
    ref = cover_and_merge(ref, mst, h * 2)
    edge_order = list(nx.dfs_edges(ref))
    dist = torch.zeros([len(edge_order)+1, 1])
    coord = torch.zeros([len(edge_order)+1, 2])
    count = 0
    for i in edge_order:
        dist[count] = i[0][0] * h * 2 + i[0][1]
        coord[count][0] = i[0][0]
        coord[count][1] = i[0][1]
        count += 1
        # dist.append(i[0][0] * h * 2 + i[0][1])
        # coord.append(i[0])
    dist[-1] = (edge_order[-1][1][0] * h * 2 + edge_order[-1][1][1])
    coord[-1][0] = edge_order[-1][1][0]
    coord[-1][1] = edge_order[-1][1][1]
    # dist.append(edge_order[-1][1][0] * h * 2 + edge_order[-1][1][1])
    # coord.append(edge_order[-1])
    assert len(dist) == len(coord)
    return dist, coord, mst


def MS_MST(mst2, mst3, mst4, h, eta=1e-3):
    A2 = nx.adjacency_matrix(mst2)
    A3 = nx.adjacency_matrix(mst3)
    A4 = nx.adjacency_matrix(mst4)
    A = A2*A3*A4
    B = A.toarray()
    Awg = A4.toarray()
    B = ((eta-1)*B + 1) * Awg
    ans = nx.from_numpy_matrix(B)
    mst = tree.maximum_spanning_tree(ans, algorithm="prim")
    ref = nx.grid_2d_graph(h * 2, h * 2)  # 4x4 grid
    ref = hamilton(ref, h * 2)
    ref = cover_and_merge(ref, mst, h * 2)
    edge_order = list(nx.dfs_edges(ref))
    dist = torch.zeros([len(edge_order) + 1, 1])
    coord = torch.zeros([len(edge_order) + 1, 2])
    count = 0
    for i in edge_order:
        dist[count] = i[0][0] * h * 2 + i[0][1]
        coord[count][0] = i[0][0]
        coord[count][1] = i[0][1]
        count += 1
        # dist.append(i[0][0] * h * 2 + i[0][1])
        # coord.append(i[0])
    dist[-1] = (edge_order[-1][1][0] * h * 2 + edge_order[-1][1][1])
    coord[-1][0] = edge_order[-1][1][0]
    coord[-1][1] = edge_order[-1][1][1]
    # dist.append(edge_order[-1][1][0] * h * 2 + edge_order[-1][1][1])
    # coord.append(edge_order[-1])
    assert len(dist) == len(coord)
    return coord


def mask2randDist(mask, h):
    G = gnn_to_graph(mask, h)
    mst = tree.minimum_spanning_tree(G, algorithm="prim")
    ref = nx.grid_2d_graph(h * 2, h * 2)  # 4x4 grid
    ref = hamilton(ref, h * 2)
    ref = cover_and_merge(ref, mst, h * 2)
    edge_order = list(nx.dfs_edges(ref))
    dist = torch.zeros([len(edge_order)+1, 1])
    coord = torch.zeros([len(edge_order)+1, 2])
    count = 0
    for i in edge_order:
        dist[count] = i[0][0] * h * 2 + i[0][1]
        coord[count][0] = i[0][0]
        coord[count][1] = i[0][1]
        count += 1
        # dist.append(i[0][0] * h * 2 + i[0][1])
        # coord.append(i[0])
    dist[-1] = (edge_order[-1][1][0] * h * 2 + edge_order[-1][1][1])
    coord[-1][0] = edge_order[-1][1][0]
    coord[-1][1] = edge_order[-1][1][1]
    # dist.append(edge_order[-1][1][0] * h * 2 + edge_order[-1][1][1])
    # coord.append(edge_order[-1])
    assert len(dist) == len(coord)
    return dist, coord, mst

def coo_diagonal(mst, h):
    mid_upper = nx.adjacency_matrix(mst).diagonal(k=1)
    upper = nx.adjacency_matrix(mst).diagonal(k=h)

    return upper, mid_upper


def transform_to_tensor(size, mode='train'):
    if mode == 'train':
        transform_train = transforms.Compose([
            transforms.Resize((size,size)),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        return transform_train
    else:
        transform_test = transforms.Compose([
            transforms.Resize((size,size)),
            transforms.ToTensor(),
        ])
        return transform_test





