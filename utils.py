import torch
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import scipy.special as ss
from scipy.sparse.linalg.eigen.arpack import eigsh
from scipy.sparse.csgraph import connected_components
import sys
import warnings
warnings.filterwarnings("ignore")

def csr_to_tensor(csr_mat):
    num_nodes = csr_mat.shape[0]
    edge_index = torch.LongTensor(csr_mat.nonzero())
    sp_tensor = torch.sparse.FloatTensor(edge_index, torch.FloatTensor(csr_mat.data), torch.Size(csr_mat.shape))
    return sp_tensor

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt,0)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized


def laplacian(W, normalized=False):
    """Return the Laplacian of the weight matrix."""
    # Degree matrix.
    d = W.sum(axis=0)
    # Laplacian matrix.
    if not normalized:
        D = sp.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        # d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = sp.diags(d.A.squeeze(), 0)
        I = sp.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is sp.csr.csr_matrix
    return L


def fourier(L, algo='eigh', k=100):
    """Return the Fourier basis, i.e. the EVD of the Laplacian."""
    # print "eigen decomposition:"
    def sort(lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:, idx]
    # if(dataset == "pubmed"):
    #     # print "loading pubmed U"
    #     rfile = open("data/pubmed_U.pkl")
    #     lamb, U = pkl.load(rfile)
    #     rfile.close()
    # else:
    if algo is 'eig':
        lamb, U = np.linalg.eig(L.toarray())
        lamb, U = sort(lamb, U)
    elif algo is 'eigh':
        lamb, U = np.linalg.eigh(L.toarray())
        lamb, U = sort(lamb, U)
    elif algo is 'eigs':
        lamb, U = sp.linalg.eigs(L, k=k, which='SM')
        lamb, U = sort(lamb, U)
    elif algo is 'eigsh':
        lamb, U = sp.linalg.eigsh(L, k=k, which='SM')
    # print "end"
    # wfile = open("data/pubmed_U.pkl","w")
    # pkl.dump([lamb,U],wfile)
    # wfile.close()
    # print "pkl U end"
    return lamb, U



def spectral_basis(adj,laplacian_normalize,sparse_ness,threshold,num_comps,lamb_re = False):
    L = laplacian(adj,normalized=laplacian_normalize)
    lamb, U = fourier(L)
    #if (sparse_ness):
    #    U[U < threshold] = 0.0

    U = sp.csr_matrix(U[:,:num_comps])
    U_transpose = sp.csr_matrix(np.transpose(U[:,:num_comps]))
    t_k = [csr_to_tensor(U), csr_to_tensor(U_transpose)]
    if lamb_re:
        return t_k, lamb[:num_comps]
    else:
        return t_k

def random_select_nodes(split_unlabeled, num_targets = 40):
    '''
    random selecting nodes
    '''
    target_nodes = np.random.choice(split_unlabeled, num_targets, replace=False).tolist()

    return target_nodes


def classification_margin(output, true_label):
    """Calculate classification margin for outputs.
    `probs_true_label - probs_best_second_class`
    Parameters
    ----------
    output: torch.Tensor
        output vector (1 dimension)
    true_label: int
        true label for this node
    Returns
    -------
    list
        classification margin for this node
    """

    probs = torch.exp(output)
    probs_true_label = probs[true_label].clone()
    probs[true_label] = 0
    probs_best_second_class = probs[probs.argmax()]
    return (probs_true_label - probs_best_second_class).item()

def select_nodes(target_gcn, split_unlabeled, data, adj, _A_obs, num_targets = 40):
    '''
    selecting nodes as reported in nettack paper:
    (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
    (ii) the 10 nodes with lowest margin (but still correctly classified) and
    (iii) 20 more nodes randomly
    '''
    target_gcn.eval()
    logits, accs = target_gcn(data.x, adj), []

    margin_dict = {}
    for idx in split_unlabeled:
        margin = classification_margin(logits[idx], data.y[idx])
        if margin < 0: # only keep the nodes correctly classified
            continue
        """check the outliers:"""
        neighbours = list(_A_obs.todense()[idx].nonzero()[1])
        y = [data.y[i] for i in neighbours]
        node_y = data.y[idx]
        aa = [yy==node_y for yy in y]
        aa = [a.data.cpu().detach().numpy() for a in aa]
        outlier_score = 1- np.sum(aa)/len(aa)
        if outlier_score >=0.5:
            continue
        margin_dict[idx] = margin
    sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
    num_margin = num_targets // 4
    num_random = num_targets // 2
    high = [x for x, y in sorted_margins[: num_margin]]
    low = [x for x, y in sorted_margins[-num_margin: ]]
    #import pdb; pdb.set_trace()
    other = [x for x, y in sorted_margins[num_margin: -num_margin]]
    other = np.random.choice(other, num_random, replace=False).tolist()
    target_nodes = other + low + high

    return target_nodes

def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool)
    mask[index] = 1
    return mask

def train_val_test_split_tabular(*arrays, train_size=0.5, val_size=0.3, test_size=0.2, stratify=None, random_state=None):
    """
    Split the arrays or matrices into random train, validation and test subsets.
    #train_test_split重写成可以分出验证集的函数

    Parameters
    ----------
    *arrays : sequence of indexables with same length / shape[0]
            Allowed inputs are lists, numpy arrays or scipy-sparse matrices.
    train_size : float, default 0.5
        Proportion of the dataset included in the train split.
    val_size : float, default 0.3
        Proportion of the dataset included in the validation split.
    test_size : float, default 0.2
        Proportion of the dataset included in the test split.
    stratify : array-like or None, default None
        If not None, data is split in a stratified fashion, using this as the class labels. #是否为标签均衡的数据
    random_state : int or None, default None
        Random_state is the seed used by the random number generator;

    Returns
    -------
    splitting : list, length=3 * len(arrays)
        List containing train-validation-test split of inputs.

    """
    if len(set(array.shape[0] for array in arrays)) != 1:
        raise ValueError("Arrays must have equal first dimension.")
    idx = np.arange(arrays[0].shape[0])
    idx_train_and_val, idx_test = train_test_split(idx,
                                                   random_state=random_state,
                                                   train_size=(train_size + val_size),
                                                   test_size=test_size,
                                                   stratify=stratify)
    if stratify is not None:
        stratify = stratify[idx_train_and_val]
    idx_train, idx_val = train_test_split(idx_train_and_val,
                                          random_state=random_state,
                                          train_size=(train_size / (train_size + val_size)),
                                          test_size=(val_size / (train_size + val_size)),
                                          stratify=stratify)
    result = []
    for X in arrays:
        result.append(X[idx_train])
        result.append(X[idx_val])
        result.append(X[idx_test])
    return result





