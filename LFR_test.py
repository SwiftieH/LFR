import os
import os.path as osp
import argparse
from targeted_attack.nettack import Nettack
import numpy as np
from tqdm import tqdm
import scipy.sparse as sp
from scipy import linalg
from utils import *
import random

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, Amazon, Coauthor, WebKB
import torch_geometric.transforms as T
import numpy as np
from models.gcn_attack import GCN_attack
from models import *
from ogb.nodeproppred import PygNodePropPredDataset

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=233, help='Random seed.')
parser.add_argument('--hidden', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--targetmodel', type=str, default='GCN',  choices=['GCN2', 'GCN', 'JK', 'UNet', 'SGC'])
parser.add_argument('--dataset', type=str, default="Cora", help='Dataset to use.')
parser.add_argument('--k_por', type=float, default=0.5, help='Proportion of low_frequency.')
parser.add_argument('--num_targets', type=int, default=40, help='Number of nodes used for poisoning.')
parser.add_argument('--gpu_id', type=int, default=0, help='ID of GPU used for training.')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--random_select', action='store_true', default=False, help='random select candidates or nettack select.')
parser.add_argument('--degree_flips', action='store_true', default=False,  help='number of perturbations')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.defensemodel = args.targetmodel + 'LFR'

np.random.seed(args.seed)
torch.manual_seed(args.seed)
random.seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

path = osp.join(osp.dirname(osp.realpath(__file__)), 'data', args.dataset)
if 'ogbn' in args.dataset:
    dataset = PygNodePropPredDataset(name = args.dataset, root = path)
elif args.dataset == 'Citeseer' or args.dataset == 'Cora' or args.dataset == 'Pubmed':
    dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
elif args.dataset == 'CS' or args.dataset == 'Physics':
    dataset = Coauthor(path, args.dataset, transform=T.NormalizeFeatures())
elif args.dataset == 'Photo' or args.dataset == 'Computers':
    dataset = Amazon(path, args.dataset, transform=T.NormalizeFeatures())
elif args.dataset == 'Cornell' or args.dataset == 'Texas' or args.dataset == 'Wisconsin':
    dataset = WebKB(path, args.dataset, transform=T.NormalizeFeatures())
else:
    raise NotImplementedError()
data = dataset.data

num_features = int(dataset.num_features)
num_classes = int(dataset.num_classes)

device = torch.device('cuda' if args.cuda else 'cpu')
data = data.to(device)
if type(data.num_nodes).__name__ == 'list':
    data.num_nodes = data.num_nodes[0]

if data.edge_attr is None:
    edge_weight = torch.ones((data.edge_index.size(1), ), dtype=torch.float32, device=data.edge_index.device)
else:
    edge_weight = data.edge_attr

adj_ori = torch.sparse.FloatTensor(data.edge_index, edge_weight, torch.Size([data.num_nodes, data.num_nodes])) #original adj
_A_obs = sp.csr_matrix((edge_weight.cpu().numpy(), (data.edge_index[0,:].cpu().numpy(), data.edge_index[1,:].cpu().numpy() )), shape=(data.num_nodes, data.num_nodes))
_A_obs = _A_obs + _A_obs.T
_A_obs[_A_obs > 1] = 1
degrees = _A_obs.sum(0).A1

_X_obs = sp.csr_matrix(data.x.data.cpu().numpy()).astype('float32')
_z_obs = data.y.cpu().numpy()

adj_ori_nor = preprocess_adj(_A_obs)
adj_ori_nor = csr_to_tensor(adj_ori_nor).to(device)

unlabeled_share = 0.8
val_share = 0.1
train_share = 1 - unlabeled_share - val_share

split_train, split_val, split_unlabeled = train_val_test_split_tabular(np.arange(data.num_nodes),
                                                                       train_size=train_share,
                                                                       val_size=val_share,
                                                                       test_size=unlabeled_share,
                                                                       stratify=data.y.cpu().numpy())
if data.y.shape[-1] == 1:
    data.y = torch.squeeze(data.y)

try:
    data.train_mask.fill_(False)
    data.val_mask.fill_(False)
    data.test_mask.fill_(False)
except:
    train_mask = torch.zeros(
        data.num_nodes, dtype=torch.bool, device=data.edge_index.device
    )
    val_mask = torch.zeros(
        data.num_nodes, dtype=torch.bool, device=data.edge_index.device
    )
    test_mask = torch.zeros(
        data.num_nodes, dtype=torch.bool, device=data.edge_index.device
    )
    setattr(data, "train_mask", train_mask)
    setattr(data, "val_mask", val_mask)
    setattr(data, "test_mask", test_mask)

data.train_mask = index_to_mask(split_train,data.num_nodes)
data.val_mask = index_to_mask(split_val,data.num_nodes)
data.test_mask = index_to_mask(split_unlabeled,data.num_nodes)


surrogate = GCN_attack(nfeat=_X_obs.shape[1], nclass=_z_obs.max().item()+1,
                nhid=args.hidden, dropout=0, with_relu=False, with_bias=False, device=device)
surrogate = surrogate.to(device)
surrogate.fit(_X_obs, _A_obs, _z_obs, split_train, train_iters=101)

model_before = globals()[args.targetmodel](num_features, num_classes).to(device)
model_before.train_eval(adj_ori, data)
print(model_before.test_eval(adj_ori, data.x, data.y, data.test_mask))
acc_model_before_clean = model_before.test_eval(adj_ori, data.x, data.y, data.test_mask)

laplacian_normalize = True
sparse_ness = False
threshold = 1e-4
k_por = args.k_por
num_comps = int(data.num_nodes * k_por)
supports = spectral_basis(_A_obs, laplacian_normalize, sparse_ness, threshold, num_comps)
supports = [support.to(device) for support in supports]

model_after = globals()[args.defensemodel](num_features, num_classes, ncomps = num_comps).to(device)
if args.dataset == 'Photo':
    adj_ori = adj_ori_nor
model_after.train_eval(adj_ori, data, supports)
print(model_after.test_eval(adj_ori, data.x, data.y, data.test_mask))
acc_model_after_clean = model_after.test_eval(adj_ori, data.x, data.y, data.test_mask)

#paras for attack
if args.random_select:
    print('Random selecting candidates.')
    target_nodes = random_select_nodes(split_unlabeled, num_targets = args.num_targets)
else:
    print('Nettack selecting candidates.')
    target_nodes = select_nodes(model_before, split_unlabeled, data, adj_ori, _A_obs, num_targets = args.num_targets)
len_targets = len(target_nodes)

pbar = tqdm(range(len_targets))
attacked = 0

poisoned_data = []

for pos in pbar:
    u = target_nodes[pos]
    n_influencers = 1
    if args.degree_flips:
        n_perturbations = int(degrees[u])
    else:
        n_perturbations = 1

    if n_perturbations < 1:
        n_perturbations = 1

    perturb_features = False
    perturb_structure = True

    nettack = Nettack(surrogate, nnodes=_A_obs.shape[0], attack_structure=perturb_structure, attack_features=perturb_features, device=device)
    nettack = nettack.to(device)
    nettack.attack(_X_obs, _A_obs, _z_obs, u, n_perturbations, direct=True, n_influencers=n_influencers, verbose=False)
    modified_adj = nettack.modified_adj

    acc_before = model_before.test_eval(adj_ori, data.x, data.y, index_to_mask([u],data.num_nodes))
    print('acc before is: {}'.format(acc_before))

    supports_after = spectral_basis(modified_adj, laplacian_normalize, sparse_ness, threshold, num_comps)
    supports_after = [support_after.to(device) for support_after in supports_after]

    edge_index_after = torch.LongTensor(modified_adj.nonzero()).to(device) 
    adj_after = torch.sparse.FloatTensor(edge_index_after, torch.ones((edge_index_after.size(1), ), dtype=torch.float32, device=data.edge_index.device), torch.Size([data.num_nodes, data.num_nodes])) #adj after attack

    adj_after_nor = preprocess_adj(modified_adj)
    adj_after_nor = csr_to_tensor(adj_after_nor).to(device)
    if args.dataset == 'Photo':
        adj_after = adj_after_nor

    #defend poisoning
    model_after.train_eval(adj_after, data, supports_after)
    acc_after = model_after.test_eval(adj_after, data.x, data.y, index_to_mask([u], data.num_nodes))

    print('acc after is: {}'.format(acc_after))
    if acc_after < 1.0:
        attacked += 1

    pbar.set_description('current attack: {}'.format(attacked))

#The accuracy of selected nodes on clean graph is 1.0
print('Accuracy of selected nodes after attack is: {}'.format(1.0 - attacked/len_targets ))
print("Dataset is : {}".format(args.dataset))
print("Target model: {}".format(args.targetmodel))
print("Defense model: {}".format(args.defensemodel))
print('Propotion used as low-frequency: {}'.format(args.k_por))
degree_str = 'degree_flips' if args.degree_flips else 'one_flip'
print('Number of perturbations: {}'.format(degree_str) )

print('Acc of original model on clean graph is {}'.format(acc_model_before_clean))
print('Acc of original model regularized by LFR on clean graph is {}'.format(acc_model_after_clean))


