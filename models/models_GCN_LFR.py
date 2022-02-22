import torch
import torch.nn.functional as F
import math
from torch.nn import Parameter
import random
from layers.gcn_conv import GCNConvLFR
import time

class GCNLFR(torch.nn.Module):
    def __init__(self, nfeat, nclass, ncomps, if_relu = True, filters = None, no_share_para = False):
        super(GCNLFR, self).__init__()
        nhid = 128
        self.gc1 = GCNConvLFR(nfeat, nhid, ncomps, filters = filters, no_share_para = False)
        self.gc2 = GCNConvLFR(nhid, nclass, ncomps, filters = filters, no_share_para = False)
        self.optimizer = torch.optim.Adam([
            dict(params=self.gc1.parameters(), weight_decay=5e-4),
            dict(params=self.gc2.parameters(), weight_decay=0)
        ], lr=0.01)  # Only perform weight-decay on first convolution.
        self.relu = if_relu
        self.loss_gcn = []
        self.loss_lfr = []
        self.losses = []
        self.loss_gcn_val = []
        self.loss_lfr_val = []
        self.losses_val = []
        self.acc_val = []
        self.acc_train = []

    def forward(self, x, adj):
        if self.relu:
            x = F.relu(self.gc1(x, adj))
        else:
            x = self.gc1(x, adj)
        x = F.dropout(x, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)

    def forward_LFR(self, x, supports):
        x = F.relu(self.gc1.forward_LFR(x, supports))
        x = F.dropout(x, training=self.training)
        x = self.gc2.forward_LFR(x, supports)
        return F.log_softmax(x, dim=1)

    @torch.no_grad()
    def test_eval(self, adj, features, y, mask):
        self.eval()
        logits, accs = self(features, adj), []
        pred = logits[mask].max(1)[1]
        acc = pred.eq(y[mask]).sum().item() / mask.sum().item()
        return acc

    def train_eval(self, adj, data, supports, alpha = 0.5):
        print('Alternate Training!')
        best_val_acc = test_acc = 0
        start_time = time.time()
        for epoch in range(1, 201):
            rand_index = random.uniform(0, 1)
            if not rand_index < alpha:
                print('updating GCN part!')
                self.train()
                self.optimizer.zero_grad()
                output = self(data.x, adj)
                loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
                loss.backward()
                self.loss_gcn.append(loss.item())
                self.losses.append(loss.item())
                self.optimizer.step()
                loss_val = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
                self.loss_gcn_val.append(loss_val.item())
                self.losses_val.append(loss_val.item())
            else:
                print('updating LFR_Net part!')
                self.train()
                self.optimizer.zero_grad()
                output = self.forward_LFR(data.x, supports)
                loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])
                loss.backward()
                self.loss_lfr.append(loss.item())
                self.losses.append(loss.item())
                self.optimizer.step()
                loss_val = F.nll_loss(output[data.val_mask], data.y[data.val_mask])
                self.loss_lfr_val.append(loss_val.item())
                self.losses_val.append(loss_val.item())
            train_acc, val_acc, tmp_test_acc = self.test_eval(adj, data.x, data.y, data.train_mask), self.test_eval(adj, data.x, data.y, data.val_mask), self.test_eval(adj, data.x, data.y, data.test_mask)
            self.acc_val.append(val_acc)
            self.acc_train.append(train_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, train_acc, best_val_acc, test_acc))
        end_time = time.time()
        print('Total training time is:{}'.format(end_time - start_time))


    def train_eval_joint(self, adj, data, supports):
        print('Join Training!')
        start_time = time.time()
        best_val_acc = test_acc = 0
        alpha = 1
        for epoch in range(1, 201):
            print('updating GCN part!')
            self.train()
            self.optimizer.zero_grad()
            output_gcn = self(data.x, adj)
            loss_gcn = F.nll_loss(output_gcn[data.train_mask], data.y[data.train_mask])
            self.loss_gcn.append(loss_gcn.item())

            print('updating LFR_Net part!')
            output_lfr = self.forward_LFR(data.x, supports)
            loss_lfr = F.nll_loss(output_lfr[data.train_mask], data.y[data.train_mask])
            self.loss_lfr.append(loss_lfr.item())
            loss = loss_gcn + alpha * loss_lfr
            loss.backward()
            self.losses.append(loss.item())
            self.optimizer.step()

            loss_gcn_val = F.nll_loss(output_gcn[data.val_mask], data.y[data.val_mask])
            self.loss_gcn_val.append(loss_gcn_val.item())
            loss_lfr_val = F.nll_loss(output_lfr[data.val_mask], data.y[data.val_mask])
            self.loss_lfr_val.append(loss_lfr_val.item())
            self.losses_val.append(loss_gcn_val.item() + alpha * loss_lfr_val.item())

            train_acc, val_acc, tmp_test_acc = self.test_eval(adj, data.x, data.y, data.train_mask), self.test_eval(adj, data.x, data.y, data.val_mask), self.test_eval(adj, data.x, data.y, data.test_mask)
            self.acc_val.append(val_acc)
            self.acc_train.append(train_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, train_acc, best_val_acc, test_acc))
        end_time = time.time()
        print('Total training time is:{}'.format(end_time - start_time))





