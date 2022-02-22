import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, ChebConv  # noqa

class GCN(torch.nn.Module):
    def __init__(self, nfeat, nclass, if_relu = True, normalize = True):
        super(GCN, self).__init__()
        nhid = 128
        self.normalize = normalize
        self.conv1 = GCNConv(nfeat, nhid, normalize = self.normalize)
        self.conv2 = GCNConv(nhid, nclass, normalize = self.normalize)
        self.optimizer = torch.optim.Adam([
            dict(params=self.conv1.parameters(), weight_decay=5e-4),
            dict(params=self.conv2.parameters(), weight_decay=0)
        ], lr=0.01)  # Only perform weight-decay on first convolution.
        self.if_relu = if_relu


    def forward(self, features, adj):
        x, edge_index, edge_weight = features, adj.coalesce().indices(), adj.coalesce().values()
        if self.if_relu == True:
            x = F.relu(self.conv1(x, edge_index, edge_weight))
        else:
            x = self.conv1(x, edge_index, edge_weight)
        x = F.dropout(x, training=self.training)
        edge_index, edge_weight = adj.coalesce().indices(), adj.coalesce().values()
        x = self.conv2(x, edge_index, edge_weight)
        return F.log_softmax(x, dim=1)


    @torch.no_grad()
    def test_eval(self, adj, features, y, mask):
        self.eval()
        logits, accs = self(features, adj), []
        pred = logits[mask].max(1)[1]
        acc = pred.eq(y[mask]).sum().item() / mask.sum().item()
        return acc

    def train_eval(self, adj, data):
        best_val_acc = test_acc = 0
        for epoch in range(1, 201):
            self.train()
            self.optimizer.zero_grad()
            output = self(data.x, adj)
            F.nll_loss(output[data.train_mask], data.y[data.train_mask]).backward()
            self.optimizer.step()

            train_acc, val_acc, tmp_test_acc = self.test_eval(adj, data.x, data.y, data.train_mask), self.test_eval(adj, data.x, data.y, data.val_mask), self.test_eval(adj, data.x, data.y, data.test_mask)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
            log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
            print(log.format(epoch, train_acc, best_val_acc, test_acc))


