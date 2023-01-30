import torch
import torch
from torch import nn
import torch_geometric as tg
from torch_geometric.loader import DataLoader
import torch.utils.data
import shutil
import sys
import rdkit 
sys.path.insert(1, '~/egnn_cof/models/egnn_clean')
sys.path.insert(1, '~/egnn_cof')
# from easydict import EasyDict as edict
import torch
from torch import nn, optim
# My imports
from loader import *
from megnn import *
from utils import *

def compute_mean_mad(series):
    values = torch.Tensor(series)
    mean = torch.mean(values)
    ma = torch.abs(values - mean)
    mad = torch.mean(ma)
    return mean, mad

# clear the processed dataset
shutil.rmtree('./processed')

n_epochs  = 2
device = torch.device("cpu")
dtype = torch.float32
dat = Cloud_Point_Dataset(root='.')

batch_size = 8

dat.shuffle()
train_dataset = dat[:5000]
test_dataset = dat[5000:]
train_loader = DataLoader(train_dataset, batch_size=batch_size, follow_batch=['x_s', 'x_t', 'positions_s', 'positions_t'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, follow_batch=['x_s', 'x_t','positions_s', 'positions_t'], shuffle=False)

prop_mean, prop_mad = compute_mean_mad(dat.dataframe['CP (C)'])

model = MEGNN(n_graphs=2, in_node_nf=len(dat.elements), in_edge_nf=0, hidden_nf=128, device=device, n_layers=7, coords_weight=1.0,
             attention=False, node_attr=1)
# model = PairEGNN(in_node_nf=len(dat.elements), in_edge_nf=0, hidden_nf=128, device=device, n_layers=7, coords_weight=1.0,
#              attention=False, node_attr=1)

optimizer = optim.Adam(model.parameters())
loss = nn.MSELoss()


def train(epoch, loader, partition='train'):
    res = {'loss': 0, 'counter': 0, 'loss_arr':[]}
    for i, data in enumerate(loader):
        if partition == 'train':
            model.train()
            optimizer.zero_grad()

        else:
            model.eval()

        one_hot_s, one_hot_t, edges_s, edges_t, atom_mask_s, atom_mask_t, \
        edge_mask_s, edge_mask_t, n_nodes_s, n_nodes_t, atom_positions_s, atom_positions_t, label = convert_to_dense(data, device, dtype)
        # MEGNN
        pred = model(h0=[one_hot_s, one_hot_t], all_edges=[edges_s, edges_t], all_edge_attr=None, node_masks=[atom_mask_s, atom_mask_t],
                    edge_masks=[edge_mask_s, edge_mask_t], n_nodes=[n_nodes_s, n_nodes_t], x=[atom_positions_s, atom_positions_t])
        #PairEGNN
        # pred = model(h0_s=one_hot_s, h0_t=one_hot_t, edges_s=edges_s, edges_t=edges_t, edge_attr=None, node_mask_s=atom_mask_s, 
        #             edge_mask_s=edge_mask_s, n_nodes_s=n_nodes_s, node_mask_t=atom_mask_t, edge_mask_t=edge_mask_t, 
        #             n_nodes_t=n_nodes_t, x_s=atom_positions_s, x_t=atom_positions_t)
        if partition == 'train':
            loss = loss(pred, label)
            loss.backward()
            optimizer.step()
        else:
            loss = loss(prop_mad * pred + prop_mean, label)

        res['loss'] += loss.item()
        res['counter'] += batch_size

        prefix = ""
        if partition != 'train':
            prefix = ">> %s \t" % partition

        if i % 2 == 0:
            print(prefix + "Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, sum(res['loss_arr'][-10:])/len(res['loss_arr'][-10:])))
    return res['loss'] / res['counter']

res = {'epochs': [], 'losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}

for epoch in range(0, n_epochs):
    train(epoch, train_loader, partition='train')
    if epoch % 2 == 0:
        val_loss = train(epoch, train_loader, partition='valid')
        test_loss = train(epoch, test_loader, partition='test')
        res['epochs'].append(epoch)
        res['losess'].append(test_loss)

        if val_loss < res['best_val']:
            res['best_val'] = val_loss
            res['best_test'] = test_loss
            res['best_epoch'] = epoch
        print("Val loss: %.4f \t test loss: %.4f \t epoch %d" % (val_loss, test_loss, epoch))
        print("Best: val loss: %.4f \t test loss: %.4f \t epoch %d" % (res['best_val'], res['best_test'], res['best_epoch']))


    # json_object = json.dumps(res, indent=4)
    # with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
    #     outfile.write(json_object)