import rdkit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import mbuild as mb
import parmed
import torch
import torchmetrics
from torch import nn, optim
import torch_geometric as tg
from torch_geometric.loader import DataLoader
import torch.utils.data
import shutil
import sys
from datetime import datetime
 
# My imports
sys.path.insert(1, '~/projects/megnn')
from megnn.datasets import *
from megnn.megnn import *
from megnn.utils import *

# clear the processed dataset
try:
    if sys.argv[1] == 'remove':
        print('Removed processed dataset')
        shutil.rmtree('./processed')
except:
    pass

n_epochs  = 100
batch_size = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
dat = PdbBind_Dataset(root='/raid6/homes/kierannp/projects/megnn/datasets/v2019-other-PL')
print('Loaded the dataset')

a = []
for d in dat:
    a.append(d.y)
kd_mean = torch.mean(torch.tensor(a))
kd_std = torch.std(torch.tensor(a))

dat.shuffle()
train_percent = .8
test_percent = .1
train_stop = int(len(dat)*train_percent)
test_stop = train_stop + int(len(dat)*test_percent)
train_dataset = dat[:train_stop]
test_dataset = dat[train_stop:]
train_loader = DataLoader(train_dataset, batch_size=batch_size, follow_batch=['x_s', 'x_t', 'positions_s', 'positions_t'], shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, follow_batch=['x_s', 'x_t', 'positions_s', 'positions_t'], shuffle=False)

model = MEGNN(n_graphs=2, in_node_nf=dat.data.x_s.size(1), in_edge_nf=0, hidden_nf=45, device=device, n_layers=5, coords_weight=1.0,
             attention=False, node_attr=1)

optimizer = optim.Adam(model.parameters())

def rmsd_loss(x, y):
    return torch.sqrt(torch.sum((x - y)**2)/x.shape[0])
def RMSELoss(yhat,y):
    return torch.sqrt(torch.mean((yhat-y)**2))

criterion = nn.MSELoss().to(device)

print('Initialized model')

def train(epoch, loader):
    epoch_loss = 0
    total_samples = 0
    model.train()
    for i, data in enumerate(loader):

        one_hot_s, one_hot_t,\
        edges_s, edges_t,\
        atom_mask_s, atom_mask_t, \
        edge_mask_s, edge_mask_t, \
        n_nodes_s, n_nodes_t, \
        atom_positions_s,atom_positions_t,\
        batch_size_s, label = convert_to_dense(data, device, dtype)

        label = (label - kd_mean) / kd_std

        if (torch.abs(one_hot_s[:,:]) > 1000).sum()>0 or (torch.abs(one_hot_t[:,:]) > 1000).sum()>0:
            continue 
        pred = model(
            h0 = [one_hot_s, one_hot_t], 
            all_edges = [edges_s, edges_t], 
            all_edge_attr = [None, None], 
            node_masks = [atom_mask_s, atom_mask_t], 
            edge_masks = [edge_mask_s, edge_mask_t],
            n_nodes = [n_nodes_s, n_nodes_t], 
            x = [atom_positions_s, atom_positions_t]
        )
        loss = criterion(pred, label)  # Compute the loss.
        if not torch.isfinite(torch.tensor(loss.item())):
            raise Exception('Loss exploded')
        epoch_loss += loss.item()
        total_samples += batch_size_s
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        if i % 10 == 0:
            print("Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, loss.item()))
    return epoch_loss/total_samples

def test(loader):
    model.eval()
    predictions, actuals = [], []
    total_samples = 0
    for data in loader:  # Iterate in batches over the training/test dataset.

        one_hot_s, one_hot_t,\
        edges_s, edges_t,\
        atom_mask_s, atom_mask_t, \
        edge_mask_s, edge_mask_t, \
        n_nodes_s, n_nodes_t, \
        atom_positions_s,atom_positions_t,\
        batch_size_s, label = convert_to_dense(data, device, dtype)

        label = (label - kd_mean) / kd_std

        pred = model(
            h0 = [one_hot_s, one_hot_t], 
            all_edges = [edges_s, edges_t], 
            all_edge_attr = [None, None], 
            node_masks = [atom_mask_s, atom_mask_t], 
            edge_masks = [edge_mask_s, edge_mask_t],
            n_nodes = [n_nodes_s, n_nodes_t], 
            x = [atom_positions_s, atom_positions_t]
        )
        total_samples += batch_size
        predictions.extend(list((kd_std*pred.detach()+kd_mean).cpu().numpy()))
        actuals.extend(list((kd_std*label.detach()+kd_mean).cpu().numpy()))
    return criterion(torch.tensor(predictions), torch.tensor(actuals)).item()

res = {'epochs': [], 'train_loss': [],'test_loss': [], 'best_test': 1e10, 'best_epoch': 0}
for epoch in range(0, n_epochs):
    train_loss = train(epoch, train_loader) 
    res['train_loss'].append(train_loss)
    if epoch % 1 == 0:
        test_loss = test(test_loader)
        res['epochs'].append(epoch)
        res['test_loss'].append(test_loss)
        if test_loss < res['best_test']:
            res['best_test'] = test_loss
            res['best_epoch'] = epoch
        print("test loss: %.4f \t epoch %d" % (test_loss, epoch))
        print("Best: test loss: %.4f \t epoch %d" % (res['best_test'], res['best_epoch']))

plt.plot(res['epochs'], res['train_loss'], label='train')
plt.plot(res['epochs'], res['test_loss'], label='test')
plt.legend()
plt.savefig('history.png')
torch.save(model.state_dict(), './models/MEGNN_pdb_RMS{}.pth'.format(datetime.now()))
