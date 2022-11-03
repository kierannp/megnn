import rdkit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
import mbuild as mb
import parmed
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

# clear the processed dataset
try:
    shutil.rmtree('./processed')
except:
    pass

n_epochs  = 5
device = torch.device("cpu")
dtype = torch.float32
dat = Cloud_Point_Dataset(root='.')

batch_size = 16

dat.shuffle()
train_dataset = dat[:4000]
test_dataset = dat[4000:5000]
valid_dataset = dat[5000:]
train_loader = DataLoader(train_dataset, batch_size=batch_size, follow_batch=['x_s', 'x_t', 'positions_s', 'positions_t'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, follow_batch=['x_s', 'x_t','positions_s', 'positions_t'], shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, follow_batch=['x_s', 'x_t','positions_s', 'positions_t'], shuffle=False)

model = MEGNN(n_graphs=2, in_node_nf=len(dat.elements), in_edge_nf=0, hidden_nf=128, device=device, n_layers=7, coords_weight=1.0,
             attention=True, node_attr=1, n_enviro=4)
# model = PairEGNN(in_node_nf=len(dat.elements), in_edge_nf=0, hidden_nf=128, device=device, n_layers=7, coords_weight=1.0,
#              attention=True, node_attr=1)

optimizer = optim.Adam(model.parameters())
loss_func = nn.MSELoss()


def train():
    model.train()
    epoch_loss = 0
    for data in train_loader:  # Iterate in batches over the training dataset.
        dense_positions_s, atom_mask_s = tg.utils.to_dense_batch(data.positions_s, data.positions_s_batch)
        dense_positions_t, atom_mask_t = tg.utils.to_dense_batch(data.positions_t, data.positions_t_batch)
        batch_size_s, n_nodes_s, _ = dense_positions_s.size()
        batch_size_t, n_nodes_t, _ = dense_positions_t.size()
        atom_positions_s = dense_positions_s.view(batch_size_s * n_nodes_s, -1).to(device, dtype)
        atom_positions_t = dense_positions_t.view(batch_size_t * n_nodes_t, -1).to(device, dtype)
        edge_mask_s = atom_mask_s.unsqueeze(1) * atom_mask_s.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask_s.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask_s *= diag_mask
        edge_mask_s = edge_mask_s.view(batch_size_s * n_nodes_s * n_nodes_s, 1).to(device)
        edge_mask_t = atom_mask_t.unsqueeze(1) * atom_mask_t.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask_t.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask_t *= diag_mask
        edge_mask_t = edge_mask_t.view(batch_size_t * n_nodes_t * n_nodes_t, 1).to(device)
        atom_mask_s = atom_mask_s.view(batch_size_s * n_nodes_s, -1).to(device)
        atom_mask_t = atom_mask_t.view(batch_size_t * n_nodes_t, -1).to(device)
        one_hot_s, one_hot_s_mask = tg.utils.to_dense_batch(data.x_s, data.x_s_batch)
        one_hot_t, one_hot_s_mask = tg.utils.to_dense_batch(data.x_t, data.x_t_batch)
        one_hot_s = one_hot_s.view(batch_size_s * n_nodes_s, -1).to(device)
        one_hot_t = one_hot_t.view(batch_size_t * n_nodes_t, -1).to(device)
        edges_s = get_adj_matrix(n_nodes_s, batch_size_s, device)
        edges_t = get_adj_matrix(n_nodes_t, batch_size_t, device)
        enviro = data.enviro.to(device, dtype)
        label = data.y.to(device, dtype)
        
        # pred = model(h0_s=one_hot_s, h0_t=one_hot_t, edges_s=edges_s, edges_t=edges_t, edge_attr=None, node_mask_s=atom_mask_s, 
        #             edge_mask_s=edge_mask_s, n_nodes_s=n_nodes_s, node_mask_t=atom_mask_t, edge_mask_t=edge_mask_t, 
        #             n_nodes_t=n_nodes_t, x_s=atom_positions_s, x_t=atom_positions_t)
        pred = model(h0=[one_hot_s, one_hot_t], x=[atom_positions_s, atom_positions_t], all_edges=[edges_s, edges_t],
                         all_edge_attr=[None, None], node_masks=[atom_mask_s, atom_mask_t],
                         edge_masks=[edge_mask_s, edge_mask_t], n_nodes=[n_nodes_s, n_nodes_t], enviro = enviro)
        loss = loss_func(pred, label)
        epoch_loss += loss
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    return epoch_loss

def test(loader):
    model.eval()

    squared_error = []
    error = 0 
    for data in loader:  # Iterate in batches over the training/test dataset.
        dense_positions_s, atom_mask_s = tg.utils.to_dense_batch(data.positions_s, data.positions_s_batch)
        dense_positions_t, atom_mask_t = tg.utils.to_dense_batch(data.positions_t, data.positions_t_batch)
        batch_size_s, n_nodes_s, _ = dense_positions_s.size()
        batch_size_t, n_nodes_t, _ = dense_positions_t.size()
        atom_positions_s = dense_positions_s.view(batch_size_s * n_nodes_s, -1).to(device, dtype)
        atom_positions_t = dense_positions_t.view(batch_size_t * n_nodes_t, -1).to(device, dtype)
        edge_mask_s = atom_mask_s.unsqueeze(1) * atom_mask_s.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask_s.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask_s *= diag_mask
        edge_mask_s = edge_mask_s.view(batch_size_s * n_nodes_s * n_nodes_s, 1).to(device)
        edge_mask_t = atom_mask_t.unsqueeze(1) * atom_mask_t.unsqueeze(2)
        diag_mask = ~torch.eye(edge_mask_t.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask_t *= diag_mask
        edge_mask_t = edge_mask_t.view(batch_size_t * n_nodes_t * n_nodes_t, 1).to(device)
        atom_mask_s = atom_mask_s.view(batch_size_s * n_nodes_s, -1).to(device)
        atom_mask_t = atom_mask_t.view(batch_size_t * n_nodes_t, -1).to(device)
        one_hot_s, one_hot_s_mask = tg.utils.to_dense_batch(data.x_s, data.x_s_batch)
        one_hot_t, one_hot_s_mask = tg.utils.to_dense_batch(data.x_t, data.x_t_batch)
        one_hot_s = one_hot_s.view(batch_size_s * n_nodes_s, -1).to(device)
        one_hot_t = one_hot_t.view(batch_size_t * n_nodes_t, -1).to(device)
        edges_s = get_adj_matrix(n_nodes_s, batch_size_s, device)
        edges_t = get_adj_matrix(n_nodes_t, batch_size_t, device)
        enviro = data.enviro.to(device, dtype)
        label = data.y.to(device, dtype)
        
        # pred = model(h0_s=one_hot_s, h0_t=one_hot_t, edges_s=edges_s, edges_t=edges_t, edge_attr=None, node_mask_s=atom_mask_s, 
        #             edge_mask_s=edge_mask_s, n_nodes_s=n_nodes_s, node_mask_t=atom_mask_t, edge_mask_t=edge_mask_t, 
        #             n_nodes_t=n_nodes_t, x_s=atom_positions_s, x_t=atom_positions_t)
        pred = model(h0=[one_hot_s, one_hot_t], x=[atom_positions_s, atom_positions_t], all_edges=[edges_s, edges_t],
                    all_edge_attr=[None, None], node_masks=[atom_mask_s, atom_mask_t],
                    edge_masks=[edge_mask_s, edge_mask_t], n_nodes=[n_nodes_s, n_nodes_t], enviro=enviro)
        loss = loss_func(pred, label)
        error += loss.detach()
    return error

training_loss = []
testing_loss = []

for epoch in range(1, n_epochs):
    train()
    train_acc = test(train_loader)
    test_acc = test(test_loader)
    training_loss.append(train_acc.detach().numpy())
    testing_loss.append(test_acc.detach().numpy())
    print(f'Epoch: {epoch:03d}, Train MSE: {train_acc:.4f}, Test MSE: {test_acc:.4f}')

plt.figure(figsize=(12,6))
plt.plot(range(1, n_epochs), training_loss,label='training')
plt.plot(range(1, n_epochs), testing_loss,label='testing')
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('result.png')
