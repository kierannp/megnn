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

# example of bayesian optimization with scikit-optimize
from numpy import mean
from sklearn.datasets import make_blobs
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from skopt.space import Integer, Categorical
from skopt.utils import use_named_args
from skopt import gp_minimize
 
# My imports
from loader import *
from megnn import *
from utils import *

# clear the processed dataset
try:
    if sys.argv[1] == 'remove':
        print('Removed processed dataset')
        shutil.rmtree('./processed')
except:
    pass



n_epochs  = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')
# Remove cached cuda memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()
dtype = torch.float32
dat = PdbBind_Dataset(root='.')
print('Loaded the dataset')
batch_size = 2

dat.shuffle()
train_percent = .8
test_percent = .1
train_stop = int(len(dat)*train_percent)
test_stop = train_stop + int(len(dat)*test_percent)
train_dataset = dat[:train_stop]
test_dataset = dat[train_stop:test_stop]
valid_dataset = dat[test_stop:]
train_loader = DataLoader(train_dataset, batch_size=batch_size, follow_batch=['x_s', 'x_t', 'positions_s', 'positions_t'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, follow_batch=['x_s', 'x_t','positions_s', 'positions_t'], shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, follow_batch=['x_s', 'x_t','positions_s', 'positions_t'], shuffle=False)

# generate 2d classification dataset
X, y = make_blobs(n_samples=500, centers=3, n_features=2)
# define the model
model = KNeighborsClassifier()
# define the space of hyperparameters to search
# define the space of hyperparameters to search
search_space = [
                    Integer(8, 128, name='hidden_nf'),
                    Integer(1, 8, name='n_layers'), 
                    Categorical([nn.ReLU(),nn.SiLU(), nn.LeakyReLU()],name='act_func'),
                    Categorical([True, False],name='attention')
                ]
 
# define the function used to evaluate a given configuration
@use_named_args(search_space)
def evaluate_model(**params):
    # calculate 5-fold cross validation
    result = cross_val_score(model, X, y, cv=5, n_jobs=-1, scoring='accuracy')
    # calculate the mean of the scores
    estimate = mean(result)
    return 1.0 - estimate
 
# perform optimization
result = gp_minimize(evaluate_model, search_space)
# summarizing finding:
print('Best Accuracy: %.3f' % (1.0 - result.fun))
print('Best Parameters: n_neighbors=%d, p=%d' % (result.x[0], result.x[1]))



model = MEGNN(n_graphs=2, in_node_nf=dat.data.x_s.size(1), in_edge_nf=0, hidden_nf=32, device=device, n_layers=3, coords_weight=1.0,
             attention=True, node_attr=1)
# model = PairEGNN(in_node_nf=len(dat.elements), in_edge_nf=0, hidden_nf=128, device=device, n_layers=7, coords_weight=1.0,
#              attention=True, node_attr=1)

optimizer = optim.Adam(model.parameters())
loss_func = nn.MSELoss()

print('Initialized model')

def train(loader):
    model.train()
    epoch_loss = 0
    for data in loader:  # Iterate in batches over the training dataset.
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
        label = data.y.to(device, dtype)
        
        # pred = model(h0_s=one_hot_s, h0_t=one_hot_t, edges_s=edges_s, edges_t=edges_t, edge_attr=None, node_mask_s=atom_mask_s, 
        #             edge_mask_s=edge_mask_s, n_nodes_s=n_nodes_s, node_mask_t=atom_mask_t, edge_mask_t=edge_mask_t, 
        #             n_nodes_t=n_nodes_t, x_s=atom_positions_s, x_t=atom_positions_t)
        pred = model(h0=[one_hot_s, one_hot_t], x=[atom_positions_s, atom_positions_t], all_edges=[edges_s, edges_t],
                         all_edge_attr=[None, None], node_masks=[atom_mask_s, atom_mask_t],
                         edge_masks=[edge_mask_s, edge_mask_t], n_nodes=[n_nodes_s, n_nodes_t])
        loss = loss_func(pred, label)
        epoch_loss += loss.item()
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
    return epoch_loss/len(loader)

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
        label = data.y.to(device, dtype)
        
        # pred = model(h0_s=one_hot_s, h0_t=one_hot_t, edges_s=edges_s, edges_t=edges_t, edge_attr=None, node_mask_s=atom_mask_s, 
        #             edge_mask_s=edge_mask_s, n_nodes_s=n_nodes_s, node_mask_t=atom_mask_t, edge_mask_t=edge_mask_t, 
        #             n_nodes_t=n_nodes_t, x_s=atom_positions_s, x_t=atom_positions_t)
        pred = model(h0=[one_hot_s, one_hot_t], x=[atom_positions_s, atom_positions_t], all_edges=[edges_s, edges_t],
                    all_edge_attr=[None, None], node_masks=[atom_mask_s, atom_mask_t],
                    edge_masks=[edge_mask_s, edge_mask_t], n_nodes=[n_nodes_s, n_nodes_t])
        loss = loss_func(pred, label)
        error += loss.item()
    return error/len(loader)

training_loss = []
testing_loss = []

for epoch in range(1, n_epochs):
    train_loss = train(train_loader)
    test_loss = test(test_loader)
    training_loss.append(train_loss)
    testing_loss.append(test_loss)
    print(f'Epoch: {epoch:03d}, Train MSE: {train_loss:.4f}, Test MSE: {test_loss:.4f}')

torch.save(model.state_dict(), './models/model.pth')

plt.figure(figsize=(12,6))
plt.plot(range(1, n_epochs), training_loss,label='training')
plt.plot(range(1, n_epochs), testing_loss,label='testing')
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.legend()
plt.savefig('result.png')
