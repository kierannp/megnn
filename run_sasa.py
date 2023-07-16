import torch
from torch import nn, optim
from torchmetrics import MeanAbsolutePercentageError
import torch_geometric as tg
from torch_geometric.loader import DataLoader
from typing import List, Optional, Union
import matplotlib.pyplot as plt
import shutil
import sys
from datetime import datetime
import argparse
# My imports
sys.path.append('~/projects/megnn')
from megnn.datasets import *
from megnn.megnn import *
from megnn.utils import *

# clear the processed dataset
try:
    shutil.rmtree('./processed')
except:
    pass


parser = argparse.ArgumentParser(description='SASA IEGNN')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_layers', type=int, default=5)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--attention', type=bool, default=False)
parser.add_argument('--n_samples', type=int, default=10000)

args, unparsed_args = parser.parse_known_args()

# hyperparameters
n_epochs  = args.epochs
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
batch_size = args.batch_size
tg.seed.seed_everything(123456)

# dataset
dat = SASA_Dataset(
    root='.', 
    smi_path='/raid6/homes/kierannp/projects/megnn/datasets/gdb11/gdb11_size09.smi',
    n_samples=args.n_samples
)
train_dataset = dat[:int(len(dat)*.8)]
test_dataset = dat[int(len(dat)*.8):]
train_loader = DataLoader(train_dataset, batch_size=batch_size, follow_batch=['x_s', 'x_t', 'positions_s', 'positions_t'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, follow_batch=['x_s', 'x_t', 'positions_s', 'positions_t'], shuffle=False)
place_holder = dat[0]

# define the model
model = MEGNN(
    n_graphs=2, 
    in_node_nf=5, 
    in_edge_nf=0, 
    hidden_nf=args.hidden_dim, 
    device=device, 
    n_layers=args.n_layers, 
    coords_weight=1.,
    attention=args.attention, 
    node_attr=1
)
print(model)

# define optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

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
    epoch_loss = 0
    total_samples = 0
    for data in loader:  # Iterate in batches over the training/test dataset.

        one_hot_s, one_hot_t,\
        edges_s, edges_t,\
        atom_mask_s, atom_mask_t, \
        edge_mask_s, edge_mask_t, \
        n_nodes_s, n_nodes_t, \
        atom_positions_s,atom_positions_t,\
        batch_size_s, label = convert_to_dense(data, device, dtype)

        pred = model(
            h0 = [one_hot_s, one_hot_t], 
            all_edges = [edges_s, edges_t], 
            all_edge_attr = [None, None], 
            node_masks = [atom_mask_s, atom_mask_t], 
            edge_masks = [edge_mask_s, edge_mask_t],
            n_nodes = [n_nodes_s, n_nodes_t], 
            x = [atom_positions_s, atom_positions_t]
        )

        epoch_loss += criterion(pred, label).item()
        total_samples += batch_size_s
    return epoch_loss / total_samples

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
plt.savefig('training.png')
# torch.save(model.state_dict(), './models/MEGNN_{}.pth'.format(datetime.now()))
