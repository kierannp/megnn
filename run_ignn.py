import torch
from torch import nn, optim
import torch_geometric as tg
from torch_geometric.loader import DataLoader
from torchmetrics import MeanAbsolutePercentageError
from typing import List, Optional, Union
import shutil
from datetime import datetime
import argparse
import sys
# My imports
sys.path.append('~/projects/megnn')
from megnn.datasets import *
from megnn.megnn import *
from megnn.utils import *

# clear the processed dataset
# try:
#     shutil.rmtree('./processed')
# except:
#     pass

parser = argparse.ArgumentParser(description='IGNN')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--epochs', type=int, default=2)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_layers', type=int, default=5)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--attention', type=bool, default=False)
parser.add_argument('--n_samples', type=int, default=10000)

args, unparsed_args = parser.parse_known_args()

# hyperparameters
n_epochs  = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
dtype = torch.float32
batch_size = 1
tg.seed.seed_everything(123456)

# dataset
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
# prop_mean, prop_mad = compute_cof_mean_mad(dat.dataframe)

# define the model
model = IGNN(
    in_node_nf=dat[0].x_s.shape[1], 
    in_edge_nf=0, 
    device=device, 
    hidden_nf=128, 
    n_layers=8,  
    node_attr=1, 
    act_fn=nn.ReLU(), 
    attention=False
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
        data.to(device)
        one_hot_s, one_hot_t,\
        edges_s, edges_t,\
        atom_mask_s, atom_mask_t, \
        edge_mask_s, edge_mask_t, \
        n_nodes_s, n_nodes_t, \
        atom_positions_s,atom_positions_t,\
        batch_size_s, label = convert_to_dense(data, device, dtype)

        # label = (label - kd_mean) / kd_std
        pred = model(
            one_hot_s, 
            edges_s, 
            None, 
            one_hot_t
        )
            # data.x_s, data.x_t
            # all_edges = [data.edge_index_s, data.edge_index_t], 
            # all_edge_attr = [None, None], 
            # n_nodes = [data.n_nodes_s, data.n_nodes_t], 
            # x = [data.positions_s, data.positions_t]
        loss = criterion(pred, data.y)  # Compute the loss.
        epoch_loss += loss.item()
        total_samples += len(data.batch)
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

        pred = model(
            h0 = [data.x_s, data.x_t], 
            all_edges = [data.edge_index_s, data.edge_index_t], 
            all_edge_attr = [None, None], 
            n_nodes = [data.n_nodes_s, data.n_nodes_t], 
            x = [data.positions_s, data.positions_t]
        )

        # epoch_loss += criterion(pred, (label - prop_mean) / prop_mad).item()*batch_size
        epoch_loss += criterion(pred, data.y).item()
        total_samples += len(data.batch)
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


torch.save(model.state_dict(), './models/GNN_{}.pth'.format(datetime.now()))
