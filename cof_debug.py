import torch
from torch import nn
import torch_geometric as tg
from torch_geometric.loader import DataLoader
from typing import List, Optional, Union
import torch.utils.data
import shutil
import sys

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

# hyperparameters
n_epochs  = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dtype = torch.float32
batch_size = 32

# dataset
dat = COF_Dataset(root='.')
dat.shuffle()
train_dataset = dat[:int(len(dat)*.8)]
test_dataset = dat[int(len(dat)*.8):]
train_loader = DataLoader(train_dataset, batch_size=batch_size, follow_batch=['x_s', 'x_t', 'positions_s', 'positions_t'], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, follow_batch=['x_s', 'x_t', 'positions_s', 'positions_t'], shuffle=False)

prop_mean, prop_mad = compute_cof_mean_mad(dat.dataframe)

# define the model
model = PairEGNN(in_node_nf=5, in_edge_nf=0, hidden_nf=128, device=device, n_layers=8, coords_weight=1.,
             attention=False, node_attr=1)
print(model)

# define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()


def train(epoch, loader):
    epoch_loss = 0
    model.train()

    for i, data in enumerate(loader):
        one_hot_s, one_hot_t, edges_s, edges_t, atom_mask_s, atom_mask_t, \
        edge_mask_s, edge_mask_t, n_nodes_s, n_nodes_t, atom_positions_s, atom_positions_t, batch_size_s, label = convert_to_dense(data, device, dtype)   

        pred = model(h0_s=one_hot_s, h0_t=one_hot_t, edges_s=edges_s, edges_t=edges_t, edge_attr=None, node_mask_s=atom_mask_s, 
                    edge_mask_s=edge_mask_s, n_nodes_s=n_nodes_s, node_mask_t=atom_mask_t, edge_mask_t=edge_mask_t, 
                    n_nodes_t=n_nodes_t, x_s=atom_positions_s, x_t=atom_positions_t)


        loss = criterion(pred, label)  # Compute the loss.
        epoch_loss += loss.item() * batch_size_s
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

        if i % 10 == 0:
            print("Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, loss.item()))
    return epoch_loss/len(loader)
    
def test(loader):
    model.eval()

    epoch_loss = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        one_hot_s, one_hot_t, edges_s, edges_t, atom_mask_s, atom_mask_t, \
        edge_mask_s, edge_mask_t, n_nodes_s, n_nodes_t, atom_positions_s, atom_positions_t, batch_size_s, label = convert_to_dense(data, device, dtype) 
        pred = model(h0_s=one_hot_s, h0_t=one_hot_t, edges_s=edges_s, edges_t=edges_t, edge_attr=None, node_mask_s=atom_mask_s, 
                    edge_mask_s=edge_mask_s, n_nodes_s=n_nodes_s, node_mask_t=atom_mask_t, edge_mask_t=edge_mask_t, 
                    n_nodes_t=n_nodes_t, x_s=atom_positions_s, x_t=atom_positions_t)

        # epoch_loss += criterion(pred, (label - prop_mean) / prop_mad).item()*batch_size
        epoch_loss += criterion(pred, label).item()*batch_size_s

    return epoch_loss /len(loader)

res = {'epochs': [], 'train_loss': [],'test_loss': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}

for epoch in range(0, n_epochs):
    train_loss = train(epoch, train_loader)
    res['train_loss'].append(train_loss)
    if epoch % 4 == 0:
        test_loss = test(test_loader)
        res['epochs'].append(epoch)
        res['test_loss'].append(test_loss)

        if test_loss < res['best_val']:
            res['best_val'] = test_loss
            res['best_test'] = test_loss
            res['best_epoch'] = epoch
        print("test loss: %.4f \t epoch %d" % (test_loss, epoch))
        print("Best: val loss: %.4f \t test loss: %.4f \t epoch %d" % (res['best_val'], res['best_test'], res['best_epoch']))


    # json_object = json.dumps(res, indent=4)
    # with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
    #     outfile.write(json_object)
