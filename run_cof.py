import torch
from torch import nn, optim
import torch_geometric as tg
from torch_geometric.loader import DataLoader
from typing import List, Optional, Union
import shutil
import sys
# from easydict import EasyDict as edict
# My imports
sys.path.append('~/projects/multi-egnn')
from megnn.datasets import *
from megnn.megnn import *
from megnn.utils import *

# clear the processed dataset
try:
    shutil.rmtree('./processed')
except:
    pass

# hyperparameters
n_epochs  = 80
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
model = MEGNN(n_graphs=2, in_node_nf=110, in_edge_nf=0, hidden_nf=256, device=device, n_layers=8, coords_weight=1.,
             attention=False, node_attr=1)
print(model)
model = model.to(device)

# define optimizer and loss function
optimizer = optim.Adam(model.parameters())
criterion = nn.MSELoss()

def train(epoch, loader):
    epoch_loss = 0
    total_samples = 0
    model.train()
    for i, data in enumerate(loader):
        conversion = convert_to_dense(data, device, dtype)

        one_hot_s = conversion[0]
        one_hot_t = conversion[1]
        edges_s = conversion[2]
        edges_t = conversion[3]
        atom_mask_s = conversion[4]
        atom_mask_t = conversion[5]
        edge_mask_s = conversion[6]
        edge_mask_t = conversion[7]
        n_nodes_s = conversion[8]
        n_nodes_t = conversion[9]
        atom_positions_s = conversion[10]
        atom_positions_t = conversion[11]
        batch_size_s = conversion[12]
        label = conversion[13]

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
        conversion = convert_to_dense(data, device, dtype)

        one_hot_s = conversion[0]
        one_hot_t = conversion[1]
        edges_s = conversion[2]
        edges_t = conversion[3]
        atom_mask_s = conversion[4]
        atom_mask_t = conversion[5]
        edge_mask_s = conversion[6]
        edge_mask_t = conversion[7]
        n_nodes_s = conversion[8]
        n_nodes_t = conversion[9]
        atom_positions_s = conversion[10]
        atom_positions_t = conversion[11]
        batch_size_s = conversion[12]
        label = conversion[13]

        pred = model(
            h0 = [one_hot_s, one_hot_t], 
            all_edges = [edges_s, edges_t], 
            all_edge_attr = [None, None], 
            node_masks = [atom_mask_s, atom_mask_t], 
            edge_masks = [edge_mask_s, edge_mask_t],
            n_nodes = [n_nodes_s, n_nodes_t], 
            x = [atom_positions_s, atom_positions_t]
        )

        # epoch_loss += criterion(pred, (label - prop_mean) / prop_mad).item()*batch_size
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


torch.save(model.state_dict(), './models/model.pth')
    # json_object = json.dumps(res, indent=4)
    # with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
    #     outfile.write(json_object)
