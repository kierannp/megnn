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
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_layers', type=int, default=5)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--attention', type=bool, default=False)
parser.add_argument('--n_samples', type=int, default=10000)
parser.add_argument('--dp', type=bool, default=False)

args, unparsed_args = parser.parse_known_args()

def create_interactional_edges(n_nodes_s, n_nodes_t, device, dtype):
    row, col = torch.empty(n_nodes_s*n_nodes_t), torch.empty(n_nodes_s*n_nodes_t)
    s_index = 0
    for i in range(row.shape[0]):
        if i % n_nodes_t == 0 and i != 0:
            s_index += 1
        row[i] = s_index
    for i in range(col.shape[0]):
        s_index = i % n_nodes_t
        col[i] = s_index
    
    return [row.to(device, torch.long), col.to(device, torch.long)]

def train(model, epoch, loader, device, dtype, criterion, optimizer):
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

        int_edges = create_interactional_edges(n_nodes_s, n_nodes_t, device, dtype)
        # label = (label - kd_mean) / kd_std
        pred = model(
            one_hot_s, 
            edges_s, 
            None, 
            n_nodes_s, 
            atom_mask_s,
            one_hot_t,
            int_edges,
        )
            # data.x_s, data.x_t
            # all_edges = [data.edge_index_s, data.edge_index_t], 
            # all_edge_attr = [None, None], 
            # n_nodes = [data.n_nodes_s, data.n_nodes_t], 
            # x = [data.positions_s, data.positions_t]
        if torch.isnan(data.y).any():
            continue
        loss = criterion(pred, data.y)  # Compute the loss.
        if loss.item() != loss.item():
            print("Something exploded!")
        epoch_loss += loss.item()
        total_samples += len(data.batch)
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        if i % 10 == 0:
            print("Epoch %d \t Iteration %d \t loss %.4f" % (epoch, i, loss.item()))
    return epoch_loss/len(loader)

def test(model, loader, device, dtype, criterion):
    model.eval()
    epoch_loss = 0
    total_samples = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data.to(device)
        one_hot_s, one_hot_t,\
        edges_s, edges_t,\
        atom_mask_s, atom_mask_t, \
        edge_mask_s, edge_mask_t, \
        n_nodes_s, n_nodes_t, \
        atom_positions_s,atom_positions_t,\
        batch_size_s, label = convert_to_dense(data, device, dtype)

        int_edges = create_interactional_edges(n_nodes_s, n_nodes_t, device, dtype)
        # label = (label - kd_mean) / kd_std
        pred = model(
            one_hot_s, 
            edges_s, 
            None, 
            n_nodes_s, 
            atom_mask_s,
            one_hot_t,
            int_edges
        )
        # epoch_loss += criterion(pred, (label - prop_mean) / prop_mad).item()*batch_size
        epoch_loss += criterion(pred, data.y).item()
        total_samples += len(data.batch)
    return epoch_loss / len(loader)
def main():
        # hyperparameters
    n_epochs  = args.epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("Using CUDA", flush=True)
    # device = torch.device('cpu')
    dtype = torch.float32
    batch_size = args.batch_size
    tg.seed.seed_everything(123456)

    # dataset
    # dataset
    # dat = COF_Dataset(
    #     root='.', 
    #     smi_path='/raid6/homes/kierannp/projects/megnn/datasets/gdb11/gdb11_size09.smi',
    #     n_samples=args.n_samples
    # )
    dat = COF_Dataset(root='.')
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
    model.to(device)

    # define optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss().to(device)
    # criterion = MeanAbsolutePercentageError().to(device)
    if args.dp and torch.cuda.device_count() > 1:
        print(f'Training using {torch.cuda.device_count()} GPUs', flush=True)
        model = torch.nn.DataParallel(model.cpu())
        model = model.cuda()
    res = {'epochs': [], 'train_loss': [],'test_loss': [], 'best_test': 1e10, 'best_epoch': 0}
    for epoch in range(0, n_epochs):
        train_loss = train(model, epoch, train_loader, device, dtype, criterion, optimizer)
        res['train_loss'].append(train_loss)
        if epoch % 1 == 0:
            test_loss = test(model, test_loader, device, dtype, criterion)
            res['epochs'].append(epoch)
            res['test_loss'].append(test_loss)
            if test_loss < res['best_test']:
                res['best_test'] = test_loss
                res['best_epoch'] = epoch
            print("test loss: %.4f \t epoch %d" % (test_loss, epoch))
            print("Best: test loss: %.4f \t epoch %d" % (res['best_test'], res['best_epoch']))


    torch.save(model.state_dict(), './models/GNN_{}.pth'.format(datetime.now()))

    print("MAPE:{}")

if __name__ == "__main__":
    main()