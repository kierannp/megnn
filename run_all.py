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
from megnn.models import *
from megnn.utils import *



parser = argparse.ArgumentParser(description='IGNN')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--n_epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--n_layers', type=int, default=5)
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--attention', type=bool, default=False)
parser.add_argument('--n_samples', type=int, default=10000)
parser.add_argument('--reload_data', type=bool, default=True)
parser.add_argument('--dataset', choices=['COF', 'SASA', 'Simple'], default='COF')
parser.add_argument('--gdb_path', default='/raid6/homes/kierannp/projects/megnn/datasets/gdb11/gdb11_size09.smi')
parser.add_argument('--model', choices=['IEGNN','MGNN','IGNN','MEGNN'], default='IEGNN')

args, unparsed_args = parser.parse_known_args()

# clear the processed dataset
if args.reload_data:
    try:
        shutil.rmtree('./processed')
    except:
        pass

def train(model, epoch, loader, device, dtype, criterion, optimizer):
    epoch_loss = 0
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

        if args.model == 'IEGNN':
            int_edges = create_interactional_edges(n_nodes_s, n_nodes_t, device)
            pred = model(
                one_hot_s, 
                edges_s, 
                edge_attr=None, 
                node_attr=None, 
                coord=atom_positions_s, 
                n_nodes_h=n_nodes_s, 
                node_mask=atom_mask_s, 
                int_h=one_hot_t, 
                int_edges=int_edges
            )
        elif args.model == 'MEGNN':
            pred = model(
                h0 = [one_hot_s, one_hot_t], 
                all_edges = [edges_s, edges_t], 
                all_edge_attr = [None, None], 
                node_masks = [atom_mask_s, atom_mask_t], 
                edge_masks = [edge_mask_s, edge_mask_t],
                n_nodes = [n_nodes_s, n_nodes_t], 
                x = [atom_positions_s, atom_positions_t]
            )
        elif args.model == 'MGNN':
            pred = model(
                h0 = [data.x_s, data.x_t], 
                all_edges = [data.edge_index_s, data.edge_index_t], 
                all_edge_attr = [None, None], 
                n_nodes = [data.n_nodes_s, data.n_nodes_t], 
                x = [data.positions_s, data.positions_t]
            )
        elif args.model == 'IGNN':
            int_edges = create_interactional_edges(n_nodes_s, n_nodes_t, device)
            pred = model(
                one_hot_s, 
                edges_s, 
                None, 
                n_nodes_s, 
                atom_mask_s,
                one_hot_t,
                int_edges
            )
        if torch.isnan(data.y).any():
            # raise Exception('Input data has NaNs')
            continue
        loss = criterion(pred, label)  # Compute the loss.
        if loss.item() != loss.item():
            raise Exception("Output went to inf, something exploded!")
        
        epoch_loss += loss.item()
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        if i % 10 == 0:
            print("Epoch %d \t Iteration %d \t loss %.6f" % (epoch, i, loss.item()))
    return epoch_loss/len(loader)

def test(model, loader, device, dtype, criterion):
    model.eval()
    epoch_loss = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data.to(device)
        one_hot_s, one_hot_t,\
        edges_s, edges_t,\
        atom_mask_s, atom_mask_t, \
        edge_mask_s, edge_mask_t, \
        n_nodes_s, n_nodes_t, \
        atom_positions_s,atom_positions_t,\
        batch_size_s, label = convert_to_dense(data, device, dtype)

        if args.model == 'IEGNN':
            int_edges = create_interactional_edges(n_nodes_s, n_nodes_t, device)
            pred = model(
                one_hot_s, 
                edges_s, 
                edge_attr=None, 
                node_attr=None, 
                coord=atom_positions_s, 
                n_nodes_h=n_nodes_s, 
                node_mask=atom_mask_s, 
                int_h=one_hot_t, 
                int_edges=int_edges
            )
        elif args.model == 'MEGNN':
            pred = model(
                h0 = [one_hot_s, one_hot_t], 
                all_edges = [edges_s, edges_t], 
                all_edge_attr = [None, None], 
                node_masks = [atom_mask_s, atom_mask_t], 
                edge_masks = [edge_mask_s, edge_mask_t],
                n_nodes = [n_nodes_s, n_nodes_t], 
                x = [atom_positions_s, atom_positions_t]
            )
        elif args.model == 'MGNN':
            pred = model(
                h0 = [data.x_s, data.x_t], 
                all_edges = [data.edge_index_s, data.edge_index_t], 
                all_edge_attr = [None, None], 
                n_nodes = [data.n_nodes_s, data.n_nodes_t], 
                x = [data.positions_s, data.positions_t]
            )
        elif args.model == 'IGNN':
            int_edges = create_interactional_edges(n_nodes_s, n_nodes_t, device)
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
    return epoch_loss / len(loader)

def main():
    # hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("Using CUDA", flush=True)
    dtype = torch.float32
    # tg.seed.seed_everything(123456)

    # dataset
    if args.dataset == "COF":
        dat = COF_Dataset(root='.')
    elif args.dataset == "SASA":
        dat = SASA_Dataset(
            root='.', 
            smi_path='/raid6/homes/kierannp/projects/megnn/datasets/gdb11/gdb11_size09.smi',
            n_samples=args.n_samples
        )
    elif args.dataset == "Simple":
        dat = Simple_Dataset(
            root='.', 
            smi_path='/raid6/homes/kierannp/projects/megnn/datasets/gdb11/gdb11_size09.smi',
            n_samples=args.n_samples
        )
    else:
        raise Exception("Unknown dataset specified")
    
    train_dataset = dat[:int(len(dat)*.8)]
    test_dataset = dat[int(len(dat)*.8):]
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, follow_batch=['x_s', 'x_t', 'positions_s', 'positions_t'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, follow_batch=['x_s', 'x_t', 'positions_s', 'positions_t'], shuffle=False)

    #model
    if args.model == 'IEGNN':
        model = IEGNN(
            in_node_nf=dat[0].x_s.shape[1], 
            in_edge_nf=0,
            device=device,
            hidden_nf=args.hidden_dim, 
            n_layers=args.n_layers,  
            node_attr=1, 
            act_fn=nn.ReLU(), 
            attention=False
        )
    elif args.model == 'MEGNN':
        model = MEGNN(
            n_graphs=2, 
            in_node_nf=dat[0].x_s.shape[1], 
            in_edge_nf=0, 
            hidden_nf=args.hidden_dim, 
            device=device,
            n_layers=args.n_layers, 
            coords_weight=1.,
            attention=True, 
            node_attr=1
        )
    elif args.model == 'MGNN':
        model = MGNN(
            n_graphs=2, 
            in_node_nf=dat[0].x_s.shape[1], 
            in_edge_nf=0, 
            hidden_nf=args.hidden_dim, 
            device=device, 
            n_layers=8
        )
    elif args.model == 'IGNN':
        model = IGNN(
            in_node_nf=dat[0].x_s.shape[1], 
            in_edge_nf=0, 
            device=device, 
            hidden_nf=args.hidden_dim, 
            n_layers=args.n_layers,  
            node_attr=1, 
            act_fn=nn.ReLU(), 
            attention=False
        )
    else:
        raise Exception('Unknown model specified')

    print(model)
    model.to(device)

    # define optimizer and loss function
    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss().to(device)
    # criterion = MeanAbsolutePercentageError().to(device)

    #training and testing loop
    res = {'epochs': [], 'train_loss': [],'test_loss': [], 'best_test': 1e10, 'best_epoch': 0}
    for epoch in range(0, args.n_epochs):
        train_loss = train(model, epoch, train_loader, device, dtype, criterion, optimizer)
        res['train_loss'].append(train_loss)
        if epoch % 1 == 0:
            test_loss = test(model, test_loader, device, dtype, criterion)
            res['epochs'].append(epoch)
            res['test_loss'].append(test_loss)
            if test_loss < res['best_test']:
                res['best_test'] = test_loss
                res['best_epoch'] = epoch
            print("test loss: %.6f \t epoch %d" % (test_loss, epoch))
            print("Best: test loss: %.6f \t epoch %d" % (res['best_test'], res['best_epoch']))


    torch.save(model.state_dict(), './models/GNN_{}.pth'.format(datetime.now()))


if __name__ == "__main__":
    main()