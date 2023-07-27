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
# from megnn.datasets import *
from megnn.old_datasets import COF_Dataset, SASA_Dataset
from megnn.datasets import Simple_Dataset
from megnn.models import *
from megnn.utils import *
import copy



parser = argparse.ArgumentParser(description='Interactional Graph Neural Network Script')
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--n_epochs', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--n_layers', type=int, default=8)
parser.add_argument('--n_node_layers', type=int, default=5)
parser.add_argument('--n_int_layers', type=int, default=8)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--n_samples', type=int, default=5000)
parser.add_argument('--reload_data', type=bool, default=True)
parser.add_argument('--dataset', choices=['COF', 'SASA', 'Simple'], default='COF')
parser.add_argument('--gdb_path', default='/raid6/homes/kierannp/projects/megnn/datasets/gdb11/gdb11_size09.smi')
parser.add_argument('--model', choices=['IEGNN','MGNN','IGNN','MEGNN', 'MMLP'], default='MEGNN')
parser.add_argument('--attention', type=bool, default=False)
parser.add_argument('--act_fn', choices=['relu','silu'], default='silu')
parser.add_argument('--prop', choices=['COF','intercept'], default='COF')
parser.add_argument('--standardize',  default=True, action='store_false')
parser.add_argument('--mix', choices=['50','25', 'both'], default='50')
parser.add_argument('--seed', type=int, default=None)


args, unparsed_args = parser.parse_known_args()

# clear the processed dataset
if args.reload_data:
    try:
        shutil.rmtree('./processed')
    except:
        pass
if args.seed is not None:
    tg.seed.seed_everything(args.seed)

def main():
    # hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("Using CUDA", flush=True)
    dtype = torch.float32

    # dataset
    if args.dataset == "COF":
        dat = COF_Dataset(
            root='.',
            prop=args.prop,
            mix=args.mix,
            standardize = args.standardize
        )
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
    
    if args.act_fn == "silu":
        act_fn = nn.SiLU().to(device)
    else:
        act_fn = nn.ReLU().to(device)
    
    train_dataset = dat[:int(len(dat)*.8)]
    test_dataset = dat[int(len(dat)*.8):]
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, follow_batch=['h_0', 'h_1', 'x_0', 'x_1'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, follow_batch=['h_0', 'h_1', 'x_0', 'x_1'], shuffle=False)

    #model
    if args.model == 'IEGNN':
        model = IEGNN(
            in_node_nf=dat[0].h_0.shape[1], 
            in_edge_nf=0,
            device=device,
            hidden_nf=args.hidden_dim, 
            n_layers=args.n_layers,  
            node_attr=1, 
            act_fn=act_fn, 
            attention=args.attention,
            n_node_layers=args.n_node_layers,
            n_int_layers=args.n_int_layers
        )
    elif args.model == 'MEGNN':
        model = MEGNN(
            n_graphs=2, 
            in_node_nf=dat[0].h_0.shape[1], 
            in_edge_nf=0, 
            hidden_nf=args.hidden_dim, 
            device=device,
            n_layers=args.n_layers, 
            coords_weight=1.,
            attention=args.attention, 
            node_attr=1,
            act_fn=act_fn
            # n_node_layers=args.n_node_layers
        )
    elif args.model == 'MGNN':
        model = MGNN(
            n_graphs=2, 
            in_node_nf=dat[0].h_0.shape[1], 
            in_edge_nf=0, 
            hidden_nf=args.hidden_dim, 
            device=device, 
            n_layers=args.n_layers,
            act_fn = act_fn
            # n_node_layers=args.n_node_layers
        )
    elif args.model == 'IGNN':
        model = IGNN(
            in_node_nf=dat[0].h_0.shape[1], 
            in_edge_nf=0, 
            device=device, 
            hidden_nf=args.hidden_dim, 
            n_layers=args.n_layers,  
            node_attr=1, 
            act_fn=act_fn, 
            attention=args.attention,
            n_node_layers=args.n_node_layers
        )
    elif args.model == 'MMLP':
        model = MMLP(
            in_nf=dat[0].h_0.shape[1] - 5, 
            device=device, 
            hidden_nf=args.hidden_dim, 
            n_layers=args.n_layers,   
            act_fn=nn.ReLU()
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
                best_model = copy.deepcopy(model)
            print("test loss: %.6f \t epoch %d" % (test_loss, epoch))
            print("Best: test loss: %.6f \t epoch %d" % (res['best_test'], res['best_epoch']))

    torch.save(model.state_dict(), './saved_models/{}_{}.pth'.format(args.model, datetime.now()))
    best_model.eval()
    predictions, actuals = [], []
    if args.prop == 'COF' and args.standardize and args.dataset == "COF":
        dat.mean = 0.13767896130130278
        dat.std = 0.014303609422571644
    if args.prop == 'intercept' and args.standardize and args.dataset == "COF":
        dat.mean = 1.643974365146937
        dat.std = 1.1279177781337186
    for i, data in enumerate(test_loader):
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
        data.to(device)

        if args.model == 'IEGNN':
            int_edges = create_interactional_edges(n_nodes_s, n_nodes_t, device)
            pred = best_model(
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
            pred = best_model(
                h0 = [one_hot_s, one_hot_t], 
                all_edges = [edges_s, edges_t], 
                all_edge_attr = [None, None], 
                node_masks = [atom_mask_s, atom_mask_t], 
                edge_masks = [edge_mask_s, edge_mask_t],
                n_nodes = [n_nodes_s, n_nodes_t], 
                x = [atom_positions_s, atom_positions_t]
            )
        elif args.model == 'MGNN':
            pred = best_model(
                h0 = [data.h_0, data.h_1], 
                all_edges = [data.edges_0.to(torch.int64), data.edges_1.to(torch.int64)],
                all_edge_attr = [None, None], 
                n_nodes = [data.n_nodes_s, data.n_nodes_t], 
                x = [data.x_0, data.x_1],
                batches=[data.h_0_batch, data.h_1_batch]
            )
        elif args.model == 'IGNN':
            int_edges = create_interactional_edges(n_nodes_s, n_nodes_t, device)
            pred = best_model(
                one_hot_s, 
                edges_s, 
                None, 
                n_nodes_s, 
                atom_mask_s,
                one_hot_t,
                int_edges
            )
        elif args.model == "MMLP":
            mol1 = one_hot_s[data.n_nodes_s - torch.ones_like(data.n_nodes_s)]
            mol2 = one_hot_t[data.n_nodes_t - torch.ones_like(data.n_nodes_t)]
            mol1 = mol1[:,5:]
            mol2 = mol2[:,5:]
            pred = best_model(
                v1 = mol1,
                v2 = mol2
            )
            pred = pred.squeeze(1)
        # loss = criterion(pred, label)  # Compute the loss.
        if torch.cuda.is_available():
            if args.standardize:
                predictions.extend(list(pred.detach().cpu().numpy()*dat.std+dat.mean))
                actuals.extend(list(label.detach().cpu().numpy()*dat.std+dat.mean))
            else:
                predictions.extend(list(pred.detach().cpu().numpy()))
                actuals.extend(list(label.detach().cpu().numpy()))
        else:
            if args.standardize:
                predictions.extend(list(pred.detach().numpy()*dat.std+dat.mean))
                actuals.extend(list(label.detach().numpy()*dat.std+dat.mean))
            else:
                predictions.extend(list(pred.detach().cpu().numpy()))
                actuals.extend(list(label.detach().cpu().numpy()))

    mape = MeanAbsolutePercentageError().to(device)
    print("MAPE: {}".format(mape(torch.tensor(predictions), torch.tensor(actuals))))
    print("MSE: {}".format(criterion(torch.tensor(predictions), torch.tensor(actuals))))

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
                h=one_hot_s, 
                edges=edges_s, 
                edge_attr=None, 
                node_attr=None, 
                edge_mask=edge_mask_s,
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
                h0 = [data.h_0, data.h_1], 
                all_edges = [data.edges_0.to(torch.int64), data.edges_1.to(torch.int64)], 
                all_edge_attr = [None, None], 
                n_nodes = [data.n_nodes_s, data.n_nodes_t], 
                x = [data.x_0, data.x_1],
                batches=[data.h_0_batch, data.h_1_batch]
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
        elif args.model == "MMLP":
            mol1 = one_hot_s[data.n_nodes_s - torch.ones_like(data.n_nodes_s)]
            mol2 = one_hot_t[data.n_nodes_t - torch.ones_like(data.n_nodes_t)]
            mol1 = mol1[:,5:]
            mol2 = mol2[:,5:]
            pred = model(
                v1 = mol1,
                v2 = mol2
            )
        if torch.isnan(data.y).any():
            # raise Exception('Input data has NaNs')
            continue
        # label = torch.exp(label)
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
                h=one_hot_s, 
                edges=edges_s, 
                edge_attr=None, 
                node_attr=None, 
                coord=atom_positions_s, 
                n_nodes_h=n_nodes_s, 
                edge_mask=edge_mask_s,
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
                h0 = [data.h_0, data.h_1], 
                all_edges = [data.edges_0.to(torch.int64), data.edges_1.to(torch.int64)], 
                all_edge_attr = [None, None], 
                n_nodes = [data.n_nodes_s, data.n_nodes_t], 
                x = [data.x_0, data.x_1],
                batches = [data.h_0_batch, data.h_1_batch]
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
        elif args.model == "MMLP":
            mol1 = one_hot_s[data.n_nodes_s - torch.ones_like(data.n_nodes_s)]
            mol2 = one_hot_t[data.n_nodes_t - torch.ones_like(data.n_nodes_t)]
            mol1 = mol1[:,5:]
            mol2 = mol2[:,5:]
            pred = model(
                v1 = mol1,
                v2 = mol2
            )
        if torch.isnan(data.y).any():
            # raise Exception('Input data has NaNs')
            continue
        # label = torch.exp(label)
        loss = criterion(pred, label)  # Compute the loss.
        if loss.item() != loss.item():
            raise Exception("Output went to inf, something exploded!")
        epoch_loss += loss.item()
    return epoch_loss / len(loader)

if __name__ == "__main__":
    main()