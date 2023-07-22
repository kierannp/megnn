import torch
import torch_geometric as tg
from rdkit import Chem
import rdkit
import networkx as nx
from glob import glob
import os
import subprocess

def compute_mean_mad(dataloaders, label_property):
    values = dataloaders['train'].dataset.data[label_property]
    meann = torch.mean(values)
    ma = torch.abs(values - meann)
    mad = torch.mean(ma)
    return meann, mad

edges_dic = {}
def get_adj_matrix(n_nodes, batch_size, device):
    if n_nodes in edges_dic:
        edges_dic_b = edges_dic[n_nodes]
        if batch_size in edges_dic_b:
            return edges_dic_b[batch_size]
        else:
            # get edges for a single sample
            rows, cols = [], []
            for batch_idx in range(batch_size):
                for i in range(n_nodes):
                    for j in range(n_nodes):
                        rows.append(i + batch_idx*n_nodes)
                        cols.append(j + batch_idx*n_nodes)

    else:
        edges_dic[n_nodes] = {}
        return get_adj_matrix(n_nodes, batch_size, device)

    edges = [torch.LongTensor(rows).to(device), torch.LongTensor(cols).to(device)]
    return edges

def preprocess_input(one_hot, charges, charge_power, charge_scale, device):
    charge_tensor = (charges.unsqueeze(-1) / charge_scale).pow(
        torch.arange(charge_power + 1., device=device, dtype=torch.float32))
    charge_tensor = charge_tensor.view(charges.shape + (1, charge_power + 1))
    atom_scalars = (one_hot.unsqueeze(-1) * charge_tensor).view(charges.shape[:2] + (-1,))
    return atom_scalars

def compute_cof_mean_mad(dataframe):
    values = torch.Tensor(dataframe['COF'])
    mean = torch.mean(values)
    ma = torch.abs(values - mean)
    mad = torch.mean(ma)
    return mean, mad

def convert_to_dense(data, device, dtype, graphs = 2):
        if graphs == 2:
            dense_positions_s, atom_mask_s = tg.utils.to_dense_batch(data.positions_s, data.positions_s_batch)
            dense_positions_t, atom_mask_t = tg.utils.to_dense_batch(data.positions_t, data.positions_t_batch)
            batch_size_s, n_nodes_s, _ = dense_positions_s.size()
            batch_size_t, n_nodes_t, _ = dense_positions_t.size()
            atom_positions_s = dense_positions_s.view(batch_size_s * n_nodes_s, -1).to(device, dtype)
            atom_positions_t = dense_positions_t.view(batch_size_t * n_nodes_t, -1).to(device, dtype)

            edge_mask_s = (atom_mask_s.unsqueeze(1) * atom_mask_s.unsqueeze(2)).to(device)
            #mask diagonal
            diag_mask = ~torch.eye(edge_mask_s.size(1), dtype=torch.bool).unsqueeze(0).to(device)
            edge_mask_s *= diag_mask
            edge_mask_s = edge_mask_s.view(batch_size_s * n_nodes_s * n_nodes_s, 1).to(device)
            edge_mask_t = (atom_mask_t.unsqueeze(1) * atom_mask_t.unsqueeze(2)).to(device)
            #mask diagonal
            diag_mask = ~torch.eye(edge_mask_t.size(1), dtype=torch.bool).unsqueeze(0).to(device)
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
            edges_s[0].to(device)
            edges_s[1].to(device)
            edges_t[0].to(device)
            edges_t[1].to(device)

            label = data.y.to(device, dtype)

            return one_hot_s, one_hot_t, edges_s, edges_t, atom_mask_s, atom_mask_t, \
            edge_mask_s, edge_mask_t, n_nodes_s, n_nodes_t, atom_positions_s, atom_positions_t, batch_size_s, label
        if graphs == 1:
            dense_positions, atom_mask = tg.utils.to_dense_batch(data.pos, data.pos_batch)
            batch_size, n_nodes, _ = dense_positions.size()
            atom_positions = dense_positions.view(batch_size * n_nodes, -1).to(device, dtype)

            edge_mask = atom_mask.unsqueeze(1) * atom_mask.unsqueeze(2)
            #mask diagonal
            diag_mask = ~torch.eye(edge_mask.size(1), dtype=torch.bool).unsqueeze(0)
            edge_mask *= diag_mask
            edge_mask = edge_mask.view(batch_size * n_nodes * n_nodes, 1).to(device)
            atom_mask = atom_mask.view(batch_size * n_nodes, -1).to(device)

            one_hot, one_hot_mask = tg.utils.to_dense_batch(data.x, data.x_batch)
            one_hot = one_hot.view(batch_size * n_nodes, -1).to(device)

            edges = get_adj_matrix(n_nodes, batch_size, device)
            edges[0].to(device)
            edges[1].to(device)

            label = data.y.to(device, dtype)

            return one_hot, edges, atom_mask, edge_mask, n_nodes, atom_positions, batch_size, label
        else:
            raise ValueError('Only dense conversion of 1 or 2 graphs is implemented')

def mol_from_graph(node_list, adj_list):
    # create empty editable mol object
    mol = Chem.RWMol()

    # add atoms to mol and keep track of index
    node_to_id = {}
    for i in node_list:
        a = Chem.Atom(i.element.symbol)
        molId = mol.AddAtom(a)
        node_to_id[i] = molId

    bond = 1
    # add bonds between adjacent atoms
    for ix, ele in enumerate(adj_list):
            if bond == 1:
                bond_type = Chem.rdchem.BondType.SINGLE
                for b in adj_list[ele]:
                    try:
                        mol.AddBond(node_to_id[ele], node_to_id[b], bond_type)
                    except:
                        pass
            # elif bond == 2:
                # bond_type = Chem.rdchem.BondType.DOUBLE
                # mol.AddBond(node_to_idx[ix], node_to_idx[iy], bond_type)

    # Convert RWMol to Mol object
    mol = mol.GetMol()

    return mol

def mol_to_nx(mol):
    G = nx.Graph()

    for atom in mol.GetAtoms():
        G.add_node(atom.GetIdx(),
                atomic_num=atom.GetAtomicNum(),
                formal_charge=atom.GetFormalCharge(),
                chiral_tag=atom.GetChiralTag(),
                hybridization=atom.GetHybridization(),
                num_explicit_hs=atom.GetNumExplicitHs(),
                is_aromatic=atom.GetIsAromatic())
    for bond in mol.GetBonds():
        G.add_edge(bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond_type=bond.GetBondType())
    return G

def atomtype(filename, root, pname):
    os.chdir('{}/{}'.format(root, pname))
    subprocess.run('grep -v HOH {} > {}_clean.pdb'.format(filename, pname+'_pocket'), shell=True)
    subprocess.run('module load gromacs && gmx pdb2gmx -f {}_clean.pdb -ff amber94 -water tip3p'.format(pname+'_pocket'), shell=True)
    if os.path.exists('topol.top') and os.path.exists('conf.gro'):
        result = True
    else:
        result = False
    subprocess.run('cd -', shell=True)
    return result

def create_interactional_edges(n_nodes_s, n_nodes_t, device):
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