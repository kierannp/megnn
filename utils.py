import torch
import torch_geometric as tg

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


def convert_to_dense(data, device, dtype):
        dense_positions_s, atom_mask_s = tg.utils.to_dense_batch(data.positions_s, data.positions_s_batch)
        dense_positions_t, atom_mask_t = tg.utils.to_dense_batch(data.positions_t, data.positions_t_batch)
        batch_size_s, n_nodes_s, _ = dense_positions_s.size()
        batch_size_t, n_nodes_t, _ = dense_positions_t.size()
        atom_positions_s = dense_positions_s.view(batch_size_s * n_nodes_s, -1).to(device, dtype)
        atom_positions_t = dense_positions_t.view(batch_size_t * n_nodes_t, -1).to(device, dtype)

        edge_mask_s = atom_mask_s.unsqueeze(1) * atom_mask_s.unsqueeze(2)
        #mask diagonal
        diag_mask = ~torch.eye(edge_mask_s.size(1), dtype=torch.bool).unsqueeze(0)
        edge_mask_s *= diag_mask
        edge_mask_s = edge_mask_s.view(batch_size_s * n_nodes_s * n_nodes_s, 1).to(device)
        edge_mask_t = atom_mask_t.unsqueeze(1) * atom_mask_t.unsqueeze(2)
        #mask diagonal
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

        return one_hot_s, one_hot_t, edges_s, edges_t, atom_mask_s, atom_mask_t, \
        edge_mask_s, edge_mask_t, n_nodes_s, n_nodes_t, atom_positions_s, atom_positions_t, batch_size_s, label