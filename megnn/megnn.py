import torch
from torch import nn
from gcl import E_GCL

def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)

class E_GCL_mask(E_GCL):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_attr_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False):
        E_GCL.__init__(self, input_nf, output_nf, hidden_nf, edges_in_d=edges_in_d, nodes_att_dim=nodes_attr_dim, act_fn=act_fn, recurrent=recurrent, coords_weight=coords_weight, attention=attention)

        del self.coord_mlp
        self.act_fn = act_fn

    def coord_model(self, coord, edge_index, coord_diff, edge_feat, edge_mask):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat) * edge_mask
        agg = unsorted_segment_sum(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord

    def forward(self, h, edge_index, coord, node_mask, edge_mask, edge_attr=None, node_attr=None, n_nodes=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)

        edge_feat = edge_feat * edge_mask

        # TO DO: edge_feat = edge_feat * edge_mask

        #coord = self.coord_model(coord, edge_index, coord_diff, edge_feat, edge_mask)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)

        return h, coord, edge_attr



class EGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1):
        super(EGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))

        self.node_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                       act_fn,
                                       nn.Linear(self.hidden_nf, 1))
        self.to(self.device)

    def forward(self, h0, x, edges, edge_attr, node_mask, edge_mask, n_nodes):
        h = self.embedding(h0)
        for i in range(0, self.n_layers):
            if self.node_attr:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr, node_attr=h0, n_nodes=n_nodes)
            else:
                h, _, _ = self._modules["gcl_%d" % i](h, edges, x, node_mask, edge_mask, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes)

        h = self.node_dec(h)
        h = h * node_mask
        h = h.view(-1, n_nodes, self.hidden_nf)
        h = torch.sum(h, dim=1)
        pred = self.graph_dec(h)
        return pred.squeeze(1)

class PairEGNN(nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, coords_weight=1.0, attention=False, node_attr=1):
        super(PairEGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers

        ### Encoders
        self.embedding_s = nn.Linear(in_node_nf, hidden_nf)
        self.embedding_t = nn.Linear(in_node_nf, hidden_nf)

        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        ### Graph convolution layers
        for i in range(0, n_layers):
            self.add_module("gcl_s_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))
            self.add_module("gcl_t_%d" % i, E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))

        ### Decoders
        self.node_dec_s = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                      act_fn,
                                      nn.Linear(self.hidden_nf, self.hidden_nf))

        self.node_dec_t = nn.Sequential(nn.Linear(self.hidden_nf, self.hidden_nf),
                                act_fn,
                                nn.Linear(self.hidden_nf, self.hidden_nf))

        self.graph_dec = nn.Sequential(nn.Linear(2*self.hidden_nf, 2*self.hidden_nf),
                                       act_fn,
                                       nn.Linear(2*self.hidden_nf, 1))
        self.to(self.device)

    def forward(self, h0_s, h0_t, edges_s, edges_t, edge_attr, node_mask_s, edge_mask_s, n_nodes_s, node_mask_t, edge_mask_t, n_nodes_t, x_s, x_t):
        h_s = self.embedding_s(h0_s)
        h_t = self.embedding_t(h0_t)

        for i in range(0, self.n_layers):
            if self.node_attr:
                h_s, _, _ = self._modules["gcl_s_%d" % i](h_s, edges_s, x_s, node_mask_s, edge_mask_s, edge_attr=edge_attr, node_attr=h0_s, n_nodes=n_nodes_s)
                h_t, _, _ = self._modules["gcl_t_%d" % i](h_t, edges_t, x_t, node_mask_t, edge_mask_t, edge_attr=edge_attr, node_attr=h0_t, n_nodes=n_nodes_t)
            else:
                h_s, _, _ = self._modules["gcl_s_%d" % i](h_s, edges_s, x_s, node_mask_s, edge_mask_s, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes_s)
                h_t, _, _ = self._modules["gcl_t_%d" % i](h_t, edges_t, x_t, node_mask_t, edge_mask_t, edge_attr=edge_attr,
                                                      node_attr=None, n_nodes=n_nodes_t)

        h_s = self.node_dec_s(h_s)
        h_s = h_s * node_mask_s
        h_s = h_s.view(-1, n_nodes_s, self.hidden_nf)
        h_s = torch.sum(h_s, dim=1)

        h_t = self.node_dec_t(h_t)
        h_t = h_t * node_mask_t
        h_t = h_t.view(-1, n_nodes_t, self.hidden_nf)
        h_t = torch.sum(h_t, dim=1)

        pred = self.graph_dec(torch.cat((h_s,h_t), dim=1))
        return pred.squeeze(1)

class MEGNN(nn.Module):
    def __init__(self, n_graphs, in_node_nf, in_edge_nf, hidden_nf, device, act_fn=nn.SiLU(), 
                n_layers=7, coords_weight=1.0, attention=True, node_attr=1):
        super(MEGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.n_graphs = n_graphs

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for j in range(n_graphs):
            for i in range(n_layers):
                self.add_module("gcl_{}_{}".format(j,i), E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))

        for i in range(n_graphs):
            self.add_module("node_dec_{}".format(i), nn.Sequential( nn.Linear(self.hidden_nf, self.hidden_nf),
                                                                    act_fn,
                                                                    nn.Linear(self.hidden_nf, self.hidden_nf)))
        self.add_module("grand_dec", nn.Sequential(nn.Linear(n_graphs * self.hidden_nf, n_graphs * self.hidden_nf),
                                    act_fn,
                                    nn.Linear(n_graphs * self.hidden_nf, 1)))
        self.to(self.device)

    def forward(self, h0, x, all_edges, all_edge_attr, node_masks, edge_masks, n_nodes, enviro=None):
        hf = []
        for j in range(self.n_graphs):
            h = self.embedding(h0[j])
            node_mask = node_masks[j]
            edge_mask = edge_masks[j]
            edges = all_edges[j]
            node_attr = h0[j]
            x_curr = x[j]
            n_node = n_nodes[j]
            if all_edge_attr is not None:
                edge_attr = all_edge_attr[j]
            else:
                edge_attr = None
            for i in range(0, self.n_layers):
                if self.node_attr:
                    h, _, _ = self._modules["gcl_{}_{}".format(j,i)](h, edges, x_curr, node_mask, edge_mask, edge_attr=edge_attr, node_attr=node_attr, n_nodes=n_node)
                else:
                    h, _, _ = self._modules["gcl_{}_{}".format(j,i)](h, edges, x_curr, node_mask, edge_mask, edge_attr=edge_attr,
                                                        node_attr=None, n_nodes=n_nodes)
            h = self._modules["node_dec_{}".format(j)](h)
            h = h * node_mask
            h = h.view(-1, n_node, self.hidden_nf)
            h = torch.sum(h, dim=1)
            hf.append(h)
        # pred = self.grand_dec(torch.cat(hf, dim=1)) if enviro is None else self.grand_dec(torch.cat((hf,enviro_out), dim=1))
        combined = torch.cat(hf, dim=1)
        pred = self.grand_dec(combined)
        return pred.squeeze(1)
        
class MEGNN_enviro(nn.Module):
    def __init__(self, n_graphs, in_node_nf, in_edge_nf, hidden_nf, device, act_fn=nn.SiLU(), 
                n_layers=7, coords_weight=1.0, attention=True, node_attr=1, n_enviro = 0):
        super(MEGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.n_graphs = n_graphs
        self.n_enviro = n_enviro
        n_enviro_dim = 0 if n_enviro == 0 else 1

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.node_attr = node_attr
        if node_attr:
            n_node_attr = in_node_nf
        else:
            n_node_attr = 0
        for j in range(n_graphs):
            for i in range(n_layers):
                self.add_module("gcl_{}_{}".format(j,i), E_GCL_mask(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_d=in_edge_nf, nodes_attr_dim=n_node_attr, act_fn=act_fn, recurrent=True, coords_weight=coords_weight, attention=attention))

        for i in range(n_graphs):
            self.add_module("node_dec_{}".format(i), nn.Sequential( nn.Linear(self.hidden_nf, self.hidden_nf),
                                                                    act_fn,
                                                                    nn.Linear(self.hidden_nf, self.hidden_nf)))
        if n_enviro != 0:
            self.add_module('enviro_enc', nn.Sequential( nn.Linear(n_enviro, self.hidden_nf),
                                                                    act_fn,
                                                         nn.Linear(self.hidden_nf, self.hidden_nf)) )
        self.add_module("grand_dec", nn.Sequential(nn.Linear((n_graphs+ n_enviro_dim) * self.hidden_nf,(n_graphs+ n_enviro_dim) * self.hidden_nf),
                                    act_fn,
                                    nn.Linear((n_graphs+ n_enviro_dim) * self.hidden_nf, 1)))
        self.to(self.device)

    def forward(self, h0, x, all_edges, all_edge_attr, node_masks, edge_masks, n_nodes, enviro=None):
        hf = []
        for j in range(self.n_graphs):
            h = self.embedding(h0[j])
            node_mask = node_masks[j]
            edge_mask = edge_masks[j]
            edges = all_edges[j]
            node_attr = h0[j]
            x_curr = x[j]
            n_node = n_nodes[j]
            if all_edge_attr is not None:
                edge_attr = all_edge_attr[j]
            else:
                edge_attr = None
            for i in range(0, self.n_layers):
                if self.node_attr:
                    h, _, _ = self._modules["gcl_{}_{}".format(j,i)](h, edges, x_curr, node_mask, edge_mask, edge_attr=edge_attr, node_attr=node_attr, n_nodes=n_node)
                else:
                    h, _, _ = self._modules["gcl_{}_{}".format(j,i)](h, edges, x_curr, node_mask, edge_mask, edge_attr=edge_attr,
                                                        node_attr=None, n_nodes=n_nodes)
            h = self._modules["node_dec_{}".format(j)](h)
            h = h * node_mask
            h = h.view(-1, n_node, self.hidden_nf)
            h = torch.sum(h, dim=1)
            hf.append(h)
        if enviro is not None:
            enviro_out = self.enviro_enc(enviro)
            hf.append(enviro_out)
        # pred = self.grand_dec(torch.cat(hf, dim=1)) if enviro is None else self.grand_dec(torch.cat((hf,enviro_out), dim=1))
        combined = torch.cat(hf, dim=1)
        pred = self.grand_dec(combined)
        return pred.squeeze(1)