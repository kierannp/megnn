import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool

class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nh),
            nn.LeakyReLU(0.2),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x)

class GCL_basic(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self):
        super(GCL_basic, self).__init__()


    def edge_model(self, source, target, edge_attr):
        pass

    def node_model(self, h, edge_index, edge_attr):
        pass

    def forward(self, x, edge_index, edge_attr=None):
        row, col = edge_index
        edge_feat = self.edge_model(x[row], x[col], edge_attr)
        x = self.node_model(x, edge_index, edge_feat)
        return x, edge_feat

class GCL(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_nf=0, act_fn=nn.ReLU(), bias=True, attention=False, t_eq=False, recurrent=True):
        super(GCL, self).__init__()
        self.attention = attention
        self.t_eq=t_eq
        self.recurrent = recurrent
        input_edge_nf = input_nf * 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge_nf + edges_in_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf, bias=bias),
            act_fn)
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(input_nf, hidden_nf, bias=bias),
                act_fn,
                nn.Linear(hidden_nf, 1, bias=bias),
                nn.Sigmoid())


        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, output_nf, bias=bias))

        #if recurrent:
            #self.gru = nn.GRUCell(hidden_nf, hidden_nf)


    def edge_model(self, source, target, edge_attr):
        edge_in = torch.cat([source, target], dim=1)
        if edge_attr is not None:
            edge_in = torch.cat([edge_in, edge_attr], dim=1)
        out = self.edge_mlp(edge_in)
        if self.attention:
            att = self.att_mlp(torch.abs(source - target))
            out = out * att
        return out

    def node_model(self, h, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=h.size(0))
        out = torch.cat([h, agg], dim=1)
        out = self.node_mlp(out)
        if self.recurrent:
            out = out + h
            #out = self.gru(out, h)
        return out

class I_GCL(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, int_node_nf, edges_in_nf=0, act_fn=nn.ReLU(), bias=True, attention=False, t_eq=False, recurrent=True):
        super(I_GCL, self).__init__()
        self.attention = attention
        self.t_eq=t_eq
        self.recurrent = recurrent
        input_edge_nf = input_nf * 2
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge_nf + edges_in_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf, bias=bias),
            act_fn)
        
        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(input_nf, hidden_nf, bias=bias),
                act_fn,
                nn.Linear(hidden_nf, 1, bias=bias),
                nn.Sigmoid())

        self.interaction_mlp = nn.Sequential(
            nn.Linear(input_edge_nf, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf, bias=bias),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf*3, hidden_nf, bias=bias),
            act_fn,
            nn.Linear(hidden_nf, output_nf, bias=bias))
        

    def edge_model(self, source, target, edge_attr):
        edge_in = torch.cat([source, target], dim=1)
        if edge_attr is not None:
            edge_in = torch.cat([edge_in, edge_attr], dim=1)
        out = self.edge_mlp(edge_in)
        if self.attention:
            att = self.att_mlp(torch.abs(source - target))
            out = out * att
        return out

    def interaction_model(self, source, target):
        edge_in = torch.cat([source, target], dim=1)
        out = self.interaction_mlp(edge_in)
        if self.attention:
            att = self.att_mlp(torch.abs(source - target))
            out = out * att
        return out
    
    def node_model(self, h, edge_index, edge_attr, int_index, int_attr):
        row, col = edge_index
        int_row, int_col = int_index
        edge_agg = unsorted_segment_sum(edge_attr, row, num_segments=h.size(0))
        int_agg = unsorted_segment_sum(int_attr, int_row, num_segments=h.size(0))
        out = torch.cat([h, edge_agg, int_agg], dim=1)
        out = self.node_mlp(out)
        if self.recurrent:
            out = out + h
        return out
    
    def forward(self, x, edge_index, int_x, int_index, edge_attr = None):
        row, col = edge_index
        int_row, int_col = int_index
        edge_feat = self.edge_model(x[row], x[col], edge_attr)
        int_feat = self.interaction_model(x[int_row], int_x[int_col])
        x = self.node_model(x, edge_index, edge_feat, int_index, int_feat)
        return x, edge_feat
    
class GCL_rf(GCL_basic):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, nf=64, edge_attr_nf=0, reg=0, act_fn=nn.LeakyReLU(0.2), clamp=False):
        super(GCL_rf, self).__init__()

        self.clamp = clamp
        layer = nn.Linear(nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)
        self.phi = nn.Sequential(nn.Linear(edge_attr_nf + 1, nf),
                                 act_fn,
                                 layer)
        self.reg = reg

    def edge_model(self, source, target, edge_attr):
        x_diff = source - target
        radial = torch.sqrt(torch.sum(x_diff ** 2, dim=1)).unsqueeze(1)
        e_input = torch.cat([radial, edge_attr], dim=1)
        e_out = self.phi(e_input)
        m_ij = x_diff * e_out
        if self.clamp:
            m_ij = torch.clamp(m_ij, min=-100, max=100)
        return m_ij

    def node_model(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_mean(edge_attr, row, num_segments=x.size(0))
        x_out = x + agg - x*self.reg
        return x_out

class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, nodes_att_dim=0, act_fn=nn.ReLU(), recurrent=True, coords_weight=1.0, attention=False, clamp=False, norm_diff=False, tanh=False):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1


        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)


        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        #if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)


    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord


    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff/(norm)

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr

class I_E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph.
    Args:
          hidden_dim: Number of hidden units.
          num_nodes: Maximum number of nodes (for self-attentive pooling).
          global_agg: Global aggregation function ('attn' or 'sum').
          temp: Softmax temperature.
    """

    def __init__(
            self, 
            input_nf, 
            output_nf, 
            hidden_nf, 
            edges_in_d=0, 
            nodes_att_dim=0, 
            act_fn=nn.ReLU(), 
            recurrent=True, 
            coords_weight=1.0, 
            attention=False, 
            clamp=False, 
            norm_diff=False, 
            tanh=False
    ):
        super(I_E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1


        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)
        
        # not implementing interactional edge features yet
        self.interaction_mlp = nn.Sequential(
            nn.Linear(input_edge, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf*2, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = []
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(nn.Linear(hidden_nf, hidden_nf))
        coord_mlp.append(act_fn)
        coord_mlp.append(layer)
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1))*3
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid())

        #if recurrent:
        #    self.gru = nn.GRUCell(hidden_nf, hidden_nf)


    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1)
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)
        out = self.edge_mlp(out)
        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def interaction_model(self, source, target):
        edge_in = torch.cat([source, target], dim=1)
        out = self.interaction_mlp(edge_in)
        if self.attention:
            att = self.att_mlp(torch.abs(source - target))
            out = out * att
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr, int_index, int_attr):
        row, col = edge_index
        int_row, int_col = int_index
        edge_agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        int_agg = unsorted_segment_sum(int_attr, int_row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, edge_agg, int_agg, node_attr], dim=1)
        else:
            agg = torch.cat([x, edge_agg, int_agg], dim=1)
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100) #This is never activated but just in case it case it explosed it may save the train
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord += agg*self.coords_weight
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum((coord_diff)**2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff/(norm)

        return radial, coord_diff

    def forward(self, h, edge_index, coord, int_h, int_index, edge_attr=None, node_attr=None):
        row, col = edge_index
        int_row, int_col = int_index
        radial, coord_diff = self.coord2radial(edge_index, coord)

        edge_feat = self.edge_model(h[row], h[col], radial, edge_attr)
        int_feat = self.interaction_model(h[int_row], int_h[int_col])
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr, int_index=int_index, int_attr=int_feat)
        # coord = self.node_coord_model(h, coord)
        # x = self.node_model(x, edge_index, x[col], u, batch)  # GCN
        return h, coord, edge_attr

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

class MEGNN(nn.Module):
    def __init__(
            self, 
            n_graphs, 
            in_node_nf, 
            in_edge_nf, 
            hidden_nf, 
            device, 
            act_fn=nn.SiLU(), 
            n_layers=7, 
            coords_weight=1.0, 
            attention=True, 
            node_attr=1
        ):
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

    def forward(self, h0, x, all_edges, all_edge_attr, node_masks, edge_masks, n_nodes):
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

class MGNN(torch.nn.Module):
    def __init__(self, n_graphs, in_node_nf, in_edge_nf, device, hidden_nf=128, n_layers=8, act_fn=nn.ReLU()):
        super(MGNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.n_graphs = n_graphs

        ### Encoder
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        for j in range(n_graphs):
            for i in range(n_layers):
                self.add_module("gcl_{}_{}".format(j,i), GCNConv(self.hidden_nf, self.hidden_nf))

        for i in range(n_graphs):
            self.add_module("node_dec_{}".format(i), nn.Sequential( nn.Linear(self.hidden_nf, self.hidden_nf),
                                                                    act_fn,
                                                                    nn.Linear(self.hidden_nf, self.hidden_nf)))
        self.add_module("grand_dec", nn.Sequential(nn.Linear(n_graphs * self.hidden_nf, n_graphs * self.hidden_nf),
                                    act_fn,
                                    nn.Linear(n_graphs * self.hidden_nf, 1)))
        self.to(self.device)


    def forward(self, h0, x, all_edges, all_edge_attr, n_nodes):
        hf = []
        for j in range(self.n_graphs):
            h = self.embedding(h0[j])
            edges = all_edges[j]
            node_attr = h0[j]
            x_curr = x[j]
            n_node = n_nodes[j]
            if all_edge_attr is not None:
                edge_attr = all_edge_attr[j]
            else:
                edge_attr = None
            for i in range(0, self.n_layers):
                h = self._modules["gcl_{}_{}".format(j,i)](h, edges)
            h = self._modules["node_dec_{}".format(j)](h)
            h = h.unsqueeze(0)
            h = torch.sum(h, dim=1)
            hf.append(h)
        combined = torch.cat(hf, dim=1)
        pred = self.grand_dec(combined)
        return pred.squeeze(1)

class IEGNN(torch.nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, device, hidden_nf=128, n_layers=8,  node_attr=1, act_fn=nn.ReLU(), attention=False):
        super(IEGNN, self).__init__()
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
            self.add_module("i_e_gcl_%d" % i, I_E_GCL(
                input_nf=self.hidden_nf, 
                output_nf=self.hidden_nf, 
                hidden_nf=self.hidden_nf, 
                nodes_att_dim=n_node_attr,
                edges_in_d=in_edge_nf, 
                act_fn=act_fn, 
                recurrent=False, 
                attention=attention,
                clamp=False,
                tanh=False
                )
            )


        self.node_dec = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf*2),
            act_fn,
            nn.Linear(self.hidden_nf*2, self.hidden_nf*2),
            act_fn,
            nn.Linear(self.hidden_nf*2, self.hidden_nf*2),
            act_fn,
            nn.Linear(self.hidden_nf*2, self.hidden_nf*2),
            act_fn,
            nn.Linear(self.hidden_nf*2, self.hidden_nf*2),
            act_fn,
            nn.Linear(self.hidden_nf*2, self.hidden_nf*2),
            act_fn,
            nn.Linear(self.hidden_nf*2, 1)
        )
        self.to(self.device)

    def forward(self, h, edges, edge_attr, node_attr, coord, n_nodes_h, node_mask, int_h, int_edges):
        h = self.embedding(h)
        int_h = self.embedding(int_h)
        for i in range(0, self.n_layers):
            h, _, _ = self._modules["i_e_gcl_{}".format(i)](
                h, 
                edge_index=edges,
                coord=coord, 
                int_h=int_h, 
                int_index=int_edges, 
                edge_attr=edge_attr, 
                node_attr=node_attr
            )
            
        h = h * node_mask
        h = h.view(-1, n_nodes_h, self.hidden_nf)
        # h = h.unsqueeze(0)
        pred = torch.sum(h, dim=1)
        pred = self._modules["node_dec"](pred)
        return pred.squeeze(1)

class IGNN(torch.nn.Module):
    def __init__(self, in_node_nf, in_edge_nf, device, hidden_nf=128, n_layers=8,  node_attr=1, act_fn=nn.ReLU(), attention=False):
        super(IGNN, self).__init__()
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
            self.add_module("i_gcl_%d" % i, I_GCL(
                self.hidden_nf, 
                self.hidden_nf, 
                self.hidden_nf, 
                int_node_nf=in_node_nf,
                edges_in_nf=in_edge_nf, 
                # int_edge_nf=0, 
                act_fn=act_fn, 
                recurrent=False, 
                attention=attention)
            )

        self.node_dec = nn.Sequential(
            nn.Linear(self.hidden_nf, self.hidden_nf*2),
            act_fn,
            nn.Linear(self.hidden_nf*2, self.hidden_nf*2),
            act_fn,
            nn.Linear(self.hidden_nf*2, self.hidden_nf*2),
            act_fn,
            nn.Linear(self.hidden_nf*2, self.hidden_nf*2),
            act_fn,
            nn.Linear(self.hidden_nf*2, self.hidden_nf*2),
            act_fn,
            nn.Linear(self.hidden_nf*2, self.hidden_nf*2),
            act_fn,
            nn.Linear(self.hidden_nf*2, 1)
        )
        self.to(self.device)

    def forward(self, h, edges, edge_attr, n_nodes_h, node_mask, int_h, int_edges):
        h = self.embedding(h)
        int_h = self.embedding(int_h)
        for i in range(0, self.n_layers):
            h, _ = self._modules["i_gcl_{}".format(i)](
                x=h, 
                edge_index=edges, 
                int_x=int_h,
                int_index=int_edges,
                edge_attr = None
            )
        h = h * node_mask
        h = h.view(-1, n_nodes_h, self.hidden_nf)
        # h = h.unsqueeze(0)
        pred = torch.sum(h, dim=1)
        pred = self._modules["node_dec"](pred)
        return pred.squeeze(1)
