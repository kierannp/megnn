import torch
from torch import nn
from torch_geometric.nn import GCNConv, global_mean_pool
import sys
from .layers import *



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
            # act_fn,
            # nn.Linear(self.hidden_nf*2, self.hidden_nf*2),
            # act_fn,
            # nn.Linear(self.hidden_nf*2, self.hidden_nf*2),
            # act_fn,
            # nn.Linear(self.hidden_nf*2, self.hidden_nf*2),
            # act_fn,
            # nn.Linear(self.hidden_nf*2, self.hidden_nf*2),
            # act_fn,
            # nn.Linear(self.hidden_nf*2, self.hidden_nf*2),
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
