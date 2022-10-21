import glob
import pandas as pd
from gmso.external.convert_networkx import to_networkx
import torch
import torch.nn.functional as F
import mbuild as mb
import networkx as nx
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from urllib.request import urlopen
from urllib.parse import quote

class PairData(Data):
    def __init__(self, edge_index_s=None, x_s=None, positions_s=None, n_nodes_s=None,edge_index_t=None, x_t=None, positions_t=None, n_nodes_t=None, y=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.positions_s = positions_s
        self.n_nodes_s = n_nodes_s

        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.positions_t = positions_t
        self.n_nodes_t = n_nodes_t

        self.y = y

    # def __inc__(self, key, value, *args, **kwargs):
    #     if key == 'edge_index_s':
    #         return self.x_s.size(0)
    #     if key == 'edge_index_t':
    #         return self.x_t.size(0)
    #     else:
    #         return super().__inc__(key, value, *args, **kwargs)

class COF_Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return glob.glob('/Users/kieran/terminal_groups_mixed/src/util/molecules/*.pdb')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass
    def replace_name_graph(self, chem_name):
        if chem_name == 'difluoromethyl' or chem_name == 'phenol' or chem_name == 'toluene':
            return
        else:
            return self.names2graph[chem_name]

    def n_nodes_tuple(self, t1, t2):
        if t1 in self.missing_mols or t2 in self.missing_mols:
            return
        else:
            return (self.name2n_nodes[t1], self.name2n_nodes[t2])
            
    def replace_n_nodes_tuple(self, row):
        return self.n_nodes_tuple(row['terminal_group_1'], row['terminal_group_2'])

    def get_positions(self, t1,t2):
        if t1 in self.missing_mols or t2 in self.missing_mols:
            return
        else:
            return (self.name2xyz[t1], self.name2xyz[t2])

    def replace_positions(self, row):
        return self.get_positions(row['terminal_group_1'], row['terminal_group_2'])

    def replace_e_index(self, chem_name):
        if self.chem_name == 'difluoromethyl' or self.chem_name == 'phenol' or self.chem_name == 'toluene':
            return
        else:
            return self.name2e_index[chem_name]

    def replace_node_att(self, chem_name):
        if self.chem_name == 'difluoromethyl' or self.chem_name == 'phenol' or self.chem_name == 'toluene':
            return
        else:
            x = torch.empty((len(self.elements),self.name2n_nodes[chem_name]), dtype=torch.int32)
            for i, p in enumerate(self.name2mol[chem_name].particles()):
                x[i] = self.element2vec[p]
            return x

    def CIRconvert(self, ids):
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
        ans = urlopen(url).read().decode('utf8')
        return ans

    def process(self):
        data_path = '/Users/kieran/iMoDELS-supplements/data/raw-data/everything.csv'
        self.dataframe = pd.read_csv(data_path, index_col=0)
        self.dataframe = self.dataframe[['terminal_group_1','terminal_group_2','terminal_group_3', 'backbone', 'frac-1','frac-2','COF','intercept']]
    
        molecules = glob.glob('/Users/kieran/terminal_groups_mixed/src/util/molecules/*.pdb')
        self.molecules = list(set(molecules))
        self.names2graph = {}
        self.mol_smiles = {}
        self.name2n_nodes = {}
        self.name2xyz = {}
        self.name2e_index = {}
        self.name2mol = {}
        self.name2x = {}

        self.missing_mols = ['difluoromethyl', 'phenol', 'toluene']
        self.mol_smiles['nitrophenyl'] = 'CC1=CC=C(C=C1)[N+]([O-])=O'
        self.mol_smiles['isopropyl'] = 'CC(C)O'
        self.mol_smiles['perfluoromethyl'] = 'CC(F)(F)F'
        self.mol_smiles['fluorophenyl'] = 'CC1=CC=C(F)C=C1'
        self.mol_smiles['carboxyl'] = '*C(=O)O'
        self.mol_smiles['amino'] = 'CN'

        for ids in set(self.dataframe['terminal_group_1']):
            try:
                self.mol_smiles[ids] = self.CIRconvert(ids)
            except:
                pass

        names = ''.join(smiles.upper() for smiles in self.mol_smiles.values())
        names += 'H'
        self.elements = [n for n in set(names) if n.isalpha()]


        vecs = F.one_hot(torch.arange(0, len(self.elements)), num_classes=len(self.elements))
        self.element2vec = {e:v for e, v in zip(self.elements, vecs)}

        for m in molecules:
            mol_name = m.split('/')[-1].split('.')[0]
            if 'ch3' in mol_name:
                mol_name = mol_name.split('-')[0]
            mol = mb.load(m)
            G = to_networkx(mol.to_gmso())
            adj = nx.adjacency_matrix(G)
            self.names2graph[mol_name] = adj
            self.name2n_nodes[mol_name] = len(list(G.nodes))
            self.name2xyz[mol_name] = mol.xyz
            self.name2mol[mol_name] = mol
            e_index = torch.empty((2,mol.n_bonds), dtype=torch.int64)
            parts = {p:i for i, p in enumerate(mol.particles())}
            for i, b in enumerate(mol.bonds()):
                e_index[0,i] = parts[b[0]]
                e_index[1,i] = parts[b[1]]
            self.name2e_index[mol_name] = e_index
            x = torch.empty((mol.n_particles,len(self.elements)), dtype=torch.float32)
            for i, p in enumerate(mol.particles()):
                x[i] = self.element2vec[p.element.symbol]
            self.name2x[mol_name] = x

        self.dataframe['n_nodes'] = self.dataframe.apply(self.replace_n_nodes_tuple, axis=1)
        self.dataframe = self.dataframe.dropna()
        self.dataframe = self.dataframe[(self.dataframe['frac-1']==.5)]
        self.dataframe.reset_index()
        # Read data into huge `Data` list.
        data_list = []
        nodes_s, edges_s, nodes_t, edges_t = [], [], [], []
        for i, row in self.dataframe.iterrows():
            nodes_s.append(self.name2n_nodes[row.terminal_group_1])
            nodes_t.append(self.name2n_nodes[row.terminal_group_2])
            edges_s.append(self.name2e_index[row.terminal_group_1])
            edges_t.append(self.name2e_index[row.terminal_group_2])
        
        for i, row in self.dataframe.iterrows():  # Iterate in batches over the training dataset.
            if row.terminal_group_1 in self.missing_mols or row.terminal_group_2 in self.missing_mols:
                continue

            x_s = torch.tensor(self.name2x[row.terminal_group_1])
            # x_s = F.pad(x_s, (0,0,0,max_nodes_s-x_s.size()[0]))
            edge_index_s = self.name2e_index[row.terminal_group_1]
            # edge_index_s = F.pad(edge_index_s, (0, max_edges_s - edge_index_s.size()[1]))
            positions_s = torch.tensor(self.name2xyz[row.terminal_group_1])
            # positions_s = F.pad(positions_s, (0, 0, 0, max_nodes_s - positions_s.size()[0]))
            n_nodes_s = self.name2n_nodes[row.terminal_group_1]

            x_t = torch.tensor(self.name2x[row.terminal_group_2])
            # x_t = F.pad(x_t, (0,0,0,max_nodes_t-x_t.size()[0]))
            edge_index_t = self.name2e_index[row.terminal_group_2]
            # edge_index_t = F.pad(edge_index_t, (0, max_edges_t - edge_index_t.size()[1]))
            positions_t = torch.tensor(self.name2xyz[row.terminal_group_2])
            # positions_t = F.pad(positions_t, (0, 0, 0, max_nodes_t - positions_t.size()[0]))
            n_nodes_t = self.name2n_nodes[row.terminal_group_2]
            # print('x_s:{}, x_t:{}, e_s:{}, e_t:{}, p_s:{}, p_t:{}'.format(x_s.size(),x_t.size(),edge_index_s.size(), edge_index_t.size(),positions_s.size(),positions_t.size()))


            p_data = PairData(edge_index_s, x_s, positions_s, n_nodes_s, edge_index_t, x_t, positions_t, n_nodes_t,  y = torch.tensor(row.COF))
            data_list.append(p_data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class Cloud_Point_Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return glob.glob('/Users/kieran/terminal_groups_mixed/src/util/molecules/*.pdb')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        pass
        #     data = PairData(edge_index_s, x_s, positions_s, n_nodes_s, edge_index_t, x_t, positions_t, n_nodes_t,  y = torch.tensor(row.COF))
        #     data_list.append(data)
        # if self.pre_filter is not None:
        #     data_list = [data for data in data_list if self.pre_filter(data)]
        # if self.pre_transform is not None:
        #     data_list = [self.pre_transform(data) for data in data_list]
        # data, slices = self.collate(data_list)
        # torch.save((data, slices), self.processed_paths[0])