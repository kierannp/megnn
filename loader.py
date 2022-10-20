import glob
import pandas as pd
from gmso.external.convert_networkx import to_networkx
import torch
import torch.nn.functional as F
import mbuild as mb
import networkx as nx
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data


COF_data_path = '/Users/kieran/iMoDELS-supplements/data/raw-data/everything.csv'
COF_data = pd.read_csv(COF_data_path, index_col=0)
COF_data = COF_data[['terminal_group_1','terminal_group_2','terminal_group_3', 'backbone', 'frac-1','frac-2','COF','intercept']]

molecules = glob.glob('/Users/kieran/terminal_groups_mixed/src/util/molecules/*.pdb')
molecules = list(set(molecules))
names2graph = {}
mol_smiles = {}
name2n_nodes = {}
name2xyz = {}
name2e_index = {}
name2mol = {}
name2x = {}

missing_mols = ['difluoromethyl', 'phenol', 'toluene']
def replace_name_graph(chem_name):
    if chem_name == 'difluoromethyl' or chem_name == 'phenol' or chem_name == 'toluene':
        return
    else:
        return names2graph[chem_name]

def n_nodes_tuple(t1, t2):
    if t1 in missing_mols or t2 in missing_mols:
        return
    else:
        return (name2n_nodes[t1], name2n_nodes[t2])
def replace_n_nodes_tuple(row):
    return n_nodes_tuple(row['terminal_group_1'], row['terminal_group_2'])

def get_positions(t1,t2):
    if t1 in missing_mols or t2 in missing_mols:
        return
    else:
        return (name2xyz[t1], name2xyz[t2])
def replace_positions(row):
    return get_positions(row['terminal_group_1'], row['terminal_group_2'])

def replace_e_index(chem_name):
    if chem_name == 'difluoromethyl' or chem_name == 'phenol' or chem_name == 'toluene':
        return
    else:
        return name2e_index[chem_name]
def replace_node_att(chem_name):
    if chem_name == 'difluoromethyl' or chem_name == 'phenol' or chem_name == 'toluene':
        return
    else:
        x = torch.empty((len(elements),name2n_nodes[chem_name]), dtype=torch.int32)
        for i, p in enumerate(name2mol[chem_name].particles()):
            x[i] = element2vec[p]
        return x

def CIRconvert(ids):
    url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
    ans = urlopen(url).read().decode('utf8')
    return ans

for ids in set(COF_data['terminal_group_1']):
    try:
        mol_smiles[ids] = CIRconvert(ids)
    except:
        pass
mol_smiles['nitrophenyl'] = 'CC1=CC=C(C=C1)[N+]([O-])=O'
mol_smiles['isopropyl'] = 'CC(C)O'
mol_smiles['perfluoromethyl'] = 'CC(F)(F)F'
mol_smiles['fluorophenyl'] = 'CC1=CC=C(F)C=C1'
mol_smiles['carboxyl'] = '*C(=O)O'
mol_smiles['amino'] = 'CN'


names = ''.join(smiles.upper() for smiles in mol_smiles.values())
names += 'H'
elements = [n for n in set(names) if n.isalpha()]


vecs = F.one_hot(torch.arange(0, len(elements)), num_classes=len(elements))
element2vec = {e:v for e, v in zip(elements, vecs)}
vec2element = {v:e for e, v in zip(elements, vecs)}

for m in molecules:
    mol_name = m.split('/')[-1].split('.')[0]
    if 'ch3' in mol_name:
        mol_name = mol_name.split('-')[0]
    mol = mb.load(m)
    G = to_networkx(mol.to_gmso())
    adj = nx.adjacency_matrix(G)
    names2graph[mol_name] = adj
    name2n_nodes[mol_name] = len(list(G.nodes))
    name2xyz[mol_name] = mol.xyz
    name2mol[mol_name] = mol
    e_index = torch.empty((2,mol.n_bonds), dtype=torch.int64)
    parts = {p:i for i, p in enumerate(mol.particles())}
    for i, b in enumerate(mol.bonds()):
        e_index[0,i] = parts[b[0]]
        e_index[1,i] = parts[b[1]]
    name2e_index[mol_name] = e_index
    x = torch.empty((mol.n_particles,len(elements)), dtype=torch.float32)
    for i, p in enumerate(mol.particles()):
        x[i] = element2vec[p.element.symbol]
    name2x[mol_name] = x

COF_data['n_nodes'] = COF_data.apply(replace_n_nodes_tuple,axis=1)
COF_data = COF_data.dropna()
COF_data = COF_data[(COF_data['frac-1']==.5)]
COF_data.reset_index()


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

class TribologyDataset(InMemoryDataset):
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
        # Read data into huge `Data` list.
        data_list = []
        nodes_s, edges_s, nodes_t, edges_t = [], [], [], []
        for i, row in COF_data.iterrows():
            nodes_s.append(name2n_nodes[row.terminal_group_1])
            nodes_t.append(name2n_nodes[row.terminal_group_2])
            edges_s.append(name2e_index[row.terminal_group_1])
            edges_t.append(name2e_index[row.terminal_group_2])
        max_edges_s = max([e_i.size()[1] for e_i in edges_s])
        max_edges_t = max([e_i.size()[1] for e_i in edges_t])
        max_nodes_s = max(nodes_s)
        max_nodes_t = max(nodes_t)
        
        for i, row in COF_data.iterrows():  # Iterate in batches over the training dataset.
            if row.terminal_group_1 in missing_mols or row.terminal_group_2 in missing_mols:
                continue

            x_s = torch.tensor(name2x[row.terminal_group_1])
            # x_s = F.pad(x_s, (0,0,0,max_nodes_s-x_s.size()[0]))
            edge_index_s = name2e_index[row.terminal_group_1]
            # edge_index_s = F.pad(edge_index_s, (0, max_edges_s - edge_index_s.size()[1]))
            positions_s = torch.tensor(name2xyz[row.terminal_group_1])
            # positions_s = F.pad(positions_s, (0, 0, 0, max_nodes_s - positions_s.size()[0]))
            n_nodes_s = name2n_nodes[row.terminal_group_1]

            x_t = torch.tensor(name2x[row.terminal_group_2])
            # x_t = F.pad(x_t, (0,0,0,max_nodes_t-x_t.size()[0]))
            edge_index_t = name2e_index[row.terminal_group_2]
            # edge_index_t = F.pad(edge_index_t, (0, max_edges_t - edge_index_t.size()[1]))
            positions_t = torch.tensor(name2xyz[row.terminal_group_2])
            # positions_t = F.pad(positions_t, (0, 0, 0, max_nodes_t - positions_t.size()[0]))
            n_nodes_t = name2n_nodes[row.terminal_group_2]
            # print('x_s:{}, x_t:{}, e_s:{}, e_t:{}, p_s:{}, p_t:{}'.format(x_s.size(),x_t.size(),edge_index_s.size(), edge_index_t.size(),positions_s.size(),positions_t.size()))


            data = PairData(edge_index_s, x_s, positions_s, n_nodes_s, edge_index_t, x_t, positions_t, n_nodes_t,  y = torch.tensor(row.COF))
            data_list.append(data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])