import os
from glob import glob
import pandas as pd
from gmso.external.convert_networkx import to_networkx
import torch
import torch.nn.functional as F
import mbuild as mb
import rdkit.Chem as Chem
import parmed as pmd
import networkx as nx
from torch_geometric.data import InMemoryDataset
from urllib.request import urlopen
from urllib.parse import quote
from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import AllChem, rdFreeSASA
import numpy as np
from mData import Multi_Coord_Data

class COF_Dataset(InMemoryDataset):
    """
    A class to represent a person.

    ...

    Attributes
    ----------
    name : str
        first name of the person
    surname : str
        family name of the person
    age : int
        age of the person

    Methods
    -------
    info(additional=""):
        Prints the person's name and age.
    """
    def __init__(self, root, prop, standardize = True, mix = '50', transform=None, pre_transform=None, pre_filter=None):
        """
        Constructs all the necessary attributes for the person object.

        Parameters
        ----------
            name : str
                first name of the person
            surname : str
                family name of the person
            age : int
                age of the person
        """
        self.prop = prop
        self.standardize = standardize
        self.mix = mix
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return glob('~/projects/terminal_groups_mixed/src/util/molecules/*.pdb')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass
    def replace_name_graph(self, chem_name):
        if t1 in self.missing_mols or t2 in self.missing_mols:
            return
        else:
            return self.names2graph[chem_name]

    def n_nodes_tuple(self, t1, t2):
        if t1 in self.missing_mols or t2 in self.missing_mols:
            return
        else:
            return (self.smiles2n_nodes[t1], self.smiles2n_nodes[t2])
            
    def replace_n_nodes_tuple(self, row):
        return self.n_nodes_tuple(row['terminal_group_1'], row['terminal_group_2'])

    def get_positions(self, t1,t2):
        if t1 in self.missing_mols or t2 in self.missing_mols:
            return
        else:
            return (self.smiles2xyz[t1], self.smiles2xyz[t2])

    def replace_positions(self, row):
        return self.get_positions(row['terminal_group_1'], row['terminal_group_2'])

    def replace_e_index(self, chem_name):
        if t1 in self.missing_mols or t2 in self.missing_mols:
            return
        else:
            return self.smiles2e_index[chem_name]

    def replace_node_att(self, chem_name):
        if t1 in self.missing_mols or t2 in self.missing_mols:
            return
        else:
            x = torch.empty((len(self.elements),self.smiles2n_nodes[chem_name]), dtype=torch.int32)
            for i, p in enumerate(self.smiles2mol[chem_name].particles()):
                x[i] = self.element2vec[p]
            return x

    def process(self):
        data_path = '~/projects/iMoDELS-supplements/data/raw-data/everything.csv'
        self.dataframe = pd.read_csv(data_path, index_col=0)
        if self.standardize:
            self.mean = self.dataframe[self.prop].mean()
            self.std = self.dataframe[self.prop].std()
            print("mean:{} std: {}".format(self.mean, self.std))
            self.dataframe[self.prop] = (self.dataframe[self.prop]-self.mean)/self.std

        features = self.dataframe.drop(['terminal_group_1','terminal_group_2','terminal_group_3', 'backbone','chainlength', 'frac-1','frac-2','COF','intercept', 'COF-std', 'intercept-std'], axis=1, inplace=False)

        self.names2graph = {}
        self.mol_smiles = {}
        self.smiles2n_nodes = {}
        self.smiles2xyz = {}
        self.smiles2e_index = {}
        self.smiles2mol = {}
        self.smiles2x = {}

        self.missing_mols = []
        self.mol_smiles['acetyl'] = 'C(=O)C'
        self.mol_smiles['amino'] = 'N'
        self.mol_smiles['carboxyl'] = 'CC(=O)O'
        self.mol_smiles['cyano'] = 'C#N'
        self.mol_smiles['cyclopropyl'] = 'C1CC1'
        self.mol_smiles['difluoromethyl'] = 'FC(F)C'
        self.mol_smiles['ethylene'] = 'C=C'
        self.mol_smiles['fluorophenyl'] = 'C1=CC=C(F)C=C1'
        self.mol_smiles['hydroxyl'] = 'O'
        self.mol_smiles['isopropyl'] = 'C(C)C'
        self.mol_smiles['methoxy'] = 'OC'
        self.mol_smiles['methyl'] = 'C'
        self.mol_smiles['nitro'] = '[N+](=O)[O-]'
        self.mol_smiles['nitrophenyl'] = 'C1=CC=C([N+](=O)[O-])C=C1'
        self.mol_smiles['perfluoromethyl'] = 'CC(F)(F)F'
        self.mol_smiles['phenol'] = 'c1ccc(cc1)O'
        self.mol_smiles['phenyl'] = 'C1=CC=CC=C1'
        self.mol_smiles['pyrrole'] = 'C1=CNC=C1'
        self.mol_smiles['toluene'] = 'Cc1ccccc1'

        names = ''.join(smiles.upper() for smiles in self.mol_smiles.values())
        names += 'H'
        self.elements = [n for n in set(names) if n.isalpha()]

        vecs = F.one_hot(torch.arange(0, len(self.elements)), num_classes=len(self.elements))
        self.element2vec = {e:v for e, v in zip(self.elements, vecs)}

        for mol_name, s in self.mol_smiles.items():
            mol = mb.load(s, smiles=True)
            G = to_networkx(mol.to_gmso())
            adj = nx.adjacency_matrix(G)
            self.names2graph[mol_name] = adj
            self.smiles2n_nodes[mol_name] = len(list(G.nodes))
            self.smiles2xyz[mol_name] = mol.xyz
            self.smiles2mol[mol_name] = mol
            e_index = torch.empty((2,mol.n_bonds), dtype=torch.int64)
            parts = {p:i for i, p in enumerate(mol.particles())}
            for i, b in enumerate(mol.bonds()):
                e_index[0,i] = parts[b[0]]
                e_index[1,i] = parts[b[1]]
            self.smiles2e_index[mol_name] = e_index
            x = torch.empty((mol.n_particles,len(self.elements)), dtype=torch.float32)
            for i, p in enumerate(mol.particles()):
                x[i] = self.element2vec[p.element.symbol]
            self.smiles2x[mol_name] = x

        self.dataframe['n_nodes'] = self.dataframe.apply(self.replace_n_nodes_tuple, axis=1)
        self.dataframe = self.dataframe.dropna()

        if self.mix == '50':
            self.dataframe = self.dataframe[(self.dataframe['frac-1']==.5)]
        if self.mix == '25':
            self.dataframe = self.dataframe[(self.dataframe['frac-1']==.25)]
        self.dataframe.reset_index()

        # Read data into huge `Data` list.
        data_list = []
        for i, row in self.dataframe.iterrows():  # Iterate in batches over the training dataset.
            x_s = self.smiles2x[row.terminal_group_1].detach().clone()
            other_features = torch.tensor(row[features.columns]).reshape(1,len(features.columns))
            other_features = other_features.expand(x_s.size(0), len(features.columns))
            x_s = torch.cat((x_s,other_features), 1)
            edge_index_s = self.smiles2e_index[row.terminal_group_1]
            positions_s = torch.tensor(self.smiles2xyz[row.terminal_group_1])
            n_nodes_s = self.smiles2n_nodes[row.terminal_group_1]
            x_t = self.smiles2x[row.terminal_group_2].detach().clone()
            other_features = torch.tensor(row[features.columns]).reshape(1,len(features.columns))
            other_features = other_features.expand(x_t.size(0), len(features.columns))
            x_t = torch.cat((x_t, other_features), 1)
            edge_index_t = self.smiles2e_index[row.terminal_group_2]
            positions_t = torch.tensor(self.smiles2xyz[row.terminal_group_2])
            n_nodes_t = self.smiles2n_nodes[row.terminal_group_2]
            m_data = Multi_Coord_Data(
                n_graphs = 2, 
                node_features = [x_s, x_t],
                coordinates = [positions_s, positions_t],
                edge_indexs = [edge_index_s, edge_index_t],
                n_nodes = [n_nodes_s, n_nodes_t],
                y = torch.tensor(row[self.prop])
            )
            data_list.append(m_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class COF_Dataset2(InMemoryDataset):
    def __init__(self, root, prop, transform=None, pre_transform=None, pre_filter=None):
        self.prop = prop
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return glob('~/projects/terminal_groups_mixed/src/util/molecules/*.pdb')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass
    def replace_name_graph(self, chem_name):
        if t1 in self.missing_mols or t2 in self.missing_mols:
            return
        else:
            return self.names2graph[chem_name]

    def n_nodes_tuple(self, t1, t2):
        if t1 in self.missing_mols or t2 in self.missing_mols:
            return
        else:
            return (self.smiles2n_nodes[t1], self.smiles2n_nodes[t2])
            
    def replace_n_nodes_tuple(self, row):
        return self.n_nodes_tuple(row['terminal_group_1'], row['terminal_group_2'])

    def get_positions(self, t1,t2):
        if t1 in self.missing_mols or t2 in self.missing_mols:
            return
        else:
            return (self.smiles2xyz[t1], self.smiles2xyz[t2])

    def replace_positions(self, row):
        return self.get_positions(row['terminal_group_1'], row['terminal_group_2'])

    def replace_e_index(self, chem_name):
        if t1 in self.missing_mols or t2 in self.missing_mols:
            return
        else:
            return self.smiles2e_index[chem_name]

    def replace_node_att(self, chem_name):
        if t1 in self.missing_mols or t2 in self.missing_mols:
            return
        else:
            x = torch.empty((len(self.elements),self.smiles2n_nodes[chem_name]), dtype=torch.int32)
            for i, p in enumerate(self.smiles2mol[chem_name].particles()):
                x[i] = self.element2vec[p]
            return x

    def CIRconvert(self, ids):
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
        ans = urlopen(url).read().decode('utf8')
        return ans

    def process(self):
        data_path = '~/projects/iMoDELS-supplements/data/raw-data/everything.csv'
        self.dataframe = pd.read_csv(data_path, index_col=0)
        self.mean = self.dataframe[self.prop].mean()
        self.std = self.dataframe[self.prop].std()
        print("mean:{} std: {}".format(self.mean, self.std))
        self.dataframe[self.prop] = (self.dataframe[self.prop]-self.mean)/self.std

        home = os.path.expanduser('~')
        # molecules = glob(home + '/projects/terminal_groups_mixed/src/util/molecules/*')
        # self.molecules = list(set(molecules))

        # features = self.dataframe.drop(['terminal_group_1','terminal_group_2','terminal_group_3', 'backbone','chainlength', 'frac-1','frac-2','COF','intercept', 'COF-std', 'intercept-std'], axis=1, inplace=False)
        features = pd.read_csv('/raid6/homes/kierannp/projects/iMoDELS-supplements/data/raw-data/descriptors-ind.csv').T
        features = features.drop(['hdonors', 'hacceptors'],axis=1, inplace=False)
        features = features.reset_index()

        self.names2graph = {}
        self.mol_smiles = {}
        self.smiles2n_nodes = {}
        self.smiles2xyz = {}
        self.smiles2e_index = {}
        self.smiles2mol = {}
        self.smiles2x = {}

        self.missing_mols = []
        self.mol_smiles['acetyl'] = 'C(=O)C'
        self.mol_smiles['amino'] = 'N'
        self.mol_smiles['carboxyl'] = 'CC(=O)O'
        self.mol_smiles['cyano'] = 'C#N'
        self.mol_smiles['cyclopropyl'] = 'C1CC1'
        self.mol_smiles['difluoromethyl'] = 'FC(F)C'
        self.mol_smiles['ethylene'] = 'C=C'
        self.mol_smiles['fluorophenyl'] = 'C1=CC=C(F)C=C1'
        self.mol_smiles['hydroxyl'] = 'O'
        self.mol_smiles['isopropyl'] = 'C(C)C'
        self.mol_smiles['methoxy'] = 'OC'
        self.mol_smiles['methyl'] = 'C'
        self.mol_smiles['nitro'] = '[N+](=O)[O-]'
        self.mol_smiles['nitrophenyl'] = 'C1=CC=C([N+](=O)[O-])C=C1'
        self.mol_smiles['perfluoromethyl'] = 'CC(F)(F)F'
        self.mol_smiles['phenol'] = 'c1ccc(cc1)O'
        self.mol_smiles['phenyl'] = 'C1=CC=CC=C1'
        self.mol_smiles['pyrrole'] = 'C1=CNC=C1'
        self.mol_smiles['toluene'] = 'Cc1ccccc1'


        for ids in set(self.dataframe['terminal_group_1']):
            try:
                if ids in self.mol_smiles:
                    continue
                self.mol_smiles[ids] = self.CIRconvert(ids)
            except:
                pass

        names = ''.join(smiles.upper() for smiles in self.mol_smiles.values())
        names += 'H'
        self.elements = [n for n in set(names) if n.isalpha()]


        vecs = F.one_hot(torch.arange(0, len(self.elements)), num_classes=len(self.elements))
        self.element2vec = {e:v for e, v in zip(self.elements, vecs)}

        for mol_name, s in self.mol_smiles.items():
            # mol_name = m.split('/')[-1].split('.')[0]
            # if 'ch3' in mol_name:
            #     mol_name = mol_name.split('-')[0]
            mol = mb.load(s, smiles=True)
            G = to_networkx(mol.to_gmso())
            adj = nx.adjacency_matrix(G)
            self.names2graph[mol_name] = adj
            self.smiles2n_nodes[mol_name] = len(list(G.nodes))
            self.smiles2xyz[mol_name] = mol.xyz
            self.smiles2mol[mol_name] = mol
            e_index = torch.empty((2,mol.n_bonds), dtype=torch.int64)
            parts = {p:i for i, p in enumerate(mol.particles())}
            for i, b in enumerate(mol.bonds()):
                e_index[0,i] = parts[b[0]]
                e_index[1,i] = parts[b[1]]
            self.smiles2e_index[mol_name] = e_index
            x = torch.empty((mol.n_particles,len(self.elements)), dtype=torch.float32)
            for i, p in enumerate(mol.particles()):
                x[i] = self.element2vec[p.element.symbol]
            self.smiles2x[mol_name] = x

        self.dataframe['n_nodes'] = self.dataframe.apply(self.replace_n_nodes_tuple, axis=1)
        self.dataframe = self.dataframe.dropna()
        self.dataframe = self.dataframe[(self.dataframe['frac-1']==.5)]
        self.dataframe.reset_index()
        # Read data into huge `Data` list.
        data_list = []
        nodes_s, edges_s, nodes_t, edges_t = [], [], [], []
        for i, row in self.dataframe.iterrows():
            nodes_s.append(self.smiles2n_nodes[row.terminal_group_1])
            nodes_t.append(self.smiles2n_nodes[row.terminal_group_2])
            edges_s.append(self.smiles2e_index[row.terminal_group_1])
            edges_t.append(self.smiles2e_index[row.terminal_group_2])
        
        for i, row in self.dataframe.iterrows():  # Iterate in batches over the training dataset.
            if row.terminal_group_1 in self.missing_mols or row.terminal_group_2 in self.missing_mols:
                continue
            x_s = torch.tensor(self.smiles2x[row.terminal_group_1])
            s_s = self.mol_smiles[row.terminal_group_1]
            other_features = features[features['index']== s_s].drop(['index'],axis=1, inplace=False)
            other_features = torch.tensor(other_features.values, dtype=torch.float)
            other_features = other_features.expand(x_s.size(0), other_features.shape[1])
            x_s = torch.cat((x_s,other_features), 1)
            edge_index_s = self.smiles2e_index[row.terminal_group_1]
            positions_s = torch.tensor(self.smiles2xyz[row.terminal_group_1])
            n_nodes_s = self.smiles2n_nodes[row.terminal_group_1]
            x_t = torch.tensor(self.smiles2x[row.terminal_group_2])
            s_t = self.mol_smiles[row.terminal_group_2]
            other_features = features[features['index']== s_t].drop(['index'],axis=1, inplace=False)
            other_features = torch.tensor(other_features.values,dtype=torch.float)
            other_features = other_features.expand(x_t.size(0), other_features.shape[1])
            x_t = torch.cat((x_t, other_features), 1)
            edge_index_t = self.smiles2e_index[row.terminal_group_2]
            positions_t = torch.tensor(self.smiles2xyz[row.terminal_group_2])
            n_nodes_t = self.smiles2n_nodes[row.terminal_group_2]
            # print('x_s:{}, x_t:{}, e_s:{}, e_t:{}, p_s:{}, p_t:{}'.format(x_s.size(),x_t.size(),edge_index_s.size(), edge_index_t.size(),positions_s.size(),positions_t.size()))

            m_data = Multi_Coord_Data(
                n_graphs = 2, 
                node_features = [x_s, x_t],
                coordinates = [positions_s, positions_t],
                edge_indexs = [edge_index_s, edge_index_t],
                n_nodes = [n_nodes_s, n_nodes_t],
                y = torch.tensor(row[self.prop])
            )
            # p_data = PairData(edge_index_s, x_s, positions_s, n_nodes_s, edge_index_t, x_t, positions_t, n_nodes_t,  y = torch.tensor(row[self.prop]))
            data_list.append(m_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class COF_Dataset3(InMemoryDataset):
    def __init__(self, root, prop, transform=None, pre_transform=None, pre_filter=None):
        self.prop = prop
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return glob('~/projects/terminal_groups_mixed/src/util/molecules/*.pdb')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass
    def replace_name_graph(self, chem_name):
        if t1 in self.missing_mols or t2 in self.missing_mols:
            return
        else:
            return self.names2graph[chem_name]

    def n_nodes_tuple(self, t1, t2):
        if t1 in self.missing_mols or t2 in self.missing_mols:
            return
        else:
            return (self.smiles2n_nodes[t1], self.smiles2n_nodes[t2])
            
    def replace_n_nodes_tuple(self, row):
        return self.n_nodes_tuple(row['terminal_group_1'], row['terminal_group_2'])

    def get_positions(self, t1,t2):
        if t1 in self.missing_mols or t2 in self.missing_mols:
            return
        else:
            return (self.smiles2xyz[t1], self.smiles2xyz[t2])

    def replace_positions(self, row):
        return self.get_positions(row['terminal_group_1'], row['terminal_group_2'])

    def replace_e_index(self, chem_name):
        if t1 in self.missing_mols or t2 in self.missing_mols:
            return
        else:
            return self.smiles2e_index[chem_name]

    def replace_node_att(self, chem_name):
        if t1 in self.missing_mols or t2 in self.missing_mols:
            return
        else:
            x = torch.empty((len(self.elements),self.smiles2n_nodes[chem_name]), dtype=torch.int32)
            for i, p in enumerate(self.smiles2mol[chem_name].particles()):
                x[i] = self.element2vec[p]
            return x

    def CIRconvert(self, ids):
        url = 'http://cactus.nci.nih.gov/chemical/structure/' + quote(ids) + '/smiles'
        ans = urlopen(url).read().decode('utf8')
        return ans

    def process(self):
        data_path = '~/projects/iMoDELS-supplements/data/raw-data/everything.csv'
        self.dataframe = pd.read_csv(data_path, index_col=0)
        self.mean = self.dataframe[self.prop].mean()
        self.std = self.dataframe[self.prop].std()
        print("mean:{} std: {}".format(self.mean, self.std))
        self.dataframe[self.prop] = (self.dataframe[self.prop]-self.mean)/self.std

        home = os.path.expanduser('~')
        # molecules = glob(home + '/projects/terminal_groups_mixed/src/util/molecules/*')
        # self.molecules = list(set(molecules))

        features = self.dataframe.drop(['terminal_group_1','terminal_group_2','terminal_group_3', 'backbone','chainlength', 'frac-1','frac-2','COF','intercept', 'COF-std', 'intercept-std'], axis=1, inplace=False)
        # features = pd.DataFrame

        self.names2graph = {}
        self.mol_smiles = {}
        self.smiles2n_nodes = {}
        self.smiles2xyz = {}
        self.smiles2e_index = {}
        self.smiles2mol = {}
        self.smiles2x = {}

        self.missing_mols = []
        self.mol_smiles['acetyl'] = 'C(=O)C'
        self.mol_smiles['amino'] = 'N'
        self.mol_smiles['carboxyl'] = 'CC(=O)O'
        self.mol_smiles['cyano'] = 'C#N'
        self.mol_smiles['cyclopropyl'] = 'C1CC1'
        self.mol_smiles['difluoromethyl'] = 'FC(F)C'
        self.mol_smiles['ethylene'] = 'C=C'
        self.mol_smiles['fluorophenyl'] = 'C1=CC=C(F)C=C1'
        self.mol_smiles['hydroxyl'] = 'O'
        self.mol_smiles['isopropyl'] = 'C(C)C'
        self.mol_smiles['methoxy'] = 'OC'
        self.mol_smiles['methyl'] = 'C'
        self.mol_smiles['nitro'] = '[N+](=O)[O-]'
        self.mol_smiles['nitrophenyl'] = 'C1=CC=C([N+](=O)[O-])C=C1'
        self.mol_smiles['perfluoromethyl'] = 'CC(F)(F)F'
        self.mol_smiles['phenol'] = 'c1ccc(cc1)O'
        self.mol_smiles['phenyl'] = 'C1=CC=CC=C1'
        self.mol_smiles['pyrrole'] = 'C1=CNC=C1'
        self.mol_smiles['toluene'] = 'Cc1ccccc1'


        for ids in set(self.dataframe['terminal_group_1']):
            try:
                if ids in self.mol_smiles:
                    continue
                self.mol_smiles[ids] = self.CIRconvert(ids)
            except:
                pass

        names = ''.join(smiles.upper() for smiles in self.mol_smiles.values())
        names += 'H'
        self.elements = [n for n in set(names) if n.isalpha()]


        vecs = F.one_hot(torch.arange(0, len(self.elements)), num_classes=len(self.elements))
        self.element2vec = {e:v for e, v in zip(self.elements, vecs)}

        for mol_name, s in self.mol_smiles.items():
            # mol_name = m.split('/')[-1].split('.')[0]
            # if 'ch3' in mol_name:
            #     mol_name = mol_name.split('-')[0]
            mol = mb.load(s, smiles=True)
            G = to_networkx(mol.to_gmso())
            adj = nx.adjacency_matrix(G)
            self.names2graph[mol_name] = adj
            self.smiles2n_nodes[mol_name] = len(list(G.nodes))
            self.smiles2xyz[mol_name] = mol.xyz
            self.smiles2mol[mol_name] = mol
            e_index = torch.empty((2,mol.n_bonds), dtype=torch.int64)
            parts = {p:i for i, p in enumerate(mol.particles())}
            for i, b in enumerate(mol.bonds()):
                e_index[0,i] = parts[b[0]]
                e_index[1,i] = parts[b[1]]
            self.smiles2e_index[mol_name] = e_index
            x = torch.empty((mol.n_particles,len(self.elements)), dtype=torch.float32)
            for i, p in enumerate(mol.particles()):
                x[i] = self.element2vec[p.element.symbol]
            self.smiles2x[mol_name] = x

        self.dataframe['n_nodes'] = self.dataframe.apply(self.replace_n_nodes_tuple, axis=1)
        self.dataframe = self.dataframe.dropna()
        # self.dataframe = self.dataframe[(self.dataframe['frac-1']==.5)]
        self.dataframe.reset_index()
        # Read data into huge `Data` list.
        data_list = []
        nodes_s, edges_s, nodes_t, edges_t = [], [], [], []
        for i, row in self.dataframe.iterrows():
            nodes_s.append(self.smiles2n_nodes[row.terminal_group_1])
            nodes_t.append(self.smiles2n_nodes[row.terminal_group_2])
            edges_s.append(self.smiles2e_index[row.terminal_group_1])
            edges_t.append(self.smiles2e_index[row.terminal_group_2])
        
        for i, row in self.dataframe.iterrows():  # Iterate in batches over the training dataset.
            if row.terminal_group_1 in self.missing_mols or row.terminal_group_2 in self.missing_mols:
                continue
            x_s = torch.tensor(self.smiles2x[row.terminal_group_1])
            other_features = torch.tensor(row[features.columns]).reshape(1,len(features.columns))
            other_features = other_features.expand(x_s.size(0), len(features.columns))
            x_s = torch.cat((x_s,other_features), 1)
            edge_index_s = self.smiles2e_index[row.terminal_group_1]
            positions_s = torch.tensor(self.smiles2xyz[row.terminal_group_1])
            n_nodes_s = self.smiles2n_nodes[row.terminal_group_1]
            x_t = torch.tensor(self.smiles2x[row.terminal_group_2])
            other_features = torch.tensor(row[features.columns]).reshape(1,len(features.columns))
            other_features = other_features.expand(x_t.size(0), len(features.columns))
            x_t = torch.cat((x_t, other_features), 1)
            edge_index_t = self.smiles2e_index[row.terminal_group_2]
            positions_t = torch.tensor(self.smiles2xyz[row.terminal_group_2])
            n_nodes_t = self.smiles2n_nodes[row.terminal_group_2]
            # print('x_s:{}, x_t:{}, e_s:{}, e_t:{}, p_s:{}, p_t:{}'.format(x_s.size(),x_t.size(),edge_index_s.size(), edge_index_t.size(),positions_s.size(),positions_t.size()))

            m_data = Multi_Coord_Data(
                n_graphs = 2, 
                node_features = [x_s, x_t],
                coordinates = [positions_s, positions_t],
                edge_indexs = [edge_index_s, edge_index_t],
                n_nodes = [n_nodes_s, n_nodes_t],
                y = torch.tensor(row[self.prop])
            )
            # p_data = PairData(edge_index_s, x_s, positions_s, n_nodes_s, edge_index_t, x_t, positions_t, n_nodes_t,  y = torch.tensor(row[self.prop]))
            data_list.append(m_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class Cloud_Point_Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, dataframe=None):
        self.dataframe = dataframe
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return glob('~/projects/terminal_groups_mixed/src/util/molecules/*.pdb')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def mol_to_nx(self, mol):
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
    
    def process(self):
        data_path = './cloud_point.xlsx'
        self.dataframe = pd.read_excel(data_path)
        uni_mols = {}
        elements = {}
        broken_smiles = []
        for n in self.dataframe['Polymer SMILES'].unique():
            try:
                mol = mb.load(pmd.rdkit.from_smiles(n))
            except:
                broken_smiles.append(n)
                continue
            for i, p in enumerate(mol):
                if p.element is None:
                    mol.remove(p)
            uni_mols[n] = mol
            for a in mol:
                if a.element.symbol not in elements.keys():
                    elements[a.element.symbol] = a.element.symbol
        for n in self.dataframe['Solvent SMILES'].unique():
            try:
                mol = mb.load(pmd.rdkit.from_smiles(n))
            except:
                broken_smiles.append(n)
                continue
            if n == 'O':
                mol = mb.lib.moieties.H2O()
            for i, p in enumerate(mol):
                if p.element is None:
                    mol.remove(p)
            uni_mols[n] = mol
            for a in mol:
                if a.element.symbol not in elements.keys():
                    elements[a.element.symbol] = a.element.symbol
        self.elements = elements

        vecs = F.one_hot(torch.arange(0, len(self.elements)), num_classes=len(self.elements))
        self.element2vec = {e:v for e, v in zip(self.elements, vecs)}

        self.smiles2e_index = {}
        self.smiles2graph = {}
        self.smiles2n_nodes = {}
        self.smiles2xyz = {}
        self.smiles2mol = {}
        self.smiles2x = {}
        for i, row in self.dataframe.iterrows():
            polymer_smiles = row['Polymer SMILES']
            solvent_smiles = row['Solvent SMILES']
            if polymer_smiles not in broken_smiles and solvent_smiles not in broken_smiles:
                if polymer_smiles not in self.smiles2graph.keys():
                    mol = uni_mols[polymer_smiles]
                    G = nx.Graph(mol.bond_graph._adj)
                    adj = nx.adjacency_matrix(G)
                    self.smiles2graph[polymer_smiles] = adj
                    self.smiles2n_nodes[polymer_smiles] = len(list(G.nodes))
                    self.smiles2xyz[polymer_smiles] = mol.xyz
                    self.smiles2mol[polymer_smiles] = mol
                    e_index = torch.empty((2,mol.n_bonds), dtype=torch.int64)
                    comp_to_index = {a.name: i for i, a in enumerate(mol)}
                    for i, line in enumerate(mol.bonds()):
                        e_index[0,i] = comp_to_index[line[0].name]
                        e_index[1,i] = comp_to_index[line[0].name]
                    self.smiles2e_index[polymer_smiles] = e_index
                    x = torch.empty((mol.n_particles,len(self.elements)), dtype=torch.float32)
                    for i, a in enumerate(mol):
                        x[i] = self.element2vec[a.element.symbol]
                    self.smiles2x[polymer_smiles] = x
                if solvent_smiles not in self.smiles2graph.keys():
                    mol = uni_mols[solvent_smiles]
                    G = nx.Graph(mol.bond_graph._adj)
                    self.smiles2graph[solvent_smiles] = adj
                    self.smiles2n_nodes[solvent_smiles] = len(list(G.nodes))
                    self.smiles2xyz[solvent_smiles] = mol.xyz
                    self.smiles2mol[solvent_smiles] = mol
                    e_index = torch.empty((2,mol.n_bonds), dtype=torch.int64)
                    comp_to_index = {a.name: i for i, a in enumerate(mol)}
                    for i, line in enumerate(mol.bonds()):
                        e_index[0,i] = comp_to_index[line[0].name]
                        e_index[1,i] = comp_to_index[line[0].name]
                    self.smiles2e_index[solvent_smiles] = e_index
                    x = torch.empty((mol.n_particles,len(self.elements)), dtype=torch.float32)
                    for i, a in enumerate(mol):
                        x[i] = self.element2vec[a.element.symbol]
                    self.smiles2x[solvent_smiles] = x

        self.dataframe = self.dataframe.dropna()
        self.dataframe.reset_index()

        # Read data into huge `Data` list.
        data_list = []
        edges_poly, edges_solv = [], []
        for i, row in self.dataframe.iterrows():
            polymer_smiles = row['Polymer SMILES']
            solvent_smiles = row['Solvent SMILES']
            enviro = [ row['PDI'], row['Mw(Da)'], row['\phi'], row['w'] ]
            if polymer_smiles not in broken_smiles and solvent_smiles not in broken_smiles:
                edges_poly.append(self.smiles2e_index[polymer_smiles])
                edges_solv.append(self.smiles2e_index[solvent_smiles])

                x_poly = torch.tensor(self.smiles2x[polymer_smiles])
                edge_index_poly = self.smiles2e_index[polymer_smiles]
                positions_poly = torch.tensor(self.smiles2xyz[polymer_smiles])
                n_nodes_poly = self.smiles2n_nodes[polymer_smiles]

                x_solv = torch.tensor(self.smiles2x[solvent_smiles])
                edge_index_solv = self.smiles2e_index[solvent_smiles]
                positions_solv = torch.tensor(self.smiles2xyz[solvent_smiles])
                n_nodes_solv = self.smiles2n_nodes[solvent_smiles]
                p_data = PairData(edge_index_poly, x_poly, positions_poly, n_nodes_poly, 
                                    edge_index_solv, x_solv, positions_solv, n_nodes_solv,  
                                    y = torch.tensor(row['CP (C)']), 
                                    enviro = torch.tensor(enviro).unsqueeze(1).reshape(1,len(enviro)))
                data_list.append(p_data)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# class PdbBind_Dataset(InMemoryDataset):
#     def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, dataframe=None):
#         self.dataframe = dataframe
#         super().__init__(root, transform, pre_transform, pre_filter)
#         self.data, self.slices = torch.load(self.processed_paths[0])

#     @property
#     def raw_file_names(self):
#         return glob('~/projects/terminal_groups_mixed/src/util/molecules/*.pdb')

#     @property
#     def processed_file_names(self):
#         return ['data.pt']

#     def download(self):
#         pass

#     def mol_to_nx(self, mol):
#         G = nx.Graph()

#         for atom in mol.GetAtoms():
#             G.add_node(atom.GetIdx(),
#                     atomic_num=atom.GetAtomicNum(),
#                     formal_charge=atom.GetFormalCharge(),
#                     chiral_tag=atom.GetChiralTag(),
#                     hybridization=atom.GetHybridization(),
#                     num_explicit_hs=atom.GetNumExplicitHs(),
#                     is_aromatic=atom.GetIsAromatic())
#         for bond in mol.GetBonds():
#             G.add_edge(bond.GetBeginAtomIdx(),
#                     bond.GetEndAtomIdx(),
#                     bond_type=bond.GetBondType())
#         return G
    
#     def coordinate_from_mol(self, molecule):
#         molecule = Chem.AddHs(molecule)
#         AllChem.EmbedMolecule(molecule)
#         AllChem.UFFOptimizeMolecule(molecule)
#         molecule.GetConformer()
#         print()
#         coords = []
#         for i, atom in enumerate(molecule.GetAtoms()):
#             positions = molecule.GetConformer().GetAtomPosition(i)
#             coords.append([positions.x, positions.y, positions.z])
#         return coords
        
#     def process_mol2(self, filename):
#         with open(filename) as f:
#             lines = f.readlines()
#             f.close()
        
#         for i, l in enumerate(lines):
#             if 'TRIPOS>ATOM' in l:
#                 atom_start_line = i+1
#             if 'TRIPOS>BOND' in l:
#                 bond_start_line = i+1
#             if '@<TRIPOS>SUB' in l:
#                 bond_end_line = i
#         coords = torch.empty((bond_start_line-atom_start_line-1, 3), dtype=torch.float32)
#         element_dict = {}
#         for i, l in enumerate(lines[atom_start_line:bond_start_line-1]):
#             parts = l.split()
            
#             element = parts[5].split('.')[0]
#             element_dict[i] = element
#             vals = torch.tensor(list(map(float, parts[2:5])))
#             coords[i,:] = vals
#         edge_list = [[],[]]
#         for i, l in enumerate(lines[bond_start_line:bond_end_line]):
#             source, target = l.split()[1:3]
#             edge_list[0].append(int(source))
#             edge_list[1].append(int(target))

#         return element_dict, coords, edge_list

#     def process(self):
#         protein_names = []
#         diss_consts = {}
#         delinquints = set()
#         all_elements = set()
#         unit_conversions = {'fM':10e-15, 'mM':10e-3, 'nM':10e-9, 'pM':10e-12, 'uM':10e-6}
#         acceptable_elements = {'N', 'Zn', 'O', 'P', 'C', 'F', 'H', 'Cl', 'S'}
#         # Read data into huge `Data` list.
#         data_list = []
#         with open('./v2019-other-PL/index/INDEX_general_PL_data.2019') as f:
#             lines = f.readlines()
#             f.close()
#         for l in lines[6:]:
#             # Just taking the protein-ligand compounds that have Kd measurements
#             experiment_value = l.split()[4]
#             if 'Kd' in experiment_value:
#                 if experiment_value[2] != '>' or experiment_value[2] != '<':
#                     if experiment_value[3:-2] != '=100':
#                         protein_names.append(l.split()[0])
#                         # Correct the units to standard Molarity units
#                         diss_consts[l.split()[0]] = math.log(float(experiment_value[3:-2]) * unit_conversions[experiment_value[-2:]])

#         for pname in protein_names:
#             files = glob('./v2019-other-PL/'+pname+'/*')
#             for f in files:
#                 if 'pocket' in f:
#                     struc = mb.load(f)
#                     elements = set(p.element.symbol for p in struc)
#                 elif 'ligand' in f and 'mol2' in f:
#                     element_dict, _, _= self.process_mol2(f)
#                     elements = set(element_dict.values())
#                 else:
#                     continue
#                 if not elements.issubset(acceptable_elements) or struc.n_particles>500:
#                     print('Unable to load: {}'.format(pname) )
#                     delinquints.add(pname)
#                     continue
#                 # Add any new elements 
#                 for a in elements:
#                     all_elements.add(a)

#         self.elements = all_elements
#         vecs = F.one_hot(torch.arange(0, len(self.elements)), num_classes=len(self.elements))
#         self.element2vec = {e:v for e, v in zip(self.elements, vecs)}

#         for pname in protein_names:
#             if pname not in delinquints:
#                 files = glob('./v2019-other-PL/'+pname+'/*')
#                 for f in files:
#                     if 'pocket' in f:
#                         prot = mb.load(f)
#                         # G = nx.Graph(prot.bond_graph._adj)

#                         # Make Edgelist for the graph
#                         prot_edge_list = torch.empty((2,prot.n_bonds), dtype=torch.int64)
#                         parts = {p:i for i, p in enumerate(prot.particles())}
#                         for i, b in enumerate(prot.bonds()):
#                             prot_edge_list[0,i] = parts[b[0]]
#                             prot_edge_list[1,i] = parts[b[1]]

#                         # Make the coordinates for the molecule
#                         prot_coords = torch.tensor(prot.xyz)

#                         # Make the node features
#                         prot_x = torch.empty((prot.n_particles,len(self.elements)), dtype=torch.float32)
#                         for i, a in enumerate(prot):
#                             prot_x[i] = self.element2vec[a.element.symbol]

#                     if 'ligand' in f and 'mol2' in f:
#                         element_dict, ligand_coords, ligand_edge_list = self.process_mol2(f)
#                         # adj = nx.adjacency_matrix(G)
#                         # e_index = torch.empty((2,mol.n_bonds), dtype=torch.int64)

#                         # Make Edgelist for the graph
#                         ligand_edge_list = torch.tensor(ligand_edge_list)

#                         # Make the coordinates for the molecule
#                         ligand_coords = ligand_coords.clone()

#                         # Make the node features
#                         ligand_x = torch.empty((len(element_dict),len(self.elements)), dtype=torch.float32)
#                         for i, a in enumerate(element_dict.values()):
#                             ligand_x[i] = self.element2vec[a]

#                 p_data = PairData(prot_edge_list, prot_x, prot_coords, prot_coords.size(0), 
#                                     ligand_edge_list, ligand_x, ligand_coords, ligand_coords.size(0),  
#                                     y = torch.tensor(diss_consts[pname]))
#                 data_list.append(p_data)


#         if self.pre_filter is not None:
#             data_list = [data for data in data_list if self.pre_filter(data)]
#         if self.pre_transform is not None:
#             data_list = [self.pre_transform(data) for data in data_list]
#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])

class SASA_Dataset(InMemoryDataset):
    def __init__(self, root, smi_path, n_samples = 1000, transform=None, pre_transform=None, pre_filter=None):
        self.smi_path = smi_path
        self.n_samples = n_samples
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return glob('~/projects/terminal_groups_mixed/src/util/molecules/*.pdb')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass
    
    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        def get_e_h_x(mol):
            e_index = torch.empty((2,mol.n_bonds), dtype=torch.float)
            h = torch.empty((mol.n_particles,len(self.elements)), dtype=torch.float)
            x = torch.empty((mol.n_particles, 3), dtype=torch.float)
            #make edge index
            parts = {p:i for i, p in enumerate(mol.particles())}
            for i, b in enumerate(mol.bonds()):
                e_index[0,i] = parts[b[0]]
                e_index[1,i] = parts[b[1]]
            #make node features
            for i, p in enumerate(mol):
                h[i] = self.element2vec[p.element.symbol]        
            #make positional features
            x = torch.tensor(mol.xyz)
            return e_index, h, x

        def calculate_sasa(smiles1, smiles2):
            """
            Compute Solvent Accessible Surface Area.
            """
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            mol1 = Chem.AddHs(mol1)
            mol2 = Chem.AddHs(mol2)
            AllChem.EmbedMolecule(mol1)
            AllChem.EmbedMolecule(mol2)

            # Get Van der Waals radii (angstrom)
            ptable = Chem.GetPeriodicTable()
            radii1 = [ptable.GetRvdw(atom.GetAtomicNum()) for atom in mol1.GetAtoms()]
            radii2 = [ptable.GetRvdw(atom.GetAtomicNum()) for atom in mol2.GetAtoms()]

            # Compute solvent accessible surface area
            sa = rdFreeSASA.CalcSASA(mol1, radii2, confIdx=-1)
            
            return sa
        
        self.df = pd.read_csv(self.smi_path, names = ['smiles','index','grp'], delimiter='	')
        self.df = self.df.dropna()
        self.df.reset_index()

        names = ''.join(s.upper() for s in self.df['smiles'].values)
        names += 'H'
        self.elements = [n for n in set(names) if n.isalpha()]
        vecs = F.one_hot(torch.arange(0, len(self.elements)), num_classes=len(self.elements))
        self.element2vec = {e:v for e, v in zip(self.elements, vecs)}

        n_molecules = self.df['smiles'].shape[0]
        n_samples = self.n_samples

        smiles_A = self.df['smiles'].values[np.random.choice(n_molecules, n_samples)]
        smiles_B = self.df['smiles'].values[np.random.choice(n_molecules, n_samples)]
        
        for i, (a,b) in enumerate(zip(smiles_A, smiles_B)):
            if i % 1000 == 0:
                print('\n{} attempted loads\n'.format(i))
            try:
                mol_a = mb.load(a, smiles=True)
                mol_b = mb.load(b, smiles=True)
                edge_index_1, h_1, x_1 = get_e_h_x(mol_a)
                edge_index_2, h_2, x_2, = get_e_h_x(mol_b)
                sasa = calculate_sasa(a, b)
                if sasa == float("nan") or sasa == float('inf') or sasa > 1000:
                    continue
            except:
                print('unable to load \n{} \n{}\n'.format(a,b))
                continue

            m_data = Multi_Coord_Data(
                n_graphs = 2, 
                node_features = [h_1, h_2],
                coordinates = [x_1, x_2],
                edge_indexs = [edge_index_1, edge_index_2],
                n_nodes = [mol_a.n_particles, mol_b.n_particles],
                y = torch.tensor(sasa)
            )
            data_list.append(m_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class SASA_Dataset2(InMemoryDataset):
    def __init__(self, root, smi_paths, n_samples = 1000, transform=None, pre_transform=None, pre_filter=None):
        self.smi_paths = smi_paths
        self.n_samples = n_samples
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return glob('~/projects/terminal_groups_mixed/src/util/molecules/*.pdb')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass
    
    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        use_vrad=True

        def get_e_h_x(mol, use_vrad=True):
            e_index = torch.empty((2,mol.n_bonds), dtype=torch.float)
            if use_vrad:
                h = torch.empty((mol.n_particles,len(self.elements)+1), dtype=torch.float)
            else:
                h = torch.empty((mol.n_particles,len(self.elements)), dtype=torch.float)
            x = torch.empty((mol.n_particles, 3), dtype=torch.float)
            #make edge index
            parts = {p:i for i, p in enumerate(mol.particles())}
            for i, b in enumerate(mol.bonds()):
                e_index[0,i] = parts[b[0]]
                e_index[1,i] = parts[b[1]]
            #make node features
            if use_vrad:
                ptable = Chem.GetPeriodicTable()
                element2vrad = {p.element.symbol:torch.tensor(ptable.GetRvdw(p.element.atomic_number)).unsqueeze(0) for p in mol}
                for i, p in enumerate(mol):
                    h[i] =  torch.cat([element2vrad[p.element.symbol], self.element2vec[p.element.symbol]])
            else:
                for i, p in enumerate(mol):
                    h[i] = self.element2vec[p.element.symbol]
            #make positional features
            x = torch.tensor(mol.xyz)
            return e_index, h, x

        def calculate_sasa(smiles1, smiles2):
            """
            Compute Solvent Accessible Surface Area.
            """
            mol1 = Chem.MolFromSmiles(smiles1)
            mol2 = Chem.MolFromSmiles(smiles2)
            mol1 = Chem.AddHs(mol1)
            mol2 = Chem.AddHs(mol2)
            AllChem.EmbedMolecule(mol1)
            AllChem.EmbedMolecule(mol2)

            # Get Van der Waals radii (angstrom)
            ptable = Chem.GetPeriodicTable()
            radii1 = [ptable.GetRvdw(atom.GetAtomicNum()) for atom in mol1.GetAtoms()]
            radii2 = [ptable.GetRvdw(atom.GetAtomicNum()) for atom in mol2.GetAtoms()]

            # Compute solvent accessible surface area
            sa = rdFreeSASA.CalcSASA(mol1, radii2, confIdx=-1)
            
            return sa
        
        self.df = pd.read_csv(self.smi_paths[0], names = ['smiles','index','grp'], delimiter='	')
        self.df = self.df.dropna()
        self.df.reset_index()

        self.df2 = pd.read_csv(self.smi_paths[1], names = ['smiles','index','grp'], delimiter='	')
        self.df2 = self.df.dropna()
        self.df2.reset_index()

        names = ''.join(s.upper() for s in self.df['smiles'].values)
        names += 'H'
        self.elements = [n for n in set(names) if n.isalpha()]
        vecs = F.one_hot(torch.arange(0, len(self.elements)), num_classes=len(self.elements))
        self.element2vec = {e:v for e, v in zip(self.elements, vecs)}

        n_molecules = self.df['smiles'].shape[0]
        n_samples = self.n_samples

        smiles_A = self.df['smiles'].values[np.random.choice(n_molecules, n_samples)]
        smiles_B = self.df2['smiles'].values[np.random.choice(n_molecules, n_samples, replace=True)]
        
        for i, (a,b) in enumerate(zip(smiles_A, smiles_B)):
            if i % 1000 == 0:
                print('\n{} attempted loads\n'.format(i))
            try:
                mol_a = mb.load(a, smiles=True)
                mol_b = mb.load(b, smiles=True)
                edge_index_1, h_1, x_1 = get_e_h_x(mol_a)
                edge_index_2, h_2, x_2, = get_e_h_x(mol_b)
                sasa = calculate_sasa(a, b)
                if sasa == float("nan") or sasa == float('inf') or sasa > 1000 or sasa is None:
                    continue
            except:
                print('unable to load \n{} \n{}\n'.format(a,b))
                continue

            m_data = Multi_Coord_Data(
                n_graphs = 2, 
                node_features = [h_1, h_2],
                coordinates = [x_1, x_2],
                edge_indexs = [edge_index_1, edge_index_2],
                n_nodes = [mol_a.n_particles, mol_b.n_particles],
                y = torch.tensor(sasa)
            )
            data_list.append(m_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class Simple_Dataset(InMemoryDataset):
    def __init__(self, root, smi_path, n_samples = 1000, transform=None, pre_transform=None, pre_filter=None):
        self.smi_path = smi_path
        self.n_samples = n_samples
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return glob('~/projects/terminal_groups_mixed/src/util/molecules/*.pdb')

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        self.df = pd.read_csv(self.smi_path, names = ['smiles','index','grp'], delimiter='	')
        self.df = self.df.dropna()
        self.df.reset_index()

        names = ''.join(s.upper() for s in self.df['smiles'].values)
        names += 'H'
        self.elements = [n for n in set(names) if n.isalpha()]
        vecs = F.one_hot(torch.arange(0, len(self.elements)), num_classes=len(self.elements))
        self.element2vec = {e:v for e, v in zip(self.elements, vecs)}

        n_molecules = self.df['smiles'].shape[0]
        n_samples = self.n_samples

        smiles_A = self.df['smiles'].values[np.random.choice(n_molecules, n_samples)]
        smiles_B = self.df['smiles'].values[np.random.choice(n_molecules, n_samples)]

        def get_e_h_x(mol):
            e_index = torch.empty((2,mol.n_bonds), dtype=torch.float)
            h = torch.empty((mol.n_particles,len(self.elements)), dtype=torch.float)
            x = torch.empty((mol.n_particles, 3), dtype=torch.float)
            #make edge index
            parts = {p:i for i, p in enumerate(mol.particles())}
            for i, b in enumerate(mol.bonds()):
                e_index[0,i] = parts[b[0]]
                e_index[1,i] = parts[b[1]]
            #make node features
            for i, p in enumerate(mol):
                h[i] = self.element2vec[p.element.symbol]        
            #make positional features
            x = torch.tensor(mol.xyz)
            return e_index, h, x
        
        for i, (a,b) in enumerate(zip(smiles_A, smiles_B)):
            if i % 1000 == 0:
                print('\n{} loaded!\n'.format(i))
            try:
                mol_a = mb.load(a, smiles=True)
                mol_b = mb.load(b, smiles=True)
                
                edge_index_1, h_1, x_1 = get_e_h_x(mol_a)
                edge_index_2, h_2, x_2, = get_e_h_x(mol_b)

            except:
                print('unable to load \n{} \n{}\n'.format(a,b))
                continue
            m_data = Multi_Coord_Data(
                n_graphs = 2, 
                node_features = [h_1, h_2],
                coordinates = [x_1, x_2],
                edge_indexs = [edge_index_1, edge_index_2],
                n_nodes = [mol_a.n_particles, mol_b.n_particles],
                y = torch.tensor(mol_a.n_particles + mol_b.n_particles)
            )
            data_list.append(m_data)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])