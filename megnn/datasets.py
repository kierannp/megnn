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
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data
from urllib.request import urlopen
from urllib.parse import quote
from rdkit.Chem import AllChem
from rdkit import Chem
import rdkit
from rdkit.Chem import AllChem, rdFreeSASA
import math
import random
import sys
sys.path.insert(1, '~/projects/megnn')
from megnn.utils import *
from megnn.previous_descriptors import *

class PairData(Data):
    def __init__(
        self, 
        edge_index_s=None, x_s=None, positions_s=None, n_nodes_s=None, #edge_attr_s=None,
        edge_index_t=None, x_t=None, positions_t=None, n_nodes_t=None,
        y=None
    ):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.positions_s = positions_s
        # self.edge_attr_s = edge_attr_s
        self.n_nodes_s = n_nodes_s

        self.edge_index_t = edge_index_t
        self.x_t = x_t
        self.positions_t = positions_t
        # self.edge_attr_t = edge_attr_t
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
        return glob('~/projects/terminal_groups_mixed/src/util/molecules/*.pdb')

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
        if self.chem_name == 'difluoromethyl' or self.chem_name == 'phenol' or self.chem_name == 'toluene':
            return
        else:
            return self.smiles2e_index[chem_name]

    def replace_node_att(self, chem_name):
        if self.chem_name == 'difluoromethyl' or self.chem_name == 'phenol' or self.chem_name == 'toluene':
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
        features = self.dataframe.drop(['terminal_group_1','terminal_group_2','terminal_group_3', 'backbone','chainlength', 'frac-1','frac-2','COF','intercept'], axis=1, inplace=False)
        home = os.path.expanduser('~')
        molecules = glob(home + '/projects/terminal_groups_mixed/src/util/molecules/*')
        self.mean = self.dataframe['COF'].mean()
        self.std = self.dataframe['COF'].std()
        # self.dataframe['COF'] = (self.dataframe['COF']-self.mean)/self.std
        self.molecules = list(set(molecules))
        self.names2graph = {}
        self.mol_smiles = {}
        self.smiles2n_nodes = {}
        self.smiles2xyz = {}
        self.smiles2e_index = {}
        self.smiles2mol = {}
        self.smiles2x = {}

        self.missing_mols = ['difluoromethyl', 'phenol', 'toluene']
        self.mol_smiles['nitrophenyl'] = 'CC1=CC=C(C=C1)[N+]([O-])=O'
        self.mol_smiles['isopropyl'] = 'CC(C)O'
        self.mol_smiles['perfluoromethyl'] = 'CC(F)(F)F'
        self.mol_smiles['fluorophenyl'] = 'CC1=CC=C(F)C=C1'
        self.mol_smiles['carboxyl'] = 'C(=O)O'
        self.mol_smiles['amino'] = 'CN'
        self.mol_smiles['acetyl'] = 'C[C]=O'
        self.mol_smiles['nitrophenyl-h'] = 'CC1=CC=C(C=C1)[N+]([O-])=O'

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

        for m in self.molecules:
            mol_name = m.split('/')[-1].split('.')[0]
            if 'ch3' in mol_name:
                mol_name = mol_name.split('-')[0]
            mol = mb.load(m)
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

            p_data = PairData(edge_index_s, x_s, positions_s, n_nodes_s, edge_index_t, x_t, positions_t, n_nodes_t,  y = torch.tensor(row.COF))
            data_list.append(p_data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class COF_Dataset2(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
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
        return self.names2graph[chem_name]

    def n_nodes_tuple(self, t1, t2):
        return (self.smiles2n_nodes[t1], self.smiles2n_nodes[t2])
            
    def replace_n_nodes_tuple(self, row):
        return self.n_nodes_tuple(self.mol_smiles[row['terminal_group_1']], self.mol_smiles[row['terminal_group_2']])

    def get_positions(self, t1,t2):
        return (self.smiles2xyz[t1], self.smiles2xyz[t2])

    def replace_positions(self, row):
        return self.get_positions(row['terminal_group_1'], row['terminal_group_2'])

    def replace_e_index(self, chem_name):
        return self.smiles2e_index[chem_name]

    def replace_node_att(self, chem_name):
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
        descriptor_path = '/raid6/homes/kierannp/projects/iMoDELS-supplements/data/raw-data/descriptors-ind.csv'
        des_data = pd.read_csv('/raid6/homes/kierannp/projects/iMoDELS-supplements/data/raw-data/descriptors-ind.csv').T
        self.dataframe = pd.read_csv(data_path, index_col=0)
        self.dataframe = self.dataframe.dropna()
        self.dataframe = self.dataframe[(self.dataframe['frac-1']==.5)]
        self.mean = self.dataframe['COF'].mean()
        self.std = self.dataframe['COF'].std()
        self.dataframe['COF'] = (self.dataframe['COF']-self.mean)/self.std
        self.dataframe.reset_index()
        self.mol_smiles = {}
        self.mol_smiles['acetyl'] = 'C(=O)C'
        self.mol_smiles['amino'] = 'N'
        self.mol_smiles['carboxyl'] = 'CC(=O)O'
        self.mol_smiles['cyano'] = 'C#N'
        self.mol_smiles['cyclopropyl'] = 'C1C[CH]1'
        self.mol_smiles['difluoromethyl'] = 'FC(F)C'
        self.mol_smiles['ethylene'] = 'CC'
        self.mol_smiles['fluorophenyl'] = 'C1=CC=C(F)C=C1'
        self.mol_smiles['hydroxyl'] = 'CO'
        self.mol_smiles['isopropyl'] = 'CC(C)O'
        self.mol_smiles['methoxy'] = 'C[O]'
        self.mol_smiles['methyl'] = '[CH3]'
        self.mol_smiles['nitro'] = '[N+](=O)[O-]'
        self.mol_smiles['nitrophenyl'] = 'C1=CC(=CC=C1[N+](=O)[O-])O'
        self.mol_smiles['perfluoromethyl'] = 'CC(F)(F)F'
        self.mol_smiles['phenol'] = 'c1ccc(cc1)O'
        self.mol_smiles['phenyl'] = 'C1=CC=CC=C1'
        self.mol_smiles['pyrrole'] = 'C1=CNC=C1'
        self.mol_smiles['toluene'] = 'CC1=CC=CC=C1'

        # features = self.dataframe.drop(['terminal_group_1','terminal_group_2','terminal_group_3', 'backbone','chainlength', 'frac-1','frac-2','COF','intercept', 'COF-std', 'intercept-std'], axis=1, inplace=False)
        home = os.path.expanduser('~')
        self.mean = self.dataframe['COF'].mean()
        self.std = self.dataframe['COF'].std()
        self.names2graph = {}
        self.smiles2n_nodes = {}
        self.smiles2xyz = {}
        self.smiles2e_index = {}
        self.smiles2mol = {}
        self.smiles2x = {}

        names = ''.join(smiles.upper() for smiles in self.mol_smiles.values())
        names += 'H'
        self.elements = [n for n in set(names) if n.isalpha()]


        vecs = F.one_hot(torch.arange(0, len(self.elements)), num_classes=len(self.elements))
        self.element2vec = {e:v for e, v in zip(self.elements, vecs)}

        for s in self.mol_smiles.values():
            mol = mb.load(s,smiles=True)
            G = to_networkx(mol.to_gmso())
            adj = nx.adjacency_matrix(G)
            self.names2graph[s] = adj
            self.smiles2n_nodes[s] = len(list(G.nodes))
            self.smiles2xyz[s] = mol.xyz
            self.smiles2mol[s] = mol
            e_index = torch.empty((2,mol.n_bonds), dtype=torch.int64)
            parts = {p:i for i, p in enumerate(mol.particles())}
            for i, b in enumerate(mol.bonds()):
                e_index[0,i] = parts[b[0]]
                e_index[1,i] = parts[b[1]]
            self.smiles2e_index[s] = e_index
            x = torch.empty((mol.n_particles,len(self.elements)), dtype=torch.float32)
            for i, p in enumerate(mol.particles()):
                x[i] = self.element2vec[p.element.symbol]
            self.smiles2x[s] = x

        self.dataframe['n_nodes'] = self.dataframe.apply(self.replace_n_nodes_tuple, axis=1)
        self.dataframe.reset_index()
        # Read data into huge `Data` list.
        data_list = []
        # nodes_s, edges_s, nodes_t, edges_t = [], [], [], []
        # for i, row in self.dataframe.iterrows():
        #     nodes_s.append(self.smiles2n_nodes[row.terminal_group_1])
        #     nodes_t.append(self.smiles2n_nodes[row.terminal_group_2])
        #     edges_s.append(self.smiles2e_index[row.terminal_group_1])
        #     edges_t.append(self.smiles2e_index[row.terminal_group_2])
        
        for i, row in self.dataframe.iterrows():  # Iterate in batches over the training dataset.
            x_s = torch.tensor(self.smiles2x[self.mol_smiles[row.terminal_group_1]])
            other_features = torch.tensor(list(rdkit_descriptors(self.mol_smiles[row.terminal_group_1],ndigits=9).values()), dtype=torch.float)
            other_features = other_features.expand(x_s.size(0), other_features.size(0))
            # other_features = torch.tensor(row[features.columns]).reshape(1,len(features.columns))
            # other_features = other_features.expand(x_s.size(0), len(features.columns))
            x_s = torch.cat((x_s,other_features), 1)
            edge_index_s = self.smiles2e_index[self.mol_smiles[row.terminal_group_1]]
            positions_s = torch.tensor(self.smiles2xyz[self.mol_smiles[row.terminal_group_1]])
            n_nodes_s = self.smiles2n_nodes[self.mol_smiles[row.terminal_group_1]]
            x_t = torch.tensor(self.smiles2x[self.mol_smiles[row.terminal_group_2]])
            other_features = torch.tensor(list(rdkit_descriptors(self.mol_smiles[row.terminal_group_2],ndigits=9).values()), dtype=torch.float)
            other_features = other_features.expand(x_t.size(0), other_features.size(0))
            # other_features = torch.tensor(row[features.columns]).reshape(1,len(features.columns))
            # other_features = other_features.expand(x_t.size(0), len(features.columns))
            x_t = torch.cat((x_t, other_features), 1)
            edge_index_t = self.smiles2e_index[self.mol_smiles[row.terminal_group_2]]
            positions_t = torch.tensor(self.smiles2xyz[self.mol_smiles[row.terminal_group_2]])
            n_nodes_t = self.smiles2n_nodes[self.mol_smiles[row.terminal_group_2]]
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

class PdbBind_Dataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, dataframe=None, debug=False):
        self.dataframe = dataframe
        self.root = root
        self.debug = debug
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
    
    def coordinate_from_mol(self, molecule):
        molecule = Chem.AddHs(molecule)
        AllChem.EmbedMolecule(molecule)
        AllChem.UFFOptimizeMolecule(molecule)
        molecule.GetConformer()
        coords = []
        for i, atom in enumerate(molecule.GetAtoms()):
            positions = molecule.GetConformer().GetAtomPosition(i)
            coords.append([positions.x, positions.y, positions.z])
        return coords
        
    def process_mol2(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            f.close()
        
        for i, l in enumerate(lines):
            if 'TRIPOS>ATOM' in l:
                atom_start_line = i+1
            if 'TRIPOS>BOND' in l:
                bond_start_line = i+1
            if '@<TRIPOS>SUB' in l:
                bond_end_line = i
        coords = torch.empty((bond_start_line-atom_start_line-1, 3), dtype=torch.float32)
        element_dict = {}
        for i, l in enumerate(lines[atom_start_line:bond_start_line-1]):
            parts = l.split()
            
            element, charge = parts[5].split('.')[0], parts[8].split('.')[0]
            element_dict[i] = (element, float(charge))
            vals = torch.tensor(list(map(float, parts[2:5])))
            coords[i,:] = vals
        edge_list = [[],[]]
        for i, l in enumerate(lines[bond_start_line:bond_end_line]):
            source, target = l.split()[1:3]
            edge_list[0].append(int(source))
            edge_list[1].append(int(target))

        return element_dict, coords, edge_list
        
    def process(self):
        protein_names = set()
        diss_consts = {}
        delinquints = set()
        all_elements = set()
        unit_conversions = {'fM':10e-15, 'mM':10e-3, 'nM':10e-9, 'pM':10e-12, 'uM':10e-6}
        acceptable_elements = {'N', 'Zn', 'O', 'P', 'C', 'F', 'H', 'Cl', 'S'}
        mass_to_element = {14:'N', 65:'Zn', 16:'O', 31:'P', 12:'C', 19:'F', 1:'H', 35:'Cl', 32:'S'}
        data_list = []

        ############### Process index file that has K_d #########################
        with open('{}/index/INDEX_general_PL_data.2019'.format(self.root)) as f:
            lines = f.readlines()
            f.close()
        for l in lines[6:]:
            # Just taking the protein-ligand compounds that have Kd measurements
            experiment_value = l.split()[4]
            if 'Kd' in experiment_value:
                if experiment_value[2] != '>' or experiment_value[2] != '<':
                    if experiment_value[3:-2] != '=100':
                        protein_names.add(l.split()[0])
                        # Correct the units to standard Molarity units
                        diss_consts[l.split()[0]] = math.log(float(experiment_value[3:-2]) * unit_conversions[experiment_value[-2:]])

        if self.debug:
            protein_names = random.sample(protein_names, 50)
        ############### Add all element types #########################
        for pname in protein_names:
            files = glob('{}/{}/*'.format(self.root, pname))
            
            for f in files:
                if 'pocket' in f:
                    struc = mb.load(f)
                    elements = set(p.element.symbol for p in struc)
                    if not elements.issubset(acceptable_elements) or struc.n_particles>500:
                        print('Unable to load: {}'.format(pname) )
                        delinquints.add(pname)
                        continue
                elif 'ligand' in f and 'mol2' in f:
                    element_dict, _, _= self.process_mol2(f)
                    elements = {e[0] for e in element_dict.values()}
                    if not elements.issubset(acceptable_elements) or len(element_dict)>500:
                        print('Unable to load: {}'.format(pname) )
                        delinquints.add(pname)
                        continue
                else:
                    continue
                # Add any new elements 
                for a in elements:
                    all_elements.add(a)

        self.elements = all_elements
        vecs = F.one_hot(torch.arange(0, len(self.elements)), num_classes=len(self.elements))
        self.element2vec = {e:v for e, v in zip(self.elements, vecs)}

        with open('{}/../../ffnonbonded.itp'.format(self.root)) as itpfile:
            itp_lines = itpfile.readlines()
            itpfile.close()
        atype_to_lj = {}
        for i, l in enumerate(itp_lines[2:54]):
            if i == 0:
                name, nbr, mass, charge, ptype, sigma, epsilon, _, _, _, _ = l.split()
            else:
                name, nbr, mass, charge, ptype, sigma, epsilon = l.split()
            atype_to_lj[name] = (float(sigma), float(epsilon))

        ############### Process graphs #########################
        for pname in protein_names:
            if pname not in delinquints:
                files = glob('{}/{}/*'.format(self.root, pname))

                if len(files) == 0:
                    print('{} files not exist'.format(pname))
                    continue
                prot_edge_list, ligand_edge_list = None, None
                for f in files:
                    if 'pocket' in f:
                        if not os.path.exists('{}/{}/topol.top'.format(self.root, pname)):
                            success = atomtype(f, self.root, pname)
                            if not success:
                                break
                        with open('{}/{}/topol.top'.format(self.root, pname)) as topfile:
                            top_lines = topfile.readlines()
                            topfile.close()
                        if not os.path.exists('{}/{}/conf.gro'.format(self.root, pname)):
                            continue
                        with open('{}/{}/conf.gro'.format(self.root, pname)) as grofile:
                            gro_lines = grofile.readlines()
                            grofile.close()
                        
                        if len(glob('{}/{}/topol_Protein*'.format(self.root, pname))) > 0:
                            continue

                        atom_start = next(i + 2 for i, l in enumerate(top_lines) if '[ atoms ]' in l)
                        atom_stop = next(i - 1 for i, l in enumerate(top_lines) if '[ bonds ]' in l)
                        bond_start = next(i + 2 for i, l in enumerate(top_lines) if '[ bonds ]' in l)
                        bond_stop = next(i - 1 for i, l in enumerate(top_lines) if '[ pairs ]' in l)
                        bond_stop = next(i - 1 for i, l in enumerate(top_lines) if '[ pairs ]' in l)
                        # pair_start = next(i + 1 for i, l in enumerate(top_lines) if '[ pairs ]' in l)
                        # pair_stop = next(i - 1 for i, l in enumerate(top_lines) if '[ angles ]' in l)

                        # Make the coordinates for the molecule
                        prot_coords = torch.empty((len(gro_lines[2:-1]), 3),dtype=torch.float32)
                        nbr_to_res = {}
                        nbr_to_atype = {}
                        for i, l in enumerate(gro_lines[2:-1]):
                            res, atype, nbr, x, y, z = l.split()
                            nbr_to_res[int(nbr)] = res
                            nbr_to_atype[int(nbr)] = atype

                            x, y, z = float(x), float(y), float(z)
                            prot_coords[i] = torch.tensor((x, y, z))

                        # Make Edgelist for the graph
                        prot_edge_list = torch.empty((2, bond_stop-bond_start), dtype=torch.int64)
                        for i, l in enumerate(top_lines[bond_start:bond_stop]):
                            source, dest, _ = l.split()
                            prot_edge_list[0,i] = int(source)
                            prot_edge_list[1,i] = int(dest)
            
                        # Make edge features

                        # prot_edge_features = torch.empty((bond_stop-bond_start, 2), dtype=torch.int64)          # two features, for Leonard Jones parameters
                        # for i, l in enumerate(top_lines[pair_start:pair_stop]):
                        #     src, dest, _ = l.split()
                        #     atype = nbr_to_atype[int(src)]
                        #     atype_to_lj


                        # Make the node features
                        prot_x = torch.empty((len(gro_lines[2:-1]), len(self.elements) + 3), dtype=torch.float32)

                        i = 0
                        for l in top_lines[atom_start:atom_stop]:
                            if 'residue' in l:
                                continue
                            if 'qtot' in l:
                                nbr, atype, res, _, atom, _, charge, mass, _, _, _ = l.split()
                            else:
                                nbr, atype, res, _, atom, _, charge, mass = l.split()
                            charge, mass = torch.tensor([float(charge)]), float(mass)
                            sigma, epsilon = atype_to_lj[atype]
                            lj = torch.tensor([sigma, epsilon])
                            ele = mass_to_element[round(mass)]

                            prot_x[i] = torch.cat((self.element2vec[ele], charge, lj), 0)
                            i += 1
                    if 'ligand' in f and 'mol2' in f:
                        element_dict, ligand_coords, ligand_edge_list = self.process_mol2(f)

                        # Make Edgelist for the graph
                        ligand_edge_list = torch.tensor(ligand_edge_list)

                        # Make the coordinates for the molecule
                        ligand_coords = ligand_coords.clone()

                        # Make the node features
                        ligand_x = torch.empty((len(element_dict),len(self.elements)+3), dtype=torch.float32)
                        for i, a in enumerate(element_dict.values()):
                            ligand_x[i] = torch.cat( (self.element2vec[a[0]], torch.tensor([a[1],0,0])), dim=0)

                if prot_edge_list is not None and ligand_edge_list is not None:
                    p_data = PairData(
                            prot_edge_list, prot_x, prot_coords, prot_coords.size(0), 
                            ligand_edge_list, ligand_x, ligand_coords, ligand_coords.size(0),  
                            y = torch.tensor(diss_consts[pname])
                        )
                    data_list.append(p_data)


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

    # def generate_interaction(self, mol1, mol2):
        
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

            # self.smiles2e_index[mol_name] = e_index
            
            # for i, p in enumerate(mol.particles()):
            #     h[i] = self.element2vec[p.element.symbol]
            # self.smiles2x[mol_name] = h
            # x_s = torch.tensor(self.smiles2x[row.terminal_group_1])
            # other_features = torch.tensor(row[features.columns]).reshape(1,len(features.columns))
            # other_features = other_features.expand(x_s.size(0), len(features.columns))
            # x_s = torch.cat((x_s,other_features), 1)
            # edge_index_s = self.smiles2e_index[row.terminal_group_1]
            # positions_s = torch.tensor(self.smiles2xyz[row.terminal_group_1])
            # n_nodes_s = self.smiles2n_nodes[row.terminal_group_1]
            # x_t = torch.tensor(self.smiles2x[row.terminal_group_2])
            # other_features = torch.tensor(row[features.columns]).reshape(1,len(features.columns))
            # other_features = other_features.expand(x_t.size(0), len(features.columns))
            # x_t = torch.cat((x_t, other_features), 1)
            # edge_index_t = self.smiles2e_index[row.terminal_group_2]
            # positions_t = torch.tensor(self.smiles2xyz[row.terminal_group_2])
            # n_nodes_t = self.smiles2n_nodes[row.terminal_group_2]
            # # print('x_s:{}, x_t:{}, e_s:{}, e_t:{}, p_s:{}, p_t:{}'.format(x_s.size(),x_t.size(),edge_index_s.size(), edge_index_t.size(),positions_s.size(),positions_t.size()))
            except:
                print('unable to load \n{} \n{}\n'.format(a,b))
                continue
            p_data = PairData(
                edge_index_s=edge_index_1, x_s=h_1, positions_s=x_1, n_nodes_s=mol_a.n_particles, #edge_attr_s=None,
                edge_index_t=edge_index_2, x_t=h_2, positions_t=x_2, n_nodes_t=mol_b.n_particles,
                y = torch.tensor(mol_a.n_particles + mol_b.n_particles)
            )
            data_list.append(p_data)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

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
                if sasa == float("nan") or sasa == float('inf') or sasa > 10000:
                    continue
            except:
                print('unable to load \n{} \n{}\n'.format(a,b))
                continue
            p_data = PairData(
                edge_index_s=edge_index_1, x_s=h_1, positions_s=x_1, n_nodes_s=mol_a.n_particles, #edge_attr_s=None,
                edge_index_t=edge_index_2, x_t=h_2, positions_t=x_2, n_nodes_t=mol_b.n_particles,
                y = torch.tensor(sasa)
            )
            data_list.append(p_data)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
