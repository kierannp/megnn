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
from torch_geometric.data import Data
from urllib.request import urlopen
from urllib.parse import quote
from rdkit.Chem import AllChem
from rdkit import Chem
import rdkit
import math

class PairData(Data):
    def __init__(self, edge_index_s=None, x_s=None, positions_s=None, 
                n_nodes_s=None,edge_index_t=None, x_t=None, positions_t=None,
                n_nodes_t=None, y=None):
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
        features = self.dataframe.drop(['terminal_group_1','terminal_group_2','terminal_group_3', 'backbone','chainlength', 'frac-1','frac-2','COF','intercept', 'COF-std', 'intercept-std'], axis=1, inplace=False)
        home = os.path.expanduser('~')
        molecules = glob(home + '/projects/terminal_groups_mixed/src/util/molecules/*')
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
        self.mol_smiles['carboxyl'] = '*C(=O)O'
        self.mol_smiles['amino'] = 'CN'
        self.mol_smiles['acetyl'] = 'C[C]=O'

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
    
    def coordinate_from_mol(self, molecule):
        molecule = Chem.AddHs(molecule)
        AllChem.EmbedMolecule(molecule)
        AllChem.UFFOptimizeMolecule(molecule)
        molecule.GetConformer()
        print()
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
            
            element = parts[5].split('.')[0]
            element_dict[i] = element
            vals = torch.tensor(list(map(float, parts[2:5])))
            coords[i,:] = vals
        edge_list = [[],[]]
        for i, l in enumerate(lines[bond_start_line:bond_end_line]):
            source, target = l.split()[1:3]
            edge_list[0].append(int(source))
            edge_list[1].append(int(target))

        return element_dict, coords, edge_list

    def process(self):
        protein_names = []
        diss_consts = {}
        delinquints = set()
        all_elements = set()
        unit_conversions = {'fM':10e-15, 'mM':10e-3, 'nM':10e-9, 'pM':10e-12, 'uM':10e-6}
        acceptable_elements = {'N', 'Zn', 'O', 'P', 'C', 'F', 'H', 'Cl', 'S'}
        # Read data into huge `Data` list.
        data_list = []
        with open('./v2019-other-PL/index/INDEX_general_PL_data.2019') as f:
            lines = f.readlines()
            f.close()
        for l in lines[6:]:
            # Just taking the protein-ligand compounds that have Kd measurements
            experiment_value = l.split()[4]
            if 'Kd' in experiment_value:
                if experiment_value[2] != '>' or experiment_value[2] != '<':
                    if experiment_value[3:-2] != '=100':
                        protein_names.append(l.split()[0])
                        # Correct the units to standard Molarity units
                        diss_consts[l.split()[0]] = math.log(float(experiment_value[3:-2]) * unit_conversions[experiment_value[-2:]])

        for pname in protein_names:
            files = glob('./v2019-other-PL/'+pname+'/*')
            for f in files:
                if 'pocket' in f:
                    struc = mb.load(f)
                    elements = set(p.element.symbol for p in struc)
                elif 'ligand' in f and 'mol2' in f:
                    element_dict, _, _= self.process_mol2(f)
                    elements = set(element_dict.values())
                else:
                    continue
                if not elements.issubset(acceptable_elements) or struc.n_particles>500:
                    print('Unable to load: {}'.format(pname) )
                    delinquints.add(pname)
                    continue
                # Add any new elements 
                for a in elements:
                    all_elements.add(a)

        self.elements = all_elements
        vecs = F.one_hot(torch.arange(0, len(self.elements)), num_classes=len(self.elements))
        self.element2vec = {e:v for e, v in zip(self.elements, vecs)}

        for pname in protein_names:
            if pname not in delinquints:
                files = glob('./v2019-other-PL/'+pname+'/*')
                for f in files:
                    if 'pocket' in f:
                        prot = mb.load(f)
                        # G = nx.Graph(prot.bond_graph._adj)

                        # Make Edgelist for the graph
                        prot_edge_list = torch.empty((2,prot.n_bonds), dtype=torch.int64)
                        parts = {p:i for i, p in enumerate(prot.particles())}
                        for i, b in enumerate(prot.bonds()):
                            prot_edge_list[0,i] = parts[b[0]]
                            prot_edge_list[1,i] = parts[b[1]]

                        # Make the coordinates for the molecule
                        prot_coords = torch.tensor(prot.xyz)

                        # Make the node features
                        prot_x = torch.empty((prot.n_particles,len(self.elements)), dtype=torch.float32)
                        for i, a in enumerate(prot):
                            prot_x[i] = self.element2vec[a.element.symbol]

                    if 'ligand' in f and 'mol2' in f:
                        element_dict, ligand_coords, ligand_edge_list = self.process_mol2(f)
                        # adj = nx.adjacency_matrix(G)
                        # e_index = torch.empty((2,mol.n_bonds), dtype=torch.int64)

                        # Make Edgelist for the graph
                        ligand_edge_list = torch.tensor(ligand_edge_list)

                        # Make the coordinates for the molecule
                        ligand_coords = ligand_coords.clone()

                        # Make the node features
                        ligand_x = torch.empty((len(element_dict),len(self.elements)), dtype=torch.float32)
                        for i, a in enumerate(element_dict.values()):
                            ligand_x[i] = self.element2vec[a]

                p_data = PairData(prot_edge_list, prot_x, prot_coords, prot_coords.size(0), 
                                    ligand_edge_list, ligand_x, ligand_coords, ligand_coords.size(0),  
                                    y = torch.tensor(diss_consts[pname]))
                data_list.append(p_data)


        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
