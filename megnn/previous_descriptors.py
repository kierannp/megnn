import os

import pandas as pd
from rdkit.Chem import Descriptors
from rdkit.Chem.Descriptors import rdMolDescriptors
from rdkit.Chem import AllChem as Chem

def rdkit_descriptors(smiles, ndigits=12, include_pc=True,
                      include_moe=False, include_h_bond=False,
                      ch3_smiles=None, barcode_seed=None,
                      vary_descriptors=None, vary_significant=None):
    """
    Parameters
    ----------
    include_pc : bool, optional, default=True
        Include partial charge descriptors (as defined in
        https://www.chemcomp.com/journal/descr.htm). These are calculated
        using Gasteiger charge assignments and VSA descriptors
        obtained from RDKit.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    Chem.EmbedMolecule(mol, Chem.ETKDG())

    descriptors = {}
    
    # Molecular weight
    descriptors['molwt'] = round(Descriptors.ExactMolWt(mol), ndigits)

    # Molecular weight (excluding H's)
    descriptors['molwt-hvy'] = round(Descriptors.HeavyAtomMolWt(mol), ndigits)

    # Number of valence electrons
    descriptors['e-valence'] = Descriptors.NumValenceElectrons(mol)

    # Balaban J value
    descriptors['balabanj'] = round(Descriptors.BalabanJ(mol), ndigits)

    # BertzCT
    descriptors['bertzct'] = round(Descriptors.BertzCT(mol), ndigits)

    # Ipc
    descriptors['ipc'] = round(Descriptors.Ipc(mol), ndigits)

    # Hall-Kier alpha
    descriptors['hk-alpha'] = Descriptors.HallKierAlpha(mol)

    # Hall-Kier kappas
    descriptors['hk-kappa1'] = round(Descriptors.Kappa1(mol), ndigits)
    descriptors['hk-kappa2'] = round(Descriptors.Kappa2(mol), ndigits)
    descriptors['hk-kappa3'] = round(Descriptors.Kappa3(mol), ndigits)

    # Chi values from Rev. Comput. Chem. 2:367-422 (1991)
    descriptors['chi0'] = round(Descriptors.Chi0(mol), ndigits)
    descriptors['chi1'] = round(Descriptors.Chi1(mol), ndigits)
    descriptors['chi0n'] = round(Descriptors.Chi0n(mol), ndigits)
    descriptors['chi1n'] = round(Descriptors.Chi1n(mol), ndigits)
    descriptors['chi2n'] = round(Descriptors.Chi2n(mol), ndigits)
    descriptors['chi3n'] = round(Descriptors.Chi3n(mol), ndigits)
    descriptors['chi4n'] = round(Descriptors.Chi4n(mol), ndigits)
    descriptors['chi0v'] = round(Descriptors.Chi0v(mol), ndigits)
    descriptors['chi1v'] = round(Descriptors.Chi1v(mol), ndigits)
    descriptors['chi2v'] = round(Descriptors.Chi2v(mol), ndigits)
    descriptors['chi3v'] = round(Descriptors.Chi3v(mol), ndigits)
    descriptors['chi4v'] = round(Descriptors.Chi4v(mol), ndigits)

    # Wildman-Crippen LogP value
    descriptors['logP'] = round(Descriptors.MolLogP(mol), ndigits)

    # Wildman-Crippen MR value
    descriptors['MR'] = round(Descriptors.MolMR(mol), ndigits)

    # Number of rotateable bonds
    descriptors['rbonds'] = Descriptors.NumRotatableBonds(mol)

    # Number of heavy atoms
    descriptors['nheavy'] = Descriptors.HeavyAtomCount(mol)

    # TPSA, J. Med. Chem. 43:3714-7, (2000)
    descriptors['tpsa'] = round(Descriptors.TPSA(mol), ndigits)

    # Labute's Approximate Surface Area, J. Mol. Graph. Mod. 18:464-77 (2000)
    descriptors['labuteASA'] = round(Descriptors.LabuteASA(mol), ndigits)

    # MOE-type descriptors using partial charges and SA contributions
    moe = {}
    moe['peoe-vsa1'] = round(Descriptors.PEOE_VSA1(mol), ndigits)
    moe['peoe-vsa2'] = round(Descriptors.PEOE_VSA2(mol), ndigits)
    moe['peoe-vsa3'] = round(Descriptors.PEOE_VSA3(mol), ndigits)
    moe['peoe-vsa4'] = round(Descriptors.PEOE_VSA4(mol), ndigits)
    moe['peoe-vsa5'] = round(Descriptors.PEOE_VSA5(mol), ndigits)
    moe['peoe-vsa6'] = round(Descriptors.PEOE_VSA6(mol), ndigits)
    moe['peoe-vsa7'] = round(Descriptors.PEOE_VSA7(mol), ndigits)
    moe['peoe-vsa8'] = round(Descriptors.PEOE_VSA8(mol), ndigits)
    moe['peoe-vsa9'] = round(Descriptors.PEOE_VSA9(mol), ndigits)
    moe['peoe-vsa10'] = round(Descriptors.PEOE_VSA10(mol), ndigits)
    moe['peoe-vsa11'] = round(Descriptors.PEOE_VSA11(mol), ndigits)
    moe['peoe-vsa12'] = round(Descriptors.PEOE_VSA12(mol), ndigits)
    moe['peoe-vsa13'] = round(Descriptors.PEOE_VSA13(mol), ndigits)
    moe['peoe-vsa14'] = round(Descriptors.PEOE_VSA14(mol), ndigits)

    # MOE-type descriptors using MR and SA contributions
    moe['smr-vsa1'] = round(Descriptors.SMR_VSA1(mol), ndigits)
    moe['smr-vsa2'] = round(Descriptors.SMR_VSA2(mol), ndigits)
    moe['smr-vsa3'] = round(Descriptors.SMR_VSA3(mol), ndigits)
    moe['smr-vsa4'] = round(Descriptors.SMR_VSA4(mol), ndigits)
    moe['smr-vsa5'] = round(Descriptors.SMR_VSA5(mol), ndigits)
    moe['smr-vsa6'] = round(Descriptors.SMR_VSA6(mol), ndigits)
    moe['smr-vsa7'] = round(Descriptors.SMR_VSA7(mol), ndigits)
    moe['smr-vsa8'] = round(Descriptors.SMR_VSA8(mol), ndigits)
    moe['smr-vsa9'] = round(Descriptors.SMR_VSA9(mol), ndigits)
    moe['smr-vsa10'] = round(Descriptors.SMR_VSA10(mol), ndigits)

    # MOE-type descriptors using LogP and SA contributions
    moe['slogP-vsa1'] = round(Descriptors.SlogP_VSA1(mol), ndigits)
    moe['slogP-vsa2'] = round(Descriptors.SlogP_VSA2(mol), ndigits)
    moe['slogP-vsa3'] = round(Descriptors.SlogP_VSA3(mol), ndigits)
    moe['slogP-vsa4'] = round(Descriptors.SlogP_VSA4(mol), ndigits)
    moe['slogP-vsa5'] = round(Descriptors.SlogP_VSA5(mol), ndigits)
    moe['slogP-vsa6'] = round(Descriptors.SlogP_VSA6(mol), ndigits)
    moe['slogP-vsa7'] = round(Descriptors.SlogP_VSA7(mol), ndigits)
    moe['slogP-vsa8'] = round(Descriptors.SlogP_VSA8(mol), ndigits)
    moe['slogP-vsa9'] = round(Descriptors.SlogP_VSA9(mol), ndigits)
    moe['slogP-vsa10'] = round(Descriptors.SlogP_VSA10(mol), ndigits)
    moe['slogP-vsa11'] = round(Descriptors.SlogP_VSA11(mol), ndigits)
    moe['slogP-vsa12'] = round(Descriptors.SlogP_VSA12(mol), ndigits)

    # MOE-type descriptors using EState indices as SA contributions
    moe['estate-vsa1'] = round(Descriptors.EState_VSA1(mol), ndigits)
    moe['estate-vsa2'] = round(Descriptors.EState_VSA2(mol), ndigits)
    moe['estate-vsa3'] = round(Descriptors.EState_VSA3(mol), ndigits)
    moe['estate-vsa4'] = round(Descriptors.EState_VSA4(mol), ndigits)
    moe['estate-vsa5'] = round(Descriptors.EState_VSA5(mol), ndigits)
    moe['estate-vsa6'] = round(Descriptors.EState_VSA6(mol), ndigits)
    moe['estate-vsa7'] = round(Descriptors.EState_VSA7(mol), ndigits)
    moe['estate-vsa8'] = round(Descriptors.EState_VSA8(mol), ndigits)
    moe['estate-vsa9'] = round(Descriptors.EState_VSA9(mol), ndigits)
    moe['estate-vsa10'] = round(Descriptors.EState_VSA10(mol), ndigits)
    moe['estate-vsa11'] = round(Descriptors.EState_VSA11(mol), ndigits)

    # MOE-type descriptors using EState indices as SA contributions
    moe['vsa-estate1'] = round(Descriptors.VSA_EState1(mol), ndigits)
    moe['vsa-estate2'] = round(Descriptors.VSA_EState2(mol), ndigits)
    moe['vsa-estate3'] = round(Descriptors.VSA_EState3(mol), ndigits)
    moe['vsa-estate4'] = round(Descriptors.VSA_EState4(mol), ndigits)
    moe['vsa-estate5'] = round(Descriptors.VSA_EState5(mol), ndigits)
    moe['vsa-estate6'] = round(Descriptors.VSA_EState6(mol), ndigits)
    moe['vsa-estate7'] = round(Descriptors.VSA_EState7(mol), ndigits)
    moe['vsa-estate8'] = round(Descriptors.VSA_EState8(mol), ndigits)
    moe['vsa-estate9'] = round(Descriptors.VSA_EState9(mol), ndigits)
    moe['vsa-estate10'] = round(Descriptors.VSA_EState10(mol), ndigits)

    if include_moe:
        descriptors.update(moe)

    # Plane of best fit, Firth et al., JCIM 52:2516-25
    descriptors['pbf'] = round(rdMolDescriptors.CalcPBF(mol), ndigits)

    # Principal moments of inertia
    descriptors['pmi1'] = round(rdMolDescriptors.CalcPMI1(mol), ndigits)
    descriptors['pmi2'] = round(rdMolDescriptors.CalcPMI2(mol), ndigits)
    descriptors['pmi3'] = round(rdMolDescriptors.CalcPMI3(mol), ndigits)

    # Normalized principal moments ratios Sauer and Schwarz JCIM 43:987-1003 (2003)
    descriptors['npr1'] = round(rdMolDescriptors.CalcNPR1(mol), ndigits)
    descriptors['npr2'] = round(rdMolDescriptors.CalcNPR2(mol), ndigits)

    # Radius of gyration
    descriptors['rg'] = round(rdMolDescriptors.CalcRadiusOfGyration(mol), ndigits)

    # Inertial shape factor
    descriptors['isf'] = round(rdMolDescriptors.CalcInertialShapeFactor(mol),
                               ndigits)

    # Eccentricity
    descriptors['eccentricity'] = round(rdMolDescriptors.CalcEccentricity(mol),
                                        ndigits)

    # Asphericity
    descriptors['asphericity'] = round(rdMolDescriptors.CalcAsphericity(mol),
                                       ndigits)

    # Spherocity Index
    descriptors['spherocity'] = round(rdMolDescriptors.CalcSpherocityIndex(mol),
                                      ndigits)

    # Charge descriptors
    if include_pc:
        Chem.ComputeGasteigerCharges(mol)
        charges = [float(atom.GetProp('_GasteigerCharge'))
                   for atom in mol.GetAtoms()]
        positive_charges = [c for c in charges if c > 0]
        negative_charges = [c for c in charges if c < 0]
        atoms = [atom for atom in mol.GetAtoms()]

        # Total positive charge
        descriptors['pc+'] = round(sum(positive_charges), ndigits)

        # Total negative charge
        descriptors['pc-'] = round(sum(negative_charges), ndigits)

        # Relative positive partial charge
        descriptors['rpc+'] = round(max(positive_charges) / sum(positive_charges), ndigits)

        # Relative negative partial charge
        descriptors['rpc-'] = round(min(negative_charges) / sum(negative_charges), ndigits)

        # Total positive van der Waals surface area
        descriptors['vsa+'] = round(sum([moe['peoe-vsa{}'.format(val)] for val in range(8,15)]), ndigits)

        # Total negative van der Waals surface area
        descriptors['vsa-'] = round(sum([moe['peoe-vsa{}'.format(val)] for val in range(1,8)]), ndigits)

        total_vsa = round(descriptors['vsa+'] + descriptors['vsa-'], ndigits)

        # Total positive polar van der Waals surface area
        descriptors['vsa-polar+'] = round(sum([moe['peoe-vsa{}'.format(val)] for val in range(12,15)]), ndigits)

        # Total negative polar van der Waals surface area
        descriptors['vsa-polar-'] = round(sum([moe['peoe-vsa{}'.format(val)] for val in range(1,4)]), ndigits)

        # Total hydrophobic van der Waals surface area
        descriptors['vsa-hyd'] = round(sum([moe['peoe-vsa{}'.format(val)] for val in range(4,12)]), ndigits)

        # Total polar van der Waals surface area
        descriptors['vsa-polar'] = round(descriptors['vsa-polar+'] + descriptors['vsa-polar-'], ndigits)

        # Fractional positive van der Waals surface area
        descriptors['vsa-fpos'] = round(descriptors['vsa+'] / total_vsa, ndigits)

        # Fractional negative van der Waals surface area
        descriptors['vsa-fneg'] = round(descriptors['vsa-'] / total_vsa, ndigits)

        # Fractional positive polar van der Waals surface area
        descriptors['vsa-fppos'] = round(descriptors['vsa-polar+'] / total_vsa, ndigits)

        # Fractional negative polar van der Waals surface area
        descriptors['vsa-fpneg'] = round(descriptors['vsa-polar-'] / total_vsa, ndigits)

        # Fractional hydrophobic van der Waals surface area
        descriptors['vsa-fhyd'] = round(descriptors['vsa-hyd'] / total_vsa, ndigits)

        # Fractional polar van der Waals surface area
        descriptors['vsa-polar'] = round(descriptors['vsa-polar'] / total_vsa, ndigits)

    # If the number of H-bond donors and acceptors are desired then
    # an additional smiles needs to be provided where groups are CH3-
    # terminated
    if include_h_bond:
        assert(ch3_smiles)
        mol_ch3 = Chem.AddHs(Chem.MolFromSmiles(ch3_smiles))
        descriptors['hdonors'] = Descriptors.NumHDonors(mol_ch3)
        descriptors['hacceptors'] = Descriptors.NumHAcceptors(mol_ch3)

    return descriptors
