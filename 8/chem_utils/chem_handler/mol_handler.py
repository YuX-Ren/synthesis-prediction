#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/1/5 11:21
# @Author  : zhangbc0315@outlook.com
# @File    : mol_handler.py
# @Software: PyCharm
import logging
from typing import Iterator, Tuple, Dict

from rdkit.Chem.rdchem import Mol, EditableMol, Atom, BondType
from rdkit.Chem import AllChem


class MolHandler:

    # region ===== correct mol =====

    @classmethod
    def link_to_atom_by_bond(cls, atom, atomic_num: int, bond_type):
        for bond in atom.GetBonds():
            if bond.GetBondType() != bond_type:
                continue
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            if atom1 == atom and atom2.GetAtomicNum() == atomic_num:
                return True, bond, atom2
            elif atom2 == atom and atom1.GetAtomicNum() == atomic_num:
                return True, bond, atom1
        return False, None, None

    @classmethod
    def count_bond_weight(cls, atom):
        res = 0
        for bond in atom.GetBonds():
            bond_type_level = int(bond.GetBondType())
            if bond_type_level <= 3:
                res += bond_type_level
            else:
                res += 1.5
        return res

    @classmethod
    def correct_n(cls, n_atom, rdmol):
        bond_weight = cls.count_bond_weight(n_atom)
        if n_atom.GetTotalNumHs() == 0:
            return
        if bond_weight + n_atom.GetTotalNumHs() < 4:
            return
        n_atom.SetNumExplicitHs(0)
        mol = AllChem.MolFromSmiles(AllChem.MolToSmiles(rdmol))
        if mol is not None:
            return
        n_atom.SetNumExplicitHs(1)

    @classmethod
    def correct_mol(cls, rdmol):
        mol = AllChem.MolFromSmiles(AllChem.MolToSmiles(rdmol))
        if mol is not None:
            return
        # print(AllChem.MolToSmiles(rdmol))
        for atom in rdmol.GetAtoms():
            if atom.GetAtomicNum() == 7:
                try:
                    cls.correct_n(atom, rdmol)
                except:
                    continue
        # print(AllChem.MolToSmiles(rdmol))

    # endregion

    @classmethod
    def get_inchis_from_smileses(cls, smileses: [str]):
        inchis = []
        for smiles in smileses:
            mol = AllChem.MolFromSmiles(smiles)
            if mol is None:
                return None
            inchis.append(AllChem.MolToInchi(mol))
        return inchis

    @classmethod
    def remove_inchi_stereo(cls, inchi: str):
        mol = AllChem.MolFromInchi(inchi)
        AllChem.RemoveStereochemistry(mol)
        return AllChem.MolToInchi(mol)

    @classmethod
    def atom_num(cls, smiles):
        mol = AllChem.MolFromSmiles(smiles)
        return mol.GetNumAtoms()

    @classmethod
    def bond_num(cls, smiles):
        mol = AllChem.MolFromSmiles(smiles)
        if mol is None:
            logging.warning(f"wrong smiles: {smiles}")
            return 0
        return mol.GetNumBonds()

    @classmethod
    def remove_charge(cls, rdmol):
        for atom in rdmol.GetAtoms():
            atom.SetFormalCharge(0)

    @classmethod
    def contain_smiles_or_error(cls, smileses: [str], smiles: str):
        if smiles in smileses:
            return True
        try:
            inchis = [AllChem.MolToInchi(AllChem.MolFromSmiles(smi)) for smi in smileses]
        except:
            return True
        if any([inc is None for inc in inchis]):
            return True
        inchi = AllChem.MolToInchi(AllChem.MolFromSmiles(smiles))
        if inchi is None:
            return True
        return inchi in inchis

    # region ===== smiles inchi =====

    @classmethod
    def smiles_from_inchi(cls, inchi: str):
        return AllChem.MolToSmiles(AllChem.MolFromInchi(inchi))

    @classmethod
    def inchi_from_smiles(cls, smiles: str):
        return AllChem.MolToInchi(AllChem.MolFromSmiles(smiles))

    # endregion

    # region ===== smiles/inchi to mol =====
    @classmethod
    def mol_from_smiles(cls, smiles: str):
        try:
            mol = AllChem.MolFromSmiles(smiles)
        except:
            mol = None
        return mol

    @classmethod
    def mol_from_inchi(cls, inchi: str):
        try:
            mol = AllChem.MolFromInchi(inchi)
            smi = AllChem.MolToSmiles(mol)
            mol = AllChem.MolFromSmiles(smi)
        except:
            mol = None
        return mol

    @classmethod
    def mol_from_inchis(cls, inchis: [str]):
        mol = None
        for inchi in inchis:
            if mol is None:
                mol = AllChem.MolFromInchi(inchi)
            else:
                mol = AllChem.CombineMols(mol, AllChem.MolFromInchi(inchi))
        return mol

    # region ===== get mol fingerprint =====

    @classmethod
    def get_mol_fp_by_rdmol(cls, rdmol: Mol):
        return AllChem.GetMorganFingerprintAsBitVect(rdmol)

    @classmethod
    def get_mol_fp_by_smiles(cls, smiles: str):
        return cls.get_mol_fp_by_rdmol(AllChem.MolFromSmiles(smiles))

    # endregion

    # region ===== get smiles =====

    @classmethod
    def get_smiles_without_map_num(cls, mol: Mol) -> str:
        copy_mol = AllChem.MolFromSmiles(AllChem.MolToSmiles(mol))
        for atom in copy_mol.GetAtoms():
            atom.SetAtomMapNum(0)
        return AllChem.MolToSmiles(copy_mol)

    # endregion

    # region ===== canonicalize map num =====

    @classmethod
    def canonical_rank_atoms(cls, mol):
        idx_to_old_map = {}
        for idx, atom in enumerate(mol.GetAtoms()):
            idx_to_old_map[idx] = atom.GetAtomMapNum()
            atom.SetAtomMapNum(0)
        atoms_ranks = list(AllChem.CanonicalRankAtoms(mol, breakTies=True))
        ranked_atom_idxes = []
        for r in range(len(atoms_ranks)):
            ranked_atom_idxes.append(atoms_ranks.index(r))
        return ranked_atom_idxes, idx_to_old_map

    @classmethod
    def canonicalize_map_num(cls, mol: Mol) -> Dict[int, int]:
        map_num_old_to_new = {}
        new_map_num = 1
        ranked_atom_idxes, idx_to_old_map = cls.canonical_rank_atoms(mol)
        for idx in ranked_atom_idxes:
            atom = mol.GetAtomWithIdx(idx)
            old_map_num = idx_to_old_map[idx]
            if old_map_num == 0:
                continue
            if old_map_num in map_num_old_to_new.keys():
                atom.SetAtomMapNum(map_num_old_to_new[old_map_num])
            else:
                atom.SetAtomMapNum(new_map_num)
                map_num_old_to_new[old_map_num] = new_map_num
                new_map_num += 1
        AllChem.CanonicalizeMol(mol)
        return map_num_old_to_new

    # endregion

    # region ===== edit mol =====

    @classmethod
    def merge_mols(cls, mols: [Mol]) -> Mol:
        res = None
        for n, mol in enumerate(mols):
            if n == 0:
                res = mol
            else:
                res = AllChem.CombineMols(res, mol)
        return res

    @classmethod
    def frag_idxes_to_mol(cls, mol: Mol, idxes: [int]):
        em = EditableMol(Mol())
        idxes_old_to_new = {}
        for i, idx in enumerate(idxes):
            em.AddAtom(mol.GetAtomWithIdx(idx))
            idxes_old_to_new[idx] = i

        for i, idx in enumerate(idxes):
            atom = mol.GetAtomWithIdx(idx)
            for bond in atom.GetBonds():
                if bond.GetBeginAtomIdx() == idx:
                    other_idx = bond.GetEndAtomIdx()
                else:
                    other_idx = bond.GetBeginAtomIdx()
                if other_idx < idx:
                    continue
                if other_idx in idxes_old_to_new.keys():
                    em.AddBond(idxes_old_to_new[idx], idxes_old_to_new[other_idx], bond.GetBondType())
        res = em.GetMol()
        res.ClearComputedProps()
        AllChem.GetSymmSSSR(res)
        res.UpdatePropertyCache(False)
        res._idxMap = idxes_old_to_new
        return res

    # endregion

    # region ===== query atom with map num =====

    @classmethod
    def contain_atom_with_map_num_in_mols(cls, mols: [Mol], map_num: int) -> bool:
        for _ in cls.query_atoms_with_map_num_in_mols(mols, map_num):
            return True
        return False

    @classmethod
    def query_atoms_with_map_num_in_mols(cls, mols: [Mol], map_num: int) -> Iterator[Atom]:
        for mol in mols:
            for atom in cls.query_atoms_with_map_num(mol, map_num):
                yield atom

    @classmethod
    def query_atoms_with_map_num(cls, mol: Mol, map_num: int) -> Iterator[Atom]:
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == map_num:
                yield atom

    # endregion


if __name__ == "__main__":
    pass
