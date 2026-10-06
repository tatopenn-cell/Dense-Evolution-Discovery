"""
Systematic lambda sweep of qmmm.propagation.propagate_relevance (the real
Diffuse2Seg p-Laplacian algorithm, arXiv:2609.06491 Algorithm 1, with the
exact degenerate-case limit worked out directly rather than an epsilon
patch) on the real Mayer bond-order graph of OCC(c1ccccc1)CCC.

Purpose: map how the aromatic-vs-alkyl differentiation signal actually
behaves across lambda, not search for a value that looks good. lam=1e-5
is the paper's own final value (Sec. 4.2), calibrated for a dense grid of
prompts later merged together -- this molecule gets exactly one seed
pair, with no merging step, so there is no reason to expect that specific
value to carry over, and no reason to hide what happens across the full
range instead.

RESULT (see docs/qmmm_utils.md for the full table and plot): the ratio
starts BELOW 1.0 (aromatic branch gets LESS relevance than alkyl) at the
paper's own lam, only crosses 1.0 around lam~0.5-1, and only reaches a
large aromatic-favoring differentiation (ratio 1.49) at lam=10, two
orders of magnitude above the paper's calibrated value.
"""
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit", "basis_set_exchange", "pyscf",
                "git+https://github.com/tatopenn-cell/Dense-Evolution.git@main"],
               check=True)
subprocess.run(["curl", "-sL",
                 "https://raw.githubusercontent.com/tatopenn-cell/Dense-Evolution-Discovery/"
                 "main/scripts/qmmm/propagation.py",
                 "-o", "/kaggle/working/propagation.py"], check=True)
sys.path.insert(0, "/kaggle/working")

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from dense_evolution.native_hf.basis import build_molecule_shells
from dense_evolution.native_hf.cartesian import cartesian_powers
from dense_evolution.native_hf.libcint_bridge import (
    build_repulsion_tensor_libcint, build_overlap_and_core_hamiltonian_libcint,
)
from dense_evolution.native_hf.scf import run_scf
from propagation import propagate_relevance

ANGSTROM_TO_BOHR = 1.8897259886
BASIS = "sto-3g"


def embed_whole(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 2026
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def ao_atom_map(atomic_numbers, geometry_bohr, basis_name):
    shells = build_molecule_shells(atomic_numbers, geometry_bohr, basis_name)
    ao_atom = []
    for s in shells:
        n_comp = len(cartesian_powers(s.degree))
        ao_atom.extend([int(s.atom_index)] * n_comp)
    return np.array(ao_atom)


def bond_order_matrix(D, S, ao_atom, n_atoms):
    DS = D @ S
    bo = np.zeros((n_atoms, n_atoms))
    for A in range(n_atoms):
        for B in range(n_atoms):
            if A == B:
                continue
            ma, mb = ao_atom == A, ao_atom == B
            bo[A, B] = (DS[np.ix_(ma, mb)] * DS[np.ix_(mb, ma)].T).sum()
    return bo


if __name__ == "__main__":
    whole_smiles = "OCC(c1ccccc1)CCC"
    mol_noH = Chem.MolFromSmiles(whole_smiles)
    o_idx = [a.GetIdx() for a in mol_noH.GetAtoms() if a.GetSymbol() == "O"][0]
    c1_idx = mol_noH.GetAtomWithIdx(o_idx).GetNeighbors()[0].GetIdx()
    c2_idx = [n.GetIdx() for n in mol_noH.GetAtomWithIdx(c1_idx).GetNeighbors() if n.GetIdx() != o_idx][0]
    aromatic_first = [n.GetIdx() for n in mol_noH.GetAtomWithIdx(c2_idx).GetNeighbors()
                       if n.GetIsAromatic()][0]
    alkyl_first = [n.GetIdx() for n in mol_noH.GetAtomWithIdx(c2_idx).GetNeighbors()
                   if n.GetIdx() not in (c1_idx,) and not n.GetIsAromatic()][0]

    whole_mol = embed_whole(whole_smiles)
    atomic_numbers = [a.GetAtomicNum() for a in whole_mol.GetAtoms()]
    conf = whole_mol.GetConformer()
    geometry_bohr = np.array([list(conf.GetAtomPosition(i)) for i in range(whole_mol.GetNumAtoms())]) * ANGSTROM_TO_BOHR
    n_atoms = len(atomic_numbers)
    print(f"Molecule: {whole_smiles}, {n_atoms} atoms\n")

    S, H_core = build_overlap_and_core_hamiltonian_libcint(atomic_numbers, geometry_bohr, BASIS)
    repulsion = build_repulsion_tensor_libcint(atomic_numbers, geometry_bohr, BASIS)
    n_electrons = sum(atomic_numbers)
    result = run_scf(S, H_core, repulsion, n_electrons, [float(z) for z in atomic_numbers], geometry_bohr)
    print(f"HF converged={result.converged}, total_energy={result.total_energy:.6f} Hartree\n")

    ao_atom = ao_atom_map(atomic_numbers, geometry_bohr, BASIS)
    D = 2 * np.asarray(result.density_matrix)
    bo = bond_order_matrix(D, np.asarray(S), ao_atom, n_atoms)

    aromatic_second = [n.GetIdx() for n in whole_mol.GetAtomWithIdx(aromatic_first).GetNeighbors()
                        if n.GetIdx() != c2_idx and n.GetIsAromatic()][0]
    alkyl_second = [n.GetIdx() for n in whole_mol.GetAtomWithIdx(alkyl_first).GetNeighbors()
                    if n.GetIdx() != c2_idx and n.GetAtomicNum() > 1][0]
    print(f"C2-aromatic bond order: {bo[c2_idx, aromatic_first]:.3f}, "
          f"C2-alkyl bond order: {bo[c2_idx, alkyl_first]:.3f}")
    print(f"aromatic ring internal bond order: {bo[aromatic_first, aromatic_second]:.3f}, "
          f"alkyl chain internal bond order: {bo[alkyl_first, alkyl_second]:.3f}\n")

    print(f"{'lambda':>10} {'iters':>7} {'aromatic_2nd':>14} {'alkyl_2nd':>12} {'ratio':>8}")
    for lam in [1e-5, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0]:
        rel = propagate_relevance(bo, [o_idx, c1_idx], n_atoms, p=1.6, lam=lam)
        ratio = rel[aromatic_second] / rel[alkyl_second] if rel[alkyl_second] > 0 else float("inf")
        print(f"{lam:>10.5g} {'':>7} {rel[aromatic_second]:>14.6f} {rel[alkyl_second]:>12.6f} {ratio:>8.4f}")
