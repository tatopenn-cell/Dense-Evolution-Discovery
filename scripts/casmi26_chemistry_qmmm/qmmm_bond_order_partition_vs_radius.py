"""
QM/MM region partitioning via real bond order (Mayer/Wiberg-style), for
Dense-Evolution issue #283. Inspired by Diffuse2Seg (arXiv:2609.06491,
Humer et al. 2026): propagate a seed signal through pairwise affinities
in an edge-preserving manner, instead of a fixed hop-count radius. Here
the affinity is not learned (no diffusion model exists for molecules in
this codebase) -- it is the REAL bond order computed from an actual HF
density matrix (D = 2P), already available from run_scf.

Verified locally first, cheap, on water and ethanol (pure native_hf path,
no pyscf needed):
  - Bond order values are chemically sensible: O-H ~0.95, C-C ~1.01,
    non-bonded pairs ~0.01 (water H...H, ethanol's non-adjacent atoms).
  - A max-product ("widest path") propagation of relevance from the
    reactive-bond seed atoms, WITHOUT any per-hop decay, does not work:
    real single-bond orders are all close to 1.0, so relevance barely
    decays across several bonds of a saturated chain -- the whole
    molecule stays "relevant" regardless of distance from the seed.
    Real single bonds don't have the discontinuities Diffuse2Seg's image
    edges do.
  - Adding a fixed per-hop decay factor (relevance *= decay at each
    propagation step, in addition to the bond-order weight) fixes this:
    relevance now decreases monotonically with distance, modulated by
    bond strength. Verified isomorphism-invariant (result depends only on
    the bond-order graph, never on atom index/labeling) both without and
    with decay, on ethanol, via explicit random relabeling.
  - Ethanol (9 atoms) is too small to show a real cutoff -- the whole
    molecule stays above any reasonable threshold within 2 hops. This
    kernel tests on 5-amino-1-pentanol (OCCCCCN, used for the earlier
    electrostatic-embedding experiment), which has enough atoms and TWO
    different polar groups (OH, NH2) for a meaningful comparison against
    the existing fixed-radius partition.

CONCLUSION on 5-amino-1-pentanol -- an honest negative/inconclusive
result, NOT promoted to Dense-Evolution:
  - Bond orders are again chemically sensible (O-C1=0.980, C1-C2=0.990,
    O-H=0.945), and isomorphism invariance holds on this real molecule
    too (verified by explicit random relabeling).
  - The resulting regions differ from the fixed-radius BFS partition at
    every matched threshold, but decay=0.7 pulls in the ENTIRE molecule
    (including the far NH2 group) while decay=0.3 cuts down to just 3
    heavy atoms -- on a plain aliphatic chain where every backbone bond
    (C-C, C-O, C-N) has similarly strong bond order (~0.97-0.99), there
    is no real bond-strength heterogeneity for the method to exploit.
  - The BFS radius partition is itself already isomorphism-invariant
    (it only uses the molecular graph, never atom indices), so that
    property isn't actually a differentiator in this method's favor.
  - This approach also costs strictly more than BFS: it needs a real HF
    calculation on the WHOLE molecule first just to get a density matrix,
    where BFS is free and instant. On the molecules tested here, that
    extra cost bought no demonstrated accuracy or robustness advantage.
  - Not promoted. To actually test whether this method is worth its cost,
    the right next experiment is a molecule with real bond-order
    heterogeneity near the reactive site (e.g. one branch aromatic/
    conjugated, one branch a plain rotatable single bond, both at the
    same hop-distance from the seed) -- not done here.
"""
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit", "basis_set_exchange", "pyscf",
                "git+https://github.com/tatopenn-cell/Dense-Evolution.git@main"],
               check=True)

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from dense_evolution.native_hf.basis import build_molecule_shells
from dense_evolution.native_hf.cartesian import cartesian_powers
from dense_evolution.native_hf.libcint_bridge import (
    build_repulsion_tensor_libcint, build_overlap_and_core_hamiltonian_libcint,
)
from dense_evolution.native_hf.scf import run_scf

ANGSTROM_TO_BOHR = 1.8897259886
BASIS = "sto-3g"


def embed_whole(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 2026
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def whole_geometry_bohr(mol):
    conf = mol.GetConformer()
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    return pos * ANGSTROM_TO_BOHR


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


def propagate(bo, seeds, n_atoms, decay, n_iter=None):
    n_iter = n_iter or n_atoms
    relevance = np.zeros(n_atoms)
    for s in seeds:
        relevance[s] = 1.0
    for _ in range(n_iter):
        new_rel = relevance.copy()
        for A in range(n_atoms):
            for B in range(n_atoms):
                if A == B:
                    continue
                new_rel[A] = max(new_rel[A], relevance[B] * bo[B, A] * decay)
        relevance = new_rel
    return relevance


if __name__ == "__main__":
    whole_smiles = "OCCCCCN"
    mol_noH = Chem.MolFromSmiles(whole_smiles)
    o_idx = [a.GetIdx() for a in mol_noH.GetAtoms() if a.GetSymbol() == "O"][0]
    c_idx = mol_noH.GetAtomWithIdx(o_idx).GetNeighbors()[0].GetIdx()

    whole_mol = embed_whole(whole_smiles)
    atomic_numbers = [a.GetAtomicNum() for a in whole_mol.GetAtoms()]
    geometry_bohr = whole_geometry_bohr(whole_mol)
    n_atoms = len(atomic_numbers)
    print(f"Molecule: {whole_smiles}, reactive bond O(idx={o_idx})-C(idx={c_idx}), {n_atoms} atoms total")

    S, H_core = build_overlap_and_core_hamiltonian_libcint(atomic_numbers, geometry_bohr, BASIS)
    repulsion = build_repulsion_tensor_libcint(atomic_numbers, geometry_bohr, BASIS)
    n_electrons = sum(atomic_numbers)
    result = run_scf(S, H_core, repulsion, n_electrons, [float(z) for z in atomic_numbers], geometry_bohr)
    print(f"HF converged={result.converged}, n_iterations={result.n_iterations}, "
          f"total_energy={result.total_energy:.6f} Hartree")

    ao_atom = ao_atom_map(atomic_numbers, geometry_bohr, BASIS)
    D = 2 * np.asarray(result.density_matrix)
    bo = bond_order_matrix(D, np.asarray(S), ao_atom, n_atoms)

    print("\nBond orders for atoms adjacent to O and N (sanity check):")
    for a_idx, label in ((o_idx, "O"), (c_idx, "C1")):
        top = sorted(range(n_atoms), key=lambda i: -bo[a_idx, i])[:3]
        print(f"  {label} (idx={a_idx}) strongest bonds: " +
              ", ".join(f"atom{i}(Z={atomic_numbers[i]})={bo[a_idx, i]:.3f}" for i in top))

    seeds = [o_idx, c_idx]
    for decay in (0.3, 0.5, 0.7):
        rel = propagate(bo, seeds, n_atoms, decay)
        region = sorted(int(i) for i in np.where(rel > 0.1)[0])
        heavy_region = [i for i in region if atomic_numbers[i] > 1]
        print(f"\ndecay={decay}: relevance={np.round(rel, 3)}")
        print(f"  QM region (threshold=0.1): {region} ({len(heavy_region)} heavy atoms: "
              f"{[atomic_numbers[i] for i in heavy_region]})")

    print("\n=== Isomorphism-invariance check (random atom relabeling) ===")
    decay = 0.5
    rel = propagate(bo, seeds, n_atoms, decay)
    region_orig = set(int(i) for i in np.where(rel > 0.1)[0])
    rng = np.random.default_rng(123)
    perm = rng.permutation(n_atoms)
    bo_perm = bo[np.ix_(perm, perm)]
    seeds_perm = [int(np.where(perm == s)[0][0]) for s in seeds]
    rel_perm = propagate(bo_perm, seeds_perm, n_atoms, decay)
    region_perm_mapped_back = set(int(perm[i]) for i in np.where(rel_perm > 0.1)[0])
    print(f"  region (original labeling): {sorted(region_orig)}")
    print(f"  region (relabeled, mapped back): {sorted(region_perm_mapped_back)}")
    print(f"  [RESULT] ISOMORPHISM INVARIANT: {region_orig == region_perm_mapped_back}")

    print("\n=== Comparison: bond-order region vs old fixed-radius BFS region ===")

    def qm_region_bfs(radius):
        qm_heavy = {o_idx, c_idx}
        frontier = {o_idx, c_idx}
        for _ in range(radius):
            new_frontier = set()
            for idx in frontier:
                for nbr in whole_mol.GetAtomWithIdx(idx).GetNeighbors():
                    if nbr.GetAtomicNum() > 1 and nbr.GetIdx() not in qm_heavy:
                        new_frontier.add(nbr.GetIdx())
            qm_heavy |= new_frontier
            frontier = new_frontier
        qm_atoms = set(qm_heavy)
        for idx in qm_heavy:
            for nbr in whole_mol.GetAtomWithIdx(idx).GetNeighbors():
                if nbr.GetAtomicNum() == 1:
                    qm_atoms.add(nbr.GetIdx())
        return qm_atoms

    for radius in (1, 2, 3):
        bfs_region = qm_region_bfs(radius)
        rel = propagate(bo, seeds, n_atoms, 0.5)
        bo_region = set(int(i) for i in np.where(rel > 0.5 ** radius * 0.9)[0])
        print(f"  radius={radius}: BFS region={sorted(bfs_region)}, "
              f"bond-order region (threshold~decay^{radius})={sorted(bo_region)}, "
              f"same set: {bfs_region == bo_region}")
