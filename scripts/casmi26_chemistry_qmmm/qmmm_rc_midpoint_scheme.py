"""
Real "RC" (redistributed charge) scheme, Lin & Truhlar 2005 (J. Phys.
Chem. A 109, 3991), tested as the next real attempt at QM/MM point 5
(electrostatic embedding). Not the naive charge-shift already tried and
found worse than plain (qmmm_electrostatic_embedding_charge_shifting.py):
that version moved the M1 boundary atom's charge onto its M2 neighbor's
OWN position. The real RC scheme redistributes it to the MIDPOINT of the
M1-M2 bond instead -- a concretely different, real difference confirmed
via two independent web searches of the actual paper's description
(full RCD's additional bond-dipole-preserving charge tuning is NOT
implemented here -- the paper is paywalled, no formula available -- this
tests the RC part only, honestly labeled as such).

Molecule: 5-amino-1-pentanol (OCCCCCN), same as every other embedding
attempt this session, for direct comparability.
"""
import subprocess
import sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "rdkit", "basis_set_exchange", "pyscf",
                "git+https://github.com/tatopenn-cell/Dense-Evolution.git@main"],
               check=True)

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from dense_evolution.native_hf.libcint_bridge import (
    build_repulsion_tensor_libcint, build_overlap_and_core_hamiltonian_libcint,
    _libcint_mol_and_perm_rescale,
)
from dense_evolution.native_hf.scf import run_scf

ANGSTROM_TO_BOHR = 1.8897259886
CH_BOND_BOHR = 1.09 * ANGSTROM_TO_BOHR
BASIS = "sto-3g"


def embed_whole(smiles):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = 2026
    AllChem.EmbedMolecule(mol, params)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def whole_geometry_bohr(mol):
    conf = mol.GetConformer()
    return np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())]) * ANGSTROM_TO_BOHR


def sliced_geometry(atomic_numbers, geom_bohr, keep_idx, boundary_pairs):
    keep_idx = sorted(keep_idx)
    new_numbers = [atomic_numbers[i] for i in keep_idx]
    new_geom = [geom_bohr[i] for i in keep_idx]
    for kept, cut in boundary_pairs:
        vec = geom_bohr[cut] - geom_bohr[kept]
        vec = vec / np.linalg.norm(vec)
        new_geom.append(geom_bohr[kept] + vec * CH_BOND_BOHR)
        new_numbers.append(1)
    return new_numbers, np.array(new_geom)


def external_point_charge_potential(atomic_numbers, geometry_bohr, basis_name, mm_charges, mm_positions_bohr):
    mol, perm, rescale = _libcint_mol_and_perm_rescale(atomic_numbers, geometry_bohr, basis_name)
    n = mol.nao
    V = np.zeros((n, n))
    for q, R in zip(mm_charges, mm_positions_bohr):
        mol.set_rinv_origin(tuple(R))
        V += -q * mol.intor("int1e_rinv")
    V = V * rescale[:, None] * rescale[None, :]
    return V[perm][:, perm]


def mm_nuclear_interaction(atomic_numbers, geometry_bohr, mm_charges, mm_positions_bohr):
    e = 0.0
    for z, r in zip(atomic_numbers, geometry_bohr):
        for q, R in zip(mm_charges, mm_positions_bohr):
            e += z * q / np.linalg.norm(r - R)
    return e


def region_hf_energy(atomic_numbers, geometry_bohr, label, embed=None):
    n_electrons = sum(atomic_numbers)
    if n_electrons % 2 != 0:
        return None
    S, H_core = build_overlap_and_core_hamiltonian_libcint(atomic_numbers, geometry_bohr, BASIS)
    if embed is not None:
        mm_charges, mm_positions = embed
        H_core = H_core + external_point_charge_potential(atomic_numbers, geometry_bohr, BASIS, mm_charges, mm_positions)
    repulsion = build_repulsion_tensor_libcint(atomic_numbers, geometry_bohr, BASIS)
    result = run_scf(S, H_core, repulsion, n_electrons, [float(z) for z in atomic_numbers], geometry_bohr)
    print(f"    [hf:{label}] n_atoms={len(atomic_numbers)} converged={result.converged}")
    if not result.converged:
        return None
    e = result.total_energy
    if embed is not None:
        e += mm_nuclear_interaction(atomic_numbers, geometry_bohr, embed[0], embed[1])
    return e


def h2_reference_energy():
    return region_hf_energy([1, 1], np.array([[0, 0, 0.0], [0, 0, 0.74 * ANGSTROM_TO_BOHR]]), "H2")


def rc_midpoint_charges(mm_atoms, mm_charges_raw, mm_positions, outer_boundary, whole_mol, geom_whole):
    """Real RC scheme (Lin & Truhlar 2005): the M1 atom's charge is moved
    to the MIDPOINT of each M1-M2 bond, not onto M2's own position (that
    was the earlier, already-failed charge_shifted() attempt)."""
    charge = dict(zip(mm_atoms, mm_charges_raw))
    pos = dict(zip(mm_atoms, mm_positions))
    extra_charges = []
    extra_positions = []
    for _kept, m1 in outer_boundary:
        q = charge[m1]
        m2_ids = [n.GetIdx() for n in whole_mol.GetAtomWithIdx(m1).GetNeighbors()
                  if n.GetIdx() in charge and n.GetIdx() != m1]
        charge[m1] = 0.0
        if m2_ids:
            share = q / len(m2_ids)
            for m2 in m2_ids:
                midpoint = 0.5 * (geom_whole[m1] + geom_whole[m2])
                extra_charges.append(share)
                extra_positions.append(midpoint)
    final_charges = [charge[i] for i in mm_atoms] + extra_charges
    final_positions = list(mm_positions) + extra_positions
    return final_charges, np.array(final_positions)


if __name__ == "__main__":
    whole_smiles = "OCCCCCN"
    mol_noH = Chem.MolFromSmiles(whole_smiles)
    o_idx = [a.GetIdx() for a in mol_noH.GetAtoms() if a.GetSymbol() == "O"][0]
    c_idx = mol_noH.GetAtomWithIdx(o_idx).GetNeighbors()[0].GetIdx()

    whole_mol = embed_whole(whole_smiles)
    atomic_numbers_whole = [a.GetAtomicNum() for a in whole_mol.GetAtoms()]
    geom_whole = whole_geometry_bohr(whole_mol)
    props = AllChem.MMFFGetMoleculeProperties(whole_mol)
    mmff_charges_whole = [props.GetMMFFPartialCharge(i) for i in range(whole_mol.GetNumAtoms())]

    e_h2 = h2_reference_energy()
    print(f"E(H2) = {e_h2:.6f} Hartree\n")

    def side_atoms(seed, blocked):
        seen = {seed}
        frontier = {seed}
        while frontier:
            new_frontier = set()
            for idx in frontier:
                for nbr in whole_mol.GetAtomWithIdx(idx).GetNeighbors():
                    if nbr.GetIdx() != blocked and nbr.GetIdx() not in seen:
                        new_frontier.add(nbr.GetIdx())
            seen |= new_frontier
            frontier = new_frontier
        return seen

    o_side = side_atoms(o_idx, c_idx)
    c_side = side_atoms(c_idx, o_idx)

    def qm_region(radius):
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
        raw_boundary = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in whole_mol.GetBonds()
                        if (b.GetBeginAtomIdx() in qm_atoms) != (b.GetEndAtomIdx() in qm_atoms)]
        boundary_pairs = [(i, j) if i in qm_atoms else (j, i) for i, j in raw_boundary]
        return qm_atoms, boundary_pairs

    print("=== Reference: whole-molecule O-C isodesmic scission ===")
    e_whole = region_hf_energy(atomic_numbers_whole, geom_whole, label="whole-ref:whole")
    atoms_o, geom_o = sliced_geometry(atomic_numbers_whole, geom_whole, o_side, [(o_idx, c_idx)])
    e_o = region_hf_energy(atoms_o, geom_o, label="whole-ref:frag_O")
    atoms_c, geom_c = sliced_geometry(atomic_numbers_whole, geom_whole, c_side, [(c_idx, o_idx)])
    e_c = region_hf_energy(atoms_c, geom_c, label="whole-ref:frag_C")
    e_whole_ref = None
    if None not in (e_whole, e_o, e_c, e_h2):
        e_whole_ref = (e_o + e_c - e_whole - e_h2) * 627.5
        print(f"[RESULT whole] {e_whole_ref:.2f} kcal/mol\n")

    for radius in (1, 2, 3):
        print(f"=== QM region radius={radius}: plain vs point-embedded vs RC-midpoint ===")
        qm_atoms, outer_boundary = qm_region(radius)
        mm_atoms = sorted(i for i in range(len(atomic_numbers_whole)) if i not in qm_atoms)
        mm_charges = [mmff_charges_whole[i] for i in mm_atoms]
        mm_positions = geom_whole[mm_atoms]
        embed = (mm_charges, mm_positions)
        rc_charges, rc_positions = rc_midpoint_charges(mm_atoms, mm_charges, mm_positions, outer_boundary, whole_mol, geom_whole)
        embed_rc = (rc_charges, rc_positions)
        print(f"  mm_charges={[round(q, 5) for q in mm_charges]}")

        qm_o = o_side & qm_atoms
        qm_c = c_side & qm_atoms
        atoms_reg, geom_reg = sliced_geometry(atomic_numbers_whole, geom_whole, qm_atoms, outer_boundary)
        atoms_of, geom_of = sliced_geometry(atomic_numbers_whole, geom_whole, qm_o, [(o_idx, c_idx)])
        atoms_cf, geom_cf = sliced_geometry(atomic_numbers_whole, geom_whole, qm_c, outer_boundary + [(c_idx, o_idx)])

        for mode, use_embed in (("plain", None), ("point_embedded", embed), ("rc_midpoint", embed_rc)):
            e_reg = region_hf_energy(atoms_reg, geom_reg, label=f"r{radius}:{mode}:region", embed=use_embed)
            e_of = region_hf_energy(atoms_of, geom_of, label=f"r{radius}:{mode}:o_frag", embed=use_embed)
            e_cf = region_hf_energy(atoms_cf, geom_cf, label=f"r{radius}:{mode}:c_frag", embed=use_embed)
            if None in (e_reg, e_of, e_cf, e_h2):
                print(f"[RESULT r={radius} {mode}] failed\n")
                continue
            e_rxn_kcal = (e_of + e_cf - e_reg - e_h2) * 627.5
            line = f"[RESULT r={radius} {mode}] {e_rxn_kcal:.2f} kcal/mol"
            if e_whole_ref is not None:
                line += f" (diff={e_rxn_kcal - e_whole_ref:.2f})"
            print(line)
        print()
