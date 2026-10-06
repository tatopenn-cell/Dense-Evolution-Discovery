"""
QM/MM electrostatic embedding (step 5), for Dense-Evolution issue #283 --
second attempt, on a molecule chosen so the truncated MM region actually
has real charge.

First attempt (1-hexanol, OCCCCCC): plain vs embedded gave IDENTICAL
numbers at every radius. Verified locally (no bug): MMFF94 assigns real
charge only to the polar O-C1-H(O) group, which is always inside the QM
region by construction (O/C1 are the reactive-bond seed atoms) -- the
truncated-away alkyl tail (C2 onward) gets exactly zero MMFF94 charge, so
embedding correctly has nothing to add there. That result didn't test
whether the embedding MECHANISM works, only that this particular molecule
had no polar MM region to embed.

This kernel uses 5-amino-1-pentanol (OCCCCCN) instead: a SECOND polar
group (NH2) far from the O-C1 reactive bond, so a small QM radius
truncates it into the MM region with a real, nonzero MMFF94 charge --
now there is something for electrostatic embedding to actually do.

Second attempt result (real MM charges now, e.g. N=-0.99): plain vs
embedded now DIFFER, but embedding made things WORSE at every radius
(diff 0.11/0.08/0.02 kcal/mol plain vs 0.27/0.26/0.09 embedded) --
plausibly the textbook QM/MM near-boundary overpolarization artifact: the
MM point charge immediately across the cut bond sits too close to the QM
density and distorts it unphysically. Real QM/MM codes fix this with
charge-shifting (redistributing that one boundary atom's charge onto its
own remaining MM neighbors instead of embedding it directly) -- added
below as a third mode (`charge_shifted`) to test whether that actually
closes the gap instead of just diagnosing it.

RESULT: this naive charge-shifting does NOT fix it, and makes it slightly
worse where it activates at all. At radius 1-2 the boundary MM atom
already had ~0 charge, so shifting changed nothing. At radius 3, where a
real 0.27 charge got redistributed, the corrected difference (0.21
kcal/mol) is worse than both plain embedding (0.09) and plain truncation
(0.02). Likely cause: the redistribution target (the boundary atom's own
remaining neighbors, here its own hydrogens) can be just as close -- or
closer -- to the QM/MM boundary as the atom being zeroed, so the
overpolarization artifact moves rather than resolves. A correct fix needs
a more careful choice of redistribution target (real QM/MM
implementations use more elaborate schemes than this first attempt) --
left open, not solved here.

Mechanism (unchanged from the first attempt): MM point charges (MMFF94
partial charges) enter the QM Hamiltonian directly as an external
potential via PySCF's int1e_rinv integral (same libcint bridge already
used for S/H_core/ERI, reusing its private _libcint_mol_and_perm_rescale
helper so the point-charge matrix lands in the identical AO order/
normalization as H_core).

Geometric fix versus the earlier (steps 1-4) prototype: that one
independently re-embedded a fresh RDKit conformer for each fragment/
QM-region SMILES -- each piece ends up with an unrelated 3D structure,
which is fine for a mechanical MMFF94 correction (only needs consistent
topology) but wrong for electrostatic embedding, which needs real
relative QM-atom/MM-charge distances. Fixed here: every piece is a
coordinate SUBSET of ONE whole-molecule conformer, embedded once; a
boundary link-H is placed by scaling the cut bond vector to a standard
C-H bond length, never by re-embedding independently. Because of this,
the "plain" (uncorrected) numbers in this kernel are a fresh baseline
under this more consistent geometry treatment, not a reprint of the
steps-1-4 numbers -- the embedded-vs-plain comparison within this kernel
is what matters, not a cross-kernel number match.

Reactive bond: O-C1 in 5-amino-1-pentanol (OCCCCCN). For a given QM radius,
three single-point HF energies are needed, all embedded in the SAME
fixed MM point-charge field (all whole-molecule atoms outside the QM
region, at their real positions, with their real MMFF94 partial
charges):
  - qm_region_whole:  the QM region intact (pre-break)
  - qm_region_o_frag: just the O side after breaking O-C1 (usually just
                       O+H, since O has no other heavy neighbor here)
  - qm_region_c_frag: the C-chain side after breaking O-C1, still capped
                       at the outer QM/MM truncation boundary too
isodesmic energy = E(o_frag) + E(c_frag) - E(qm_region_whole) - E(H2)
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
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def whole_geometry_bohr(mol):
    conf = mol.GetConformer()
    pos = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    return pos * ANGSTROM_TO_BOHR


def sliced_geometry(atomic_numbers, geom_bohr, keep_idx, boundary_pairs):
    """keep_idx: atom indices (into the whole molecule) to keep as-is,
    real coordinates. boundary_pairs: list of (kept_idx, cut_idx) -- for
    each, a new capping H is placed along the kept->cut bond direction at
    a standard C-H bond length, replacing the cut atom (and whatever was
    beyond it)."""
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
        print(f"    [hf:{label}] odd electron count ({n_electrons}) -- skipping")
        return None
    nuclear_charges = [float(z) for z in atomic_numbers]
    S, H_core = build_overlap_and_core_hamiltonian_libcint(atomic_numbers, geometry_bohr, BASIS)
    if embed is not None:
        mm_charges, mm_positions = embed
        H_core = H_core + external_point_charge_potential(atomic_numbers, geometry_bohr, BASIS, mm_charges, mm_positions)
    repulsion = build_repulsion_tensor_libcint(atomic_numbers, geometry_bohr, BASIS)
    result = run_scf(S, H_core, repulsion, n_electrons, nuclear_charges, geometry_bohr)
    print(f"    [hf:{label}] n_atoms={len(atomic_numbers)} converged={result.converged} "
          f"total_energy={result.total_energy:.6f} Hartree")
    if not result.converged:
        return None
    e = result.total_energy
    if embed is not None:
        mm_charges, mm_positions = embed
        e += mm_nuclear_interaction(atomic_numbers, geometry_bohr, mm_charges, mm_positions)
    return e


def h2_reference_energy():
    atomic_numbers = [1, 1]
    geometry_bohr = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74 * ANGSTROM_TO_BOHR]])
    return region_hf_energy(atomic_numbers, geometry_bohr, label="H2")


if __name__ == "__main__":
    # 5-amino-1-pentanol: a SECOND polar group (NH2) far from the O-C1
    # reactive bond, so a small QM radius truncates it into the MM region
    # with a real, nonzero MMFF94 charge -- unlike hexanol's pure alkyl
    # tail, this actually tests whether electrostatic embedding matters
    # when there IS real polarity to embed.
    whole_smiles = "OCCCCCN"
    mol_noH = Chem.MolFromSmiles(whole_smiles)
    o_idx = [a.GetIdx() for a in mol_noH.GetAtoms() if a.GetSymbol() == "O"][0]
    c_idx = mol_noH.GetAtomWithIdx(o_idx).GetNeighbors()[0].GetIdx()

    whole_mol = embed_whole(whole_smiles)
    atomic_numbers_whole = [a.GetAtomicNum() for a in whole_mol.GetAtoms()]
    geom_whole = whole_geometry_bohr(whole_mol)
    print(f"Molecule: {whole_smiles}, reactive bond O(idx={o_idx})-C(idx={c_idx}), "
          f"{len(atomic_numbers_whole)} atoms total\n")

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

    print("=== Reference: whole-molecule O-C isodesmic scission (shared conformer) ===")
    e_whole = region_hf_energy(atomic_numbers_whole, geom_whole, label="whole-ref:whole")
    atoms_o, geom_o = sliced_geometry(atomic_numbers_whole, geom_whole, o_side, [(o_idx, c_idx)])
    e_o = region_hf_energy(atoms_o, geom_o, label="whole-ref:frag_O")
    atoms_c, geom_c = sliced_geometry(atomic_numbers_whole, geom_whole, c_side, [(c_idx, o_idx)])
    e_c = region_hf_energy(atoms_c, geom_c, label="whole-ref:frag_C")
    e_whole_ref = None
    if None not in (e_whole, e_o, e_c, e_h2):
        e_whole_ref = e_o + e_c - e_whole - e_h2
        print(f"[RESULT whole] {e_whole_ref * 627.5:.2f} kcal/mol\n")

    def charge_shifted(mm_atoms, mm_charges_raw, outer_boundary, whole_mol):
        """Standard QM/MM 'redistributed charge' fix for the near-boundary
        overpolarization artifact: the MM atom immediately across the cut
        bond has its charge zeroed for embedding purposes, redistributed
        equally onto ITS OWN remaining MM neighbors (never onto the QM
        region), preserving total charge while removing the too-close
        point charge that was distorting the QM density."""
        charge = dict(zip(mm_atoms, mm_charges_raw))
        for _kept, cut in outer_boundary:
            q = charge[cut]
            neighbor_ids = [n.GetIdx() for n in whole_mol.GetAtomWithIdx(cut).GetNeighbors()
                             if n.GetIdx() in charge and n.GetIdx() != cut]
            charge[cut] = 0.0
            if neighbor_ids:
                share = q / len(neighbor_ids)
                for nid in neighbor_ids:
                    charge[nid] += share
        return [charge[i] for i in mm_atoms]

    for radius in (1, 2, 3):
        print(f"=== QM region radius={radius}: plain vs embedded vs charge-shifted ===")
        qm_atoms, outer_boundary = qm_region(radius)
        mm_atoms = sorted(i for i in range(len(atomic_numbers_whole)) if i not in qm_atoms)
        mm_charges = [mmff_charges_whole[i] for i in mm_atoms]
        mm_positions = geom_whole[mm_atoms]
        embed = (mm_charges, mm_positions)
        mm_charges_shifted = charge_shifted(mm_atoms, mm_charges, outer_boundary, whole_mol)
        embed_shifted = (mm_charges_shifted, mm_positions)
        print(f"  mm_atoms={mm_atoms}, mm_charges={[round(q, 5) for q in mm_charges]}")
        print(f"  mm_charges_shifted={[round(q, 5) for q in mm_charges_shifted]}")

        qm_o = o_side & qm_atoms
        qm_c = c_side & qm_atoms

        atoms_reg, geom_reg = sliced_geometry(atomic_numbers_whole, geom_whole, qm_atoms, outer_boundary)
        atoms_of, geom_of = sliced_geometry(atomic_numbers_whole, geom_whole, qm_o, [(o_idx, c_idx)])
        atoms_cf, geom_cf = sliced_geometry(atomic_numbers_whole, geom_whole, qm_c,
                                             outer_boundary + [(c_idx, o_idx)])

        for mode, use_embed in (("plain", None), ("embedded", embed), ("charge_shifted", embed_shifted)):
            e_reg = region_hf_energy(atoms_reg, geom_reg, label=f"r{radius}:{mode}:region", embed=use_embed)
            e_of = region_hf_energy(atoms_of, geom_of, label=f"r{radius}:{mode}:o_frag", embed=use_embed)
            e_cf = region_hf_energy(atoms_cf, geom_cf, label=f"r{radius}:{mode}:c_frag", embed=use_embed)
            if None in (e_reg, e_of, e_cf, e_h2):
                print(f"[RESULT r={radius} {mode}] failed\n")
                continue
            e_rxn = e_of + e_cf - e_reg - e_h2
            e_rxn_kcal = e_rxn * 627.5
            line = f"[RESULT r={radius} {mode}] {e_rxn_kcal:.2f} kcal/mol"
            if e_whole_ref is not None:
                line += f" (whole-ref = {e_whole_ref * 627.5:.2f}, diff = {e_rxn_kcal - e_whole_ref * 627.5:.2f} kcal/mol)"
            print(line)
        print()
