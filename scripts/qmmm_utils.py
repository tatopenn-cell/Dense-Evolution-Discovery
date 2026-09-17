"""
Reusable QM/MM region-partitioning utilities for Dense-Evolution issue
#283, consolidating everything steps 1-4/5 actually validated into one
importable module instead of duplicating this logic in every new
experiment script.

Two real bugs already found and fixed live here permanently:

1. Hydrogens must always follow their own heavy atom, never be
   independently BFS-expanded -- doing so spuriously cuts terminal C-H
   bonds and leaves an empty MM fragment (found on hexanol).

2. A boundary bond that would cut INTO an aromatic ring must instead pull
   the WHOLE ring into the QM region -- otherwise RDKit's FragmentOnBonds
   leaves one aromatic atom outside its ring, capped with H, which is not
   a valid molecule (`AtomKekulizeException: non-ring atom marked
   aromatic`, found on OCC(c1ccccc1)CCC at radius=2).

A third, more subtle failure is guarded against rather than silently
"fixed", since there is no correct number to substitute: the MMFF94
ONIOM-style correction (`dE_MMFF94(whole) - dE_MMFF94(QM-region)`) is
only meaningful when the whole molecule and the small QM-region-sized
fragment have the SAME aromatic ring content. On OCC(c1ccccc1)CCC at
radius=1, the whole molecule has one aromatic ring and the small fragment
has none -- subtracting their MMFF94 energies compares a resonance-
stabilized system against one that never had that resonance to begin
with, since MMFF94's classical parameters cannot represent quantum
resonance stabilization energy. Measured effect: a 0.68 kcal/mol
uncorrected error became -7.05 kcal/mol corrected -- worse than doing
nothing. At radius>=2 (ring included on both sides) the same correction
behaved exactly like it did on 1-hexanol (small, genuinely halves the
error). `mmff94_correction` below refuses the correction (returns
`applied=False`) whenever the aromatic ring counts on the two sides
differ, instead of silently returning a number known to sometimes be
catastrophically wrong.

Electrostatic embedding (issue #283 point 5) is deliberately NOT included
here: five different treatments were tried (plain point charge,
charge-shifting, Gaussian-smeared over the whole MM region, Gaussian-
smeared on just the M1 boundary atom, M1 charge deletion) and every one
made the isodesmic-energy error worse than no embedding at all, on the
one molecule with a real MM charge to embed (5-amino-1-pentanol). The
sign convention was independently verified correct (a minimal He-atom
test: E(+1 charge nearby) < E(isolated) < E(-1 charge nearby), exactly as
physics requires) -- the failure is not a bug in this codebase. See
docs/qmmm_bond_order_and_embedding.md for the full record. Plain
truncation (no embedding) remains the right default until this is
understood.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

ANGSTROM_TO_BOHR = 1.8897259886
CH_BOND_BOHR = 1.09 * ANGSTROM_TO_BOHR


def partition_qm_mm_region(mol, seed_heavy_atoms, radius):
    """BFS outward from `seed_heavy_atoms` (a set/list of atom indices)
    across `radius` heavy-atom hops. Hydrogens always follow their own
    heavy atom afterward (never independently expanded -- see module
    docstring). Any boundary bond that would cut into an aromatic ring
    pulls that whole ring into the QM region first, iterating until
    stable, before hydrogens are added and the boundary is finalized.

    Returns (qm_atoms: set[int], boundary_pairs: list[(kept_idx, cut_idx)]),
    one pair per bond crossing the QM/MM boundary, `kept_idx` on the QM
    side.
    """
    qm_heavy = set(seed_heavy_atoms)
    frontier = set(seed_heavy_atoms)
    for _ in range(radius):
        new_frontier = set()
        for idx in frontier:
            for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
                if nbr.GetAtomicNum() > 1 and nbr.GetIdx() not in qm_heavy:
                    new_frontier.add(nbr.GetIdx())
        qm_heavy |= new_frontier
        frontier = new_frontier

    ring_info = mol.GetRingInfo()
    changed = True
    while changed:
        changed = False
        for bond in mol.GetBonds():
            if not bond.GetIsAromatic():
                continue
            a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if (a in qm_heavy) != (b in qm_heavy):
                for ring in ring_info.AtomRings():
                    if a in ring or b in ring:
                        newly = set(ring) - qm_heavy
                        if newly:
                            qm_heavy |= newly
                            changed = True

    qm_atoms = set(qm_heavy)
    for idx in qm_heavy:
        for nbr in mol.GetAtomWithIdx(idx).GetNeighbors():
            if nbr.GetAtomicNum() == 1:
                qm_atoms.add(nbr.GetIdx())

    raw_boundary = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()
                    if (b.GetBeginAtomIdx() in qm_atoms) != (b.GetEndAtomIdx() in qm_atoms)]
    boundary_pairs = [(i, j) if i in qm_atoms else (j, i) for i, j in raw_boundary]
    return qm_atoms, boundary_pairs


def sliced_geometry(atomic_numbers, geom_bohr, keep_idx, boundary_pairs):
    """A coordinate SUBSET of one whole-molecule conformer (never an
    independently re-embedded fragment -- that would give unrelated 3D
    structures across fragments, wrong for anything electrostatic).
    Boundary bonds get a capping H placed along the kept->cut bond
    direction at a standard C-H bond length."""
    keep_idx = sorted(keep_idx)
    new_numbers = [atomic_numbers[i] for i in keep_idx]
    new_geom = [geom_bohr[i] for i in keep_idx]
    for kept, cut in boundary_pairs:
        vec = geom_bohr[cut] - geom_bohr[kept]
        vec = vec / np.linalg.norm(vec)
        new_geom.append(geom_bohr[kept] + vec * CH_BOND_BOHR)
        new_numbers.append(1)
    return new_numbers, np.array(new_geom)


def _region_mol_capped(mol, qm_atoms, boundary_pairs):
    """The QM region as its own RDKit molecule, boundary bonds capped
    with H, for MMFF94 energy evaluation (a fresh embedding is fine here
    -- MMFF94 correction only needs consistent topology, not real
    relative QM/MM distances, unlike electrostatic embedding)."""
    em = Chem.RWMol(mol)
    for kept, cut in boundary_pairs:
        for bond in list(em.GetBonds()):
            if {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()} == {kept, cut}:
                em.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                cap = em.AddAtom(Chem.Atom(1))
                em.AddBond(kept, cap, Chem.BondType.SINGLE)
    keep = set(qm_atoms) | {em.GetNumAtoms() - 1 - i for i in range(len(boundary_pairs))}
    frag = Chem.RWMol(em)
    remove = sorted((i for i in range(frag.GetNumAtoms()) if i not in keep), reverse=True)
    for i in remove:
        frag.RemoveAtom(i)
    frag = frag.GetMol()
    Chem.SanitizeMol(frag)
    frag = Chem.AddHs(frag)
    params = AllChem.ETKDGv3()
    params.randomSeed = 2026
    AllChem.EmbedMolecule(frag, params)
    AllChem.MMFFOptimizeMolecule(frag)
    return frag


def _aromatic_ring_count(mol):
    ring_info = mol.GetRingInfo()
    return sum(1 for ring in ring_info.AtomRings()
               if all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring))


def mmff94_correction(whole_mol, qm_atoms, boundary_pairs, e_qm_kcal, mmff_delta_whole_kcal):
    """ONIOM-style correction `dE_MMFF94(whole) - dE_MMFF94(QM-region)`,
    added to `e_qm_kcal`, GUARDED against the aromatic-ring mismatch
    failure described in the module docstring: refuses to apply
    (`applied=False`) when the whole molecule and the QM-region fragment
    don't have the same number of aromatic rings, since that comparison
    is not physically meaningful (MMFF94 cannot represent the quantum
    resonance energy a ring difference implies).

    `mmff_delta_whole_kcal`: the whole-molecule MMFF94 reaction delta,
    computed once by the caller (same for every radius on a given
    molecule) -- not recomputed here to avoid re-doing that work per call.
    """
    ring_count_whole = _aromatic_ring_count(whole_mol)
    region_mol = _region_mol_capped(whole_mol, qm_atoms, boundary_pairs)
    ring_count_region = _aromatic_ring_count(region_mol)

    if ring_count_whole != ring_count_region:
        return {
            "correction_kcal": 0.0,
            "corrected_kcal": e_qm_kcal,
            "applied": False,
            "reason": (f"aromatic ring count mismatch (whole={ring_count_whole}, "
                       f"region={ring_count_region}) -- MMFF94 correction not trusted "
                       f"across a ring boundary"),
        }

    props_region = AllChem.MMFFGetMoleculeProperties(region_mol)
    e_region = AllChem.MMFFGetMoleculeForceField(region_mol, props_region).CalcEnergy()
    props_whole = AllChem.MMFFGetMoleculeProperties(whole_mol)
    e_whole = AllChem.MMFFGetMoleculeForceField(whole_mol, props_whole).CalcEnergy()
    mmff_delta_region = e_region - e_whole
    correction = mmff_delta_whole_kcal - mmff_delta_region
    return {
        "correction_kcal": correction,
        "corrected_kcal": e_qm_kcal + correction,
        "applied": True,
        "reason": None,
    }
