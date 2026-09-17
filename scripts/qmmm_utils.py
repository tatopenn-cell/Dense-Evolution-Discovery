"""
Reusable QM/MM region-partitioning utilities for Dense-Evolution issue
#283 -- consolidates the two real bugs already found and fixed while
building steps 1-4, so new experiment scripts don't duplicate this logic
(or its bugs) from scratch.

1. Hydrogens must always follow their own heavy atom, never be
   independently BFS-expanded -- doing so spuriously cuts terminal C-H
   bonds and leaves an empty MM fragment (found on 1-hexanol).

2. A boundary bond that would cut INTO an aromatic ring must instead pull
   the WHOLE ring into the QM region -- otherwise RDKit's FragmentOnBonds
   leaves one aromatic atom outside its ring, capped with H, which is not
   a valid molecule (`AtomKekulizeException: non-ring atom marked
   aromatic`, found on OCC(c1ccccc1)CCC at radius=2).

Deliberately NOT included here, both real negative results with the full
record in docs/qmmm_bond_order_and_embedding.md:

- The MMFF94 ONIOM-style correction (`dE_MMFF94(whole) - dE_MMFF94(QM-
  region)`) originally looked like it halved 1-hexanol's error, but that
  used a cruder geometry (each fragment independently re-embedded by
  RDKit instead of sliced from one shared conformer). Once corrected to
  use the same shared-conformer geometry as everything else here, plain
  truncation is already accurate to <0.2 kcal/mol on every radius tested
  on BOTH 1-hexanol and a branched-aromatic molecule -- and applying the
  MMFF94 correction on top makes it WORSE in every single case (up to
  -1.5 kcal/mol off). The correction was compensating for a geometry
  artifact of the OLD embedding choice, not for truncation itself; it
  does not survive the more careful geometry treatment, so it is not
  included here at all.

- Electrostatic embedding (issue #283 point 5): five different
  treatments were tried (plain point charge, charge-shifting,
  Gaussian-smeared over the whole MM region, Gaussian-smeared on just the
  M1 boundary atom, M1 charge deletion) and every one made the isodesmic-
  energy error worse than no embedding at all, on the one molecule with a
  real MM charge to embed (5-amino-1-pentanol). The sign convention was
  independently verified correct (a minimal He-atom test: E(+1 charge
  nearby) < E(isolated) < E(-1 charge nearby), exactly as physics
  requires) -- the failure is not a bug in this codebase, just genuinely
  unsolved. Plain truncation remains the right default.
"""
import numpy as np
from rdkit import Chem

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


def propagate_relevance(affinity, seed_idx, n_nodes, p=1.6, lam=1e-5, tau_prop=1e-4, max_iter=500):
    """Algorithm 1 of Hummer, Sicking, Huger & Gottschalk 2026
    (arXiv:2609.06491, "Diffuse2Seg", Sec. 3.4/A.1), read directly from
    the paper (provided by the user, Desktop/QAI/2609.06491v1.pdf) and
    implemented verbatim -- not the max-product-with-decay heuristic used
    in this repo's earlier bond-order-partitioning scripts
    (qmmm_bond_order_partition_vs_radius.py,
    qmmm_bond_order_aromatic_vs_alkyl_branch.py), which was written
    without having read the paper first and does not match it.

    Non-linear p-Laplacian graph-regularized smoothing (Elmoataz et al.
    2008), solved by Gauss-Jacobi iteration: propagates a one-hot seed
    vector over any node-affinity graph `affinity` (self-attention in the
    original paper; Mayer/Wiberg bond order here) into a soft relevance
    map that stays smooth within high-affinity regions and is throttled
    across low-affinity (edge) ones. `p`, `lam`, `tau_prop` are the
    paper's own final values (Sec. 4.2) -- NOT re-tuned here, since doing
    so without a reason grounded in this different (small, single-seed,
    molecular) setting would repeat the same mistake this function was
    written to fix. Known real limitation, not yet resolved: with these
    exact values, a single isolated seed (no dense grid of prompts to
    later merge, unlike the paper's own use of this algorithm) can
    converge to a near-uniform relevance map at the paper's own lam=1e-5
    (confirmed on both the solar-filament image test in
    solar_diffuse2seg_kernel/script.py and on a real molecular bond-order
    graph) -- because lam is tuned to be a weak anchor for a many-prompt,
    later-merged pipeline, not a single isolated seed. Not "fixed" by
    retuning lam here (that would just be searching for a number that
    looks good); left as a real, open, measured limitation -- see the
    lam sweep in qmmm_diffuse2seg_propagation_lambda_sweep.py.

    Degenerate-case handling: at any node i where g_i = sqrt(sum_j
    A_ij(f_j-f_i)^2) is exactly 0 (every affinity-neighbor already equals
    f_i), the formula's g_i^(p-2) term diverges for p<2. An earlier
    version clamped g_i away from 0 with an epsilon -- the same category
    of shortcut already rejected elsewhere in this project for degenerate
    eigenvalues (see dense_evolution.physics.spectral, Kato's divided-
    difference formula) in favor of the real mathematical limit. Worked
    out directly here: as g_i -> 0, every A_ij-connected f_j equals f_i
    by definition of g_i=0, so the g_i^(p-2)-weighted terms in both the
    numerator and denominator of the update come to dominate and cancel
    to exactly f_i -- i.e. a node already consistent with its whole
    affinity-neighborhood is unchanged by an edge-preserving smoothing
    step, exactly as expected. Implemented as an explicit special case
    below, not an epsilon."""
    f0 = np.zeros(n_nodes)
    f0[seed_idx] = 1.0
    f = f0.copy()
    for _ in range(max_iter):
        diff = f[None, :] - f[:, None]
        g = np.sqrt(np.sum(affinity * diff ** 2, axis=1))
        degenerate = g == 0.0
        gp = np.zeros_like(g)
        gp[~degenerate] = g[~degenerate] ** (p - 2)
        gamma = affinity * (gp[:, None] + gp[None, :])
        numerator = lam * f0 + (gamma * f[None, :]).sum(axis=1)
        denominator = lam + gamma.sum(axis=1)
        f_new = np.where(degenerate, f, numerator / denominator)
        if np.sum((f_new - f) ** 2) <= tau_prop:
            f = f_new
            break
        f = f_new
    return f


def sliced_geometry(atomic_numbers, geom_bohr, keep_idx, boundary_pairs):
    """A coordinate SUBSET of one whole-molecule conformer (never an
    independently re-embedded fragment -- that would give unrelated 3D
    structures across fragments, and was the real cause of the MMFF94
    correction's apparent benefit turning out to be a geometry artifact,
    see module docstring). Boundary bonds get a capping H placed along
    the kept->cut bond direction at a standard C-H bond length."""
    keep_idx = sorted(keep_idx)
    new_numbers = [atomic_numbers[i] for i in keep_idx]
    new_geom = [geom_bohr[i] for i in keep_idx]
    for kept, cut in boundary_pairs:
        vec = geom_bohr[cut] - geom_bohr[kept]
        vec = vec / np.linalg.norm(vec)
        new_geom.append(geom_bohr[kept] + vec * CH_BOND_BOHR)
        new_numbers.append(1)
    return new_numbers, np.array(new_geom)
