"""Isodesmic bond-scission energy via dense_evolution.native_hf, tested as
a candidate-discriminating signal for the Kaggle CASMI26 molecule-ID
competition (enveda-CASMI26-molecule-id-mass-spectra) -- the chemistry-
informed feature neither MSAlign (arXiv:2605.19752, quantumrag) nor the
public PyTorch reference implementation (abrilrisso/MassSpecGym-Molecular-
Identification-from-MS-MS-Spectra) use, since both work purely from raw
spectral peaks.

native_hf.scf.run_scf is RHF-only (raises on an odd electron count -- no
radical/open-shell support). Rather than needing UHF, this uses the
standard H-capping scheme for bond scission: break one acyclic single bond,
cap BOTH resulting fragments with an H atom at the cut site (RDKit
FragmentOnBonds + direct dummy-atom-to-H atomic-number patch, not string
SMILES substitution -- RDKit's dummy atoms carry isotope labels like [1*],
so a naive "[*]" -> "[H]" string replace silently fails to match and
leaves an atomic-number-0 dummy in the fragment). This models heterolytic
cleavage with proton transfer -- the dominant CID mechanism in
positive-mode ESI-MS/MS (the "even-electron rule"), which is exactly this
competition's adduct ([M+H]+) -- so both fragments stay closed-shell,
matching what native_hf can actually compute.

NOT a real bond dissociation energy, despite reading like one at first:
whole_molecule + H2 -> fragment_A_H + fragment_B_H is an ISODESMIC
reaction (bond types are conserved/redistributed on both sides), not a
bare homolytic bond-breaking. First found on ethanol at HF/STO-3G: this
scheme gives NEGATIVE values for the C-C and C-O bonds (-16.7 and -14.3
kcal/mol) -- checked against real experimental heats of formation for the
corresponding reaction (ethanol -> methane + methanol), which is itself
mildly exothermic (~-9.6 kcal/mol) in reality -- so the negative sign is a
real property of this reaction scheme, not a bug. Named accordingly.

RESULT (verified on a real Kaggle CPU kernel, not assumed): tested whether
this signal discriminates between 3 real C3H8O isomers (1-propanol,
2-propanol, methyl ethyl ether) -- exactly the CASMI26 retrieval scenario,
candidates tied on molecular formula/precursor mass, needing a signal to
tell structures apart.
  - HF/STO-3G: sum_isodesmic spread across the 3 isomers = 2.5 kcal/mol --
    DOES NOT discriminate (within HF/STO-3G's own typical error margin).
  - HF/6-31G*: spread = 10.8 kcal/mol -- DISCRIMINATES. 1-propanol
    -62.6 kcal/mol, 2-propanol -58.5 kcal/mol, methyl ethyl ether
    -69.2 kcal/mol -- chemically sensible: the ether (no O-H) separates
    clearly from both alcohols, and the two alcohol regioisomers still
    separate by ~4 kcal/mol from each other.

Real bottleneck found and fixed getting to the 6-31G* result: with only
build_repulsion_tensor_libcint (the ERI tensor) routed through PySCF/
libcint, the larger basis set made native_hf's own JAX-jitted one-electron
integral assembly (S, H_core) run out of memory during XLA JIT compilation
on a real Kaggle CPU kernel (4 heavy atoms, 6-31G*) -- confirmed directly
("LLVM compilation error: Cannot allocate memory"), not assumed from the
STO-3G timing alone. Fixed by extending the SAME libcint bridge to S/H_core
too (build_overlap_and_core_hamiltonian_libcint, promoted to
dense_evolution.native_hf.libcint_bridge -- see Dense-Evolution PR
referenced in this repo's own promotion tracking) -- reuses
build_repulsion_tensor_libcint's own AO permutation/rescale machinery
rather than duplicating it, since S/T/V_nuc live in the identical AO basis
as the ERI tensor and need the identical native_hf<->libcint reordering.

HF/6-31G* is still a rough method by real quantum-chemistry standards --
this is a discriminating RELATIVE signal for candidate re-ranking, not a
chemically-accurate absolute energy. Runs entirely on Kaggle (a real
Kaggle CPU kernel, ~6 minutes for the full 3-isomer 6-31G* test) --
deliberately never run locally after an unrelated 3GB local dataset
download caused a real PC freeze earlier in this line of work.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from dense_evolution.native_hf.basis import build_molecule_shells, n_cartesian_functions
from dense_evolution.native_hf.cartesian import cartesian_powers
from dense_evolution.native_hf.libcint_bridge import (
    build_repulsion_tensor_libcint, build_overlap_and_core_hamiltonian_libcint,
)
from dense_evolution.native_hf.scf import run_scf

ANGSTROM_TO_BOHR = 1.8897259886
BASIS = "6-31g*"


def embed_3d(mol):
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 2026
    if AllChem.EmbedMolecule(mol, params) != 0:
        return None
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


def mol_to_atoms_geometry(mol):
    conf = mol.GetConformer()
    atomic_numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    positions_angstrom = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    return atomic_numbers, positions_angstrom * ANGSTROM_TO_BOHR


def hf_energy(atomic_numbers, geometry_bohr, charge=0):
    n_electrons = sum(atomic_numbers) - charge
    if n_electrons % 2 != 0:
        return None  # open-shell -- native_hf (RHF-only) can't compute this
    nuclear_charges = [float(z) for z in atomic_numbers]
    S, H_core = build_overlap_and_core_hamiltonian_libcint(atomic_numbers, geometry_bohr, BASIS)
    repulsion = build_repulsion_tensor_libcint(atomic_numbers, geometry_bohr, BASIS)
    result = run_scf(S, H_core, repulsion, n_electrons, nuclear_charges, geometry_bohr)
    return result.total_energy


def h2_reference_energy():
    mol = embed_3d(Chem.MolFromSmiles("[H][H]"))
    if mol is None:
        atomic_numbers = [1, 1]
        geometry_bohr = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74 * ANGSTROM_TO_BOHR]])
    else:
        atomic_numbers, geometry_bohr = mol_to_atoms_geometry(mol)
    return hf_energy(atomic_numbers, geometry_bohr)


def isodesmic_scission_energy(smiles, bond_idx, e_h2):
    """whole + H2 -> fragment_A_H + fragment_B_H. See module docstring for
    why this is an isodesmic reaction energy, not a bond dissociation
    energy, and why the sign can legitimately be negative."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    whole = embed_3d(mol)
    if whole is None:
        return None
    atoms_whole, geom_whole = mol_to_atoms_geometry(whole)
    e_whole = hf_energy(atoms_whole, geom_whole)
    if e_whole is None:
        return None

    frag_mol = Chem.FragmentOnBonds(Chem.MolFromSmiles(smiles), [bond_idx], addDummies=True)
    frags = Chem.GetMolFrags(frag_mol, asMols=True, sanitizeFrags=False)
    if len(frags) != 2:
        return None

    frag_energies = []
    for frag in frags:
        rw = Chem.RWMol(frag)
        for atom in rw.GetAtoms():
            if atom.GetAtomicNum() == 0:  # dummy atom left by FragmentOnBonds
                atom.SetAtomicNum(1)      # cap the cut site with H directly
                atom.SetIsotope(0)
                atom.SetNoImplicit(False)
                atom.SetFormalCharge(0)
        frag_mol_capped = rw.GetMol()
        try:
            Chem.SanitizeMol(frag_mol_capped)
        except Exception:
            return None
        frag_3d = embed_3d(frag_mol_capped)
        if frag_3d is None:
            return None
        atoms_frag, geom_frag = mol_to_atoms_geometry(frag_3d)
        e_frag = hf_energy(atoms_frag, geom_frag)
        if e_frag is None:
            return None  # this fragment is open-shell -- skip, can't score
        frag_energies.append(e_frag)

    return sum(frag_energies) - e_whole - e_h2


def molecule_stability_fingerprint(smiles, e_h2):
    """Sum of isodesmic scission energy over every acyclic single bond --
    a per-molecule scalar summarizing how favorable H-transfer cleavage is
    across the whole structure. Skips bonds where either fragment is
    open-shell rather than failing the whole molecule."""
    mol = Chem.MolFromSmiles(smiles)
    total, n_scored = 0.0, 0
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        e = isodesmic_scission_energy(smiles, bond.GetIdx(), e_h2)
        if e is not None:
            total += e
            n_scored += 1
    return total, n_scored


if __name__ == "__main__":
    # Requires PySCF (pip install pyscf) -- no Windows wheel, run on Linux
    # (verified on a real Kaggle CPU kernel; see module docstring for the
    # actual verified numbers reproduced by this exact script).
    print("Step 0: H2 reference energy")
    e_h2 = h2_reference_energy()
    print(f"  E(H2) = {e_h2:.6f} Hartree")

    print("\nStep 1: ranking test -- three C3H8O isomers (same formula/mass, different structure)")
    isomers = {"1-propanol": "CCCO", "2-propanol": "CC(C)O", "methyl ethyl ether": "CCOC"}
    results = {}
    for name, smi in isomers.items():
        total, n_scored = molecule_stability_fingerprint(smi, e_h2)
        results[name] = total
        print(f"  {name} ({smi}): sum_isodesmic = {total:.4f} Hartree ({total * 627.5:.1f} kcal/mol) "
              f"over {n_scored} scored bonds")

    spread = max(results.values()) - min(results.values())
    print(f"\n[RESULT] spread across isomers: {spread:.4f} Hartree ({spread * 627.5:.1f} kcal/mol)")
    print(f"  {'DISCRIMINATES' if spread > 0.01 else 'DOES NOT MEANINGFULLY DISCRIMINATE'} between these isomers")
