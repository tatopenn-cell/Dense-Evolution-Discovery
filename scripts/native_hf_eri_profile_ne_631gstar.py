import time
import numpy as np

from dense_evolution.native_hf.basis import build_molecule_shells, n_cartesian_functions
from dense_evolution.native_hf.assembly import (
    _pad_primitives, _shell_pair_schwarz_bounds, _shell_offsets, _quartet_block,
)
from dense_evolution.native_hf.coulomb import electron_repulsion

_BOHR_PER_ANGSTROM = 1.0 / 0.52917721067
_LETTER = {0: "s", 1: "p", 2: "d", 3: "f"}


def _sig(degrees):
    return "".join(_LETTER[d] for d in degrees)


def main():
    geometry_bohr = np.array([[0.0, 0.0, 0.0]])
    shells = build_molecule_shells([10], geometry_bohr, "6-31g*")
    n_ao = n_cartesian_functions(shells)
    print(f"shells: {len(shells)}, AOs: {n_ao}")
    for i, s in enumerate(shells):
        print(f"  shell {i}: degree={s.degree} ({_LETTER[s.degree]}) n_primitives={s.exponents.shape[0]}")

    max_primitives = max(s.exponents.shape[0] for s in shells)
    offsets = _shell_offsets(shells)
    screening_tol = 1e-12

    t0 = time.perf_counter()
    schwarz = _shell_pair_schwarz_bounds(shells, max_primitives)
    t_schwarz_bounds = time.perf_counter() - t0
    print(f"\nschwarz bound computation (all shell pairs): {t_schwarz_bounds:.3f}s")

    n_after_symmetry = 0
    n_after_schwarz = 0
    quartets = []
    for i, sa in enumerate(shells):
        for j in range(i + 1):
            sb = shells[j]
            ij_index = i * (i + 1) // 2 + j
            for k, sc in enumerate(shells):
                for l in range(k + 1):
                    sd = shells[l]
                    kl_index = k * (k + 1) // 2 + l
                    if ij_index < kl_index:
                        continue
                    n_after_symmetry += 1
                    if schwarz[(i, j)] * schwarz[(k, l)] < screening_tol:
                        continue
                    n_after_schwarz += 1
                    quartets.append((i, j, k, l))

    print(f"\n1) quartets after 8-way symmetry: {n_after_symmetry}")
    print(f"   quartets after Schwarz screening: {n_after_schwarz}")

    compiled_seen = {}   # exact ordered degree-tuple -> compile+first-exec wall time
    warm_times = {}      # exact ordered degree-tuple -> list of warm per-call times
    sig_counts = {}       # sorted-letter signature -> count of quartets
    call_order = []

    t_loop0 = time.perf_counter()
    for (i, j, k, l) in quartets:
        sa, sb, sc, sd = shells[i], shells[j], shells[k], shells[l]
        degrees = (sa.degree, sb.degree, sc.degree, sd.degree)
        sig = "".join(sorted(_sig(degrees)))
        sig_counts[sig] = sig_counts.get(sig, 0) + 1

        ea, ca = _pad_primitives(sa.exponents, sa.coefficients, max_primitives)
        eb, cb = _pad_primitives(sb.exponents, sb.coefficients, max_primitives)
        ec, cc = _pad_primitives(sc.exponents, sc.coefficients, max_primitives)
        ed, cd = _pad_primitives(sd.exponents, sd.coefficients, max_primitives)

        t_call0 = time.perf_counter()
        block = np.array(
            _quartet_block(
                (ea, eb, ec, ed), (ca, cb, cc, cd),
                (sa.center, sb.center, sc.center, sd.center),
                degrees, electron_repulsion,
            )
        )
        dt = time.perf_counter() - t_call0
        call_order.append((degrees, dt))

        if degrees not in compiled_seen:
            compiled_seen[degrees] = dt
        else:
            warm_times.setdefault(degrees, []).append(dt)

    t_loop_total = time.perf_counter() - t_loop0
    t_total = time.perf_counter() - t0

    print(f"\n2) warm (post-compile) time per exact degree-tuple, one isolated call each:")
    warm_avg_by_tuple = {}
    for degrees, times in warm_times.items():
        avg = sum(times) / len(times)
        warm_avg_by_tuple[degrees] = avg
        print(f"   {_sig(degrees)} (order {degrees}): n_calls_after_first={len(times)}, "
              f"avg_warm={avg*1000:.3f}ms, min={min(times)*1000:.3f}ms, max={max(times)*1000:.3f}ms")
    for degrees in compiled_seen:
        if degrees not in warm_times:
            print(f"   {_sig(degrees)} (order {degrees}): only ever called ONCE (no warm repeat in this molecule)")

    print(f"\n   grouped by sorted signature (ignoring a/b/c/d order), quartet counts:")
    for sig, cnt in sorted(sig_counts.items()):
        print(f"   {sig}: {cnt} quartets")

    print(f"\n3) (count per exact tuple) x (avg warm time per exact tuple):")
    exact_tuple_counts = {}
    for (degrees, _) in call_order:
        exact_tuple_counts[degrees] = exact_tuple_counts.get(degrees, 0) + 1
    total_execution_estimate = 0.0
    for degrees, count in exact_tuple_counts.items():
        warm = warm_avg_by_tuple.get(degrees)
        if warm is None:
            continue
        product = count * warm
        total_execution_estimate += product
        print(f"   {_sig(degrees)} {degrees}: count={count}, warm_each={warm*1000:.3f}ms, "
              f"product={product:.4f}s")
    print(f"   sum of (count x warm_avg) over tuples with a warm sample: {total_execution_estimate:.4f}s")
    print(f"   this covers {100*total_execution_estimate/t_loop_total:.1f}% of the {t_loop_total:.4f}s main quartet loop")

    print(f"\n4) compilation time (first-call time, includes exec of that one call), per distinct exact tuple:")
    total_compile_first_calls = sum(compiled_seen.values())
    for degrees, first_time in sorted(compiled_seen.items(), key=lambda kv: -kv[1]):
        warm = warm_avg_by_tuple.get(degrees)
        if warm is not None:
            compile_only = first_time - warm
            print(f"   {_sig(degrees)} {degrees}: first_call={first_time*1000:.3f}ms, "
                  f"warm_avg={warm*1000:.3f}ms, compile_only_est={compile_only*1000:.3f}ms")
        else:
            print(f"   {_sig(degrees)} {degrees}: first_call={first_time*1000:.3f}ms "
                  f"(no warm repeat -- can't separate compile from exec for this one)")
    print(f"   {len(compiled_seen)} distinct exact (a,b,c,d)-ordered degree-tuples compiled")
    print(f"   sum of all first-call times: {total_compile_first_calls:.4f}s")

    accounted = total_compile_first_calls + total_execution_estimate
    dispatch_other = t_loop_total - accounted
    print(f"\n5) time outside compile+execution (Python dispatch, np.array(), device syncs):")
    print(f"   main quartet loop wall time: {t_loop_total:.4f}s")
    print(f"   accounted for by (compile first-calls) + (count x warm_avg): {accounted:.4f}s")
    print(f"   unaccounted / dispatch overhead: {dispatch_other:.4f}s ({100*dispatch_other/t_loop_total:.1f}%)")

    print(f"\n--- summary ---")
    print(f"schwarz bound phase: {t_schwarz_bounds:.3f}s")
    print(f"main quartet loop:   {t_loop_total:.3f}s")
    print(f"total (this script): {t_total:.3f}s")


if __name__ == "__main__":
    main()
