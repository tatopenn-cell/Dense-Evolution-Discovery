# Making the offline split of a regression tree online without changing its result

Salvatore Pennacchio, Dense-Armor `v1.1.31`, October 2026.

## Problem

The textbook way to choose the split of a numerical attribute in a regression tree is offline: keep all
the samples, sort them by the attribute, and compute the left and right sums of the target for every
threshold. It needs the whole stream in memory. Dense-Armor works online: one sample at a time, bounded
memory, and the same answer as the offline procedure.

Ikonomovska (2012, PhD thesis, chapter 6) gives the online structure for it, the extended binary search
tree (E-BST). Each node keeps the count, the sum and the sum of squares of the target for the samples that
reached it with a value at most equal to its key (page 74, "the counts in all the nodes on the way are
additionally updated"). The statistics of a node depend on the order in which the samples arrived, while
the offline answer does not. The thesis gives the pseudocode (Algorithm 3, page 75) and one example
(figure 7), and no value computed by the offline procedure. A test written from the same scheme as the
code checks the code against itself. The open question was therefore: does the online split statistic
equal the offline one, and how can it be checked when the order changes the nodes?

## Reference for the check

The Lagrange interpolating polynomial through the observed points, `p_n(f;x) = sum_k f(x_k) l_k(x)` with
`l_k(x) = prod_{j != k} (x - x_j)/(x_k - x_j)` (Gautschi 2012, eq. (6)-(7), page 351; remainder eq. (11),
page 353), reproduces every observation exactly and does not depend on the order in which the points were
seen. Computing the standard deviation reduction (SDR) of a threshold from all the observations at once
gives a value that holds for any insertion order, and the E-BST statistic is compared with it threshold by
threshold. Using the interpolation as the reference for the order-dependent tree is my choice: "Lagrange"
and "interpolat" appear in none of the 222 pages of the thesis.

## Result

Four samples `(0,0), (1,1), (2,2), (3,3)` in this order, split `x <= 1`: reference SDR 0.6180, E-BST
statistic taken node by node (as `FIMTDDRegressor` v1.1.31 computes it) 0.1826, difference 0.4354.

The sequence of figure 7 of the thesis, `(0.95, 0.71), (0.43, 0.56), (7.44, 2.3), (1.03, 0.18), (8.25, 3.45)`:

| threshold | reference | node by node (v1.1.31) | node plus ancestors |
|---|---|---|---|
| 0.43 | 0.2030 | 0.2030 | 0.2030 |
| 0.95 | 0.3963 | 0.3963 | 0.3963 |
| 1.03 | 0.8751 | 0.2848 | 0.8751 |
| 7.44 | 0.5909 | 0.0179 | 0.5909 |
| best threshold | 1.03 | 8.25 | 1.03 |

The node-by-node statistic is the one in `dense_armor/utility/tree/fimt.py` (lines 162-190): the left side
is the `l` of the single node. "Node plus ancestors" adds the `l` of every ancestor from which the path to
the node turns right, which is the running total of Algorithm 3 along the path. On 500 random insertion
orders of 12 points (`y = 2x` plus noise) the node plus ancestors statistic equals the reference to
1.3e-11 and picks the reference threshold 500 times out of 500; the node-by-node statistic picks it 94
times out of 500.

The threshold 8.25 that v1.1.31 returns on the figure 7 sequence has no sample to its right in the
reference; its node-by-node split puts the single sample 8.25 on the left and the four earlier ones on the
right.

## Reproduce

```python
import numpy as np
from dense_armor.utility.tree import fimt as F


def build(seq):
    t = F._Ebst()
    tot = F._SideStats()
    for x, y in seq:
        t.insert(x, y)
        tot.n += 1
        tot.sum_y += y
        tot.sum_y2 += y * y
    return t, tot


def sdr(L, R):
    a = np.array(L + R)
    return a.std() - len(L) / len(a) * np.std(L) - len(R) / len(a) * np.std(R)


def batch(seq):
    out = {}
    for k in sorted({x for x, _ in seq}):
        L = [y for x, y in seq if x <= k]
        R = [y for x, y in seq if x > k]
        if L and R:
            out[k] = sdr(L, R)
    return out


def node_by_node(seq):
    t, tot = build(seq)
    out = {}
    for k, l in t._inorder():
        rn = tot.n - l.n
        if rn < 1:
            continue
        r = F._SideStats(n=rn, sum_y=tot.sum_y - l.sum_y, sum_y2=tot.sum_y2 - l.sum_y2)
        out[k] = tot.sd() - l.n / tot.n * l.sd() - rn / tot.n * r.sd()
    return out


def with_ancestors(seq):
    t, tot = build(seq)
    out = {}

    def walk(node, an, asy, asq):
        if node is None:
            return
        walk(node.left, an, asy, asq)
        n, s, q = an + node.l.n, asy + node.l.sum_y, asq + node.l.sum_y2
        rn = tot.n - n
        if rn >= 1:
            L = F._SideStats(n=n, sum_y=s, sum_y2=q)
            R = F._SideStats(n=rn, sum_y=tot.sum_y - s, sum_y2=tot.sum_y2 - q)
            out[node.key] = tot.sd() - n / tot.n * L.sd() - rn / tot.n * R.sd()
        walk(node.right, n, s, q)

    walk(t.root, 0, 0.0, 0.0)
    return out


fig7 = [(0.95, 0.71), (0.43, 0.56), (7.44, 2.3), (1.03, 0.18), (8.25, 3.45)]
b, e, a = batch(fig7), node_by_node(fig7), with_ancestors(fig7)
print("fig7  thr   batch  node  ancestors")
for k in sorted(b):
    print(f"{k:5} {b[k]:.4f} {e.get(k, float('nan')):.4f} {a.get(k, float('nan')):.4f}")
print("best batch", max(b, key=b.get), "best node", max(e, key=e.get), "best ancestors", max(a, key=a.get))

rng = np.random.default_rng(0)
same_node = same_anc = 0
N = 500
maxerr = 0.0
for _ in range(N):
    xs = rng.permutation(12).astype(float)
    seq = [(x, float(2 * x + rng.normal())) for x in xs]
    b, e, a = batch(seq), node_by_node(seq), with_ancestors(seq)
    same_node += max(b, key=b.get) == max(e, key=e.get)
    same_anc += max(b, key=b.get) == max(a, key=a.get)
    maxerr = max(maxerr, max(abs(b[k] - a[k]) for k in a if k in b))
print("random 12 points, 500 orders: best threshold equals batch -> node", same_node, "ancestors", same_anc, "max |ancestors-batch|", f"{maxerr:.2e}")
```

## Limits

- The tests of v1.1.31 (`test_sdr_paper_example_matches_the_temporal_offset`,
  `test_ebst_offset_makes_the_paper_difference_explicit`) fix the node-by-node values. They quote the
  thesis rule of page 74, and the table of the first one lists `(1, 0.18)` as the left side of 1.03,
  where the reference has the three samples at or below 1.03.
- The check covers one attribute and a regression target of the form `y = 2x` plus noise. It does not
  measure the effect on the accuracy of the full tree.
- The page citation in `fimt.py`, "page 71 (PDF 85)", is the printed page 74 (PDF 88).

## Sources

- E. Ikonomovska, *Algorithms for Learning Regression Trees and Ensembles on Evolving Data Streams*, PhD
  thesis, Jozef Stefan International Postgraduate School, 2012,
  https://kt.ijs.si/wp-content/uploads/2021/11/phd_elena_ikonomovska.pdf (pages 74-75).
- W. Gautschi, "Interpolation before and after Lagrange", Rend. Sem. Mat. Univ. Pol. Torino 70(4),
  347-368, 2012, https://seminariomatematico.polito.it/rendiconti/70-4/347.pdf (eq. (6)-(7) page 351,
  eq. (11) page 353).
- Dense-Armor `v1.1.31`, commit `b1b32ec7866e641ab0188feb663c2404f1658370`.
