# Lagrange interpolation as the reference for the E-BST split statistics

Salvatore Pennacchio, Dense-Armor `v1.1.31`, October 2026.

## Question

FIMT-DD (Ikonomovska 2012, thesis, chapter 6) chooses a split of a numerical attribute from an
extended binary search tree (E-BST): every node keeps the count, the sum and the sum of squares of the
target for the samples that reached it with a value less than or equal to its key. Does the split
statistic that `FIMTDDRegressor` computes from this tree equal the statistic computed from the same
samples in batch?

## Idea

The Lagrange interpolating polynomial through the n + 1 nodes, `p(x) = sum_k y_k ell_k(x)` with
`ell_k(x) = prod_{j != k} (x - x_j) / (x_k - x_j)` (Gautschi 2012, eq. (6)-(7), page 351), reproduces
the observed values exactly at the nodes and a polynomial of degree at most n everywhere (eq. (6)).
It gives a reference that uses all the observations at once and no insertion order. Taking it as the
reference for the E-BST is my choice: "Lagrange" and "interpolat" appear in none of the 222 pages of
the thesis.

## Setup

Four samples `(x, y) = (0,0), (1,1), (2,2), (3,3)` inserted in this order. The line `y = x` is the
Lagrange polynomial of degree 3 through them. The reference statistic is the standard deviation
reduction (SDR) of the split at `x <= 1` computed with all four samples:
`SDR = sd(all) - (n_L/n) sd(L) - (n_R/n) sd(R)`.

```python
import numpy as np
from dense_armor.utility.tree import fimt as F

t = F._Ebst()
tot = F._SideStats()
for x in (0.0, 1.0, 2.0, 3.0):
    t.insert(x, x)
    tot.n += 1
    tot.sum_y += x
    tot.sum_y2 += x * x


def sdr(a, b):
    c = np.array(a + b)
    return c.std() - len(a) / len(c) * np.std(a) - len(b) / len(c) * np.std(b)


print("batch", round(sdr([0, 1], [2, 3]), 4))

for thr, l in t._inorder():
    rn = tot.n - l.n
    if rn < 1:
        continue
    r = F._SideStats(n=rn, sum_y=tot.sum_y - l.sum_y, sum_y2=tot.sum_y2 - l.sum_y2)
    print("node", thr, l.n, round(tot.sd() - l.n / tot.n * l.sd() - rn / tot.n * r.sd(), 4))

cn = cs = cq = 0
for thr, l in t._inorder():
    cn += l.n
    cs += l.sum_y
    cq += l.sum_y2
    rn = tot.n - cn
    if rn < 1:
        continue
    L = F._SideStats(n=cn, sum_y=cs, sum_y2=cq)
    r = F._SideStats(n=rn, sum_y=tot.sum_y - cs, sum_y2=tot.sum_y2 - cq)
    print("accumulated", thr, cn, round(tot.sd() - cn / tot.n * L.sd() - rn / tot.n * r.sd(), 4))
```

## Results

| split `x <=` | batch (reference) | `_inorder`, one node (v1.1.31) | accumulated in order |
|---|---|---|---|
| 0 | 0.5057 | 0.5057 | 0.5057 |
| 1 | 0.6180 | 0.1826 | 0.6180 |
| 2 | 0.5057 | 0.1826 | 0.5057 |

The batch values at 0 and 2 are the same by symmetry. At `x <= 1` the node-by-node value differs from
the reference by 0.4354 (0.6180 - 0.1826).

## Where the difference comes from

Thesis, page 75 (PDF 89), Algorithm 3 `FindBestSplit`, in-order traversal:

```
sumTotalLeft  = sumTotalLeft  + Node->sumLeft
sumTotalRight = sumTotalRight - Node->sumLeft
countRightTotal = countRightTotal - Node->countLeft
```

The left side of a split is the running total of the nodes visited so far. In
`dense_armor/utility/tree/fimt.py` the method that scores the splits (lines 162-179) takes
`left = node.l` of the single node that `_inorder` yields (lines 181-190), without the running total.
On this chain of four nodes, accumulating as the algorithm states gives the batch value in every row of
the table. This is the check on this chain only: the thesis text on the E-BST (page 74, PDF 88: "the
counts in all the nodes on the way are additionally updated") also holds for trees that are not a
chain, and whether the running total is the exact left statistic there is not tested here.

## What the test of v1.1.31 checks

`test/tree/test_fimt.py::test_ebst_offset_makes_the_paper_difference_explicit` asserts the 0.1826
that the code computes, and takes the Lagrange value 0.6180 as a constant written in the test, not
computed. The batch value above confirms that 0.6180 is the right number for the reference; the 0.1826
is the score of the node-by-node statistic, which this note finds different from Algorithm 3.

## Open

- Run the accumulation on random insertion orders (non-chain trees) against the batch SDR, at the
  quantity the split choice uses, and measure how often the best threshold differs.
- Correct the page citation in the code: the sentence "the counts in all the nodes on the way are
  additionally updated" is on printed page 74 (PDF 88), the code quotes page 71 (PDF 85).

## Sources

- E. Ikonomovska, *Algorithms for Learning Regression Trees and Ensembles on Evolving Data Streams*,
  PhD thesis, Jozef Stefan International Postgraduate School, 2012,
  https://kt.ijs.si/wp-content/uploads/2021/11/phd_elena_ikonomovska.pdf (pages 74-75).
- W. Gautschi, "Interpolation before and after Lagrange", Rend. Sem. Mat. Univ. Pol. Torino 70(4),
  347-368, 2012, eq. (6)-(7) page 351, eq. (11) page 353.
- Dense-Armor `v1.1.31`, commit `b1b32ec7866e641ab0188feb663c2404f1658370`.
