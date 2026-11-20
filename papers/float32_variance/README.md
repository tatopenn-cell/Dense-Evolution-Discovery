# Variance in float32 with a first-element shift, a local centring and a compensated tree sum

- Paper, in this folder: [`paper.md`](paper.md), a copy of the published `docs/papers/float32_variance.md` with the image links pointing to `docs/papers/assets/float32/`; if the two differ, the one in `docs/` is the one published on the site
- Paper, published: `docs/papers/float32_variance.md` (the problem, related work, the kernel, the tests of the library page on a CPU and a GPU, declared limits, reproduction)
- Note it extends: `docs/dense_armor_float32_variance.md`
- Code and numbers: `benchmarks/stable_variance_float32.py`, `stable_variance_float32.json` (CPU) and `stable_variance_float32_gpu.json` (GPU) of the library
- Status: written; not submitted.
- The LaTeX package of the paper: `main.tex`, `references.bib`, `figures/` and the compiled `float32_variance.pdf`; `float32_variance_arxiv_source.zip` is the source package for arXiv (the tex, the bbl, the bib and the figures); `build_tex.py` writes the tex from `docs/papers/float32_variance.md`, and `tectonic main.tex` compiles it
- The standalone program of the paper: `reproduce_standalone.py` (the kernel of `stable_var` written out, JAX and NumPy only; it computes the examples and the accuracy runs of the paper) and its output `reproduce_standalone_output.txt`; both are in Appendix A and B of the PDF and in the arXiv source zip
