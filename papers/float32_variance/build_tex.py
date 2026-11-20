"""Build the LaTeX source package of the paper from docs/papers/float32_variance.md.

Writes main.tex, references.bib and figures/ next to this file, ready to be compiled
(pdflatex or xelatex, then bibtex) or sent to arXiv. The text of the paper is the one of the
markdown file; nothing is typed twice.

    python papers/float32_variance/build_tex.py
"""

import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SRC = REPO / "docs" / "papers" / "float32_variance.md"
ASSETS = REPO / "docs" / "papers" / "assets" / "float32"

BIB = [
    ("chan1983", "article", {"author": "Chan, T. F. and Golub, G. H. and LeVeque, R. J.", "title": "Algorithms for Computing the Sample Variance: Analysis and Recommendations", "journal": "The American Statistician", "volume": "37", "number": "3", "pages": "242--247", "year": "1983"}),
    ("ogita2005", "article", {"author": "Ogita, T. and Rump, S. M. and Oishi, S.", "title": "Accurate Sum and Dot Product", "journal": "SIAM Journal on Scientific Computing", "volume": "26", "number": "6", "pages": "1955--1988", "year": "2005"}),
    ("healy2022", "misc", {"author": "Healy, J.", "title": "Accurate and consistent calculation of the mean and variance in {Monte-Carlo} simulations", "howpublished": "arXiv:2206.10662v7", "year": "2022"}),
    ("shewchuk1997", "article", {"author": "Shewchuk, J. R.", "title": "Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates", "journal": "Discrete and Computational Geometry", "volume": "18", "number": "3", "pages": "305--363", "year": "1997"}),
    ("tfmoments", "misc", {"title": "{TensorFlow}, tensorflow/python/ops/nn\\_impl.py, function moments, lines 1480-1487", "howpublished": "master, read on 2026-10-29", "year": "2026"}),
    ("tfissue", "misc", {"title": "{TensorFlow}, issue \\#117286: the inaccuracy of tf.nn.moments in float32", "year": "2026"}),
    ("kahan1965", "article", {"author": "Kahan, W.", "title": "Further remarks on reducing truncation errors", "journal": "Communications of the ACM", "volume": "8", "number": "1", "pages": "40", "year": "1965"}),
    ("micikevicius2018", "inproceedings", {"author": "Micikevicius, P. and others", "title": "Mixed Precision Training", "booktitle": "ICLR 2018", "year": "2018"}),
    ("zhang2019", "inproceedings", {"author": "Zhang, B. and Sennrich, R.", "title": "Root Mean Square Layer Normalization", "booktitle": "NeurIPS 2019", "year": "2019"}),
]
KEYS = [b[0] for b in BIB]

SUP = dict(zip("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-"))
SUB = dict(zip("₀₁₂₃₄₅₆₇₈₉ᵢ", "0123456789i"))
SYM = {
    "·": r"$\cdot$", "−": r"$-$", "≤": r"$\leq$", "≥": r"$\geq$", "×": r"$\times$", "≈": r"$\approx$",
    "→": r"$\rightarrow$", "Σ": r"$\Sigma$", "μ": r"$\mu$", "σ": r"$\sigma$", "√": r"$\surd$", "±": r"$\pm$",
    "—": "---", "–": "--", "’": "'", "‘": "`", "“": "``", "”": "''", "…": r"\ldots{}", "≠": r"$\neq$",
}
SPECIAL = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}", "^": r"\textasciicircum{}", "\\": r"\textbackslash{}"}
CITE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)(?:,\s*([A-Za-z][^\]]*))?\]")


def esc(t):
    return "".join(SPECIAL.get(c, c) for c in t)


def uni(t):
    t = re.sub("[⁰¹²³⁴⁵⁶⁷⁸⁹⁻]+", lambda m: "$^{" + "".join(SUP[c] for c in m.group()) + "}$", t)
    t = re.sub("[₀₁₂₃₄₅₆₇₈₉ᵢ]+", lambda m: "$_{" + "".join(SUB[c] for c in m.group()) + "}$", t)
    for k, v in SYM.items():
        t = t.replace(k, v)
    return t


def plain(t):
    out, pos = [], 0
    for m in CITE.finditer(t):
        out.append(uni(esc(t[pos:m.start()])))
        keys = ",".join(KEYS[int(n) - 1] for n in re.split(r"\s*,\s*", m.group(1)))
        note = f"[{esc(m.group(2))}]" if m.group(2) else ""
        out.append(f"\\cite{note}{{{keys}}}")
        pos = m.end()
    out.append(uni(esc(t[pos:])))
    return "".join(out)


APPENDIX = (
    "\n\n\\appendix\n\\section{The standalone program}\n\n"
    "The program computes the examples and the accuracy runs of this paper with no library of ours: the kernel of "
    "\\texttt{stable\\_var} is written out in it, and it needs only JAX and NumPy. Save it as "
    "\\texttt{reproduce\\_standalone.py} and run \\texttt{python reproduce\\_standalone.py}.\n\n"
    "\\lstinputlisting[language=Python,basicstyle=\\ttfamily\\tiny,breaklines=true,showstringspaces=false,numbers=left]{reproduce_standalone.py}\n\n"
    "\\section{The output of the program}\n\n"
    "The output of the run on a CPU; the last digits can differ with the version of JAX and with the machine.\n\n"
    "\\lstinputlisting[basicstyle=\\ttfamily\\tiny,breaklines=true]{reproduce_standalone_output.txt}"
)

TOK = re.compile(r"`([^`]+)`|\$([^$]+)\$|\*\*(.+?)\*\*|\*(.+?)\*")


def fence(lang, block):
    if lang == "latex":
        return "\\begin{equation}\n" + "\n".join(block) + "\n\\end{equation}"
    if lang == "algorithm":
        cap = block[0][len("%caption: "):]
        return "\\begin{algorithm}[H]\n\\caption{" + cap + "}\n\\begin{algorithmic}[1]\n" + "\n".join(block[1:]) + "\n\\end{algorithmic}\n\\end{algorithm}"
    return "\\begin{verbatim}\n" + "\n".join(block) + "\n\\end{verbatim}"


def inline(t):
    out, pos = [], 0
    for m in TOK.finditer(t):
        out.append(plain(t[pos:m.start()]))
        if m.group(1) is not None:
            out.append("\\texttt{" + uni(esc(m.group(1))) + "}")
        elif m.group(2) is not None:
            out.append("$" + m.group(2) + "$")
        elif m.group(3) is not None:
            out.append("\\textbf{" + inline(m.group(3)) + "}")
        else:
            out.append("\\emph{" + inline(m.group(4)) + "}")
        pos = m.end()
    out.append(plain(t[pos:]))
    return "".join(out)


def table(rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, body = cells[0], cells[2:]
    n = len(head)
    wide = n >= 5
    o = ["\\begin{table}[H]", "\\centering", "\\small"]
    if wide:
        o.append("\\resizebox{\\textwidth}{!}{%")
    o.append("\\begin{tabular}{" + "l" * n + "}")
    o.append("\\toprule")
    o.append(" & ".join(inline(c) for c in head) + " \\\\")
    o.append("\\midrule")
    for r in body:
        o.append(" & ".join(inline(c) for c in r) + " \\\\")
    o.append("\\bottomrule")
    o.append("\\end{tabular}" + ("}" if wide else ""))
    o.append("\\end{table}")
    return "\n".join(o)


def body(lines):
    o, i = [], 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
        elif ln.startswith("```"):
            j = i + 1
            while j < len(lines) and not lines[j].startswith("```"):
                j += 1
            o.append(fence(ln[3:].strip(), lines[i + 1 : j]))
            i = j + 1
        elif ln.startswith("|"):
            j = i
            while j < len(lines) and lines[j].startswith("|"):
                j += 1
            o.append(table(lines[i:j]))
            i = j
        elif ln.startswith("!["):
            m = re.match(r"!\[(.*?)\]\((.*?)\)", ln)
            name = Path(m.group(2)).name
            shutil.copy(ASSETS / name, HERE / "figures" / name)
            cap = ""
            j = i + 1
            while j < len(lines) and not lines[j].strip():
                j += 1
            if j < len(lines) and lines[j].startswith("*Figure"):
                cap = re.sub(r"^Figure \d+\.\s*", "", lines[j].strip().strip("*"))
                i = j
            o.append("\\begin{figure}[H]\n\\centering\n\\includegraphics[width=0.85\\linewidth]{figures/" + name + "}\n\\caption{" + inline(cap or m.group(1)) + "}\n\\end{figure}")
            i += 1
        elif re.match(r"^- ", ln):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(lines[i][2:])
                i += 1
            o.append("\\begin{itemize}\n" + "\n".join("\\item " + inline(x) for x in items) + "\n\\end{itemize}")
        elif re.match(r"^\d+\. ", ln):
            items = []
            while i < len(lines) and re.match(r"^\d+\. ", lines[i]):
                items.append(re.sub(r"^\d+\. ", "", lines[i]))
                i += 1
            o.append("\\begin{enumerate}\n" + "\n".join("\\item " + inline(x) for x in items) + "\n\\end{enumerate}")
        else:
            j = i
            par = []
            while j < len(lines) and lines[j].strip() and not lines[j].startswith(("|", "![", "- ", "#", "```")) and not re.match(r"^\d+\. ", lines[j]):
                par.append(lines[j].strip())
                j += 1
            o.append(inline(" ".join(par)))
            i = j
    return "\n\n".join(o)


def main():
    text = SRC.read_text(encoding="utf-8").replace("\r\n", "\n")
    text = text.split("\n## References")[0]
    lines = text.split("\n")
    title = lines[0][2:].strip()
    author, date = [x.strip().rstrip(".") for x in lines[2].split(",", 1)]
    (HERE / "figures").mkdir(exist_ok=True)
    parts = re.split(r"(?m)^(## .*|### .*)$", "\n".join(lines[3:]))
    out = []
    for k in range(1, len(parts), 2):
        head, chunk = parts[k], parts[k + 1].split("\n")
        name = re.sub(r"^#+\s*(\d+(\.\d+)?\.?\s+)?", "", head)
        if head.startswith("## Abstract"):
            out.append("\\begin{abstract}\n" + body(chunk) + "\n\\end{abstract}")
            continue
        cmd = "\\section" if head.startswith("## ") else "\\subsection"
        out.append(cmd + "{" + inline(name) + "}\n\n" + body(chunk))
    tex = (
        "\\documentclass[11pt]{article}\n\\usepackage[utf8]{inputenc}\n\\usepackage[T1]{fontenc}\n\\usepackage{lmodern}\n"
        "\\usepackage{amsmath,amssymb,graphicx,booktabs,float}\n\\usepackage{algorithm}\n\\usepackage{algpseudocode}\n\\usepackage{listings}\n\\usepackage[numbers]{natbib}\n\\usepackage[margin=2.5cm]{geometry}\n"
        "\\usepackage{hyperref}\n\\setlength{\\parskip}{0.5em}\\setlength{\\parindent}{0pt}\n\\sloppy\\emergencystretch=3em\n"
        f"\\title{{{inline(title)}}}\n\\author{{{esc(author)}}}\n\\date{{{esc(date)}}}\n\n\\begin{{document}}\n\\maketitle\n\n"
        + "\n\n".join(out)
        + APPENDIX
        + "\n\n\\bibliographystyle{unsrtnat}\n\\bibliography{references}\n\n\\end{document}\n"
    )
    (HERE / "main.tex").write_text(tex, encoding="utf-8")
    bib = []
    for key, kind, f in BIB:
        bib.append("@" + kind + "{" + key + ",\n" + ",\n".join(f"  {k} = {{{v}}}" for k, v in f.items()) + "\n}\n")
    (HERE / "references.bib").write_text("\n".join(bib), encoding="utf-8")
    print("written main.tex, references.bib and", len(list((HERE / "figures").glob("*.png"))), "figures")


if __name__ == "__main__":
    main()
