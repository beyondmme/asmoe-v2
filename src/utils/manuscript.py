"""Render the Springer Nature manuscript as ONE self-contained .tex file (the template forbids \\input).

    paper/template/sn-article.tex   skeleton with %%@MACROS and %%@SECTION <name> markers
    paper/sections/<name>.tex        authored text; may contain \\input{tables/<x>}
    paper/_generated/tables/*.tex    tables + autogen_macros.tex built from results/ (overleaf_sync.build_report)
    paper/_generated/figures/*.png   figures built from results/
    paper/figures_manual/*           hand-made figures (architecture diagram), copied as-is
  ->
    paper/build/sn-article.tex       single file, everything inlined
    paper/build/figures/*.png, sn-bibliography.bib, sn-jnl.cls, sn-mathphys-num.bst

paper/build/ is exactly what the Overleaf project should contain.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, List

INPUT_RE = re.compile(r"\\input\{tables/([^}]+?)(?:\.tex)?\}")


def _strip_comment(line: str) -> str:
    """Code part of a LaTeX line (text before an unescaped %)."""
    m = re.search(r"(?<!\\)%", line)
    return line if m is None else line[:m.start()]


def _expand_tables(text: str, tables_dir: Path, missing: List[str]) -> str:
    out = []
    for line in text.splitlines():
        code = _strip_comment(line)
        m = INPUT_RE.search(code)
        if not m:
            out.append(line)
            continue
        f = tables_dir / f"{m.group(1)}.tex"
        if f.exists():
            out.append(f.read_text(encoding="utf-8").rstrip())
        else:
            missing.append(f.name)
            out.append(f"% [table {m.group(1)} not generated yet]")
    return "\n".join(out)


def render_manuscript(paper_dir: Path, generated_dir: Path, build_dir: Path) -> Dict:
    paper_dir, generated_dir, build_dir = Path(paper_dir), Path(generated_dir), Path(build_dir)
    template = (paper_dir / "template" / "sn-article.tex").read_text(encoding="utf-8")
    tables = generated_dir / "tables"
    missing: List[str] = []
    out_lines: List[str] = [
        "%% ==========================================================================================",
        "%% GENERATED FILE -- do not edit in Overleaf: changes are overwritten by the next sync.",
        "%% Source: https://github.com/beyondmme/asmoe-potato-v2  (paper/template/sn-article.tex,",
        "%% paper/sections/*.tex, numbers/tables/figures from results/ via `python train.py paper`).",
        "%% Springer Nature LaTeX template v3.1 (December 2024), sn-jnl class.",
        "%% ==========================================================================================",
    ]
    body = template.splitlines()
    first = next(i for i, l in enumerate(body) if l.lstrip().startswith("\\documentclass"))
    for line in body[first:]:
        s = line.strip()
        if s == "%%@MACROS":
            f = tables / "autogen_macros.tex"
            out_lines.append(f.read_text(encoding="utf-8").rstrip() if f.exists() else "% [macros not generated]")
        elif s.startswith("%%@SECTION "):
            name = s.split(None, 1)[1].strip()
            sec = paper_dir / "sections" / f"{name}.tex"
            if not sec.exists():
                raise FileNotFoundError(sec)
            out_lines.append(f"%%%%%%%%%%%%%%%% paper/sections/{name}.tex %%%%%%%%%%%%%%%%")
            out_lines.append(_expand_tables(sec.read_text(encoding="utf-8"), tables, missing).rstrip())
        else:
            out_lines.append(line)
    tex = "\n".join(out_lines) + "\n"
    leftover = [l for l in tex.splitlines() if re.search(r"\\input\b|\\include\{", _strip_comment(l))]
    if leftover:
        raise ValueError(f"\\input left in manuscript: {leftover[:3]}")

    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "sn-article.tex").write_text(tex, encoding="utf-8")
    for f in (paper_dir / "template").iterdir():
        if f.suffix in {".cls", ".bst"}:
            shutil.copy2(f, build_dir / f.name)
    shutil.copy2(paper_dir / "sn-bibliography.bib", build_dir / "sn-bibliography.bib")
    figs = generated_dir / "figures"
    if figs.exists():
        shutil.copytree(figs, build_dir / "figures", dirs_exist_ok=True)
    manual = paper_dir / "figures_manual"   # hand-made figures (e.g. the architecture diagram)
    if manual.exists():
        shutil.copytree(manual, build_dir / "figures", dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("README*", ".gitkeep"))
    used = set(re.findall(r"figures/[\w.\-]+\.png", tex)) - set(re.findall(r"\\IfFileExists\{([^}]+)\}", tex))
    have = {f"figures/{p.name}" for p in (build_dir / "figures").glob("*.png")} if (build_dir / "figures").exists() else set()
    return {"tex": str(build_dir / "sn-article.tex"), "missing_tables": missing,
            "figures_pending": sorted(used - have), "lines": len(tex.splitlines())}
