"""Static sanity checks for paper/build/sn-article.tex (no LaTeX installation needed).

Checks: balanced braces, matched \\begin/\\end, every \\ref has a \\label, every \\cite key is in the .bib,
every auto-generated macro used in the text is defined, no \\input left. Exit code 1 on failure.
CI additionally compiles the PDF with a full TeX Live.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

build = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "paper" / "build")
tex = (build / "sn-article.tex").read_text(encoding="utf-8")
bib = (build / "sn-bibliography.bib").read_text(encoding="utf-8")
code = "\n".join(re.split(r"(?<!\\)%", l, maxsplit=1)[0] for l in tex.splitlines())
# drop verbatim-ish \detokenize{...} and \url{...} content from brace counting noise
errors = []

depth = 0
for i, ch in enumerate(code):
    if ch == "{" and code[i - 1] != "\\":
        depth += 1
    elif ch == "}" and code[i - 1] != "\\":
        depth -= 1
        if depth < 0:
            errors.append(f"unbalanced '}}' near: {code[max(0, i - 60):i + 1]!r}")
            depth = 0
if depth:
    errors.append(f"{depth} unclosed '{{'")

stack = []
for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", code):
    kind, env = m.groups()
    if kind == "begin":
        stack.append(env)
    elif not stack or stack.pop() != env:
        errors.append(f"mismatched \\end{{{env}}} at char {m.start()}")
if stack:
    errors.append(f"unclosed environments: {stack}")

labels = set(re.findall(r"\\label\{([^}]+)\}", code))
for ref in set(re.findall(r"\\ref\{([^}]+)\}", code)):
    if ref not in labels:
        errors.append(f"\\ref{{{ref}}} has no \\label")

keys = set(re.findall(r"@\w+\{([^,\s]+),", bib))
for group in re.findall(r"\\cite[pt]?\{([^}]+)\}", code):
    for k in group.split(","):
        if k.strip() not in keys:
            errors.append(f"\\cite key '{k.strip()}' missing from sn-bibliography.bib")

defined = set(re.findall(r"\\(?:re|provide|new)command\{\\(\w+)\}", code))
for name in set(re.findall(r"\\((?:Vtwo|Base|Delta|Abl|Legacy|Random)\w*)", code)):
    if name not in defined and name not in {"Delta"}:
        errors.append(f"macro \\{name} used but not defined")

if re.search(r"\\input\b|\\include\{", code):
    errors.append("\\input/\\include present (template requires a single file)")

optional = set(re.findall(r"\\IfFileExists\{([^}]+)\}", code))   # guarded: a fallback is drawn instead
for fig in set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", code)) - optional:
    if "#" not in fig and not (build / fig).exists():
        errors.append(f"\\includegraphics file missing: {fig}")

print(f"checked {build / 'sn-article.tex'}: {len(labels)} labels, {len(keys)} bib entries, {len(defined)} macros")
if errors:
    print("\n".join("ERROR: " + e for e in errors))
    sys.exit(1)
print("OK")
