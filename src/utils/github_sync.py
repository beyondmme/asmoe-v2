"""Commit + push experiment results from a Kaggle session back to the GitHub repository.

Authentication: a GitHub token in ``GITHUB_TOKEN`` (fine-grained PAT with *Contents: read and write* on the
repository). It is passed via ``GIT_CONFIG_*`` environment variables, never written to disk or argv.
Concurrent sessions (several folds at once) are handled with pull --rebase + retry; each fold only touches
its own ``results/<exp>/fold_<k>/`` folder, so rebases never conflict.
"""
from __future__ import annotations

import base64
import os
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def git_env(token: Optional[str] = None) -> Dict[str, str]:
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    token = (token or os.environ.get("GITHUB_TOKEN") or "").strip()   # pasted secrets often end in a newline
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.https://github.com/.extraHeader",
                   GIT_CONFIG_VALUE_0=f"Authorization: Basic {basic}")
    return env


def git(args: List[str], cwd: Optional[Path] = None, token: Optional[str] = None, check: bool = True):
    r = subprocess.run(["git", *args], cwd=cwd, env=git_env(token), capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args[:2])} failed: {(r.stderr or r.stdout).strip()[:400]}")
    return r


def clone(repo_url: str, dest: Path, branch: str = "main", token: Optional[str] = None) -> Path:
    dest = Path(dest)
    if (dest / ".git").exists():
        git(["pull", "--rebase", "--autostash", "origin", branch], dest, token)
    else:
        git(["clone", "--branch", branch, repo_url, str(dest)], None, token)
    return dest


def explain(error: str, token: Optional[str] = None) -> str:
    """Turn a git/GitHub error into the fix the user has to make."""
    e = error.lower()
    has_token = bool((token or os.environ.get("GITHUB_TOKEN") or "").strip())
    if "could not read username" in e or "terminal prompts disabled" in e:
        if has_token:   # GitHub answered 401 to the token, so git fell back to asking for a password
            return ("GITHUB_TOKEN was REJECTED by GitHub (mistyped, expired, revoked, or no access to this repo): "
                    "create a new fine-grained token (Repository access = this repo, Contents = Read and write) "
                    "and paste it into the Kaggle secret")
        return ("no token in this session: Kaggle Add-ons -> Secrets -> GITHUB_TOKEN must be TICKED (attached) for "
                "this notebook, then run a new version")
    if "403" in e or "permission to" in e or "denied" in e:
        return ("token works but may not write to this repo: create a fine-grained token with Resource owner = the "
                "repo owner, Repository access = this repo, Permissions -> Contents = Read and write; then update "
                "the Kaggle secret")
    if "401" in e or "authentication failed" in e or "invalid username or password" in e:
        return "token rejected (wrong, revoked or expired): create a new one and update the Kaggle secret"
    if "could not resolve host" in e or "unable to access" in e:
        return "no internet: notebook Settings -> Internet ON"
    if "rejected" in e or "non-fast-forward" in e or "fetch first" in e:
        return "the repo moved on while training; re-run the push (it rebases automatically)"
    return "see the git error above"


def check_push_access(repo_dir: Path, branch: str = "main", token: Optional[str] = None) -> Dict:
    """Dry-run push: proves the token can write to the repo before hours of training are spent."""
    r = git(["push", "--dry-run", "origin", f"HEAD:{branch}"], Path(repo_dir), token, check=False)
    out = (r.stderr or r.stdout).strip()
    behind = any(k in out for k in ("non-fast-forward", "fetch first", "rejected"))
    ok = r.returncode == 0 or behind          # "behind" still means we are allowed to write
    return {"ok": ok, "error": "" if ok else out[-300:], "hint": "" if ok else explain(out, token)}


def push_results(repo_dir: Path, paths: Sequence[str], message: str, branch: str = "main",
                 token: Optional[str] = None, retries: int = 5) -> Dict:
    repo_dir = Path(repo_dir)
    git(["add", "--", *paths], repo_dir, token)
    if not git(["diff", "--cached", "--name-only"], repo_dir, token).stdout.strip():
        return {"committed": False, "pushed": False, "reason": "nothing new"}
    git(["-c", "user.name=asmoe-kaggle-bot", "-c", "user.email=asmoe-kaggle-bot@users.noreply.github.com",
         "commit", "-m", message], repo_dir, token)
    last = ""
    for attempt in range(retries):
        git(["pull", "--rebase", "--autostash", "origin", branch], repo_dir, token, check=False)
        r = git(["push", "origin", f"HEAD:{branch}"], repo_dir, token, check=False)
        if r.returncode == 0:
            return {"committed": True, "pushed": True, "attempts": attempt + 1}
        last = (r.stderr or r.stdout).strip()[:300]
        time.sleep(5 * (attempt + 1))
    return {"committed": True, "pushed": False, "error": last, "hint": explain(last, token)}
