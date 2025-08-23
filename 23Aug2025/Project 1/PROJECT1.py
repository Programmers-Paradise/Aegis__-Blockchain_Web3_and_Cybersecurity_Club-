"""
PROJECT1.py

AI-powered repository Security Analyzer that combines lightweight static checks
with a **local LLM** (via Ollama) to review source files and produce actionable
findings. Works as a CLI and offers a Streamlit dashboard.

Quickstart
----------
1) Install deps (choose any local model you have in Ollama, e.g. `qwen2.5:7b`):

   pip install streamlit pydantic pygments tomlkit

   # Optional: for Python-specific checks
   pip install bandit

2) Ensure Ollama is running locally (default: http://localhost:11434) and you
   have at least one model pulled, e.g.

   ollama pull qwen2.5:7b

3) CLI usage:

   python security_analyzer.py scan /path/to/repo --model qwen2.5:7b --out report.md

4) Streamlit UI:

   streamlit run PROJECT1.py

Notes
-----
- The analyzer is language-agnostic for baseline checks (regex + heuristics). It
  also runs optional Bandit if Python files are present and Bandit is installed.
- Findings are saved as JSON and Markdown; a SARIF-ish JSON is emitted for CI.
- No internet access is required; the LLM is queried via local Ollama.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import fnmatch
import io
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import requests  # for Ollama REST
except Exception:
    requests = None

try:
    from pydantic import BaseModel
except Exception:
    BaseModel = object  # soft dependency fallback

try:
    import bandit
    from bandit.core import tester as bandit_tester
    from bandit.core import config as bandit_config
    from bandit.core import manager as bandit_manager
except Exception:
    bandit = None

try:
    import streamlit as st
    import matplotlib.pyplot as plt
except Exception:
    st = None

# ------------------------------ Models ------------------------------
SEVERITIES = ["LOW", "MEDIUM", "HIGH", "CRITICAL"]

@dataclass
class Finding:
    file: str
    line: int
    col: int
    rule_id: str
    title: str
    severity: str
    cwe: Optional[str] = None
    excerpt: Optional[str] = None
    advice: Optional[str] = None
    engine: str = "heuristic"  # heuristic | bandit | llm

@dataclass
class FileSummary:
    path: str
    language: str
    loc: int
    sha1: Optional[str] = None

@dataclass
class Report:
    repo_root: str
    model: str
    created_at: str
    files: List[FileSummary] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)

# --------------------------- Helpers / IO ---------------------------
SUPPORTED_GLOBS = [
    "**/*.py", "**/*.js", "**/*.ts", "**/*.tsx", "**/*.jsx",
    "**/*.go", "**/*.rb", "**/*.php", "**/*.java", "**/*.cs",
    "**/*.c", "**/*.cpp", "**/*.rs", "**/*.sh", "**/*.yaml", "**/*.yml",
    "**/*.json", "**/*.sql"
]
DEFAULT_IGNORE = {".git", ".hg", ".svn", ".venv", "node_modules", "dist", "build", "target", ".mypy_cache", ".pytest_cache"}

LANG_BY_EXT = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx",
    ".jsx": "jsx", ".go": "go", ".rb": "ruby", ".php": "php", ".java": "java",
    ".cs": "csharp", ".c": "c", ".cpp": "cpp", ".rs": "rust", ".sh": "bash",
    ".yaml": "yaml", ".yml": "yaml", ".json": "json", ".sql": "sql"
}

SECRET_PATTERNS = [
    (r"AKIA[0-9A-Z]{16}", "AWS Access Key ID"),
    (r"(?i)aws.?secret.?access.?key\s*[:=]\s*['\"]?[A-Za-z0-9/+=]{20,}['\"]?", "AWS Secret Key"),
    (r"(?i)secret_key\s*[:=]", "Generic secret key placeholder"),
    (r"(?i)api[_-]?key\s*[:=]", "Generic API key"),
    (r"(?i)Authorization: Bearer [A-Za-z0-9\-_.]+", "Bearer token inline"),
]

DANGEROUS_PATTERNS = [
    (r"\beval\s*\(", "Use of eval()"),
    (r"\bexec\s*\(", "Use of exec()"),
    (r"subprocess\.(Popen|call|run)\(.*shell\s*=\s*True", "shell=True in subprocess"),
    (r"requests\.[gs]et\(.*verify\s*=\s*False", "requests verify=False"),
    (r"pickle\.(load|loads)\(", "Unsafe pickle deserialization"),
    (r"yaml\.load\(.*\)", "Potential unsafe yaml.load()"),
    (r"MD5\(|sha1\(", "Weak hash MD5/SHA1"),
    (r"SELECT .* FROM .*\+.*|INSERT .*\+.*|UPDATE .*\+.*", "String-concat SQL"),
    (r"\bSystem\.setProperty\(\"javax\.net\.ssl\.trustStore\"", "Custom trust store (Java)")
]

BACKDOOR_MARKERS = [
    (r"//\s*TODO:\s*disable auth in prod", "Prod auth bypass TODO"),
    (r"if\s*\(\s*user\.role\s*==\s*'admin'\s*\)\s*return\s*true;", "Unconditional admin")
]

SQLI_HINTS = [
    (r"(['\"]\s*\+\s*user_input\s*\+\s*['\"])", "Concatenating untrusted input into SQL"),
]

def read_text(path: Path, max_bytes: int = 400_000) -> str:
    try:
        data = path.read_bytes()[:max_bytes]
        return data.decode("utf-8", errors="replace")
    except Exception:
        return ""

def get_local_models():
    import requests
    try:
        resp = requests.get("http://localhost:11434/api/tags").json()
        return [m["name"] for m in resp.get("models", [])]
    except Exception:
        # fallback defaults
        return ["qwen2.5-coder:7b", "deepseek-r1:7b"]


def iter_files(root: Path, globs: List[str] = SUPPORTED_GLOBS, ignore: set = DEFAULT_IGNORE) -> Iterable[Path]:
    for g in globs:
        for p in root.glob(g):
            if any(part in ignore for part in p.parts):
                continue
            if p.is_file():
                yield p


def summarize_file(path: Path) -> FileSummary:
    text = read_text(path)
    loc = text.count("\n") + 1 if text else 0
    lang = LANG_BY_EXT.get(path.suffix.lower(), "unknown")
    return FileSummary(path=str(path), language=lang, loc=loc)


# --------------------------- Heuristic pass --------------------------
HEURISTIC_RULES: List[Tuple[str, str, str, Optional[str], List[Tuple[str, str]]]] = [
    ("SEC001", "Hardcoded secret", "HIGH", "CWE-798", SECRET_PATTERNS),
    ("SEC010", "Dangerous function/param", "HIGH", "CWE-94", DANGEROUS_PATTERNS),
    ("SEC020", "Suspicious backdoor marker", "CRITICAL", "CWE-912", BACKDOOR_MARKERS),
    ("SEC030", "Possible SQL injection", "HIGH", "CWE-89", SQLI_HINTS),
]


def heuristic_scan(path: Path, text: str) -> List[Finding]:
    findings: List[Finding] = []
    for rule_id, title, sev, cwe, patt_list in HEURISTIC_RULES:
        for patt, desc in patt_list:
            for m in re.finditer(patt, text, flags=re.MULTILINE):
                line = text.count("\n", 0, m.start()) + 1
                col = m.start() - (text.rfind("\n", 0, m.start()) + 1)
                excerpt = text[max(0, m.start() - 80): m.end() + 80].replace("\n", " ")
                findings.append(Finding(
                    file=str(path), line=line, col=col, rule_id=rule_id,
                    title=f"{title}: {desc}", severity=sev, cwe=cwe,
                    excerpt=excerpt, advice=None, engine="heuristic"
                ))
    return findings


# ----------------------------- Bandit pass ---------------------------
def run_bandit(paths: List[Path]) -> List[Finding]:
    if bandit is None:
        return []
    cfg = bandit_config.BanditConfig()
    mgr = bandit_manager.BanditManager(cfg, "file")
    for p in paths:
        if p.suffix.lower() == ".py":
            mgr.discover_files([str(p)])
    mgr.run_tests()
    findings: List[Finding] = []
    for issue in mgr.get_issue_list():
        findings.append(Finding(
            file=issue.fname, line=issue.lineno, col=0,
            rule_id=f"BANDIT_{issue.test_id}", title=issue.text,
            severity=str(issue.severity), cwe=None,
            excerpt=None, advice=None, engine="bandit"
        ))
    return findings


# ------------------------------- LLM pass ----------------------------
DEFAULT_SYSTEM_PROMPT = (
    "You are a senior application security engineer. Review code for security, reliability, and production risks. "
    "Return specific, actionable findings: severity (LOW/MEDIUM/HIGH/CRITICAL), CWE if clear, file/line spans, "
    "why it's risky, and a safe fix.")

LLM_USER_TEMPLATE = textwrap.dedent(
    """
    Project context:
    - File path: {file}
    - Language: {language}
    - Lines of code: {loc}

    Known heuristic hits in this file (from a separate static pass):
    {heuristic_summary}

    Now read this code and produce a compact JSON list of findings (array), where each item has:
    "file", "title", "severity", "cwe" (optional), "line_start", "line_end", "explanation", "recommendation".

    Only output JSON. Be conservative with CRITICAL. Prefer concrete line references.
    --- BEGIN CODE ---
    {snippet}
    --- END CODE ---
    """
)


def ollama_generate(model: str, prompt: str, system: Optional[str] = None, base_url: str = "http://localhost:11434") -> str:
    if requests is None:
        raise RuntimeError("The 'requests' package is required for Ollama calls. Install with 'pip install requests'.")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    if system:
        payload["system"] = system
    resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=600)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "")


def llm_scan_file(model: str, fsum: FileSummary, text: str, heuristic_hits: List[Finding]) -> List[Finding]:
    # Truncate very large files to fit typical context; keep head and tail.
    max_chars = 16_000
    if len(text) > max_chars:
        head = text[:12_000]
        tail = text[-3_000:]
        text = head + "\n\n/* ...snip... */\n\n" + tail

    heur_summary = [
        {
            "rule_id": h.rule_id,
            "title": h.title,
            "line": h.line,
            "severity": h.severity
        } for h in heuristic_hits if h.file == fsum.path
    ]
    prompt = LLM_USER_TEMPLATE.format(
        file=fsum.path,
        language=fsum.language,
        loc=fsum.loc,
        heuristic_summary=json.dumps(heur_summary, ensure_ascii=False, indent=2),
        snippet=text
    )
    raw = ollama_generate(model=model, prompt=prompt, system=DEFAULT_SYSTEM_PROMPT)
    # Extract JSON (best effort)
    m = re.search(r"\[.*\]", raw, flags=re.DOTALL)
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    out: List[Finding] = []
    for it in arr:
        out.append(Finding(
            file=it.get("file", fsum.path),
            line=int(it.get("line_start", 1)),
            col=0,
            rule_id="LLM_REVIEW",
            title=it.get("title", "LLM finding"),
            severity=str(it.get("severity", "MEDIUM")).upper(),
            cwe=it.get("cwe"),
            excerpt=None,
            advice=it.get("recommendation"),
            engine="llm",
        ))
    return out


# ------------------------------ Orchestrator -------------------------
def analyze_repo(root: Path, model: str, include: Optional[List[str]] = None, exclude: Optional[List[str]] = None, use_bandit: bool = True, max_workers: int = 4) -> Report:
    root = root.resolve()
    globs = include if include else SUPPORTED_GLOBS
    ignore = DEFAULT_IGNORE | set(exclude or [])

    files = [p for p in iter_files(root, globs, ignore)]
    summaries = [summarize_file(p) for p in files]

    # Heuristic pass
    heur_findings: List[Finding] = []
    for p in files:
        txt = read_text(p)
        heur_findings.extend(heuristic_scan(p, txt))

    # Bandit pass (optional)
    bandit_findings: List[Finding] = run_bandit(files) if use_bandit else []

    # LLM pass in parallel (per file)
    llm_findings: List[Finding] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for s in summaries:
            txt = read_text(Path(s.path))
            hhits = [f for f in heur_findings if f.file == s.path]
            futs.append(ex.submit(llm_scan_file, model, s, txt, hhits))
        for f in concurrent.futures.as_completed(futs):
            try:
                llm_findings.extend(f.result())
            except Exception as e:
                print(f"[llm] error: {e}", file=sys.stderr)

    report = Report(
        repo_root=str(root),
        model=model,
        created_at=dt.datetime.utcnow().isoformat() + "Z",
        files=summaries,
        findings=heur_findings + bandit_findings + llm_findings,
    )
    return report


# ------------------------------ Reporting ---------------------------
def to_markdown(report: Report) -> str:
    sev_order = {s: i for i, s in enumerate(SEVERITIES)}
    sorted_findings = sorted(report.findings, key=lambda f: (sev_order.get(f.severity, 99), f.file, f.line))
    buf = io.StringIO()
    buf.write(f"# Security Analysis Report\n\n")
    buf.write(f"**Repo:** `{report.repo_root}`  ")
    buf.write(f"**Model:** `{report.model}`  ")
    buf.write(f"**Generated:** {report.created_at}\n\n")

    # Summary counts
    counts: Dict[str, int] = {s: 0 for s in SEVERITIES}
    for f in report.findings:
        counts[f.severity] = counts.get(f.severity, 0) + 1
    buf.write("## Summary\n\n")
    for s in SEVERITIES:
        buf.write(f"- **{s}**: {counts.get(s,0)}\n")
    buf.write("\n---\n\n## Findings\n\n")

    for f in sorted_findings:
        buf.write(f"### {f.title} \n")
        buf.write(f"- **Severity:** {f.severity}\n")
        if f.cwe:
            buf.write(f"- **CWE:** {f.cwe}\n")
        buf.write(f"- **File:** `{f.file}` (line {f.line})\n")
        buf.write(f"- **Engine:** {f.engine}\n")
        if f.excerpt:
            buf.write(f"\n```\n{textwrap.shorten(f.excerpt, width=400, placeholder=' ...')}\n```\n")
        if f.advice:
            buf.write(f"\n**Recommendation:** {f.advice}\n")
        buf.write("\n---\n\n")

    return buf.getvalue()


def to_sarif(report: Report) -> Dict:
    # Minimal SARIF-ish structure for CI systems
    runs = [{
        "tool": {"driver": {"name": "AI Security Analyzer", "informationUri": "https://localhost", "version": "0.1"}},
        "results": []
    }]
    for f in report.findings:
        runs[0]["results"].append({
            "ruleId": f.rule_id,
            "level": f.severity.lower(),
            "message": {"text": f.title},
            "locations": [{
                "physicalLocation": {
                    "artifactLocation": {"uri": f.file},
                    "region": {"startLine": f.line}
                }
            }]
        })
    return {"version": "2.1.0", "runs": runs}


# ------------------------------ CLI ---------------------------------
def cmd_scan(args: argparse.Namespace) -> None:
    report = analyze_repo(
        root=Path(args.path),
        model=args.model,
        include=args.include,
        exclude=args.exclude,
        use_bandit=not args.no_bandit,
        max_workers=args.workers,
    )
    out_json = args.json or "report.json"
    out_md = args.out or "report.md"
    out_sarif = args.sarif or "report.sarif.json"

    Path(out_json).write_text(json.dumps(report, default=lambda o: o.__dict__, indent=2))
    Path(out_md).write_text(to_markdown(report), encoding="utf-8")
    Path(out_sarif).write_text(json.dumps(to_sarif(report), indent=2))

    print(f"Saved:\n - {out_md}\n - {out_json}\n - {out_sarif}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="AI-powered Security Analyzer (local LLM + heuristics)")
    sub = p.add_subparsers(dest="cmd")

    s = sub.add_parser("scan", help="Scan a repository")
    s.add_argument("path", type=str, help="Path to repository root")
    s.add_argument("--model", type=str, default="qwen2.5:7b", help="Ollama model name")
    s.add_argument("--include", nargs="*", default=None, help="Glob patterns to include")
    s.add_argument("--exclude", nargs="*", default=[], help="Directory names to ignore")
    s.add_argument("--workers", type=int, default=4, help="Parallel file scans")
    s.add_argument("--no-bandit", action="store_true", help="Disable Bandit for Python files")
    s.add_argument("--out", type=str, default=None, help="Markdown report path")
    s.add_argument("--json", type=str, default=None, help="JSON report path")
    s.add_argument("--sarif", type=str, default=None, help="SARIF-like JSON path")
    s.set_defaults(func=cmd_scan)

    return p


# --------------------------- Streamlit UI ---------------------------
# Run with: streamlit run security_analyzer.py
if __name__ == "__main__" and (len(sys.argv) == 1 or sys.argv[1].startswith("-")):
    # If executed without subcommand, try to run Streamlit app context.
    if st is None:
        print("Streamlit not installed. Install with 'pip install streamlit' or run the CLI: python security_analyzer.py scan <path>")
        sys.exit(0)

    st.set_page_config(page_title="AI Security Analyzer", layout="wide")
    st.title("🛡️ AI Powered Security Analyzer (Local LLM)")

    repo_path = st.text_input("Repository path", value=str(Path.cwd()))
    colA, colB, colC = st.columns(3)
    with colA:
        models = get_local_models()
        model = st.selectbox("Ollama model", options=models, index=0)
    with colB:
        workers = st.slider("Parallel workers", 1, 12, 4)
    with colC:
        use_bandit = st.checkbox("Use Bandit for Python", value=True)

    include = st.multiselect("Include globs", SUPPORTED_GLOBS, default=["**/*.py", "**/*.js", "**/*.ts", "**/*.go", "**/*.java", "**/*.rs", "**/*.cs", "**/*.php", "**/*.sh", "**/*.yaml", "**/*.yml", "**/*.json", "**/*.sql"]) 
    exclude_add = st.text_input("Extra directories to exclude (comma-separated)", value=".venv, node_modules, dist, build")

    run = st.button("🔍 Analyze Repository", type="primary", use_container_width=True)

    if run:
        with st.spinner("Scanning files and calling local LLM (Ollama)..."):
            ex_dirs = [x.strip() for x in exclude_add.split(",") if x.strip()]
            try:
                report = analyze_repo(Path(repo_path), model, include, ex_dirs, use_bandit, workers)
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

        # Summary
        st.subheader("Summary")
        sev_counts = {s: 0 for s in SEVERITIES}
        for f in report.findings:
            sev_counts[f.severity] = sev_counts.get(f.severity, 0) + 1
        cols = st.columns(4)
        for i, s in enumerate(SEVERITIES):
            cols[i].metric(s, sev_counts.get(s, 0))

        # Files table
        st.subheader("Files Scanned")
        st.dataframe([{"path": f.path, "language": f.language, "loc": f.loc} for f in report.files], use_container_width=True)

        # Findings table
        st.subheader("Findings")
        rows = []
        for f in report.findings:
            rows.append({
                "file": f.file, "line": f.line, "severity": f.severity, "rule": f.rule_id, "title": f.title, "engine": f.engine, "cwe": f.cwe or ""
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

        # Save artifacts
        out_dir = Path("reports"); out_dir.mkdir(exist_ok=True)
        md_path = out_dir / "report.md"
        json_path = out_dir / "report.json"
        sarif_path = out_dir / "report.sarif.json"
        Path(md_path).write_text(to_markdown(report), encoding="utf-8")
        Path(json_path).write_text(json.dumps(report, default=lambda o: o.__dict__, indent=2))
        Path(sarif_path).write_text(json.dumps(to_sarif(report), indent=2))

        st.success("Artifacts saved in ./reports")
        st.code(str(md_path))
        st.code(str(json_path))
        st.code(str(sarif_path))

# Standard CLI entry
if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] != "-m":
    parser = build_parser()
    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()
