"""MkDocs hook: auto-generate the Use Cases overview table + per-use-case
pages from ``use_cases/<name>/README.md`` frontmatter.

Add a use case by dropping a ``README.md`` with YAML frontmatter into
``use_cases/<name>/``::

    ---
    title: My Use Case
    domain: Field — short descriptor          # one cell in the overview table
    summary: One or two sentences shown in the overview table and page.
    status: published                         # omit, or 'draft' to hide it
    order: 10                                 # optional sort key (default 100)
    guides:                                   # optional walkthrough links
      - text: Walkthrough
        path: demo.md                         # path within the use-case folder
    ---

It then appears in the overview table, gets its own docs page, and lands in
the "Use Cases" nav group — no edits to mkdocs.yml or the docs tree required.

Demo-bundle links (a downloadable ``.tar.gz`` snapshot of the use case's
folder, hosted on Google Drive) come from ``use_cases/links.csv`` (columns:
folder name, URL) keyed by folder name, and are rendered only on the
generated docs page (alongside the GitHub source link) — READMEs carry no
hardcoded copy, so there's nothing to keep in sync.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import yaml
from mkdocs.structure.files import File

log = logging.getLogger("mkdocs.hooks.use_cases")

TABLE_MARKER = "<!-- USE_CASES_TABLE -->"
INDEX_URI = "use-cases/index.md"

# Populated in on_config, consumed in on_files / on_page_markdown within the
# same build.  Module-level is fine: each build re-runs on_config first.
_use_cases: list[dict] = []


def _load_demo_urls(uc_dir: Path) -> dict[str, str]:
    csv_path = uc_dir / "links.csv"
    if not csv_path.is_file():
        return {}
    with csv_path.open(encoding="utf-8", newline="") as f:
        return {
            name.strip(): url.strip() for name, url in csv.reader(f) if name.strip()
        }


def _parse_frontmatter(text: str) -> dict | None:
    if not text.startswith("---"):
        return None
    parts = text.split("---", 2)
    if len(parts) < 3:
        return None
    try:
        data = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return None
    return data if isinstance(data, dict) else None


def _discover(config) -> list[dict]:
    uc_dir = Path(config.docs_dir).parent / "use_cases"
    found: list[dict] = []
    if not uc_dir.is_dir():
        return found
    demo_urls = _load_demo_urls(uc_dir)
    for folder in sorted(p for p in uc_dir.iterdir() if p.is_dir()):
        readme = folder / "README.md"
        if not readme.is_file():
            continue
        fm = _parse_frontmatter(readme.read_text(encoding="utf-8"))
        if not fm:
            continue
        if str(fm.get("status", "published")).lower() == "draft":
            continue
        missing = [k for k in ("title", "domain", "summary") if not fm.get(k)]
        if missing:
            log.warning(
                "use_cases/%s/README.md missing frontmatter %s — skipped",
                folder.name,
                missing,
            )
            continue
        found.append(
            {
                "name": folder.name,
                "title": str(fm["title"]).strip(),
                "domain": str(fm["domain"]).strip(),
                "summary": " ".join(str(fm["summary"]).split()),
                "order": fm.get("order", 100),
                "guides": fm.get("guides") or [],
                "demo_url": demo_urls.get(folder.name),
            }
        )
    found.sort(key=lambda u: (u["order"], u["title"].lower()))
    return found


def on_config(config):
    global _use_cases
    _use_cases = _discover(config)
    log.info("use_cases: discovered %d published use case(s)", len(_use_cases))

    # Append a nav entry per use case under the existing "Use Cases" group.
    for item in config.nav or []:
        if isinstance(item, dict) and isinstance(item.get("Use Cases"), list):
            children = item["Use Cases"]
            present = {next(iter(c.values())) for c in children if isinstance(c, dict)}
            for uc in _use_cases:
                uri = f"use-cases/{uc['name']}.md"
                if uri not in present:
                    children.append({uc["title"]: uri})
    return config


def _render_page(uc: dict, repo_url: str) -> str:
    gh = (repo_url or "").rstrip("/")
    out = [f"# {uc['title']}", "", f"**Domain:** {uc['domain']}", ""]
    if gh:
        out += [
            f"**Source:** [`use_cases/{uc['name']}/`]"
            f"({gh}/tree/main/use_cases/{uc['name']}/)",
            "",
        ]
    if uc["demo_url"]:
        out += [f"**Demo bundle:** [Download `.tar.gz`]({uc['demo_url']})", ""]
    out += [uc["summary"], ""]
    guides = [g for g in uc["guides"] if isinstance(g, dict)]
    if guides:
        out += ["## Guides", ""]
        for g in guides:
            path = g.get("path", "")
            text = g.get("text") or path
            if gh and path:
                out.append(f"- [{text}]({gh}/blob/main/use_cases/{uc['name']}/{path})")
            elif text:
                out.append(f"- {text}")
        out.append("")
    return "\n".join(out)


def on_files(files, config):
    for uc in _use_cases:
        files.append(
            File.generated(
                config,
                f"use-cases/{uc['name']}.md",
                content=_render_page(uc, config.repo_url),
            )
        )
    return files


def _render_table(use_cases: list[dict]) -> str:
    if not use_cases:
        return "_No published use cases yet._"
    rows = ["| Use case | Domain | Summary |", "|----------|--------|---------|"]
    for uc in use_cases:
        domain = uc["domain"].replace("|", "\\|")
        summary = uc["summary"].replace("|", "\\|")
        rows.append(f"| [{uc['title']}]({uc['name']}.md) | {domain} | {summary} |")
    return "\n".join(rows)


def on_page_markdown(markdown, page, config, files):
    if page.file.src_uri != INDEX_URI:
        return markdown
    table = _render_table(_use_cases)
    if TABLE_MARKER in markdown:
        return markdown.replace(TABLE_MARKER, table)
    return f"{markdown}\n\n{table}"
