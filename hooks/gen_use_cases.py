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
    ---

It then appears in the overview table, gets its own docs page, and lands in
the "Use Cases" nav group — no edits to mkdocs.yml or the docs tree required.

The README body is the walkthrough and is inlined into the generated page
rather than linked out to GitHub: its frontmatter and leading ``# Title`` line
are stripped (the generated page supplies its own), and every relative
link/image is rewritten — to another use case's generated page when it points
at that use case's folder or README, otherwise to a GitHub blob/tree URL, since
only ``docs/`` itself is served by the built site.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import yaml
from mkdocs.structure.files import File

log = logging.getLogger("mkdocs.hooks.use_cases")

TABLE_MARKER = "<!-- USE_CASES_TABLE -->"
INDEX_URI = "use-cases/index.md"

_FENCE_RE = re.compile(r"^(```|~~~)")
_H1_RE = re.compile(r"^#\s")
_LINK_RE = re.compile(r"(!?\[[^\]]*\]\()([^()\s]+)(\))")

# Populated in on_config, consumed in on_files / on_page_markdown within the
# same build.  Module-level is fine: each build re-runs on_config first.
_use_cases: list[dict] = []


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


def _strip_frontmatter(text: str) -> str:
    if not text.startswith("---"):
        return text
    parts = text.split("---", 2)
    return parts[2].lstrip("\n") if len(parts) >= 3 else text


def _strip_leading_h1(text: str) -> str:
    lines = text.splitlines()
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    if i >= len(lines) or not _H1_RE.match(lines[i]):
        return text  # no leading title line — leave the body as-is
    i += 1
    while i < len(lines) and not lines[i].strip():
        i += 1
    return "\n".join(lines[i:])


def _rewrite_link_target(target: str, uc: dict, gh: str, uc_names: set[str]) -> str:
    if re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith("#"):
        return target  # absolute URL (incl. mailto:) or a same-page anchor
    path, sep, fragment = target.partition("#")
    repo_root = uc["dir"].parent.parent
    resolved = (uc["dir"] / path).resolve()
    try:
        rel = resolved.relative_to(repo_root)
    except ValueError:
        return target  # escapes the repo entirely — leave it alone
    use_cases_dir = uc["dir"].parent
    if resolved.parent == use_cases_dir and resolved.name in uc_names:
        return f"{resolved.name}.md"  # links at a sibling use case's folder
    if (
        resolved.name == "README.md"
        and resolved.parent.parent == use_cases_dir
        and resolved.parent.name in uc_names
    ):
        return f"{resolved.parent.name}.md"  # links at a sibling's README
    kind = "tree" if resolved.is_dir() else "blob"
    return f"{gh}/{kind}/main/{rel.as_posix()}" + (sep + fragment if sep else "")


def _inline_readme(uc: dict, gh: str, uc_names: set[str]) -> str:
    text = (uc["dir"] / "README.md").read_text(encoding="utf-8")
    text = _strip_leading_h1(_strip_frontmatter(text))
    in_fence = False
    lines = []
    for line in text.splitlines():
        if _FENCE_RE.match(line.strip()):
            in_fence = not in_fence
            lines.append(line)
            continue
        if in_fence:
            lines.append(line)
            continue
        line = _LINK_RE.sub(
            lambda m: m[1] + _rewrite_link_target(m[2], uc, gh, uc_names) + m[3],
            line,
        )
        lines.append(line)
    return "\n".join(lines)


def _discover(config) -> list[dict]:
    uc_dir = Path(config.docs_dir).parent / "use_cases"
    found: list[dict] = []
    if not uc_dir.is_dir():
        return found
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
                "dir": folder.resolve(),
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


def _render_page(uc: dict, repo_url: str, uc_names: set[str]) -> str:
    gh = (repo_url or "").rstrip("/")
    out = [f"# {uc['title']}", "", f"**Domain:** {uc['domain']}", ""]
    if gh:
        out += [
            f"**Source:** [`use_cases/{uc['name']}/`]"
            f"({gh}/tree/main/use_cases/{uc['name']}/)",
            "",
        ]
    out += [uc["summary"], "", _inline_readme(uc, gh, uc_names), ""]
    return "\n".join(out)


def on_files(files, config):
    uc_names = {uc["name"] for uc in _use_cases}
    for uc in _use_cases:
        files.append(
            File.generated(
                config,
                f"use-cases/{uc['name']}.md",
                content=_render_page(uc, config.repo_url, uc_names),
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
