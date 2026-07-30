#!/usr/bin/env python3
"""Validate the skills and plugin manifests in this repo.

Checks:
- every entry in skills/ is a directory containing a SKILL.md
- SKILL.md has YAML frontmatter with `name` matching its directory and a
  non-empty `description`, within Claude Code's limits (64 / 1024 chars)
- every Python code fence in SKILL.md is syntactically valid
- .claude-plugin/marketplace.json and plugin.json are valid and consistent
- every skill is listed in README.md

Run from anywhere: `python3 scripts/validate_skills.py`. Exits non-zero on
the first report of any error. Requires PyYAML.
"""

import ast
import json
import re
import sys
import textwrap
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
MAX_NAME = 64
MAX_DESCRIPTION = 1024
PYTHON_FENCE_RE = re.compile(
    r"^```python[ \t]*\n(?P<source>.*?)^```[ \t]*$",
    re.MULTILINE | re.DOTALL,
)

errors: list[str] = []


def err(msg: str) -> None:
    errors.append(msg)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def parse_frontmatter(path: Path) -> tuple[dict | None, str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or lines[0].strip() != "---":
        err(f"{rel(path)}: missing YAML frontmatter (must start with '---')")
        return None, ""
    close = next((i for i in range(1, len(lines)) if lines[i].strip() == "---"), None)
    if close is None:
        err(f"{rel(path)}: frontmatter never closed with '---'")
        return None, ""
    body = "\n".join(lines[close + 1 :]).strip()
    try:
        data = yaml.safe_load("\n".join(lines[1:close]))
    except yaml.YAMLError as e:
        err(f"{rel(path)}: frontmatter is not valid YAML: {e}")
        return None, body
    if not isinstance(data, dict):
        err(f"{rel(path)}: frontmatter must be a YAML mapping")
        return None, body
    return data, body


def check_python_fences(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    for match in PYTHON_FENCE_RE.finditer(text):
        source = textwrap.dedent(match.group("source"))
        first_source_line = text.count("\n", 0, match.start("source")) + 1
        try:
            ast.parse(source)
        except SyntaxError as exc:
            line = first_source_line + (exc.lineno or 1) - 1
            err(f"{rel(path)}:{line}: invalid Python example: {exc.msg}")


def check_skill(skill_dir: Path) -> None:
    skill_md = skill_dir / "SKILL.md"
    if not skill_md.is_file():
        err(f"{rel(skill_dir)}: missing SKILL.md")
        return
    fm, body = parse_frontmatter(skill_md)
    if fm is None:
        return

    name = fm.get("name")
    if not isinstance(name, str) or not name:
        err(f"{rel(skill_md)}: frontmatter needs a non-empty string `name`")
    else:
        if name != skill_dir.name:
            err(f"{rel(skill_md)}: name '{name}' != directory '{skill_dir.name}'")
        if not NAME_RE.fullmatch(name):
            err(f"{rel(skill_md)}: name '{name}' is not kebab-case")
        if len(name) > MAX_NAME:
            err(f"{rel(skill_md)}: name is {len(name)} chars (max {MAX_NAME})")

    description = fm.get("description")
    if not isinstance(description, str) or not description.strip():
        err(f"{rel(skill_md)}: frontmatter needs a non-empty string `description`")
    elif len(description) > MAX_DESCRIPTION:
        err(
            f"{rel(skill_md)}: description is {len(description)} chars "
            f"(max {MAX_DESCRIPTION})"
        )

    if not body:
        err(f"{rel(skill_md)}: body is empty")

    check_python_fences(skill_md)


def load_json(path: Path) -> dict | None:
    if not path.is_file():
        err(f"{rel(path)}: missing")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        err(f"{rel(path)}: invalid JSON: {e}")
        return None
    if not isinstance(data, dict):
        err(f"{rel(path)}: must be a JSON object")
        return None
    return data


def check_manifests() -> None:
    marketplace = load_json(ROOT / ".claude-plugin" / "marketplace.json")
    plugin = load_json(ROOT / ".claude-plugin" / "plugin.json")

    if marketplace is not None:
        if not marketplace.get("name"):
            err("marketplace.json: missing `name`")
        if not (marketplace.get("owner") or {}).get("name"):
            err("marketplace.json: missing `owner.name`")
        plugins = marketplace.get("plugins")
        if not isinstance(plugins, list) or not plugins:
            err("marketplace.json: `plugins` must be a non-empty list")
            plugins = []
        for entry in plugins:
            if not entry.get("name"):
                err("marketplace.json: plugin entry missing `name`")
            source = entry.get("source")
            if not source:
                err("marketplace.json: plugin entry missing `source`")
            elif not (ROOT / source).is_dir():
                err(f"marketplace.json: plugin source '{source}' does not exist")
        if plugin is not None and plugins:
            root_entries = [e for e in plugins if e.get("source") == "./"]
            for entry in root_entries:
                if entry.get("name") != plugin.get("name"):
                    err(
                        f"plugin.json name '{plugin.get('name')}' != marketplace "
                        f"entry '{entry.get('name')}' for source './'"
                    )

    if plugin is not None and not plugin.get("name"):
        err("plugin.json: missing `name`")


def main() -> int:
    skills_root = ROOT / "skills"
    if not skills_root.is_dir():
        err("skills/ directory is missing")
        skill_dirs = []
    else:
        skill_dirs = sorted(p for p in skills_root.iterdir() if p.is_dir())
        for stray in sorted(p for p in skills_root.iterdir() if not p.is_dir()):
            err(f"{rel(stray)}: stray file — skills/ entries must be directories")
        if not skill_dirs:
            err("skills/ contains no skills")

    for skill_dir in skill_dirs:
        check_skill(skill_dir)

    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for skill_dir in skill_dirs:
        if skill_dir.name not in readme:
            err(f"README.md: does not mention skill '{skill_dir.name}'")

    check_manifests()

    if errors:
        print(f"FAILED — {len(errors)} error(s):")
        for e in errors:
            print(f"  - {e}")
        return 1
    print(f"OK — {len(skill_dirs)} skill(s) validated")
    return 0


if __name__ == "__main__":
    sys.exit(main())
