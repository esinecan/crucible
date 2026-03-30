from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Section:
    """A logical section of a document, pre-chunking."""

    heading: str
    text: str
    position: int
    metadata: dict = field(default_factory=dict)


def parse_markdown(path: Path) -> list[Section]:
    content = path.read_text(encoding="utf-8", errors="replace")
    parts = re.split(r"^(#{1,6}\s+.+)$", content, flags=re.MULTILINE)

    sections: list[Section] = []
    current_heading = path.stem
    current_text = ""
    pos = 0

    for part in parts:
        if re.match(r"^#{1,6}\s+", part):
            if current_text.strip():
                sections.append(
                    Section(
                        heading=current_heading,
                        text=current_text.strip(),
                        position=pos,
                    )
                )
                pos += 1
            current_heading = part.strip().lstrip("#").strip()
            current_text = ""
        else:
            current_text += part

    if current_text.strip():
        sections.append(
            Section(heading=current_heading, text=current_text.strip(), position=pos)
        )

    return sections or [
        Section(heading=path.stem, text=content.strip(), position=0)
    ]


def parse_json_file(path: Path) -> list[Section]:
    content = path.read_text(encoding="utf-8", errors="replace")
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return [Section(heading=path.stem, text=content, position=0)]

    sections: list[Section] = []
    if isinstance(data, dict):
        for pos, (key, value) in enumerate(data.items()):
            text = (
                json.dumps(value, indent=2, ensure_ascii=False)
                if not isinstance(value, str)
                else value
            )
            sections.append(Section(heading=str(key), text=text, position=pos))
    elif isinstance(data, list):
        for pos, item in enumerate(data):
            text = (
                json.dumps(item, indent=2, ensure_ascii=False)
                if not isinstance(item, str)
                else item
            )
            heading = (
                str(
                    item.get("name", item.get("id", item.get("title", f"item_{pos}")))
                )
                if isinstance(item, dict)
                else f"item_{pos}"
            )
            sections.append(Section(heading=heading, text=text, position=pos))
    else:
        sections.append(Section(heading=path.stem, text=str(data), position=0))

    return sections


def parse_text(path: Path) -> list[Section]:
    content = path.read_text(encoding="utf-8", errors="replace")
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    if not paragraphs:
        return [Section(heading=path.stem, text=content.strip(), position=0)]

    return [
        Section(heading=path.stem, text=p, position=i)
        for i, p in enumerate(paragraphs)
    ]


PARSERS: dict[str, callable] = {
    ".md": parse_markdown,
    ".markdown": parse_markdown,
    ".json": parse_json_file,
    ".txt": parse_text,
    ".text": parse_text,
    ".csv": parse_text,
    ".yaml": parse_text,
    ".yml": parse_text,
    ".sh": parse_text,
    ".bash": parse_text,
    ".ts": parse_text,
    ".js": parse_text,
    ".py": parse_text,
}


def parse_file(path: Path) -> list[Section]:
    parser = PARSERS.get(path.suffix.lower(), parse_text)
    return parser(path)
