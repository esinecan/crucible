from __future__ import annotations

from dataclasses import dataclass, field

from .parsers import Section


@dataclass
class ChunkData:
    text: str
    heading: str
    position: int
    metadata: dict = field(default_factory=dict)


def chunk_sections(
    sections: list[Section],
    max_chars: int = 2000,
    overlap_chars: int = 200,
) -> list[ChunkData]:
    """Split sections into chunks. Small sections pass through; large ones get windowed."""
    chunks: list[ChunkData] = []
    global_pos = 0

    for section in sections:
        text = section.text
        if not text.strip():
            continue

        if len(text) <= max_chars:
            chunks.append(
                ChunkData(
                    text=text,
                    heading=section.heading,
                    position=global_pos,
                    metadata=section.metadata,
                )
            )
            global_pos += 1
        else:
            start = 0
            while start < len(text):
                end = min(start + max_chars, len(text))
                if end < len(text):
                    search_start = end - max_chars // 5
                    last_break = text.rfind("\n", search_start, end)
                    if last_break == -1:
                        last_break = text.rfind(". ", search_start, end)
                    if last_break > search_start:
                        end = last_break + 1

                chunk_text = text[start:end].strip()
                if chunk_text:
                    chunks.append(
                        ChunkData(
                            text=chunk_text,
                            heading=section.heading,
                            position=global_pos,
                            metadata=section.metadata,
                        )
                    )
                    global_pos += 1

                start = end - overlap_chars if end < len(text) else end

    return chunks
