from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tqdm import tqdm

from ..config import Config
from ..embeddings import embed_batch
from ..graph.client import CrucibleGraph
from ..models import Chunk, Corpus, Document
from .chunker import chunk_sections
from .parsers import PARSERS, parse_file


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _make_id(*parts: str) -> str:
    raw = ":".join(parts)
    return hashlib.sha256(raw.encode()).hexdigest()[:20]


def ingest_directory(
    source_path: str,
    corpus_name: str,
    config: Config | None = None,
    description: str = "",
    embed_batch_size: int = 100,
) -> dict:
    """Ingest a directory into the knowledge graph. Resumable via checkpoint file."""
    config = config or Config()
    graph = CrucibleGraph(config)

    corpus_id = _make_id(corpus_name)
    graph.upsert_corpus(
        Corpus(
            id=corpus_id,
            name=corpus_name,
            description=description,
            source_path=source_path,
        )
    )

    source = Path(source_path)
    progress_file = source / ".crucible_progress.json"
    done: set[str] = set()
    if progress_file.exists():
        done = set(json.loads(progress_file.read_text()))

    files = sorted(
        f
        for f in source.rglob("*")
        if f.is_file()
        and f.suffix.lower() in PARSERS
        and not f.name.startswith(".")
        and ".git" not in f.parts
        and str(f) not in done
    )

    stats = {"documents": 0, "chunks": 0, "skipped": 0, "errors": 0}

    for fpath in tqdm(files, desc=f"Ingesting {corpus_name}"):
        try:
            rel_path = str(fpath.relative_to(source))
            doc_id = _make_id(corpus_id, rel_path)

            sections = parse_file(fpath)
            if not sections:
                stats["skipped"] += 1
                continue

            chunk_datas = chunk_sections(
                sections,
                max_chars=config.chunk_max_chars,
                overlap_chars=config.chunk_overlap_chars,
            )
            if not chunk_datas:
                stats["skipped"] += 1
                continue

            texts = [cd.text for cd in chunk_datas]
            embeddings = embed_batch(texts, config, batch_size=embed_batch_size)

            doc = Document(
                id=doc_id,
                corpus_id=corpus_id,
                path=rel_path,
                title=fpath.stem,
                source_type=fpath.suffix.lstrip("."),
                content_hash=_file_hash(fpath),
            )
            chunks = [
                Chunk(
                    id=_make_id(doc_id, str(cd.position)),
                    document_id=doc_id,
                    text=cd.text,
                    position=cd.position,
                    heading=cd.heading,
                    embedding=embeddings[i],
                    metadata=cd.metadata,
                )
                for i, cd in enumerate(chunk_datas)
            ]

            graph.upsert_document(doc)
            graph.upsert_chunks(chunks)
            graph.link_sequential(doc_id)

            stats["documents"] += 1
            stats["chunks"] += len(chunks)

            done.add(str(fpath))
            if stats["documents"] % 10 == 0:
                progress_file.write_text(json.dumps(sorted(done)))

        except Exception as e:
            print(f"  Error: {fpath.name}: {e}")
            stats["errors"] += 1

    progress_file.write_text(json.dumps(sorted(done)))
    graph.close()
    return stats
