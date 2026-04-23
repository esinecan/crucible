"""CLI entry point: python -m crucible <command>"""
from __future__ import annotations

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="crucible", description="Universal knowledge base builder"
    )
    sub = parser.add_subparsers(dest="command")

    ing = sub.add_parser("ingest", help="Ingest a directory into the graph")
    ing.add_argument("path", help="Source directory")
    ing.add_argument("--name", required=True, help="Corpus name")
    ing.add_argument("--description", default="")

    exp = sub.add_parser("explore", help="Run the insight engine")
    exp.add_argument("--corpus", default="", help="Corpus ID (optional)")
    exp.add_argument("--max", type=int, default=5, help="Max insights per cycle")
    exp.add_argument("--cycles", type=int, default=1, help="Number of cycles")
    exp.add_argument("--strategy", default="", help="Force strategy (bridge/outlier/hub/meta)")
    exp.add_argument("--no-eval", action="store_true", help="Disable adversarial evaluation (heuristic scores only)")
    exp.add_argument("--dry-run", action="store_true", help="Run strategies + eval, write files, skip Neo4j writes")
    exp.add_argument("--no-persist", action="store_true", help="Skip Neo4j writes")
    exp.add_argument("--output-dir", default="", help="Custom output directory for cycle artifacts")

    ans = sub.add_parser("answer", help="MCTS reasoning over the knowledge graph")
    ans.add_argument("query", help="Question to answer")
    ans.add_argument("--iterations", type=int, default=0, help="Max MCTS iterations (0=config default)")
    ans.add_argument("--depth", type=int, default=0, help="Max tree depth (0=config default)")
    ans.add_argument("--feedback", action="store_true", help="Feed answer scores back to insight bandit")
    ans.add_argument("--corpus", default="", help="Corpus name or id; routes action-bandit state per corpus (legacy: empty = CWD reward log)")

    band = sub.add_parser("bandit", help="Show Thompson Sampling posteriors")
    band.add_argument("--corpus", default="", help="Show per-corpus posteriors (default: legacy CWD bandit)")
    sub.add_parser("embed-insights", help="Backfill embeddings on existing insights")

    reemb = sub.add_parser("re-embed", help="Re-embed corpus with current backend")
    reemb.add_argument("--corpus", required=True, help="Corpus ID")
    reemb.add_argument("--include-insights", action="store_true", help="Also re-embed insights")
    reemb.add_argument("--include-entities", action="store_true", help="Also re-embed entities")
    reemb.add_argument("--force", action="store_true", help="Re-embed all (default: only items with missing embeddings)")
    reemb.add_argument("--batch-api", action="store_true", help="Use Gemini Batch API (50%% cheaper, async)")
    reemb.add_argument("--poll", type=int, default=30, help="Poll interval for batch API")

    onto = sub.add_parser("ontology", help="Generate ontology from corpus insights")
    onto.add_argument("--corpus", required=True, help="Corpus ID")
    onto.add_argument("--output", default="", help="Save ontology JSON to file")

    ext = sub.add_parser("extract", help="Extract entities using ontology")
    ext.add_argument("--corpus", required=True, help="Corpus ID")
    ext.add_argument("--ontology", default="", help="Ontology JSON file (generates if absent)")
    ext.add_argument("--batch", type=int, default=50, help="Chunks per batch")
    ext.add_argument("--limit", type=int, default=0, help="Max chunks to process (0=all)")
    ext.add_argument("--no-embed", action="store_true", help="Skip entity embedding")
    ext.add_argument("--workers", type=int, default=10, help="Concurrent extraction threads")

    rate_cmd = sub.add_parser("rate", help="Rate an insight (human feedback)")
    rate_cmd.add_argument("insight_id", help="Insight ID")
    rate_cmd.add_argument("score", type=float, help="Score 0.0-1.0")

    syn = sub.add_parser(
        "synthesize-entities",
        help="LLM-merge per-chunk Mentions into canonical Entity descriptions",
    )
    syn.add_argument("--corpus", required=True, help="Corpus ID")
    syn.add_argument("--limit", type=int, default=0, help="Max entities to consider (0=all dirty)")
    syn.add_argument("--max-calls", type=int, default=0, help="Override max LLM calls (0=config default)")

    sub.add_parser("stats", help="Show graph statistics")
    sub.add_parser("serve", help="Start MCP server")

    args = parser.parse_args()

    if args.command == "ingest":
        from .ingestion.pipeline import ingest_directory

        result = ingest_directory(
            args.path, args.name, description=args.description
        )
        print(f"\nIngestion complete: {result}")

    elif args.command == "explore":
        from .config import Config
        from .graph.client import CrucibleGraph
        from .insight.engine import InsightEngine

        config = Config()
        g = CrucibleGraph(config)
        do_persist = not (args.dry_run or args.no_persist)
        # Resolve --corpus to a canonical id so the engine can route bandit
        # state to <state_dir>/<corpus_id>/ instead of inheriting the global
        # CWD reward log. Empty --corpus means cross-corpus run; falls back
        # to legacy paths.
        corpus_id_for_state = ""
        if args.corpus:
            try:
                corpus_id_for_state = g.resolve_corpus_id(args.corpus)
            except ValueError:
                corpus_id_for_state = args.corpus  # opaque pass-through; engine still uses it for path
        engine = InsightEngine(
            config, g,
            use_evaluator=not args.no_eval,
            corpus_id=corpus_id_for_state,
        )
        if args.no_eval:
            print("Evaluation disabled (heuristic mode)")
        else:
            print(f"Adversarial evaluation enabled (model: {config.eval_model})")
        if args.dry_run:
            print("Dry-run mode: files written, Neo4j writes skipped")
        total = 0
        for c in range(args.cycles):
            strat, insights, cycle_dir = engine.run_cycle(
                # Use resolved id so persisted Insight nodes have the canonical
                # hashed corpus_id; the historical fallback (raw user input)
                # left insights with un-resolvable corpus_id values that never
                # matched downstream queries.
                corpus_id=corpus_id_for_state or None,
                max_insights=args.max,
                strategy=args.strategy or None,
                output_dir=args.output_dir or None,
                persist=do_persist,
            )
            total += len(insights)
            print(f"\n--- Cycle {c+1} [{strat}] → {len(insights)} insights ---")
            if insights:
                print(f"  artifacts: {cycle_dir}")
            for ins in insights:
                print(f"\n{'=' * 60}")
                print(f"[{ins.strategy}] score={ins.score:.3f}")
                print(ins.text[:300])
        if total == 0:
            print("No insights discovered.")
        else:
            print(f"\n{total} insights across {args.cycles} cycles.")
            post = engine.bandit.posteriors()
            print("\nBandit posteriors:")
            for arm, p in sorted(post.items(), key=lambda x: x[1]["mean"], reverse=True):
                print(f"  {arm:8s}  mean={p['mean']:.3f}  α={p['alpha']:.0f} β={p['beta']:.0f}  obs={p['observations']:.0f}")
        g.close()

    elif args.command == "answer":
        from .config import Config
        from .graph.client import CrucibleGraph
        from .reasoning.mcts import MCTSEngine

        config = Config()
        g = CrucibleGraph(config)
        # Resolve --corpus to canonical id for action-bandit state routing.
        # Empty --corpus keeps the legacy CWD-based answer_rewards.jsonl
        # behavior so existing scripts and trees stay readable.
        corpus_id_for_state = ""
        if args.corpus:
            try:
                corpus_id_for_state = g.resolve_corpus_id(args.corpus)
            except ValueError:
                corpus_id_for_state = args.corpus
        engine = MCTSEngine(config, g, corpus_id=corpus_id_for_state)
        print(f"MCTS search: \"{args.query}\"")
        print(f"  max_iterations={args.iterations or config.mcts_max_iterations}, "
              f"max_depth={args.depth or config.mcts_max_depth}, "
              f"uct_c={config.mcts_uct_c}")
        print()
        tree = engine.search(
            args.query,
            max_iterations=args.iterations or None,
            max_depth=args.depth or None,
        )
        print()
        print(tree.summary())

        if args.feedback:
            from .insight.engine import InsightEngine

            ie = InsightEngine(config, g, use_evaluator=False)
            rewarded = ie.feedback_from_answer(tree.evidence_ids, tree.best_score)
            if rewarded:
                print(f"\nFeedback: {rewarded} insights rewarded (bonus from answer score {tree.best_score:.3f})")
        g.close()

    elif args.command == "re-embed":
        import logging
        import sys

        from tqdm import tqdm

        from .config import Config
        from .embeddings import embed_batch
        from .graph.client import CrucibleGraph

        logging.basicConfig(level=logging.INFO, format="%(message)s")
        config = Config()
        g = CrucibleGraph(config)
        args.corpus = g.resolve_corpus_id(args.corpus)

        if args.batch_api:
            from .embeddings import batch_embed_via_api

            items: list[tuple[str, str]] = []
            chunks = g.get_all_chunk_texts(args.corpus)
            for ch in chunks:
                items.append((f"chunk:{ch['id']}", ch["text"]))
            print(f"Chunks: {len(chunks)}")

            if args.include_insights:
                with g._driver.session() as s:
                    insights = [dict(r) for r in s.run(
                        "MATCH (i:Insight) WHERE i.corpus_id = $cid OR i.corpus_id IS NULL OR i.corpus_id = '' "
                        "RETURN i.id AS id, i.text AS text", cid=args.corpus)]
                for ins in insights:
                    items.append((f"insight:{ins['id']}", ins["text"]))
                print(f"Insights: {len(insights)}")

            if args.include_entities:
                with g._driver.session() as s:
                    entities = [dict(r) for r in s.run(
                        "MATCH (e:Entity {corpus_id: $cid}) RETURN e.id AS id, e.name AS name, e.description AS desc",
                        cid=args.corpus)]
                for ent in entities:
                    items.append((f"entity:{ent['id']}", f"{ent['name']}: {ent['desc']}"))
                print(f"Entities: {len(entities)}")

            print(f"Total: {len(items)}")
            results = batch_embed_via_api(items, config, poll_interval=args.poll)

            chunk_updates, insight_updates, entity_updates = [], [], []
            for key, emb in results.items():
                kind, node_id = key.split(":", 1)
                if kind == "chunk":
                    chunk_updates.append((node_id, emb))
                elif kind == "insight":
                    insight_updates.append((node_id, emb))
                elif kind == "entity":
                    entity_updates.append((node_id, emb))

            if chunk_updates:
                for i in range(0, len(chunk_updates), 500):
                    g.set_chunk_embeddings(chunk_updates[i: i + 500])
                print(f"Updated {len(chunk_updates)} chunk embeddings")
            if insight_updates:
                g.set_insight_embeddings(insight_updates)
                print(f"Updated {len(insight_updates)} insight embeddings")
            if entity_updates:
                g.set_entity_embeddings(entity_updates)
                print(f"Updated {len(entity_updates)} entity embeddings")
        else:
            # Sync mode: batches of 20 via embed_content
            batch_size = 20
            only_missing = not args.force
            chunks = g.get_all_chunk_texts(args.corpus, only_missing=only_missing)
            print(f"Chunks to embed: {len(chunks)}{' (missing only)' if only_missing else ' (all)'}", flush=True)
            if chunks:
                for i in tqdm(range(0, len(chunks), batch_size), desc="Chunks", file=sys.stderr):
                    batch = chunks[i: i + batch_size]
                    texts = [ch["text"] for ch in batch]
                    embeddings = embed_batch(texts, config, batch_size=batch_size)
                    g.set_chunk_embeddings([(ch["id"], emb) for ch, emb in zip(batch, embeddings)])

            if args.include_insights:
                if only_missing:
                    insights = g.get_unembedded_insights(batch_size=999)
                else:
                    with g._driver.session() as s:
                        insights = [dict(r) for r in s.run(
                            "MATCH (i:Insight) WHERE i.corpus_id = $cid OR i.corpus_id IS NULL OR i.corpus_id = '' "
                            "RETURN i.id AS id, i.text AS text", cid=args.corpus)]
                print(f"Insights to embed: {len(insights)}", flush=True)
                if insights:
                    texts = [ins["text"] for ins in insights]
                    embeddings = embed_batch(texts, config, batch_size=batch_size)
                    g.set_insight_embeddings([(ins["id"], emb) for ins, emb in zip(insights, embeddings)])

            if args.include_entities:
                if only_missing:
                    entities = g.get_unembedded_entities(args.corpus)
                else:
                    with g._driver.session() as s:
                        entities = [dict(r) for r in s.run(
                            "MATCH (e:Entity {corpus_id: $cid}) RETURN e.id AS id, e.name AS name, e.description AS desc",
                            cid=args.corpus)]
                print(f"Entities to embed: {len(entities)}", flush=True)
                if entities:
                    texts = [f"{ent['name']}: {ent['desc']}" for ent in entities]
                    embeddings = embed_batch(texts, config, batch_size=batch_size)
                    g.set_entity_embeddings([(ent["id"], emb) for ent, emb in zip(entities, embeddings)])

        print("Done.", flush=True)
        g.close()

    elif args.command == "ontology":
        from pathlib import Path

        from .config import Config
        from .graph.client import CrucibleGraph
        from .extraction.ontology import OntologyGenerator

        config = Config()
        g = CrucibleGraph(config)
        args.corpus = g.resolve_corpus_id(args.corpus)
        gen = OntologyGenerator(config, g)
        ontology = gen.generate(args.corpus)
        print(f"Generated ontology: {len(ontology.classes)} classes, {len(ontology.relations)} relations\n")
        print("Classes:")
        for c in ontology.classes:
            parent = f" (-> {c.parent})" if c.parent else ""
            print(f"  {c.name}{parent}: {c.description}")
        print("\nRelations:")
        for r in ontology.relations:
            print(f"  {r.name}: {r.domain} -> {r.range} ({r.description})")
        if args.output:
            Path(args.output).write_text(ontology.to_json())
            print(f"\nSaved to {args.output}")
        g.close()

    elif args.command == "extract":
        from pathlib import Path

        from .config import Config
        from .graph.client import CrucibleGraph
        from .extraction.extractor import EntityExtractor
        from .extraction.ontology import Ontology, OntologyGenerator

        config = Config()
        g = CrucibleGraph(config)
        args.corpus = g.resolve_corpus_id(args.corpus)

        if args.ontology:
            ontology = Ontology.from_json(Path(args.ontology).read_text())
            print(f"Loaded ontology: {len(ontology.classes)} classes, {len(ontology.relations)} relations")
        else:
            print("No ontology file provided, generating from insights...")
            gen = OntologyGenerator(config, g)
            ontology = gen.generate(args.corpus)
            print(f"Generated: {len(ontology.classes)} classes, {len(ontology.relations)} relations")

        extractor = EntityExtractor(config, g, ontology)
        limit_msg = f" (limit: {args.limit})" if args.limit else ""
        print(f"\nExtracting entities from corpus {args.corpus}{limit_msg}...")
        stats = extractor.extract_corpus(
            args.corpus,
            batch_size=args.batch,
            embed=not args.no_embed,
            limit=args.limit,
            workers=args.workers,
        )
        print(f"\nExtraction complete:")
        print(f"  Chunks processed: {stats['chunks']}")
        print(f"  Entities created: {stats['entities']}")
        print(f"  Relations created: {stats['relations']}")
        print(f"  Errors: {stats['errors']}")
        es = g.entity_stats()
        print(f"\nGraph totals: {es.get('entities', 0)} entities, {es.get('relations', 0)} relations")
        g.close()

    elif args.command == "rate":
        from .config import Config
        from .graph.client import CrucibleGraph
        from .insight.engine import InsightEngine

        config = Config()
        g = CrucibleGraph(config)
        engine = InsightEngine(config, g, use_evaluator=False)
        arm = engine.apply_human_feedback(args.insight_id, args.score)
        if arm:
            print(f"Feedback applied: {arm} strategy, score={args.score} (3x weight)")
            post = engine.bandit.posteriors()
            print("\nUpdated posteriors:")
            for a, p in sorted(post.items(), key=lambda x: x[1]["mean"], reverse=True):
                print(f"  {a:14s}  mean={p['mean']:.3f}  α={p['alpha']:.0f} β={p['beta']:.0f}  obs={p['observations']:.0f}")
        else:
            print(f"Insight not found: {args.insight_id}")
        g.close()

    elif args.command == "synthesize-entities":
        import logging

        from .config import Config
        from .extraction.synthesize import EntitySynthesizer
        from .graph.client import CrucibleGraph

        logging.basicConfig(level=logging.INFO, format="%(message)s")
        config = Config()
        g = CrucibleGraph(config)
        args.corpus = g.resolve_corpus_id(args.corpus)
        synth = EntitySynthesizer(config, g)
        max_calls = args.max_calls if args.max_calls else None
        stats = synth.synthesize_dirty(
            args.corpus, limit=args.limit, max_calls=max_calls,
        )
        print(f"Dirty entities found:  {stats['dirty_total']}")
        print(f"Synthesized:           {stats['synthesized']}")
        print(f"Re-embedded:           {stats['embedded']}")
        print(f"Errors:                {stats['errors']}")
        if stats["remaining"]:
            print(f"Remaining (budget hit): {stats['remaining']}")
        g.close()

    elif args.command == "stats":
        from .config import Config
        from .graph.client import CrucibleGraph

        config = Config()
        g = CrucibleGraph(config)
        s = g.stats()
        print(f"Corpora:     {s.get('corpora', 0)}")
        print(f"Documents:   {s.get('docs', 0)}")
        print(f"Chunks:      {s.get('chunks', 0)}")
        print(f"Insights L1: {s.get('insights_l1', 0)}")
        print(f"Insights L2: {s.get('insights_l2', 0)}")
        es = g.entity_stats()
        if es.get("entities", 0) > 0:
            print(f"Entities:    {es['entities']}")
            print(f"Relations:   {es['relations']}")
        g.close()

    elif args.command == "embed-insights":
        from .config import Config
        from .graph.client import CrucibleGraph
        from .insight.engine import InsightEngine

        config = Config()
        g = CrucibleGraph(config)
        engine = InsightEngine(config, g)
        # Set layer=1 on all existing insights that don't have a layer
        fixed = g.set_insight_layers(1)
        if fixed:
            print(f"Set layer=1 on {fixed} existing insights")
        count = engine.embed_insights()
        print(f"Embedded {count} insights")
        g.close()

    elif args.command == "bandit":
        from .config import Config
        from .graph.client import CrucibleGraph
        from .insight.engine import InsightEngine

        config = Config()
        g = CrucibleGraph(config)
        corpus_id_for_state = ""
        if args.corpus:
            try:
                corpus_id_for_state = g.resolve_corpus_id(args.corpus)
            except ValueError:
                corpus_id_for_state = args.corpus
        engine = InsightEngine(config, g, corpus_id=corpus_id_for_state)
        scope = f"corpus={corpus_id_for_state}" if corpus_id_for_state else "legacy/CWD"
        post = engine.bandit.posteriors()
        print(f"Thompson Sampling posteriors ({scope}):")
        for arm, p in sorted(post.items(), key=lambda x: x[1]["mean"], reverse=True):
            print(f"  {arm:8s}  mean={p['mean']:.3f}  α={p['alpha']:.0f} β={p['beta']:.0f}  obs={p['observations']:.0f}")
        g.close()

    elif args.command == "serve":
        from .server import main as serve

        serve()

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
