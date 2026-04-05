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
    exp.add_argument("--eval", action="store_true", help="Enable adversarial evaluation")

    ans = sub.add_parser("answer", help="MCTS reasoning over the knowledge graph")
    ans.add_argument("query", help="Question to answer")
    ans.add_argument("--iterations", type=int, default=0, help="Max MCTS iterations (0=config default)")
    ans.add_argument("--depth", type=int, default=0, help="Max tree depth (0=config default)")
    ans.add_argument("--feedback", action="store_true", help="Feed answer scores back to insight bandit")

    sub.add_parser("bandit", help="Show Thompson Sampling posteriors")
    sub.add_parser("embed-insights", help="Backfill embeddings on existing insights")

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
        engine = InsightEngine(config, g, use_evaluator=args.eval)
        if args.eval:
            print(f"Adversarial evaluation enabled (model: {config.eval_model})")
        total = 0
        for c in range(args.cycles):
            strat, insights = engine.run_cycle(
                corpus_id=args.corpus or None,
                max_insights=args.max,
                strategy=args.strategy or None,
            )
            total += len(insights)
            print(f"\n--- Cycle {c+1} [{strat}] → {len(insights)} insights ---")
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
        engine = MCTSEngine(config, g)
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

            ie = InsightEngine(config, g)
            rewarded = ie.feedback_from_answer(tree.evidence_ids, tree.best_score)
            if rewarded:
                print(f"\nFeedback: {rewarded} insights rewarded (bonus from answer score {tree.best_score:.3f})")
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
        engine = InsightEngine(config, g)
        post = engine.bandit.posteriors()
        print("Thompson Sampling posteriors:")
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
