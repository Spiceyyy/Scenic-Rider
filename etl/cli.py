"""Command line entry point:  python -m etl {init-db,validate,run} ..."""
import argparse
import json
import logging
import sys

from . import config


def _print_issues(summary):
    print(f"Valid rows: {summary['rows_valid']}")
    for i in summary["issues"]:
        print(f"  [{i['severity'].upper():5}] {i['entity']:8} {i['check']:24} {i['count']:>7}"
              f"  e.g. {i['examples'][:2]}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m etl", description="Scenic Viber PostGIS ETL")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db", help="create schema/tables (idempotent)")
    for name, help_ in (("validate", "extract + validate only, no database"),
                        ("run", "full ETL into PostGIS")):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("--graph", action="append",
                        help=f"GraphML file (repeatable). Default: {config.DEFAULT_GRAPH.name}")
        if name == "run":
            sp.add_argument("--strict", action="store_true", help="abort if any ERROR check fails")
            sp.add_argument("--json", action="store_true", help="print full stats as JSON")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    from .pipeline import prepare, run  # heavy imports after arg parsing

    graphs = getattr(args, "graph", None) or [str(config.DEFAULT_GRAPH)]

    if args.cmd == "validate":
        failed = False
        for g in graphs:
            print(f"== {g}")
            meta, result = prepare(g)
            _print_issues(result.summary())
            failed |= bool(result.errors)
        return 1 if failed else 0

    from .load import apply_schema, connect

    conn = connect(config.database_url(), config.db_schema())
    try:
        apply_schema(conn)
        if args.cmd == "init-db":
            print(f"Schema '{config.db_schema()}' is ready.")
            return 0
        for g in graphs:
            print(f"== {g}")
            stats = run(conn, g, strict=args.strict)
            if args.json:
                print(json.dumps(stats, indent=2, default=str, ensure_ascii=False))
            else:
                _print_issues(stats["validation"])
                for table, c in stats["load"].items():
                    print(f"  {table:13} inserted={c['inserted']:>6} updated={c['updated']:>6} "
                          f"unchanged={c['unchanged']:>6}")
        return 0
    finally:
        conn.close()
