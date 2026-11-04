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


def _bbox(text):
    try:
        x1, y1, x2, y2 = (float(v) for v in text.split(","))
    except ValueError:
        raise argparse.ArgumentTypeError("expected MINLON,MINLAT,MAXLON,MAXLAT")
    if not (x1 < x2 and y1 < y2):
        raise argparse.ArgumentTypeError("min values must be smaller than max values")
    return (x1, y1, x2, y2)


def _sector_filters(sp):
    sp.add_argument("--sector", action="append",
                    help="only this sector slug (repeatable); see `python -m etl sectors`")
    sp.add_argument("--city", help="only sectors in this city")


SECTORS_SQL = """
SELECT sc.city, sc.slug, count(ss.u) AS segments,
       count(s.score) AS scored,
       count(*) FILTER (WHERE m.method_code = 'model') AS from_images,
       to_char(sc.osm_downloaded_at, 'YYYY-MM-DD') AS downloaded
FROM sector sc
LEFT JOIN sector_segment ss USING (sector_id)
LEFT JOIN scenic_score s USING (u, v, key)
LEFT JOIN score_method m USING (method_id)
GROUP BY sc.sector_id ORDER BY sc.city, sc.slug
"""


def _print_sectors(conn):
    with conn.cursor() as cur:
        cur.execute(SECTORS_SQL)
        rows = cur.fetchall()
        cur.execute("""SELECT count(*) FROM road_segment r WHERE NOT EXISTS (
                         SELECT 1 FROM sector_segment ss WHERE (ss.u, ss.v, ss.key) = (r.u, r.v, r.key))""")
        unassigned = cur.fetchone()[0]
    if not rows:
        print("No sectors yet. Add one with: python -m etl add-sector \"<place>, <city>, <country>\"")
    else:
        print(f"{'city':12} {'sector':28} {'segments':>8} {'scored':>7} {'from images':>12}  downloaded")
        for city, slug, segs, scored, images, dl in rows:
            pct = f"{images / segs:.0%}" if segs else "-"
            print(f"{city:12} {slug:28} {segs:>8} {scored:>7} {images:>7} ({pct:>4})  {dl or 'linked only'}")
    print(f"Segments not in any sector: {unassigned}")


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
    sp = sub.add_parser("score", help="fetch Mapillary images for unscored/fallback streets "
                                      "and write scores to the database (runs the model)")
    sp.add_argument("--limit", type=int, default=50, help="max streets this run (default 50)")
    sp.add_argument("--retry-after-days", type=int, default=30,
                    help="skip streets that had no images within this many days (default 30)")
    sp.add_argument("--sleep", type=float, default=0.5, help="seconds between streets")
    sp.add_argument("--dry-run", action="store_true", help="only count what would be searched")
    sp.add_argument("--rescore-legacy", action="store_true",
                    help="also re-score model scores loaded from old graph files (no recorded photo)")
    sp.add_argument("--bbox", type=_bbox, metavar="MINLON,MINLAT,MAXLON,MAXLAT",
                    help="only streets inside this box")
    _sector_filters(sp)
    sp = sub.add_parser("green", help="load parks/woods from OpenStreetMap for the sectors")
    _sector_filters(sp)
    sp = sub.add_parser("export-graph", help="write a GraphML file from the database")
    sp.add_argument("--out", default=str(config.PROJECT_ROOT / "graphs" / "db_export.graphml"))
    _sector_filters(sp)

    sp = sub.add_parser("add-sector", help="download a sector from OpenStreetMap into the database")
    sp.add_argument("place", help='OSM place name, e.g. "Le Sud-Ouest, Montreal, Canada"')
    sp.add_argument("--city", help="city name (default: 2nd part of the place name)")
    sp.add_argument("--network-type", default="bike", help="OSMnx network type (default: bike)")
    sp.add_argument("--link-only", action="store_true",
                    help="do not download; tag segments already in the database that lie "
                         "inside this sector's boundary")
    sp.add_argument("--strict", action="store_true", help="abort if any ERROR check fails")
    sub.add_parser("sectors", help="list sectors and how much of each is scored")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    from .pipeline import prepare, run  # heavy imports after arg parsing
    config.configure_osmnx()

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
        if args.cmd in ("score", "export-graph", "green"):
            from .load import resolve_sector_ids
            with conn.cursor() as cur:
                sector_ids = resolve_sector_ids(cur, args.sector, args.city)
            conn.rollback()
        if args.cmd == "score":
            from .score import run_scoring
            counts = run_scoring(conn, limit=args.limit, retry_days=args.retry_after_days,
                                 sleep=args.sleep, dry_run=args.dry_run, sector_ids=sector_ids,
                                 rescore_legacy=args.rescore_legacy, bbox=args.bbox)
            print(f"Done: {counts}")
            return 0
        if args.cmd == "green":
            from .green import load_green
            for slug, c in load_green(conn, sector_ids).items():
                print(f"  {slug:28} green areas found={c['found']:>4} "
                      f"inserted={c['inserted']:>4} updated={c['updated']:>4}")
            return 0
        if args.cmd == "export-graph":
            from .export import export_graph
            info = export_graph(conn, args.out, sector_ids)
            print(f"Wrote {info['path']}: {info['nodes']} nodes, {info['edges']} edges, "
                  f"{info['scored_edges']} scored")
            return 0
        if args.cmd == "add-sector":
            from .pipeline import add_sector
            print(f"== {args.place}" + (" (link only)" if args.link_only else " (downloading...)"))
            stats = add_sector(conn, args.place, city=args.city, network_type=args.network_type,
                               strict=args.strict, link_only=args.link_only)
            if "validation" in stats:
                _print_issues(stats["validation"])
                for table, c in stats["load"].items():
                    print(f"  {table:13} inserted={c['inserted']:>6} updated={c['updated']:>6} "
                          f"unchanged={c['unchanged']:>6}")
            s = stats["sector"]
            print(f"Sector '{stats['slug']}' ({stats['city']}): {s['segments']} segments, "
                  f"{s['scored']} scored ({s['model_scored']} from images), "
                  f"{stats['new_links']} newly linked")
            return 0
        if args.cmd == "sectors":
            _print_sectors(conn)
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
