"""Bounded, provenance-preserving access to old and new observations.

[Role] Index immutable source files and expose rows without copying captures.
[Invariants] A registered file is not automatically a calibration. Predictions,
  validation targets and primitive observations retain distinct kinds.
[Timeouts & Locks] WAL and busy_timeout=5000; streaming hashes/parsers and
  indexed bounded SQL queries. No whole-file reads.
[Side-Effect Policy] Writes only the requested local observation index.
"""
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time

try:
    import ijson
except ImportError:  # pragma: no cover - stdlib fallback when ijson is absent
    ijson = None


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_prefix(value, parts):
    if not parts:
        yield value
        return
    head, *tail = parts
    if head == "item":
        if isinstance(value, list):
            for item in value:
                yield from _iter_prefix(item, tail)
    elif isinstance(value, dict) and head in value:
        yield from _iter_prefix(value[head], tail)


def json_items(path, prefix=""):
    if ijson is not None:
        with Path(path).open("rb") as stream:
            yield from ijson.items(stream, prefix, use_float=True)
        return
    with Path(path).open("rb") as stream:
        doc = json.load(stream)
    parts = [part for part in prefix.split(".") if part] if prefix else []
    yield from _iter_prefix(doc, parts)


def json_kvitems(path, prefix=""):
    if ijson is not None:
        with Path(path).open("rb") as stream:
            yield from ijson.kvitems(stream, prefix, use_float=True)
        return
    parts = [part for part in prefix.split(".") if part] if prefix else []
    for item in json_items(path, ".".join(parts)):
        if isinstance(item, dict):
            yield from item.items()


def json_top_fields(path, wanted):
    if ijson is not None:
        from ijson.common import ObjectBuilder

        result, builder, selected, depth = {}, None, None, 0
        with Path(path).open("rb") as stream:
            for prefix, event, value in ijson.parse(stream, use_float=True):
                if builder is None:
                    if prefix not in wanted:
                        continue
                    if event not in {"start_map", "start_array"}:
                        result[prefix] = value
                        continue
                    selected, builder, depth = prefix, ObjectBuilder(), 0
                builder.event(event, value)
                if event in {"start_map", "start_array"}:
                    depth += 1
                elif event in {"end_map", "end_array"}:
                    depth -= 1
                    if not depth:
                        result[selected], builder = builder.value, None
        return result
    with Path(path).open("rb") as stream:
        doc = json.load(stream)
    if not isinstance(doc, dict):
        return {}
    return {key: doc[key] for key in wanted if key in doc}


def profiler_rows(path):
    from .kernels import kernel_rows

    for row in kernel_rows(path):
        event = row.event
        yield {
            "line": row.line, "duration_ms": row.duration_ns / 1e6,
            "kernel": row.name, "operator": row.kind,
            "input_shapes": event.input_shapes.strip('"'),
            "input_dtypes": event.input_dtypes, "input_formats": event.input_formats,
            "output_shapes": event.output_shapes.strip('"'),
            "output_dtypes": event.output_dtypes, "device": event.device_id,
            "stream": row.stream, "core": row.core,
            "block_dim": "" if row.blocks is None else str(row.blocks),
            "boundary": "profiled_device_kernel",
        }


def source_kind(path):
    path = Path(path)
    # Source role is a discovery aid. Neither this classification nor a
    # filename ever authorizes applying a timing to an operator.
    if path.name.endswith(".heldout.json"):
        return "validation"
    if path.suffix == ".csv":
        with path.open("rb") as stream:
            header = stream.readline(16384)
        if b"Duration(us)" in header or b"Task Duration(us)" in header:
            return "profiler"
    if ("mindie-sim-data/sim/" in str(path) or "predict" in path.name
            or "true128k-results-full" in str(path)):
        return "prediction"
    if path.suffix in {".json", ".jsonl", ".ndjson"}:
        return "structured_source"
    if path.suffix in {".sqlite", ".db"}:
        return "database"
    return "supporting_source"


class ObservationIndex:
    def __init__(self, path):
        self.connection = sqlite3.connect(path, timeout=5)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=5000")
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS sources(
              path TEXT PRIMARY KEY, bytes INTEGER, mtime_ns INTEGER,
              sha256 TEXT, kind TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS catalogues(
              path TEXT PRIMARY KEY, kind TEXT NOT NULL, schema_json TEXT NOT NULL);
        """)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.connection.close()

    def register(self, path, kind=None):
        path = Path(path).resolve()
        stat = path.stat()
        previous = self.connection.execute(
            "SELECT bytes,mtime_ns,sha256,kind FROM sources WHERE path=?", (str(path),)).fetchone()
        role = kind or source_kind(path)
        if previous and previous[:2] == (stat.st_size, stat.st_mtime_ns) and previous[3] == role:
            return
        self.connection.execute("INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?)",
                                (str(path), stat.st_size, stat.st_mtime_ns, digest_file(path), role))

    def scan(self, root):
        count = 0
        for directory, names, files in os.walk(root):
            names[:] = sorted(name for name in names if name not in {".git", ".venv", "__pycache__"})
            for name in sorted(files):
                path = Path(directory, name)
                if path.is_file() and path.suffix not in {".pyc", ".pid", ".cpid", ".stop"}:
                    self.register(path)
                    count += 1
                    if count % 200 == 0:
                        self.connection.commit()
        self.connection.commit()
        return self.inventory()

    def attach_profiler_catalogue(self, path):
        path = Path(path).resolve()
        with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True) as db:
            names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if not {"sources", "variants", "events", "costs"} <= names:
                raise ValueError("external_profiler_schema_not_supported")
            schema = {row[0]: row[1] for row in db.execute(
                "SELECT name,sql FROM sqlite_master WHERE type='table'")}
        self.register(path, "profiler_catalogue")
        with self.connection:
            self.connection.execute("INSERT OR REPLACE INTO catalogues VALUES(?,?,?)",
                                    (str(path), "profiler_variants/1", json.dumps(schema, sort_keys=True)))

    def inventory(self):
        return {kind: {"files": count, "bytes": size}
                for kind, count, size in self.connection.execute(
                    "SELECT kind,COUNT(*),SUM(bytes) FROM sources GROUP BY kind")}

    def find_sources(self, pattern, limit=50):
        return [dict(zip(("path", "bytes", "sha256", "kind"), row)) for row in
                self.connection.execute("SELECT path,bytes,sha256,kind FROM sources WHERE path LIKE ? LIMIT ?",
                                        ("%" + pattern + "%", min(limit, 1000)))]


class ProfilerCatalogue:
    """Read the retained 2026 capture database with no dependency on old code."""

    def __init__(self, path):
        self.connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
        self.connection.execute("PRAGMA busy_timeout=5000")

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.connection.close()

    def variants(self, *, limit=50, after=None):
        sql = """SELECT v.id,v.signature,v.context,v.capture,c.calls,c.median_ns,c.min_ns,c.max_ns
                 FROM variants v JOIN costs c ON c.variant=v.id WHERE v.id>? ORDER BY v.id LIMIT ?"""
        cursor = self.connection.execute(sql, (bytes.fromhex(after) if after else b"", min(limit, 1000)))
        for key, signature, context, capture, calls, median, minimum, maximum in cursor:
            yield {"id": key.hex(), "signature": json.loads(signature), "context": json.loads(context),
                   "capture": capture, "calls": calls, "median_ms": median / 1e6,
                   "min_ms": minimum / 1e6, "max_ms": maximum / 1e6}

    def samples(self, variant, limit=1000, *, query_seconds=5):
        # Old events has no variant index. Read through a bounded cursor rather
        # than building an in-memory reverse index or modifying the raw store.
        deadline = time.monotonic() + query_seconds
        self.connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 10000)
        try:
            cursor = self.connection.execute(
                "SELECT e.duration_ns,s.path,s.member,e.row_number FROM events e "
                "JOIN sources s ON s.id=e.source WHERE e.variant=? LIMIT ?",
                (bytes.fromhex(variant), min(limit, 10000)))
            yield from ({"duration_ms": duration / 1e6, "source": path, "member": member, "row": row}
                        for duration, path, member, row in cursor)
        finally:
            self.connection.set_progress_handler(None, 0)
