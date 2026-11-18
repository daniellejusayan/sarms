"""The spatio-temporal hypergraph ledger (STHG): the system's memory of a repository.

Nodes are components (and files under them). Every commit is a hyperedge over the nodes it touched. Beside
the hyperedges the ledger keeps the import graph as it was at sampled commits (state) and how it moved between
them (delta), every label the Differ produced, every action the loop took, and every certificate issued.
Nothing downstream reads git again: the certificate engine, the Differ's re-checks, the simulator and the
reports all read this store. Append-only; `rebuild` recreates it from the repository.

    from sarms.ledger.hypergraph import Ledger
    L = Ledger("data/kedro/rms/sthg")              # creates ledger.duckdb (+ parquet mirror on export)
    L.ingest_stream(df)                            # hyperedges + incidence from sarms.mining.gitlog
    L.ingest_graph(he, hc)                         # graph_state from the learner's sampled import graphs
    idxs, ts, comps, sizes, edges = L.history()    # exactly what sarms.rules.certify expects
    L.record_action(sha, "adopt", rule_id, {...}); L.record_labels(rows); L.record_certificates(doc)

Spec: dated amendment A-2026-11-07-1 to docs/spec/ledger.sql (additive tables; the frozen `ledger` table is untouched).
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import duckdb
import pandas as pd

DDL = """
CREATE TABLE IF NOT EXISTS nodes       (id VARCHAR PRIMARY KEY, kind VARCHAR, parent VARCHAR, first_seen TIMESTAMPTZ, last_seen TIMESTAMPTZ);
CREATE TABLE IF NOT EXISTS hyperedges  (sha VARCHAR PRIMARY KEY, idx INTEGER, ts TIMESTAMPTZ, author_hash VARCHAR, n_files INTEGER,
                                        n_components INTEGER, bulk BOOLEAN, source VARCHAR);
CREATE TABLE IF NOT EXISTS incidence   (sha VARCHAR, node VARCHAR, added INTEGER, deleted INTEGER, PRIMARY KEY (sha, node));
CREATE TABLE IF NOT EXISTS graph_state (idx INTEGER, sha VARCHAR, ts TIMESTAMPTZ, comp_from VARCHAR, comp_to VARCHAR, n INTEGER,
                                        PRIMARY KEY (idx, comp_from, comp_to));
CREATE TABLE IF NOT EXISTS components  (idx INTEGER, sha VARCHAR, ts TIMESTAMPTZ, comp VARCHAR, n_modules INTEGER, PRIMARY KEY (idx, comp));
CREATE TABLE IF NOT EXISTS graph_delta (idx INTEGER, sha VARCHAR, comp_from VARCHAR, comp_to VARCHAR, delta INTEGER, PRIMARY KEY (idx, comp_from, comp_to));
CREATE TABLE IF NOT EXISTS labels      (sha VARCHAR, rule_id VARCHAR, edge_from VARCHAR, edge_to VARCHAR, label VARCHAR, enforce BOOLEAN,
                                        retired_reason VARCHAR, ts TIMESTAMPTZ);
CREATE TABLE IF NOT EXISTS actions     (seq INTEGER PRIMARY KEY, sha VARCHAR, kind VARCHAR, rule_id VARCHAR, payload JSON, recorded_at TIMESTAMPTZ);
CREATE TABLE IF NOT EXISTS certificates(rule_id VARCHAR, run_sha VARCHAR, axioms JSON, summary JSON, per_commit JSON, checker_hash VARCHAR,
                                        recorded_at TIMESTAMPTZ, PRIMARY KEY (rule_id, run_sha));
"""


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


class Ledger:
    def __init__(self, path):
        self.dir = Path(path)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db = duckdb.connect(str(self.dir / "ledger.duckdb"))
        self.db.execute(DDL)

    # ---- ingestion -------------------------------------------------------------------------------
    def ingest_stream(self, df: pd.DataFrame, bulk_files: int = 50, source_of=None):
        """Hyperedges and incidence from the noise-filtered change stream (idx, sha, ts, author, path, added, deleted, component)."""
        if df.empty:
            return 0
        known = {r[0] for r in self.db.execute("SELECT sha FROM hyperedges").fetchall()}
        df = df[~df.sha.isin(known)]
        if df.empty:
            return 0
        hyper = df.groupby("sha").agg(idx=("idx", "first"), ts=("ts", "first"), author=("author", "first"),
                                      n_files=("path", "nunique"), n_components=("component", "nunique")).reset_index()
        hyper["author_hash"] = hyper.author.map(_hash)
        hyper["bulk"] = hyper.n_files > bulk_files
        hyper["source"] = hyper.author.map(source_of) if source_of else "human"
        self.db.register("h", hyper[["sha", "idx", "ts", "author_hash", "n_files", "n_components", "bulk", "source"]])
        self.db.execute("INSERT INTO hyperedges SELECT * FROM h")
        inc = pd.concat([
            df.groupby(["sha", "path"]).agg(added=("added", "sum"), deleted=("deleted", "sum")).reset_index().rename(columns={"path": "node"}),
            df.dropna(subset=["component"]).groupby(["sha", "component"]).agg(added=("added", "sum"), deleted=("deleted", "sum")).reset_index().rename(columns={"component": "node"}),
        ]).drop_duplicates(["sha", "node"])
        self.db.register("i", inc)
        self.db.execute("INSERT OR IGNORE INTO incidence SELECT sha, node, added, deleted FROM i")
        nodes = []
        for path, g in df.groupby("path"):
            comp = g.component.dropna()
            nodes.append({"id": path, "kind": "file", "parent": comp.iloc[0] if len(comp) else None, "first_seen": g.ts.min(), "last_seen": g.ts.max()})
        for comp, g in df.dropna(subset=["component"]).groupby("component"):
            nodes.append({"id": comp, "kind": "component", "parent": comp.split(".")[0], "first_seen": g.ts.min(), "last_seen": g.ts.max()})
        self.db.register("n", pd.DataFrame(nodes))
        self.db.execute("INSERT OR REPLACE INTO nodes SELECT * FROM n")
        return int(len(hyper))

    def ingest_graph(self, he: pd.DataFrame, hc: pd.DataFrame):
        """Import graph at sampled commits (the learner's history tables, or any (idx, sha, ts, comp_from, comp_to, n) frame)."""
        if hc.empty:
            return 0
        known = {r[0] for r in self.db.execute("SELECT DISTINCT idx FROM components").fetchall()}
        hc, he = hc[~hc.idx.isin(known)], he[~he.idx.isin(known)]
        if hc.empty:
            return 0
        self.db.register("hc", hc[["idx", "sha", "ts", "comp", "n_modules"]])
        self.db.execute("INSERT OR IGNORE INTO components SELECT * FROM hc")
        if len(he):
            self.db.register("he", he[["idx", "sha", "ts", "comp_from", "comp_to", "n"]])
            self.db.execute("INSERT OR IGNORE INTO graph_state SELECT * FROM he")
        # deltas between consecutive sampled states
        idxs = sorted(self.db.execute("SELECT DISTINCT idx FROM components ORDER BY idx").fetchall())
        idxs = [i[0] for i in idxs]
        state = {i: set() for i in idxs}
        for i, a, b in self.db.execute("SELECT idx, comp_from, comp_to FROM graph_state").fetchall():
            state[i].add((a, b))
        sha_of = dict(self.db.execute("SELECT idx, sha FROM components").fetchall())
        self.db.execute("DELETE FROM graph_delta")
        rows = []
        for prev, cur in zip(idxs[:-1], idxs[1:], strict=True):
            for a, b in state[cur] - state[prev]:
                rows.append({"idx": cur, "sha": sha_of[cur], "comp_from": a, "comp_to": b, "delta": 1})
            for a, b in state[prev] - state[cur]:
                rows.append({"idx": cur, "sha": sha_of[cur], "comp_from": a, "comp_to": b, "delta": -1})
        if rows:
            self.db.register("gd", pd.DataFrame(rows))
            self.db.execute("INSERT OR IGNORE INTO graph_delta SELECT * FROM gd")
        return int(hc.idx.nunique())

    def ingest_learn_dir(self, learn_dir):
        learn_dir = Path(learn_dir)
        he = pd.read_parquet(learn_dir / "history_edges.parquet")
        hc = pd.read_parquet(learn_dir / "history_components.parquet")
        return self.ingest_graph(he, hc)

    # ---- records written by the loop --------------------------------------------------------------
    def record_action(self, sha, kind, rule_id=None, payload=None):
        seq = self.db.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM actions").fetchone()[0]
        self.db.execute("INSERT INTO actions VALUES (?, ?, ?, ?, ?, now())", [seq, sha, kind, rule_id, json.dumps(payload or {}, default=str)])
        return seq

    def record_labels(self, rows, ts=None):
        for r in rows:
            self.db.execute("INSERT INTO labels VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                            [r["sha"], r.get("rule_id"), r["edge_from"], r["edge_to"], r["label"], bool(r.get("enforce", True)),
                             r.get("retired_reason"), r.get("ts", ts)])

    def record_certificates(self, doc, run_sha, checker_hash):
        for rid, c in doc["rules"].items():
            self.db.execute("INSERT OR REPLACE INTO certificates VALUES (?, ?, ?, ?, ?, ?, now())",
                            [rid, run_sha, json.dumps(c["axioms"]), json.dumps(c["summary"], default=str), json.dumps(c["per_commit"], default=str), checker_hash])

    # ---- reading -----------------------------------------------------------------------------------
    def history(self):
        """(idxs, ts, comps, sizes, edges): the tuple sarms.rules.certify.evaluate works on."""
        comps_df = self.db.execute("SELECT idx, ts, comp, n_modules FROM components ORDER BY idx").df()
        edges_df = self.db.execute("SELECT idx, comp_from, comp_to FROM graph_state").df()
        idxs = sorted(comps_df.idx.unique())
        ts = {int(i): pd.Timestamp(t).tz_convert("UTC") if pd.Timestamp(t).tzinfo else pd.Timestamp(t, tz="UTC")
              for i, t in comps_df.groupby("idx").ts.first().items()}
        comps = {int(i): set(g.comp) for i, g in comps_df.groupby("idx")}
        sizes = {(int(r.idx), r.comp): int(r.n_modules) for r in comps_df.itertuples(index=False)}
        edges = defaultdict(set)
        for r in edges_df.itertuples(index=False):
            if r.comp_from != r.comp_to:
                edges[int(r.idx)].add((r.comp_from, r.comp_to))
        for i in idxs:
            edges.setdefault(int(i), set())
        return [int(i) for i in idxs], ts, comps, sizes, dict(edges)

    def stream(self) -> pd.DataFrame:
        return self.db.execute("""SELECT h.idx, h.sha, h.ts, i.node AS path, i.added, i.deleted, n.parent AS component
                                  FROM hyperedges h JOIN incidence i ON h.sha = i.sha JOIN nodes n ON n.id = i.node
                                  WHERE n.kind = 'file' ORDER BY h.idx""").df()

    def excused(self) -> set:
        """Crossings the loop handled: (idx, from, to) for every apply/retire action whose payload names the pair."""
        idx_of = dict(self.db.execute("SELECT sha, idx FROM components").fetchall())
        out = set()
        for sha, payload in self.db.execute("SELECT sha, payload FROM actions WHERE kind IN ('apply', 'retire')").fetchall():
            p = json.loads(payload) if isinstance(payload, str) else (payload or {})
            if sha in idx_of and p.get("from") and p.get("to"):
                out.add((int(idx_of[sha]), p["from"], p["to"]))
        return out

    def actions(self) -> pd.DataFrame:
        return self.db.execute("SELECT * FROM actions ORDER BY seq").df()

    def summary(self) -> dict:
        q = lambda s: self.db.execute(s).fetchone()[0]  # noqa: E731
        return {"hyperedges": q("SELECT count(*) FROM hyperedges"), "bulk": q("SELECT count(*) FROM hyperedges WHERE bulk"),
                "files": q("SELECT count(*) FROM nodes WHERE kind='file'"), "components": q("SELECT count(*) FROM nodes WHERE kind='component'"),
                "graph_snapshots": q("SELECT count(DISTINCT idx) FROM components"), "graph_edges_now": q(
                    "SELECT count(*) FROM graph_state WHERE idx = (SELECT max(idx) FROM components)"),
                "labels": q("SELECT count(*) FROM labels"), "actions": q("SELECT count(*) FROM actions"), "certificates": q("SELECT count(*) FROM certificates")}

    def export_parquet(self):
        for t in ("nodes", "hyperedges", "incidence", "graph_state", "components", "graph_delta", "labels", "actions", "certificates"):
            self.db.execute(f"COPY {t} TO '{self.dir / (t + '.parquet')}' (FORMAT PARQUET)")

    def state_hash(self) -> str:
        """Determinism check: a hash of the whole graph state and the actions, for ledger replay (invariant I4)."""
        parts = []
        for t in ("components", "graph_state", "actions"):
            parts.append(self.db.execute(f"SELECT * FROM {t} ORDER BY ALL").df().to_csv(index=False))
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()

    def close(self):
        self.db.close()

