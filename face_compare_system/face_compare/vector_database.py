"""Transactional SQLite templates with revisioned, exact cosine search.

SQLite is the source of truth; the normalized NumPy matrix is a rebuildable
cache, not a second independently writable index. Deletion is recoverable.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
import uuid

import cv2
import numpy as np

from .database import FaceDatabase, PersonSummary, normalize_person_name
from .aggregation import aggregate_identity_scores


class SQLiteVectorDatabase:
    def __init__(self, root, *, save_face_images=True, feature_signature=None, expected_dimension=None):
        if not feature_signature or not expected_dimension:
            raise ValueError("向量库必须指定模型签名和维度")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.feature_signature, self.expected_dimension = feature_signature, expected_dimension
        self.save_face_images = save_face_images
        self.path = self.root / "vectors.sqlite3"
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.path, timeout=10, isolation_level=None, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS people (
                id TEXT PRIMARY KEY, name TEXT NOT NULL, name_key TEXT NOT NULL,
                created_at TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 1);
            CREATE UNIQUE INDEX IF NOT EXISTS active_name ON people(name_key) WHERE active=1;
            CREATE TABLE IF NOT EXISTS samples (
                id TEXT PRIMARY KEY, person_id TEXT NOT NULL REFERENCES people(id),
                vector BLOB NOT NULL, image BLOB, quality TEXT NOT NULL, created_at TEXT NOT NULL);
            CREATE INDEX IF NOT EXISTS sample_person ON samples(person_id);
            CREATE TABLE IF NOT EXISTS archives (
                id TEXT PRIMARY KEY, people TEXT NOT NULL, restored INTEGER NOT NULL DEFAULT 0);
        """)
        self._cache_revision = -1
        self._sets = []
        self._matrix = np.empty((0, expected_dimension), np.float32)
        self._ranges = []
        self._aggregation_groups = ()
        try:
            with self._transaction():
                signature = self._get("signature")
                if signature is None:
                    self._set("signature", feature_signature)
                    self._set("dimension", str(expected_dimension))
                    self._set("schema", "1")
                    self._set("revision", "0")
                    self._import_legacy()
                elif (signature != feature_signature or self._get("dimension") != str(expected_dimension)
                      or self._get("schema") != "1"):
                    raise RuntimeError("向量库模型/维度/版本不一致，请使用独立目录")
            self._check_legacy()
        except BaseException:
            self._conn.close()
            raise

    @contextmanager
    def _transaction(self):
        with self._lock:
            if self._conn.in_transaction:
                yield
                return
            try:
                self._conn.execute("BEGIN IMMEDIATE")
                yield
                self._conn.execute("COMMIT")
            except BaseException:
                if self._conn.in_transaction:
                    self._conn.execute("ROLLBACK")
                self._cache_revision = -1
                raise

    def write_guard(self):
        """Cover service-level identity validation and persistence with one writer lock."""
        return self._transaction()

    def _get(self, key):
        row = self._conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def _set(self, key, value):
        self._conn.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, value))

    def _bump(self):
        self._set("revision", str(int(self._get("revision"))+1))

    @property
    def revision(self):
        with self._lock:
            self._check_legacy()
            return int(self._get("revision"))

    def _validate_vector(self, vector):
        vector = np.asarray(vector, dtype=np.float32)
        norm = float(np.linalg.norm(vector))
        if (vector.shape != (self.expected_dimension,) or not np.isfinite(vector).all()
                or not np.isfinite(norm) or norm < 1e-8):
            raise ValueError("特征维数、数值或范数无效")
        return vector

    def _legacy_digest(self):
        path = self.root / "people.json"
        return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else "absent"

    def _check_legacy(self):
        if self._get("legacy_digest") != self._legacy_digest():
            raise RuntimeError("旧版JSON人脸库在迁移后被修改；请关闭旧程序并核对两份数据，拒绝静默覆盖")

    def _import_legacy(self):
        """One-time copy in the initialization transaction; originals remain unchanged."""
        before = self._legacy_digest()
        if before != "absent":
            source = FaceDatabase(self.root, save_face_images=self.save_face_images,
                                  feature_signature=self.feature_signature, expected_dimension=self.expected_dimension)
            for person, vectors in source.feature_sets():
                self._conn.execute("INSERT INTO people VALUES (?,?,?,?,1)",
                                   (person.person_id, person.name, person.name.casefold(), person.created_at))
                records = next(p["samples"] for p in source._metadata["people"] if p["id"] == person.person_id)
                for record, vector in zip(records, vectors):
                    image = None
                    if record.get("image_path"):
                        path = (self.root / record["image_path"]).resolve()
                        if not path.is_relative_to(self.root):
                            raise ValueError("旧样本图像路径越出人脸库")
                        if path.is_file():
                            image = path.read_bytes()
                    self._conn.execute("INSERT INTO samples VALUES (?,?,?,?,?,?)", (
                        record["id"], person.person_id, self._validate_vector(vector).astype("<f4").tobytes(),
                        image, json.dumps(record.get("quality", {})), record["created_at"]))
            if before != self._legacy_digest():
                raise RuntimeError("迁移时旧库发生变化，请关闭旧程序后重试")
        self._set("legacy_digest", before)

    def refresh_cache(self):
        with self._lock:
            self._check_legacy()
            own_transaction = not self._conn.in_transaction
            if own_transaction:
                self._conn.execute("BEGIN")
            try:
                revision = int(self._get("revision"))
                if revision == self._cache_revision:
                    return
                rows = self._conn.execute("""SELECT p.id,p.name,p.created_at,s.vector FROM people p
                    JOIN samples s ON s.person_id=p.id WHERE p.active=1 ORDER BY p.name_key,p.id,s.id""").fetchall()
                groups = {}
                for pid, name, created, blob in rows:
                    vector = self._validate_vector(np.frombuffer(blob, dtype="<f4")).copy()
                    vector.setflags(write=False)
                    groups.setdefault(pid, [name, created, []])[2].append(vector)
                sets, vectors, ranges = [], [], []
                for pid, (name, created, group) in groups.items():
                    person = PersonSummary(pid, name, len(group), created)
                    sets.append((person, tuple(group)))
                    ranges.append((len(vectors), len(vectors)+len(group), person))
                    vectors.extend(group)
                matrix = np.vstack(vectors) if vectors else np.empty((0, self.expected_dimension), np.float32)
                if len(matrix):
                    matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
                # Group identities by *actual* template count once per revision.
                # This preserves every template and the exact per-identity rule,
                # while removing thousands of tiny Python/NumPy reductions from
                # each query. Indices are rebuilt in the same snapshot as vectors.
                buckets = {}
                for index, (start, end, _person) in enumerate(ranges):
                    buckets.setdefault(end - start, []).append((index, start))
                aggregation_groups = []
                for count, members in buckets.items():
                    identities, starts = np.asarray(members, dtype=np.intp).T
                    indices = starts[:, None] + np.arange(count, dtype=np.intp)
                    identities.setflags(write=False)
                    indices.setflags(write=False)
                    aggregation_groups.append((identities, indices))
                self._sets, self._matrix, self._ranges = sets, matrix, ranges
                self._aggregation_groups = tuple(aggregation_groups)
                self._cache_revision = revision
            finally:
                if own_transaction:
                    self._conn.execute("COMMIT")

    def feature_sets(self):
        with self._lock:
            self.refresh_cache()
            return list(self._sets)

    def list_people(self):
        return [person for person, _ in self.feature_sets()]

    def sample_count(self):
        return sum(person.sample_count for person in self.list_people())

    @property
    def calibrated_threshold(self):
        return None  # Neural threshold selection uses held-out validation only.

    def rank_candidates(self, feature, nearest_samples=3, *, strategy="topk_median"):
        """Exact search of every identity; no approximate top-2 margin distortion."""
        query = self._validate_vector(feature)
        with self._lock:
            self.refresh_cache()
            distances = np.clip((1-self._matrix @ (query/np.linalg.norm(query)))/2, 0, 1)
            scores = np.empty(len(self._ranges), dtype=np.float64)
            for identities, indices in self._aggregation_groups:
                scores[identities] = aggregate_identity_scores(distances[indices], strategy, nearest_samples)
            # Keep the historical tuple tie-break (distance, person_id, name).
            # In particular, do not use a template-only top-K as a proxy for the
            # true second identity required by open-set ambiguity rejection.
            return sorted((float(score), person.person_id, person.name)
                          for score, (_start, _end, person) in zip(scores, self._ranges))

    def add_samples(self, name, samples):
        name = normalize_person_name(name)
        if not samples:
            raise ValueError("至少需要一张有效样本")
        encoded = []
        for sample in samples:
            vector = self._validate_vector(sample.feature).astype("<f4").tobytes()
            image = None
            if self.save_face_images:
                ok, data = cv2.imencode(".jpg", sample.crop, [cv2.IMWRITE_JPEG_QUALITY, 94])
                if not ok:
                    raise ValueError("样本图像无法编码")
                image = data.tobytes()
            encoded.append((vector, image, json.dumps(sample.quality.to_dict(), allow_nan=False)))
        with self._transaction():
            self._check_legacy()
            row = self._conn.execute("SELECT id FROM people WHERE name_key=? AND active=1", (name.casefold(),)).fetchone()
            pid = row[0] if row else uuid.uuid4().hex
            now = datetime.now(timezone.utc).isoformat()
            if not row:
                self._conn.execute("INSERT INTO people VALUES (?,?,?,?,1)", (pid, name, name.casefold(), now))
            for vector, image, quality in encoded:
                self._conn.execute("INSERT INTO samples VALUES (?,?,?,?,?,?)", (uuid.uuid4().hex, pid, vector, image, quality, now))
            self._bump()
        return next(person for person in self.list_people() if person.person_id == pid)

    def archive_people(self, person_id=None):
        with self._transaction():
            self._check_legacy()
            rows = self._conn.execute("SELECT id,name FROM people WHERE active=1").fetchall()
            ids = [pid for pid, _ in rows if person_id is None or pid == person_id]
            if not ids:
                raise ValueError("未找到要删除的人员")
            aid = uuid.uuid4().hex
            self._conn.execute("INSERT INTO archives(id,people) VALUES (?,?)", (aid, json.dumps(ids)))
            self._conn.executemany("UPDATE people SET active=0 WHERE id=?", [(pid,) for pid in ids])
            self._bump()
            directory = self.root / "archives"
            directory.mkdir(exist_ok=True)
            path = directory / f"people-{aid}.json"
            with path.open("x", encoding="utf-8") as handle:
                json.dump({"archive_id": aid, "database": self.path.name,
                           "people": [dict(id=pid, name=name) for pid, name in rows if pid in ids],
                           "note": "恢复索引；原照片与向量仍在vectors.sqlite3中，需连同数据库备份"}, handle, ensure_ascii=False, indent=2)
        return path

    def restore_archive(self, archive_path):
        aid = json.loads(Path(archive_path).read_text(encoding="utf-8"))["archive_id"]
        try:
            with self._transaction():
                self._check_legacy()
                row = self._conn.execute("SELECT people,restored FROM archives WHERE id=?", (aid,)).fetchone()
                if not row or row[1]:
                    raise ValueError("恢复索引不属于此库或已经恢复")
                ids = json.loads(row[0])
                self._conn.executemany("UPDATE people SET active=1 WHERE id=?", [(pid,) for pid in ids])
                self._conn.execute("UPDATE archives SET restored=1 WHERE id=?", (aid,))
                self._bump()
        except sqlite3.IntegrityError as exc:
            raise ValueError("已有同名人员，恢复将产生冲突；未修改数据") from exc

    def close(self):
        with self._lock:
            self._conn.close()
