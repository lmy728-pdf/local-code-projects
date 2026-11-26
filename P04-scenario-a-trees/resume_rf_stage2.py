"""Resume Scenario A RF Stage 2 with SQLite writes on the server's local disk."""

import argparse
from contextlib import closing, contextmanager
import multiprocessing
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory, mkdtemp
from time import monotonic

import run


MODE = "main"
LEARNER = "rf"
SYNC_SECONDS = 120
SERVER_TMP = Path("/tmp")
SIDECARS = ("-journal", "-wal", "-shm")


def sidecar(path, suffix):
    return path.with_name(path.name + suffix)


def checkpoint_keys(path):
    # Open read/write on local storage so SQLite can roll back an interrupted write.
    with closing(sqlite3.connect(path)) as db:
        status = db.execute("PRAGMA quick_check").fetchall()
        if status != [("ok",)]:
            raise ValueError(f"SQLite integrity check failed for {path}: {status}")
        config = db.execute("SELECT signature, mode, learner, stage FROM config").fetchone()
        if config != (run.science_signature(), MODE, LEARNER, 2):
            raise ValueError(f"Stage 2 checkpoint provenance differs: {path}")
        keys = set(db.execute("SELECT n, law, rep FROM records"))
        db.execute("PRAGMA journal_mode=DELETE")
    return keys


def prepare_checkpoint(shared, local):
    local_keys = checkpoint_keys(local) if local.exists() else set()
    if not shared.exists():
        return set(), local_keys

    with TemporaryDirectory(prefix="import-", dir=local.parent) as folder:
        imported = Path(folder) / shared.name
        shutil.copyfile(shared, imported)
        for suffix in ("-journal", "-wal"):
            original = sidecar(shared, suffix)
            if original.exists():
                shutil.copyfile(original, sidecar(imported, suffix))
        shared_keys = checkpoint_keys(imported)
        if not local.exists() or local_keys < shared_keys:
            os.replace(imported, local)
            return shared_keys, shared_keys
        if not shared_keys <= local_keys:
            raise ValueError("Local and /mnt checkpoints diverge; both have been preserved.")
    return shared_keys, local_keys


def publish_checkpoint(local, shared):
    # Called between commits, with the single project writer locked out of other runs.
    if any(sidecar(local, suffix).exists() for suffix in SIDECARS):
        raise OSError(f"Local SQLite checkpoint still has a journal: {local}")
    shared.parent.mkdir(parents=True, exist_ok=True)
    pending = None
    moved = []
    try:
        with NamedTemporaryFile(prefix=f".{shared.name}.", suffix=".tmp",
                                dir=shared.parent, delete=False) as output:
            pending = Path(output.name)
            with local.open("rb") as source:
                shutil.copyfileobj(source, output, length=2**20)
            output.flush()
            os.fsync(output.fileno())
        if pending.stat().st_size != local.stat().st_size:
            raise OSError(f"Incomplete checkpoint copy: {pending}")

        # Preserve any journal left by the old direct-to-/mnt writer with its database.
        leftovers = [sidecar(shared, suffix) for suffix in SIDECARS
                     if sidecar(shared, suffix).exists()]
        if leftovers:
            recovery = Path(mkdtemp(prefix="rf-stage2-recovery-", dir=shared.parent))
            shutil.copyfile(shared, recovery / shared.name)
            for original in leftovers:
                archived = recovery / original.name
                os.replace(original, archived)
                moved.append((archived, original))
            print(f"Original SQLite files preserved in {recovery}", flush=True)
        os.replace(pending, shared)
        pending = None
    except BaseException:
        for archived, original in reversed(moved):
            if archived.exists() and not original.exists():
                os.replace(archived, original)
        raise
    finally:
        if pending is not None:
            try:
                pending.unlink(missing_ok=True)
            except OSError:
                pass


class SnapshotConnection:
    """Keep the original runner's transaction boundaries and progress display."""

    def __init__(self, connection, on_commit):
        self.connection = connection
        self.on_commit = on_commit

    def __getattr__(self, name):
        return getattr(self.connection, name)

    def __enter__(self):
        self.connection.__enter__()
        return self

    def __exit__(self, *exc):
        result = self.connection.__exit__(*exc)
        if exc[0] is None:
            self.on_commit()
        return result


def resume(workers, check_only):
    identity = run.digest(os.fsencode(run.ROOT))[:24]
    local_dir = SERVER_TMP / f"rank-dml-rf-stage2-{os.getuid()}-{identity}"
    local_dir.mkdir(mode=0o700, exist_ok=True)
    local = local_dir / "rf.sqlite"
    original_db_path = run.db_path
    original_write_db = run.write_db
    shared = original_db_path(MODE, LEARNER, 2)
    keys = run.keys_for(MODE)
    with run.read_db(original_db_path(MODE, LEARNER, 1), MODE, LEARNER, 1) as cache:
        missing = set(keys).difference(cache.execute("SELECT n, law, rep FROM records"))
    if missing:
        raise ValueError(f"Stage 1 lacks {len(missing):,} required records.")
    shared_keys, local_keys = prepare_checkpoint(shared, local)
    if not local_keys <= set(keys):
        raise ValueError("The Stage 2 checkpoint contains unexpected replication IDs.")
    synced = len(shared_keys)
    print(f"Stage 1 verified: {len(keys):,}/{len(keys):,}; Stage 2: {len(local_keys):,} "
          f"recoverable, {synced:,} on /mnt. Local SQLite: {local}", flush=True)
    if check_only:
        return

    next_sync = monotonic() + SYNC_SECONDS

    def sync(db, force=False):
        nonlocal next_sync, synced
        if not force and monotonic() < next_sync:
            return
        next_sync = monotonic() + SYNC_SECONDS
        count, = db.execute("SELECT COUNT(*) FROM records").fetchone()
        if count == synced and shared.exists() and not any(
            sidecar(shared, suffix).exists() for suffix in SIDECARS
        ):
            return
        try:
            publish_checkpoint(local, shared)
        except OSError as exc:
            if force:
                raise OSError(f"Could not sync /mnt. Latest {count:,} records remain in {local}") from exc
            print(f"Could not sync /mnt; continuing on local disk ({count:,} saved): {exc}", flush=True)
        else:
            synced = count
            print(f"Stage 2 synced to /mnt: {count:,}/{len(keys):,}", flush=True)

    def local_db_path(mode, learner, phase):
        if (mode, learner, phase) == (MODE, LEARNER, 2):
            return local
        return original_db_path(mode, learner, phase)

    @contextmanager
    def write_with_snapshots(mode, learner, phase):
        with original_write_db(mode, learner, phase) as db:
            yield SnapshotConnection(db, lambda: sync(db))

    run.db_path = local_db_path
    run.write_db = write_with_snapshots
    try:
        print(f"main | rf | stage 2 | {len(keys):,} replications | {workers} workers", flush=True)
        run.run_stage(MODE, LEARNER, 2, keys, workers, resume=True)
    finally:
        run.db_path = original_db_path
        run.write_db = original_write_db
        if local.exists():
            with closing(sqlite3.connect(local)) as db:
                sync(db, force=True)


def terminate(signum, frame):
    raise KeyboardInterrupt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, choices=range(1, 17), default=16)
    parser.add_argument("--check-only", action="store_true", help="Validate checkpoints without computing estimates")
    args = parser.parse_args()
    if not sys.platform.startswith("linux"):
        parser.error("Run this helper on the Linux instance that has the /mnt project.")
    identity = run.digest(os.fsencode(run.ROOT))[:24]
    lock = run.FileLock(SERVER_TMP / f"rank-dml-scenario-a-{os.getuid()}-{identity}.lock")
    signal.signal(signal.SIGTERM, terminate)
    try:
        with lock.acquire(timeout=0):
            try:
                resume(run.effective_workers(args.workers), args.check_only)
            except KeyboardInterrupt:
                print("Stage 2 stopped. Committed results are available for resume.", flush=True)
    except run.Timeout as exc:
        raise SystemExit("This project already has an active run.") from exc


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
