"""Read-only integrity verification for original project materials."""
import hashlib
import json
from pathlib import Path


def verify_manifest(manifest_path=None):
    path = Path(manifest_path) if manifest_path else Path(__file__).parent / "audit/source_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    failures = []
    for item in manifest["files"]:
        source = Path(item["absolute_path"])
        if not source.is_file():
            failures.append({"path": str(source), "reason": "missing"})
        elif hashlib.sha256(source.read_bytes()).hexdigest() != item["sha256"]:
            failures.append({"path": str(source), "reason": "sha256_mismatch"})
    return {"ok": not failures, "checked_files": len(manifest["files"]),
            "failures": failures}


if __name__ == "__main__":
    result = verify_manifest()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["ok"] else 1)
