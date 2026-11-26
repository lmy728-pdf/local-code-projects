"""Read one result and its corresponding residual cache, without modifying the database."""
from pathlib import Path
import io,json,sqlite3
import numpy as np
def read_one(stage="smoke"):
    root=Path(__file__).resolve().parents[1]
    folder=root/("replications" if stage=="main" else stage)
    with sqlite3.connect((folder/"records.sqlite").as_uri()+"?mode=ro",uri=True) as db:
        row=db.execute("SELECT payload FROM records ORDER BY n,law,rep LIMIT 1").fetchone()
        if row is None:raise RuntimeError("No records")
        rec=json.loads(row[0]);batch=rec.get("cache_batch")
        if not batch:return rec,None
        payload=db.execute("SELECT payload FROM cache_batches WHERE id=?",(batch,)).fetchone()[0]
    prefix=f'n{rec["n"]}_{rec["law"]}_r{rec["replication"]}__'
    with np.load(io.BytesIO(payload),allow_pickle=False) as z:
        cache={key[len(prefix):]:z[key] for key in z.files if key.startswith(prefix)}
    return rec,cache
if __name__=="__main__":
    import sys
    rec,cache=read_one(sys.argv[1] if len(sys.argv)>1 else "smoke")
    print(json.dumps({k:v for k,v in rec.items() if k!="models"},ensure_ascii=False,indent=2))
    print("Cache arrays:",{} if cache is None else {k:v.shape for k,v in cache.items()})
