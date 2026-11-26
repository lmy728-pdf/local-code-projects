"""Make audited historical inputs portable, without changing historical files."""
from pathlib import Path
import csv,sqlite3,zlib,hashlib
from tqdm import tqdm
ROOT=Path(__file__).resolve().parents[1]
def pack():
    with (ROOT/"audit/existing_data_audit.csv").open(encoding="utf-8-sig",newline="") as f:
        rows=[x for x in csv.DictReader(f) if x["status"]=="REUSE"]
    dest=ROOT/"cache/reused_inputs.sqlite"
    with sqlite3.connect(dest) as db:
        db.execute("CREATE TABLE IF NOT EXISTS inputs(n INTEGER,law TEXT,rep INTEGER,cache BLOB,record BLOB,PRIMARY KEY(n,law,rep))")
        for row in tqdm(rows,desc="Pack audited inputs",unit="rep",ascii=True):
            key=(int(row["n"]),row["law"],int(row["replication"]))
            if db.execute("SELECT 1 FROM inputs WHERE n=? AND law=? AND rep=?",key).fetchone():continue
            raw=Path(row["cache"]).read_bytes();rec=Path(row["record"]).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row["cache_sha256"] or hashlib.sha256(rec).hexdigest()!=row["record_sha256"]:
                raise RuntimeError("Historical input changed: "+str(key))
            db.execute("INSERT INTO inputs VALUES (?,?,?,?,?)",(*key,raw,zlib.compress(rec)))
        assert db.execute("SELECT COUNT(*) FROM inputs").fetchone()[0]==10000
    print("Portable audited input archive:",dest,flush=True)
if __name__=="__main__":pack()

