"""Compare full 2-worker smoke with a paused and resumed 1-worker smoke."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
import json
import sqlite3
import pandas as pd
from report import records
from run import db_path

def normalized(row):
    item=json.loads(json.dumps(row))
    item.pop("nuisance_seconds")
    for m in item["methods"]:
        m.pop("seconds")
    return item

def snapshot(other):
    with sqlite3.connect(db_path(other,"smoke").as_uri()+"?mode=ro",uri=True) as conn:
        rows=conn.execute("SELECT json FROM replications ORDER BY scenario,n,rep").fetchall()
    partial=[json.loads(x[0]) for x in rows]
    keys=[(r["scenario"],r["n"],r["rep"]) for r in partial]
    assert keys==[("D1",500,k) for k in range(4)]+[("D1",1000,0)],keys
    (other/"paused.json").write_text(json.dumps(partial,allow_nan=False),encoding="utf-8")
    print("Paused after five unique saved tasks.",flush=True)

def verify(other):
    primary=records(root,"smoke")
    single=records(other,"smoke")
    paused=json.loads((other/"paused.json").read_text(encoding="utf-8"))
    assert single[:5]==paused,"Resume modified already committed rows"
    assert [normalized(r) for r in primary]==[normalized(r) for r in single]
    results=root/"results/smoke"
    interval_rows=pd.read_csv(results/"replications/replications.csv")
    assert len(interval_rows)==408
    assert len(interval_rows.drop_duplicates(["scenario","n","rep","method","inference"]))==408
    assert {"failure","valid","valid_point","covered","seed_key","even_weight_share"}<=set(interval_rows.columns)
    assert len(pd.read_csv(results/"replications/failures.csv"))==sum(~interval_rows.valid)
    summary=pd.read_csv(results/"summary/summary.csv")
    assert len(summary)==102 and summary.R.eq(4).all()
    assert ((summary.coverage>=0)&(summary.coverage<=1)).all()
    assert len(list((results/"figures").glob("*.pdf")))==3
    assert not db_path(root,"main").exists() and not db_path(root,"pilot").exists()
    raw=(root/"sources/Rank DML.tex").read_bytes()
    assert raw==(Path.home()/"Downloads/Rank DML.tex").read_bytes()
    print(f"Exact numerical agreement after removing runtime fields: {len(single)} replications.")
    print(f"Resume preserved all {len(paused)} committed records. 408 interval rows, "
          f"{sum(~interval_rows.valid)} invalid, 3 PDFs, 0 pilot or main replications.")

if __name__=="__main__":
    cmd,path=sys.argv[1:]
    {"snapshot":snapshot,"verify":verify}[cmd](Path(path).resolve())

