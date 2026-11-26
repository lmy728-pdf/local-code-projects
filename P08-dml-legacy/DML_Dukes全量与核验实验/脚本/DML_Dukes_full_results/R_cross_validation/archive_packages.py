"""Archive the exact installed Windows binaries, without installing anything."""
import concurrent.futures, hashlib, json, pathlib, urllib.request
import pandas as pd
ROOT=pathlib.Path(__file__).resolve().parent
table=pd.read_csv(ROOT/'r_package_versions.csv')
packages=table[table.LibPath.str.endswith('/R_library')]
dest=ROOT/'downloads'/'packages';dest.mkdir(exist_ok=True)
def fetch(row):
    name=f'{row.Package}_{row.Version}.zip'
    url=f'https://mirrors.tuna.tsinghua.edu.cn/CRAN/bin/windows/contrib/4.6/{name}'
    path=dest/name
    if not path.exists():
        with urllib.request.urlopen(url,timeout=90) as response:path.write_bytes(response.read())
    return dict(package=row.Package,version=row.Version,file='downloads/packages/'+name,url=url,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
    records=list(pool.map(fetch,packages.itertuples(index=False)))
(ROOT/'package_lock.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
print('Archived',len(records),'exact package binaries')
