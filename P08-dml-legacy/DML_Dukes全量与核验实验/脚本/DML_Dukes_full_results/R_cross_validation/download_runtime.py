import pathlib,urllib.request,hashlib,json
ROOT=pathlib.Path(__file__).resolve().parent
(ROOT/'downloads').mkdir(exist_ok=True)
items={
 'R-4.6.1-win.exe':'https://mirrors.tuna.tsinghua.edu.cn/CRAN/bin/windows/base/R-4.6.1-win.exe',
 'dri_functions.r':'https://raw.githubusercontent.com/ordukes/DRInference/e7dfb038117aea3d95ca1371e2e85557ff9f25d8/dri_functions.r',
}
records=[]
for name,url in items.items():
    path=ROOT/'downloads'/name
    if not path.exists():
        print('Downloading '+name,flush=True)
        with urllib.request.urlopen(url,timeout=60) as remote, path.with_suffix(path.suffix+'.part').open('wb') as target:
            while chunk:=remote.read(1024*1024): target.write(chunk)
        path.with_suffix(path.suffix+'.part').replace(path)
    records.append(dict(file=name,url=url,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    print(records[-1],flush=True)
(ROOT/'downloads/manifest.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
