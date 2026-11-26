import sys,json,zipfile,shutil,hashlib
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run import ROOT,save,source_hash

archive=ROOT/'archive'/'forced_restart_20261114';archive.mkdir(exist_ok=True)
folder=ROOT/'runs'/'g0_pilot'
if (archive/'recovery.json').exists():raise RuntimeError('Recovery already prepared; do not repeat')
manifest=json.loads((ROOT/'reports'/'freeze_manifest.json').read_text(encoding='utf-8'))
for rel,expected in manifest['sources'].items():
    if rel!='run.py':assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==expected,rel
bad=[];retained=0
def quarantine(p):
    p=p.resolve();dest=(archive/p.relative_to(ROOT)).resolve()
    assert p.is_relative_to(ROOT) and dest.is_relative_to(ROOT)
    dest.parent.mkdir(parents=True,exist_ok=True);shutil.move(str(p),str(dest))
for path in sorted((folder/'records').glob('*.json')):
    cache=folder/'cache'/(path.stem+'.npz')
    try:
        data=json.loads(path.read_text(encoding='utf-8'));assert len(data['records'])==10
        with zipfile.ZipFile(cache) as z:assert z.testzip() is None
        retained+=1
    except Exception as ex:
        bad.append(dict(record=path.name,reason=str(ex)));quarantine(path)
        if cache.exists():quarantine(cache)
for name in ('config.json','progress.json'):
    p=folder/name
    if p.exists():shutil.copy2(p,archive/name)
config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
old=config['source_hash'];new=source_hash();config.update(source_hash=new,workers=12)
save(folder/'config.json',config)
save(archive/'recovery.json',dict(retained_datasets=retained,quarantined=bad,remaining=6000-retained,previous_source_hash=old,new_source_hash=new,previous_workers=16,new_workers=12,change='run.py CLI default 12 and allowed range 1..12 only; statistical implementation unchanged; same seeds',original_start=config['start']))
print(dict(retained=retained,quarantined=len(bad),remaining=6000-retained,workers=12),flush=True)
