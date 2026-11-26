"""Validate original-file immutability and final artifacts; no experiment calls."""
import os,sys
sys.dont_write_bytecode=True
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import csv,json,hashlib
from concurrent.futures import ThreadPoolExecutor
from html.parser import HTMLParser
from urllib.parse import unquote,urlsplit
import pandas as pd
from audit_records import OUT

def verify_source(row):
    p=Path(row['path'])
    if not p.exists():return str(p),'missing',False
    stat=p.stat()
    if stat.st_size!=int(row['bytes']) or stat.st_mtime_ns!=int(row['mtime_ns']):return str(p),'metadata_changed',False
    digest=row.get('sha256','');checked=len(digest)==64
    if checked:
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
        if h.hexdigest()!=digest:return str(p),'hash_changed',True
    return '',None,checked

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ['href','src'] and v:self.links.append(v)

def main():
    with (OUT/'00_总览与验收/数据来源与文件索引.csv').open(encoding='utf-8-sig',newline='') as f:sources=list(csv.DictReader(f))
    issues=[];hashed=0
    source_report=OUT/'00_总览与验收/原始文件未改动核验.json'
    if '--artifacts-only' in sys.argv:
        previous=json.loads(source_report.read_text(encoding='utf-8'))
        assert previous['original_files']==len(sources) and previous['source_changes']==[]
        issues=previous['source_changes'];hashed=previous['original_hashes_rechecked']
    else:
        with ThreadPoolExecutor(max_workers=12) as pool:
            for path,error,checked in pool.map(verify_source,sources):
                hashed+=checked
                if error:issues.append({'path':path,'error':error})
        source_report.write_text(json.dumps(dict(original_files=len(sources),original_hashes_rechecked=hashed,source_changes=issues),ensure_ascii=False,indent=2),encoding='utf-8')
    print('Original files checked:',len(sources),'hashes:',hashed,'differences:',len(issues),flush=True)
    expected={'T01':120,'T02':480,'T03':1200,'T04':960,'T05':936,'T06':4068,'T07':2760,'T08':4455,'T09':432,'T10':280,'T11':140,'T12':96,'T13':19800,'T14':1980}
    stats=[]
    for tag,count in expected.items():
        paths=list(OUT.glob('*/'+tag+'_*.csv'));assert len(paths)==1,(tag,paths)
        p=paths[0];df=pd.read_csv(p);assert len(df)==count,(p,len(df),count)
        if tag in ['T01','T02','T03','T04','T07']:assert set(df.role)=={'main'} and set(df.stage).issubset({'G1','G2','G3','G4'})
        if tag=='T07':assert set(df.stage)=={'G1','G2','G3','G4'}
        for c in ['coverage','coverage_valid_only','RF_coverage','Spline_coverage']:
            if c in df:assert df[c].dropna().between(0,1).all(),(tag,c)
        if tag in ['T01','T02','T03','T04']:assert (df.attempted==1000).all() and (df.valid_intervals==1000).all()
        if tag in ['T05','T10']:assert (df.bootstrap_resamples==2000).all()
        if tag=='T05':assert (df.common_estimate_success==1000).all() and len(df[df.role=='main'])==720
        if tag=='T06':assert (df.common_interval_success==1000).all()
        if tag=='T10':assert (df.common_success==1000).all()
        if tag=='T13':assert (df[df.applicable]['count']==0).all()
        stats.append(dict(table=tag,rows=len(df),status='PASS'))
    broken=[];linkcount=0
    for p in OUT.rglob('*.html'):
        parser=Links();parser.feed(p.read_text(encoding='utf-8'))
        for href in parser.links:
            if href.startswith(('#','http:','https:','data:','mailto:')):continue
            target=(p.parent/unquote(urlsplit(href).path)).resolve();linkcount+=1
            if not target.exists():broken.append(dict(source=str(p),link=href,resolved=str(target)))
    assert len(list((OUT/'06_论文图表').rglob('*.pdf')))==40
    pdfchecks=json.loads((OUT/'05_运行与复现/PDF结构核验.json').read_text(encoding='utf-8'))
    assert len(pdfchecks)==40 and all(x['embedded_raster_images']==0 and x['pages']==1 for x in pdfchecks)
    result=dict(original_files=len(sources),original_hashes_rechecked=hashed,source_changes=issues,table_checks=stats,html_local_links_checked=linkcount,broken_links=broken,new_vector_PDFs=40,
      scientific_scope='all saved inference records, cached second-stage rep0 per condition, no nuisance refitting or Monte Carlo simulation',status='PASS' if not issues and not broken else 'FAIL')
    (OUT/'00_总览与验收/最终文件核验.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    assert not issues,issues[:5]
    assert not broken,broken[:5]
    print(json.dumps({k:v for k,v in result.items() if k not in ['table_checks','scientific_scope']},ensure_ascii=False))

if __name__=='__main__':main()
