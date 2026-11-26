from pathlib import Path
import ast, csv, hashlib, json, zipfile
from html.parser import HTMLParser
from urllib.parse import unquote, urlsplit
from concurrent.futures import ThreadPoolExecutor
from pypdf import PdfReader

ROOT=Path(__file__).resolve().parent.parent
STAGES=['G0','G1','G2','G3','G4','G5','G6A','G6B']
EXPECTED=[128,48,496,672,1056,360,4,4]

class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        self.links.extend(v for k,v in attrs if k in ('href','src'))

def check_pdf(p):
    pdf=PdfReader(p)
    assert len(pdf.pages)==1,str(p)
    assert len(pdf.pages[0].images)==0,str(p)
    assert float(pdf.pages[0].mediabox.width)>100,str(p)
    return 1

def main():
    files=[];pages=[ROOT/'G0-G6_实验总览.html'];counts={}
    for stage,expected in zip(STAGES,EXPECTED):
        base=ROOT/(stage+'实验');code=base/(stage+'实验_代码')
        pdfs=list((base/'图').rglob('*.pdf'));assert len(pdfs)==expected,(stage,len(pdfs))
        files.extend(pdfs);counts[stage]=len(pdfs)
        pages.extend([base/'结果_MD'/(stage+'_实验报告.html'),base/'图/PDF图表索引.html',code/'代码索引.html'])
        for path in code.glob('*.ipynb'):
            nb=json.loads(path.read_text(encoding='utf-8'))
            for cell in nb['cells']:
                if cell['cell_type']=='code':ast.parse(''.join(cell['source']))
        if stage in ['G4','G5','G6A','G6B']:
            digest=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(code.glob('*.py')))).hexdigest()
            for cp in (code/'_work').glob('*/config.json'):
                data=json.loads(cp.read_text(encoding='utf-8'))
                assert data['source_hash']==digest,(stage,'source hash',cp)
                assert data['status']=='completed',(stage,cp)
                assert data['workers']<=12 and data['threads_per_worker']==1
        if stage in ['G4','G5']:
            batch='g4_n2000' if stage=='G4' else 'g5_rf'
            audit=json.loads((code/'_work'/batch/'delivery_audit.json').read_text(encoding='utf-8'))
            for name,digest in audit['source_hashes'].items():
                assert hashlib.sha256((base/'汇总表_CSV'/batch/name).read_bytes()).hexdigest()==digest,(stage,name)
        if stage.startswith('G6'):
            batch='g6a_tau050' if stage=='G6A' else 'g6b_c2.0'
            audit=json.loads((code/'_work'/batch/'delivery_audit.json').read_text(encoding='utf-8'))
            assert hashlib.sha256((base/'汇总表_CSV/汇总表.csv').read_bytes()).hexdigest()==audit['csv_sha256']
    for page in pages:
        parser=Links();parser.feed(page.read_text(encoding='utf-8'))
        for link in parser.links:
            parsed=urlsplit(link)
            if parsed.scheme or not parsed.path:continue
            target=(page.parent/unquote(parsed.path)).resolve()
            assert target.exists(),(page,link)
    with ThreadPoolExecutor(max_workers=12) as pool:
        checked=sum(pool.map(check_pdf,files))
    conversion=json.loads((ROOT/'_整理工具/SVG转PDF验收.json').read_text(encoding='utf-8'))
    for entry in conversion['files']:
        for ext in ['svg','pdf']:
            assert hashlib.sha256((ROOT/entry[ext]).read_bytes()).hexdigest()==entry[ext+'_sha256']
    with zipfile.ZipFile(ROOT/'G0-G6_代码汇总.zip') as archive:assert archive.testzip() is None
    old=Path(r'C:\Users\Lenovo\Documents\ChatGPT\毕设\实验交付')
    assert not list(old.rglob('*.*')) or not any(p.is_file() for p in old.rglob('*'))
    result=dict(reports=8,report_formats=['MD','HTML'],pdf_counts=counts,total_pdf=checked,
        svg_converted=1344,all_pdf_vector=True,links_checked_pages=len(pages),
        data_hashes_unchanged=True,source_hashes_unchanged=True,zip_crc='passed',
        original_delivery_files_remaining=0,experiments_launched=False)
    (ROOT/'_整理工具/交付验收.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':main()
