"""Inspect and render generated PDF figures (bundled document Python)."""
from pathlib import Path
import json,subprocess,concurrent.futures
from pypdf import PdfReader
from PIL import Image,ImageDraw
OUT=Path(__file__).resolve().parents[2]
RUNTIME=Path(r'C:\Program Files\WindowsApps\OpenAI.CodexPrimaryRuntime.v26-1112-737-0_26.1112.737.0_x64__3k8sg7r9htsxt\dependencies')
PRE=OUT/'06_论文图表/预览';PRE.mkdir(exist_ok=True)
files=sorted((OUT/'06_论文图表').rglob('*.pdf'))
def inspect(p):
    pdf=PdfReader(p);assert len(pdf.pages)==1
    page=pdf.pages[0];images=list(page.images);assert not images,str(p)
    target=PRE/p.stem
    dpi='120' if p.stem in ['vector_d3_rho0.5_learned_vector_RMSE','RF_RMSE_ratio_learned'] else '50'
    subprocess.run([str(RUNTIME/'native/poppler/Library/bin/pdftoppm.exe'),'-f','1','-singlefile','-r',dpi,'-png',str(p),str(target)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    return {'path':str(p),'pages':1,'embedded_raster_images':len(images),'text_characters':len(page.extract_text()),'preview':str(target)+'.png'}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(inspect,files))
for block in range((len(rows)+7)//8):
    canvas=Image.new('RGB',(1440,1460),'#dddddd');draw=ImageDraw.Draw(canvas)
    for i,r in enumerate(rows[block*8:block*8+8]):
        im=Image.open(r['preview']);im.thumbnail((700,330));x=(i%2)*720;y=(i//2)*365
        canvas.paste(im,(x+(700-im.width)//2,y+25));draw.text((x+8,y+5),Path(r['path']).name,fill='black')
    canvas.save(PRE/f'contact_{block+1}.png')
(OUT/'05_运行与复现/PDF结构核验.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'PDFs':len(rows),'single_page_vector':True,'contact_sheets':(len(rows)+7)//8}))
