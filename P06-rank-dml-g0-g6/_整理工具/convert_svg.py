"""Convert existing SVGs to vector PDFs without recalculating experiment data."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import sys, json, hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE / 'dependencies'))


def convert(source):
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF
    from pypdf import PdfReader
    source = Path(source)
    target = source.with_suffix('.pdf')
    drawing = svg2rlg(str(source))
    if drawing is None:
        raise ValueError(str(source))
    renderPDF.drawToFile(drawing, str(target))
    pdf = PdfReader(target)
    assert len(pdf.pages) == 1 and len(pdf.pages[0].images) == 0
    page = pdf.pages[0]
    assert abs(float(page.mediabox.width)-drawing.width) < .1
    assert abs(float(page.mediabox.height)-drawing.height) < .1
    return dict(svg=str(source.relative_to(ROOT)), pdf=str(target.relative_to(ROOT)),
                svg_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                pdf_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                vector=True, pages=1)


if __name__ == '__main__':
    paths = [p for g in range(4) for p in (ROOT/f'G{g}实验'/'图').rglob('*.svg')]
    assert len(paths) == 1344
    results = []
    with ProcessPoolExecutor(max_workers=12) as pool:
        jobs = [pool.submit(convert, str(p)) for p in paths]
        for future in as_completed(jobs):
            results.append(future.result())
            if len(results) % 100 == 0 or len(results) == len(paths):
                print(f'Converted and verified {len(results)}/{len(paths)}', flush=True)
    (HERE/'SVG转PDF验收.json').write_text(json.dumps(dict(count=len(results),
        converter='svglib 2.4.0 + ReportLab', files=sorted(results,key=lambda r:r['svg'])),
        ensure_ascii=False,indent=2),encoding='utf-8')
