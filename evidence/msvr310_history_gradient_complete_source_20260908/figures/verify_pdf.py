from collections import Counter
from pathlib import Path
import hashlib
import json
import re
import fitz

root=Path('C:/Users/gb/.codex_tmp/history_gradient_complete_figures_20260908')
receipt=json.loads((root/'plot_receipt.json').read_bytes())
pdf=root/'msvr310_history_candidate_gradients_source.pdf'
assert hashlib.sha256(pdf.read_bytes()).hexdigest()==receipt['artifacts'][pdf.name]
doc=fitz.open(pdf)
assert len(doc)==1
page=doc[0]
lines=[s.strip().replace('\u2212','-') for s in page.get_text().splitlines()]
means=Counter(f"{p['mean']:.3f}" for p in receipt['points'])
actual=Counter(s for s in lines if re.fullmatch(r'-?\d+\.\d{3}',s))
assert actual==means
expected_counts=Counter(f"(n={p['defined']})" for p in receipt['points'])
assert Counter(s for s in lines if s.startswith('(n='))==expected_counts
outside=[]
for block in page.get_text('dict')['blocks']:
    if 'lines' not in block:
        continue
    for line in block['lines']:
        for span in line['spans']:
            x0,y0,x1,y1=span['bbox']
            if x0<0 or y0<0 or x1>page.rect.width or y1>page.rect.height:
                outside.append(span)
assert not outside
render=root/'pdf_inspection.png'
assert not render.exists()
page.get_pixmap(matrix=fitz.Matrix(1.5,1.5),alpha=False).save(render)
result=dict(status='PASS_ALL_SOURCE_PDF_CELL_VALUES_COUNTS_AND_PAGE_BOUNDS',
    pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),pages=len(doc),
    page_size_points=[page.rect.width,page.rect.height],checked_mean_cells=sum(means.values()),
    checked_n_cells=sum(expected_counts.values()),embedded_images=len(page.get_images()),
    out_of_page_text_spans=0,render_sha256=hashlib.sha256(render.read_bytes()).hexdigest(),
    scope='PDF values/counts/bounds and rendering; direct visual inspection still recorded separately.')
(root/'pdf_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
