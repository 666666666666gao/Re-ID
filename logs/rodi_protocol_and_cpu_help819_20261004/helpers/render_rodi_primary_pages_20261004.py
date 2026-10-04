from pathlib import Path
import pymupdf

root = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/rodi_primary_pdf_20261004')
document = pymupdf.open(root / 'RoDI.pdf')
for number in (6, 7, 12, 13):
    path = root / f'page_{number}.png'
    document[number - 1].get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5)).save(path)
    print(path)
