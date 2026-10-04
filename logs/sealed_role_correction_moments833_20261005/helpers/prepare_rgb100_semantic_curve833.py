"""Adapt the existing descriptive plot to the already accepted RGBNT100 endpoint."""
import ast
from pathlib import Path

root = Path('C:/Users/gb/.codex_tmp')
source = (root / 'plot_global_task_role_semantic827.py').read_text(encoding='utf-8')
replacements = (
    ('global_task_role_first_full826', 'global_task_role_rgb100_semantic_full829'),
    ('global_task_role_semantic_curve827', 'global_task_role_rgb100_semantic_curve833'),
    ('RGBNT201', 'RGBNT100'),
    ('2649', '3129'),
    ('== 8', '== 5'),
    ('epoch 8', 'epoch 5'),
    ('2.8973-point', '1.8082-point'),
)
for old, new in replacements:
    assert old in source, old
    source = source.replace(old, new)
ast.parse(source)
target = root / 'plot_global_task_role_rgb100_semantic833.py'
assert not target.exists()
target.write_text(source, encoding='utf-8')
print(target)
