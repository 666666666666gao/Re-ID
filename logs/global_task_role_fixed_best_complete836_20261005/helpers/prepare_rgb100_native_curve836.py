import ast
from pathlib import Path

private=Path('C:/Users/gb/.codex_tmp')
source=(private/'plot_global_task_role_rgb100_semantic833.py').read_text(encoding='utf-8')
for old,new in [('global_task_role_rgb100_semantic_full829','global_task_role_rgb100_native_full829'),
 ('global_task_role_rgb100_semantic_curve833','global_task_role_rgb100_native_curve836'),
 ('full_semantic_RGBNT100','full_native_RGBNT100'),('== 5','== 26'),('epoch 5','epoch 26'),
 ('1.8082-point','1.2152-point'),('SEMANTIC_50_EPOCHS','NATIVE_50_EPOCHS'),('semantic','native')]:
    assert old in source,old
    source=source.replace(old,new)
ast.parse(source)
target=private/'plot_global_task_role_rgb100_native836.py'
assert not target.exists()
target.write_text(source,encoding='utf-8')
print(target)
