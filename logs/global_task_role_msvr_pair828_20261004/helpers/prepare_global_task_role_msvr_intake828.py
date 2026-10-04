"""Prepare dataset-specific saved-distance intake from already-used collectors."""
from pathlib import Path
import ast
p=Path('C:/Users/gb/.codex_tmp')
for source,target,oldpacket,newpacket in (
 ('collect_global_task_role_first_full826.py','collect_global_task_role_msvr_semantic828.py','global_task_role_first_full826','global_task_role_msvr_semantic_full828'),
 ('collect_global_task_role_native_full827.py','collect_global_task_role_msvr_native828.py','global_task_role_native_full827','global_task_role_msvr_native_full828')):
 s=(p/source).read_text(encoding='utf-8').replace('RGBNT201','MSVR310').replace(oldpacket,newpacket)
 s=s.replace("==1 for r in steps","==3 for r in steps")
 s=s.replace('FIRST_FORMAL50_AND_FIRST_STRICT_EVALUATION_VERIFIED','MSVR310_SEMANTIC_FORMAL50_AND_FIRST_STRICT_EVALUATION_VERIFIED')
 s=s.replace('Only first endpoint closed.','This MSVR310 semantic endpoint closed.')
 s=s.replace('Only MSVR310 pair closed;','RGBNT201 and MSVR310 pairs closed;')
 t=p/target;assert not t.exists();ast.parse(s);t.write_text(s,encoding='utf-8')
 print(t)
