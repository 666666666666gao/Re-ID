from pathlib import Path
import hashlib,runpy,sys
stage=Path('/tmp/trifusion_global_task_role_cpu_20261004')
for name,digest in {'trifusion/global_task_role_heads.py': '2298763b70b48718411db52e55f32dabddade5af0e56457a85eeabab4f9fcefd', 'check_global_task_role_heads.py': 'f5fc4bd385574488a6d8e59ccfbf5832f23408026c47fbc07c2298a944dbe127'}.items():assert hashlib.sha256((stage/name).read_bytes()).hexdigest()==digest
sys.path.insert(0,'/data/gaob/Re-ID/Trifusion/modeling')
import trifusion
trifusion.__path__.insert(0,str(stage/'trifusion'))
sys.argv=[str(stage/'check_global_task_role_heads.py'),'--output',str(stage/'WITNESS.json')]
runpy.run_path(sys.argv[0],run_name='__main__')
