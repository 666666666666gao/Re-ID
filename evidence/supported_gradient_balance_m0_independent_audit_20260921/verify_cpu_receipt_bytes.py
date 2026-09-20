import os
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
os.nice(10)
from pathlib import Path
import json,hashlib,datetime
p=Path('/root/trifusion-storage/artifacts/msvr310_supported_gradient_balance_v1_r2_seed42_1381639/m0_cpu.json')
b=p.read_bytes();t=b.decode('utf-8')
print(json.dumps(dict(status='RAW_CPU_RECEIPT_BYTE_WITNESS',path=str(p),bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),
    lf=b.count(b'\n'),crlf=b.count(b'\r\n'),carriage_returns=b.count(b'\r'),
    text_read_then_utf8_sha256=hashlib.sha256(p.read_text(encoding='utf-8').encode('utf-8')).hexdigest(),
    status_field=json.loads(t)['status'],generated_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    model_forwards=0,optimizer_updates=0,image_reads=0,q1_result_reads=0),indent=2))
