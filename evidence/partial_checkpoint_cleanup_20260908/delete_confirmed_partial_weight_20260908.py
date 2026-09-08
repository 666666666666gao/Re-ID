from pathlib import Path
from datetime import datetime
import hashlib,json,os,shutil,zipfile

base=Path('/root/autodl-tmp/trifusion-v2/artifacts/rgbnt100_signal_source_oof_v1_r2_seed42_e699eac').resolve(strict=True)
target=base/'m0/fold_0/signal_m0.pth'
assert not target.is_symlink() and target.resolve(strict=True).is_relative_to(base)
assert target.stat().st_size==101712000
digest=hashlib.sha256(target.read_bytes()).hexdigest()
assert digest=='ad2d141c92ff38d40f9b75d8ee98c438a8d419d16ba5f2dde1eb81c0c1da4557'
assert not zipfile.is_zipfile(target)
pipeline=json.loads((base/'terminal.json').read_bytes())
assert pipeline['exit_code']==1 and pipeline['stages'][-1]['stage']=='m0' and pipeline['stages'][-1]['exit_code']==1
assert not Path('/proc/84049').exists()
log=(base/'m0.log').read_text()
assert 'PytorchStreamWriter failed writing file' in log and 'unexpected pos 101687680' in log
repo=Path('/root/autodl-tmp/trifusion-v2/TriFusion-ReID')
for p in (repo/'configs').rglob('*'):
    if p.is_file():assert str(base).encode() not in p.read_bytes(),str(p)
successor=Path('/root/trifusion-storage/artifacts/rgbnt100_signal_source_oof_v1_r2_storage_retry_seed42_20260906')
retained=[]
for fold in range(3):
    for rel in (f'm0/fold_{fold}/signal_m0.pth',f'baseline/fold_{fold}/signal_epoch30.pth'):
        p=successor/rel
        assert p.stat().st_size>300000000 and zipfile.is_zipfile(p)
        retained.append(dict(path=str(p),bytes=p.stat().st_size))
receipt_path=base/'partial_checkpoint_cleanup_20260908.json'
assert not receipt_path.exists()
before=shutil.disk_usage(base).free
receipt=dict(status='VERIFIED_PARTIAL_CHECKPOINT_PENDING_DELETE',at=datetime.now().astimezone().isoformat(),path=str(target),bytes=target.stat().st_size,sha256=digest,reason='Failed interrupted torch ZIP write; failed M0 exited; invalid ZIP; no active config references; six complete successor checkpoints retained.',retained=retained,free_before=before)
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
target.unlink()
assert not target.exists()
receipt.update(status='DELETED_CONFIRMED_PARTIAL_CHECKPOINT',deleted_files=1,deleted_bytes=101712000,free_after=shutil.disk_usage(base).free)
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
