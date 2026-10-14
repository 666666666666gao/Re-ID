"""Reuse the fixed-best measurements for the single absolute-value condition."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import diagnose_pretrained_value_delta_best as previous
from tools import run_absolute_pretrained_value as producer

SCHEMA='trifusion-absolute-pretrained-value-fixed-best-diagnosis-v1'


def require_inputs(seal):
    assert seal['schema']==SCHEMA
    for name,digest in seal['source_sha256'].items():
        assert producer.previous.inner.runner.sha256(ROOT/name)==digest,name
    for name,digest in seal['artifact_sha256'].items():
        assert producer.previous.inner.runner.sha256(Path(name))==digest,name
    state=json.loads(Path(seal['campaign']).read_text())
    assert state['status']=='COMPLETE' and state['accepted']==state['expected']==1
    assert json.loads(Path(seal['controller_exit']).read_text())['exit_code']==0
    assert json.loads(Path(seal['report']).read_text())['status']=='COMPLETE_ABSOLUTE_VALUE_SAVED_DISTANCE_DIAGNOSIS'


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',choices=('uniform_reference',),required=True)
    parser.add_argument('--seal',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.seal=args.seal.resolve();args.output_dir=args.output_dir.resolve()
    producer.inner=producer.previous.inner
    previous.SCHEMA=SCHEMA
    previous.producer=producer
    previous.require_inputs=require_inputs
    previous.diagnose(args,json.loads(args.seal.read_text()))
