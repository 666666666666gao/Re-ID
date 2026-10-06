"""Fresh-process continuation of one missing endpoint; reuse unchanged core."""
import argparse
from datetime import datetime
import hashlib,json,os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import diagnose_row_mass_fixed_best as core


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=('semantic','native'),required=True)
    parser.add_argument('--diagnosis',type=Path,required=True)
    args=parser.parse_args()
    diagnosis=args.diagnosis.resolve()
    assert str(ROOT)=='/data/gaob/Re-ID/Trifusion' and os.environ['CUDA_VISIBLE_DEVICES']=='0,1'
    seal=json.loads((diagnosis/'INPUT_SEAL.json').read_text())
    assert all(core.sha(ROOT/name)==digest for name,digest in seal['diagnostic_source_sha256'].items())
    assert all(core.sha(ROOT/name)==digest for name,digest in seal['original_source_sha256'].items())
    campaign=Path(seal['original_campaign'])
    state=json.loads((campaign/'campaign.json').read_text())
    matrix=json.loads((campaign/'accepted_matrix.json').read_text())
    assert state['status']=='COMPLETE' and state['report_exit_code']==0 and matrix['accepted']==6
    core.panel.configure()
    core.panel.previous.require_controls()
    row=next(r for r in matrix['rows'] if (r['dataset'],r['variant'])==('RGBNT100',args.variant))
    assert core.panel.previous.accepted_row(campaign,'RGBNT100',args.variant)==row
    job=next(j for j in state['jobs'] if (j['phase'],j['dataset'],j['variant'])==('full','RGBNT100',args.variant))
    output=diagnosis/('RGBNT100_'+args.variant)
    assert not output.exists()
    core.entry.configure()
    result=core.run_endpoint(row,job,output)
    assert result['dataset']=='RGBNT100' and len(result['modes'])==3
    assert all(core.sha(ROOT/name)==digest for name,digest in seal['diagnostic_source_sha256'].items())
    assert all(core.sha(ROOT/name)==digest for name,digest in seal['original_source_sha256'].items())
    receipt=dict(status='FRESH_PROCESS_ENDPOINT_COMPLETE',variant=args.variant,
        completed_at=datetime.now().astimezone().isoformat(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        original_input_seal_sha256=core.sha(diagnosis/'INPUT_SEAL.json'),
        endpoint_sha256=core.sha(output/'ENDPOINT.json'),
        boundary='Unchanged fixed-best core; fresh configuration/process matching original evaluator. No retraining, original four reruns or tolerance change.')
    core.write(output/'FRESH_PROCESS_RECEIPT.json',receipt)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
