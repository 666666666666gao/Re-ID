"""F3 initial real-record check: equal state/deployment/CE, active metric change."""
import argparse
import json
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools import run_metric_feature_scale as scale
from tools import queue_metric_feature_scale as panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign',type=Path,required=True)
    parser.add_argument('--dataset',choices=panel.DATASETS,required=True)
    args = parser.parse_args()
    panel.configure()
    scale.configure()
    campaign = args.campaign.resolve()
    protocol_path = panel.PROTOCOLS/f'{args.dataset}.json'
    protocol = scale.foundation.runner.read_protocol(protocol_path,args.dataset)
    bindings = [panel.expected_binding(campaign,args.dataset,variant) for variant in panel.RECIPES]
    assert bindings[0]['initial_model_state_sha256']==bindings[1]['initial_model_state_sha256']
    historical_path = panel.PREVIOUS/'initialization'/f'{args.dataset}_normalized.json'
    historical = json.loads(historical_path.read_text())['binding']
    assert historical['initial_model_state_sha256']==bindings[0]['initial_model_state_sha256']
    outputs = []
    records = scale.foundation.runner.records_for(protocol,'train')[:2]
    loader = scale.foundation.runner.loader_for(protocol,records,training=False,method='PLAIN_V8')
    batch = scale.foundation.runner._eval_batch(next(iter(loader)),args.dataset)
    for variant in panel.RECIPES:
        values = argparse.Namespace(dataset=args.dataset,recipe=variant,seed=42,epochs=50,
            protocol=protocol_path,signal_source=panel.SOURCE,clip_weight=panel.WEIGHTS/'ViT-B-16.pt',
            initialization=campaign/'initialization'/f'{args.dataset}_{variant}.json',
            output_dir=ROOT/'trained-model/f3_pair_unused',
            baseline_sha256=scale.foundation.runner.sha256(panel.WEIGHTS/'ViT-B-16.pt'))
        model,_cfg,binding = scale.build_core(values,protocol)
        assert binding==panel.expected_binding(campaign,args.dataset,variant)
        model.eval()
        with torch.inference_mode():
            deployed = model(batch)
            auxiliary = model(batch,return_aux=True)
            assert torch.equal(deployed,auxiliary['ce_feature'])
            outputs.append({key:value.cpu() for key,value in dict(deployment=deployed,**auxiliary).items()})
        del model
        torch.cuda.empty_cache()
    for key in ('deployment','ce_feature','logits'):
        assert torch.equal(outputs[0][key],outputs[1][key]),key
    assert torch.equal(outputs[0]['fused'],outputs[0]['ce_feature'])
    assert torch.allclose(torch.nn.functional.normalize(outputs[1]['fused'],dim=1),outputs[1]['ce_feature'],atol=1e-7,rtol=0)
    assert not torch.equal(outputs[0]['fused'],outputs[1]['fused'])
    output = campaign/f'initial_forward_pair_{args.dataset}.json'
    assert not output.exists()
    output.write_text(json.dumps({'schema':panel.SCHEMA,'status':'INITIAL_FORWARD_PAIR_PASS','dataset':args.dataset,
        'records':records,'initial_model_state_sha256':bindings[0]['initial_model_state_sha256'],
        'historical_f2_initialization_sha256':scale.foundation.runner.sha256(historical_path),
        'deployment_prediction_sha256':scale.foundation.clean.tensor_digest(outputs[0]['deployment']),
        'ce_prediction_sha256':scale.foundation.clean.tensor_digest(outputs[0]['logits']),
        'deployment_max_abs_difference':float((outputs[0]['deployment']-outputs[1]['deployment']).abs().max()),
        'ce_logits_max_abs_difference':float((outputs[0]['logits']-outputs[1]['logits']).abs().max()),
        'metric_feature_norms':{variant:outputs[i]['fused'].norm(dim=1).tolist() for i,variant in enumerate(panel.RECIPES)},
        'boundary':'Two real train records and identical prepared batch, eval mode, no optimizer or official scoring; CE/deployment equality and active metric-input change only.'},indent=2)+'\n')
    print(output,flush=True)


if __name__=='__main__':
    main()
