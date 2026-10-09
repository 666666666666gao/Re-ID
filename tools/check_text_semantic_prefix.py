"""Fixed FP32 full77/prefix12 output and pseudo-word VJP qualification."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'modeling'))
from tools.run_signal_baseline_dev import _configure_signal_source
from trifusion.text_semantic_prior import (
    FrozenTextPackage, PERSON_TOKENS, VEHICLE_TOKENS, text_package_state,
)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('signal-source', 'clip-weight', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    args.signal_source, args.clip_weight, args.output = (
        path.resolve() for path in (args.signal_source, args.clip_weight, args.output))
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '0,1' and not args.output.exists()
    _configure_signal_source(args.signal_source)
    from modeling.clip import model as clip_module
    assert Path(clip_module.__file__).resolve() == args.signal_source / 'modeling/clip/model.py'
    public = torch.jit.load(str(args.clip_weight), map_location='cpu').state_dict()
    rows = []
    for package in ('pretrained', 'random'):
        state = text_package_state(public, package)
        for label, tokens in (('person', PERSON_TOKENS), ('vehicle', VEHICLE_TOKENS)):
            text = FrozenTextPackage(clip_module, state, tokens).to('cuda:0')
            delta = ((torch.arange(4096, device='cuda:0', dtype=torch.float32)
                      .reshape(2, 4, 512) / 4096) - .5) * .02
            upstream = torch.sin(torch.arange(1024, device='cuda:0', dtype=torch.float32)).reshape(2, 512)
            outputs, gradients = {}, {}
            before = {name: value.detach().clone() for name, value in text.state_dict().items()}
            for length in (12, 77):
                value = delta.clone().requires_grad_(True)
                template = text.template_embedding[None].expand(2, -1, -1)
                embeddings = torch.cat((template[:, :5], template[:, 5:9] + value, template[:, 9:]), dim=1)
                if length == 77:
                    padding = state['token_embedding.weight'][0].to('cuda:0').expand(2, 65, 512)
                    embeddings = torch.cat((embeddings, padding), dim=1)
                with torch.autocast('cuda', enabled=False):
                    output = text.encode_embeddings(embeddings)
                    gradient, = torch.autograd.grad(output, value, upstream)
                assert bool(torch.isfinite(output).all()) and bool(torch.isfinite(gradient).all())
                assert bool(gradient.count_nonzero())
                outputs[length], gradients[length] = output.detach(), gradient.detach()
            assert torch.allclose(outputs[12], outputs[77], atol=1e-5, rtol=1e-5)
            assert torch.allclose(gradients[12], gradients[77], atol=1e-4, rtol=1e-4)
            assert all(torch.equal(before[name], value) for name, value in text.state_dict().items())
            assert all(p.grad is None and not p.requires_grad for p in text.parameters())
            rows.append(dict(package=package, template=label,
                output_max_abs_difference=float((outputs[12] - outputs[77]).abs().max()),
                input_vjp_max_abs_difference=float((gradients[12] - gradients[77]).abs().max()),
                frozen_state_unchanged=True, persistent_tensors=len(text.state_dict())))
            del text, before, outputs, gradients
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(status='PREFIX_OUTPUT_AND_INPUT_VJP_PASS', rows=rows,
        clip_sha256=sha(args.clip_weight), neural_forwards=8, input_vjps=8, optimizer_updates=0,
        output_atol=1e-5, output_rtol=1e-5, input_vjp_atol=1e-4, input_vjp_rtol=1e-4,
        completed_at=datetime.now().astimezone().isoformat(),
        boundary='Fixed synthetic input engineering check; not real model M0 or retrieval result.'), indent=2) + '\n')
    print(json.dumps(dict(status='PREFIX_OUTPUT_AND_INPUT_VJP_PASS', cases=len(rows))), flush=True)


if __name__ == '__main__':
    main()
