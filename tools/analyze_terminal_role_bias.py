"""CPU-only algebra on sealed terminal LayerNorm bias and saved region tensors."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import time

import torch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_regions(regions, weights):
    batch_size = regions.shape[0]
    parts = [regions[:, role].reshape(batch_size, 3, 2, 2, 2, 2, 128)
             .mean(dim=(3, 5)).reshape(batch_size, 3, 4, 128) for role in range(3)]
    inputs = torch.cat(parts, dim=-1)
    return torch.einsum('bmrf,mrof->bmro', inputs, weights).reshape(batch_size, 1536)


def analyze(seal_path, output):
    started = time.perf_counter()
    assert os.environ['CUDA_VISIBLE_DEVICES'] == ''
    assert not output.exists()
    seal = json.loads(seal_path.read_text())
    assert seal['schema'] == 'trifusion-terminal-bias-algebra-v1'
    root = Path('/data/gaob/Re-ID/Trifusion')
    assert all(sha(root/name) == value for name, value in seal['source_sha256'].items())
    assert all(sha(Path(name)) == value for name, value in seal['artifact_sha256'].items())
    records = {}
    with torch.no_grad():
        for row in seal['rows']:
            payload = torch.load(row['checkpoint'], map_location='cpu', weights_only=True)
            assert payload['epoch'] == row['selected_epoch'] == 8
            state = payload['state']
            beta = state['evidence_model.roles.output_norms.2.bias']
            weight = state['evidence_model.readout.weight']
            gain = float(state['evidence_model.readout_gain'])
            assert beta.shape == (128,) and weight.shape == (1536, 384)
            assert beta.dtype == weight.dtype == torch.float32 and gain != 0.0
            assert all(torch.isfinite(value).all() for value in (beta, weight))
            readout = torch.load(row['readout'], map_location='cpu', weights_only=True)
            assert torch.equal(weight, readout['readout_weight']) and gain == readout['gain']
            weights = weight.double().reshape(3, 4, 128, 384)
            # Only the terminal Mamba output bias changes; CNN/T and all bridges stay fixed.
            mapping = weights[..., 256:384].reshape(1536, 128)
            bias_c = mapping @ beta.double()
            singular_values = torch.linalg.svdvals(mapping)
            rank_tolerance = max(mapping.shape) * torch.finfo(torch.float64).eps * singular_values[0]
            splits = {}
            for split in ('query', 'gallery'):
                cache = torch.load(row['features'][split], map_location='cpu', weights_only=True)
                regions = cache['regions'].double()
                c, h = cache['correction'].double(), cache['raw_fused'].double()
                assert regions.shape[1:] == (3, 3, 16, 128)
                natural = read_regions(regions, weights)
                reconstruction_error = float((natural - c).abs().max())
                assert reconstruction_error <= 1e-5
                # A fixed first-coordinate unit change checks the actual linear path.
                changed = regions.clone()
                changed[:, 2, :, :, 0] += 1.0
                delta = read_regions(changed, weights) - natural
                translation_error = float((delta - mapping[:, 0]).abs().max())
                assert translation_error <= 1e-12
                mean_c = c.mean(dim=0)
                mean_norm = float(mean_c.norm())
                energy = float(c.square().sum())
                assert mean_norm > 0 and energy > 0
                norm_h = h.norm(dim=1, keepdim=True)
                assert (norm_h > 0).all()
                unit_h = h / norm_h
                # Analytic deployed-vector response; no changed ranking or task autograd.
                direction_h = gain * mapping[:, 0]
                response = (direction_h - unit_h * (unit_h @ direction_h)[:, None]) / norm_h
                splits[split] = dict(
                    samples=len(c), readout_reconstruction_max_abs=reconstruction_error,
                    fixed_unit_bias_translation_max_abs=translation_error,
                    correction_mean_norm=mean_norm,
                    correction_rms_norm=float((energy / len(c)) ** 0.5),
                    terminal_bias_c_norm=float(bias_c.norm()),
                    bias_c_norm_over_common_c_norm=float(bias_c.norm()) / mean_norm,
                    isolated_bias_energy_over_c_energy=len(c) * float(bias_c.square().sum()) / energy,
                    deployed_unit_bias_response_rms_norm=float(response.square().sum(dim=1).mean().sqrt()),
                    deployed_unit_bias_response_max_norm=float(response.norm(dim=1).max()),
                    note='Common mean is descriptive only; never used for model calibration, ranking or a test-mean transformation. Component/reference energy ratio is not an explained-information fraction.')
                del cache, regions, natural, changed, delta, c, h, response
            records[row['exposure']] = dict(
                selected_epoch=8, gain=gain, terminal_bias_norm=float(beta.double().norm()),
                terminal_bias_max_abs=float(beta.abs().max()),
                mapping_shape=list(mapping.shape), mapping_frobenius_norm=float(mapping.norm()),
                singular_value_max=float(singular_values[0]), singular_value_min=float(singular_values[-1]),
                numerical_mapping_rank=int((singular_values > rank_tolerance).sum()),
                numerical_rank_tolerance=float(rank_tolerance),
                singular_values=singular_values.tolist(), splits=splits)
            del payload, state, beta, weight, readout, weights, mapping, singular_values
    assert all(sha(root/name) == value for name, value in seal['source_sha256'].items())
    assert all(sha(Path(name)) == value for name, value in seal['artifact_sha256'].items())
    result = dict(
        status='SEALED_TERMINAL_BIAS_ALGEBRA_COMPLETE', schema=seal['schema'],
        input_seal_sha256=sha(seal_path), records=records, torch_version=torch.__version__,
        scope='Existing checkpoint state and saved final regions only; no model construction, NN forward, loss/autograd, optimizer or altered retrieval score',
        neural_forward_calls=0, optimizer_updates=0, new_training_started=False,
        elapsed_seconds_through_result=time.perf_counter()-started,
        completed_at=datetime.now().astimezone().isoformat(),
        limits='Source algebra identifies only the terminal LN bias parameter subspace. It does not explain actual drift, total common correction, harm, or a useful method. Parameter weight decay still acts. No training gradients or histories are reconstructed.')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'], records=records), allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seal', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    analyze(args.seal, args.output)
