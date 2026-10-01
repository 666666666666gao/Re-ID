"""Prepare exact public-visual resets of the three saved module-free baselines."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.queue_correspondence_roles import BASELINES, WEIGHTS
from tools.collect_correspondence_roles import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output_dir.exists()
    clip_path = WEIGHTS / 'ViT-B-16.pt'
    public = torch.jit.load(str(clip_path), map_location='cpu').state_dict()
    visual = {key.removeprefix('visual.'): value for key, value in public.items()
              if key.startswith('visual.')}
    assert len(visual) == 152
    args.output_dir.mkdir(parents=True)
    rows = {}
    prefix = 'clip_vision_encoder.base.'
    for dataset, (filename, expected) in BASELINES.items():
        original = WEIGHTS / filename
        assert sha(original) == expected
        saved = torch.load(original, map_location='cpu', weights_only=True)
        assert {key.removeprefix(prefix) for key in saved if key.startswith(prefix)} == set(visual)
        reset = {key: value.clone() for key, value in saved.items()}
        grid = (16, 8) if dataset == 'RGBNT201' else (8, 16)
        changed = []
        for name, value in visual.items():
            key = prefix + name
            if name == 'positional_embedding':
                patches = value[1:].reshape(1, 14, 14, 768).permute(0, 3, 1, 2)
                patches = F.interpolate(patches, size=grid, mode='bilinear')
                value = torch.cat((value[:1], patches.permute(0, 2, 3, 1).reshape(128, 768)))
            value = value.to(dtype=saved[key].dtype)
            assert value.shape == saved[key].shape and bool(torch.isfinite(value).all())
            reset[key] = value.clone()
            changed.append({'key': key, 'shape': list(value.shape), 'different_from_reid': not torch.equal(value, saved[key])})
        unchanged = [key for key in saved if not key.startswith(prefix)]
        assert all(torch.equal(reset[key], saved[key]) for key in unchanged)
        assert 'clip_vision_encoder.cv_embed' in unchanged
        target = args.output_dir / (dataset + '_public_visual.pth')
        torch.save(reset, target)
        restored = torch.load(target, map_location='cpu', weights_only=True)
        assert set(restored) == set(saved)
        assert all(torch.equal(restored[key], reset[key]) for key in reset)
        rows[dataset] = {'reid_visual': {'path': str(original), 'sha256': expected},
                         'public_visual': {'path': str(target), 'sha256': sha(target)},
                         'public_visual_tensors': changed, 'unchanged_nonvisual_keys': unchanged,
                         'all_nonvisual_bitwise_equal': True, 'strict_save_reload_equal': True,
                         'grid': list(grid)}
    result = {'schema': 'trifusion-visual-start-inputs-v1',
              'prepared_at': datetime.now().astimezone().isoformat(),
              'public_clip_path': str(clip_path), 'public_clip_sha256': sha(clip_path),
              'source_sha256': sha(Path(__file__)), 'datasets': rows,
              'boundary': 'Only152visual tensors replaced; trained camera and other states retained. Not a trained baseline or wholly public-only system.'}
    (args.output_dir / 'INPUTS.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(args.output_dir), 'datasets': list(rows),
                      'inputs_sha256': sha(args.output_dir / 'INPUTS.json')}))


if __name__ == '__main__':
    main()
