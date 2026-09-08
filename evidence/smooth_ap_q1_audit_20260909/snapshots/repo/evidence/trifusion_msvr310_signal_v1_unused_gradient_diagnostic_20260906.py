import hashlib
import inspect
import json
from pathlib import Path
import sys
import time
import torch

root = Path("/root/autodl-tmp/trifusion-v2/TriFusion-ReID")
sys.path.insert(0, str(root))
from tools.train_msvr310_signal_oof import configure, records_for, loader_for, new_model, sha256
from tools.run_signal_preserving_v5 import _set_seed, _module_state_sha256
from tools.build_v12_complete_path_oof_targets import _signal_training_loss

out = Path("/root/autodl-tmp/trifusion-v2/artifacts/msvr310_signal_source_oof_v1_seed42_2dcbe85/gradient_diagnostic.json")
assert not out.exists()
config_path = root / "configs/MSVR310/Signal-source-oof-v1.json"
config = json.loads(config_path.read_text())
cfg, binding = configure(config)
from layers.make_loss import make_loss

protocol = json.loads((root / config["protocol"]).read_text())
fold = protocol["folds"][0]
_set_seed(42)
records = records_for(config, protocol, fold, True)
loader = loader_for(records, True)
model = new_model(cfg, fold)
loss_fn, _ = make_loss(cfg, num_classes=model.num_classes)
before = _module_state_sha256(model)
model.train()
started = time.perf_counter()
images, labels, cameras, scenes, paths = next(iter(loader))
assert labels.numel() == 64
assert sorted(torch.unique(labels, return_counts=True)[1].tolist()) == [8] * 8
images = {key: value.cuda() for key, value in images.items()}
labels, cameras, scenes = (v.cuda() for v in (labels, cameras, scenes))
with torch.autocast("cuda", dtype=torch.float16):
    output = model(images, label=labels, cam_label=cameras, view_label=scenes,
                   training=True, sge=cfg.MODEL.stageName)
    loss = _signal_training_loss(output, loss_fn=loss_fn, labels=labels, cameras=cameras,
                                stage=cfg.MODEL.stageName, gram_weight=cfg.MODEL.Gram_Loss_weight,
                                patch_weight=cfg.MODEL.PAT_Loss_weight)
scaler = torch.amp.GradScaler("cuda", init_scale=256.0)
scaler.scale(loss).backward()
rows = []
for name, parameter in model.named_parameters():
    if parameter.requires_grad:
        owner = model.get_submodule(name.rsplit(".", 1)[0])
        rows.append({"name": name, "shape": list(parameter.shape), "numel": parameter.numel(),
                     "gradient_present": parameter.grad is not None,
                     "gradient_finite": bool(torch.isfinite(parameter.grad).all()) if parameter.grad is not None else None,
                     "owner_type": type(owner).__name__, "owner_source": inspect.getsourcefile(type(owner))})
report = {"scope": "One disposable fold0 source batch gradient-path diagnosis; no optimizer update or saved checkpoint",
          **binding, "config_sha256": sha256(config_path), "script_sha256": sha256(__file__),
          "seed": 42, "fold": 0, "loss": float(loss.detach()),
          "trainable_parameters": sum(r["numel"] for r in rows), "trainable_tensors": len(rows),
          "without_gradient": [r for r in rows if not r["gradient_present"]],
          "parameters": rows, "source_paths": list(paths), "source_identity_labels": labels.tolist(),
          "source_image_records_forwarded": 64, "backward_calls": 1, "optimizer_updates": 0,
          "checkpoint_writes": 0, "heldout_dev_official_image_access": 0,
          "initial_model_state_sha256": before, "after_forward_model_state_sha256": _module_state_sha256(model),
          "buffer_note": "Scratch-model training forward can update BN buffers; no model checkpoint is saved.",
          "elapsed_seconds": time.perf_counter() - started}
out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({k: v for k, v in report.items() if k not in ("parameters", "source_paths", "source_identity_labels")}))
