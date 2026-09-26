import json

import torch

from tools.signal_amp_audit import audited_scaler_class, summarize_amp_audit


def test_audit_counts_skipped_steps_without_changing_adam(tmp_path):
    def run(scaler_type):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.Adam([parameter], lr=0.01)
        scaler = scaler_type("cpu", init_scale=8, growth_interval=2)
        for multiplier in (1.0, float("inf"), 1.0, 1.0):
            optimizer.zero_grad()
            scaler.scale(parameter.square().sum() * multiplier).backward()
            scaler.step(optimizer)
            scaler.update()
        return parameter.detach(), optimizer.state[parameter], scaler.get_scale()

    reference = run(torch.amp.GradScaler)
    path = tmp_path / "audit.jsonl"
    observed = run(audited_scaler_class(torch.amp.GradScaler, path))
    assert torch.equal(reference[0], observed[0])
    assert all(torch.equal(reference[1][key], observed[1][key]) for key in reference[1])
    assert reference[2] == observed[2]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert [row["optimizer_updated"] for row in rows] == [True, False, True, True]
    summary = summarize_amp_audit(path)
    assert summary["attempted_updates"] == 4
    assert summary["actual_optimizer_updates"] == 3
    assert summary["skipped_attempts"] == [2]
    assert summary["scale_decreases"] == 1
