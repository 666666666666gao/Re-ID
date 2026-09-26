"""Observe native Signal AMP updates without changing its loss or optimizer."""

import json
from pathlib import Path


def audited_scaler_class(base_scaler, output_path):
    output_path = Path(output_path)

    class AuditedGradScaler(base_scaler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.audit_optimizer = None
            self.audit_actual_updates = 0
            self.audit_attempts = 0
            self.audit_pending = None
            self.audit_file = output_path.open("x", encoding="utf-8")

        def _record_actual_update(self, optimizer, args, kwargs):
            self.audit_actual_updates += 1

        def step(self, optimizer, *args, **kwargs):
            assert self.audit_pending is None
            if self.audit_optimizer is None:
                self.audit_optimizer = optimizer
                self.audit_hook = optimizer.register_step_post_hook(self._record_actual_update)
            assert optimizer is self.audit_optimizer  # Native Signal has center loss disabled.
            self.audit_attempts += 1
            rates = [group["lr"] for group in optimizer.param_groups]
            self.audit_pending = {
                "attempt": self.audit_attempts,
                "scale_before": self.get_scale(),
                "actual_updates_before": self.audit_actual_updates,
                "lr_min": min(rates),
                "lr_max": max(rates),
            }
            return super().step(optimizer, *args, **kwargs)

        def update(self, *args, **kwargs):
            assert self.audit_pending is not None
            result = super().update(*args, **kwargs)
            row = self.audit_pending
            row["scale_after"] = self.get_scale()
            row["actual_updates_after"] = self.audit_actual_updates
            row["optimizer_updated"] = self.audit_actual_updates > row["actual_updates_before"]
            self.audit_file.write(json.dumps(row) + "\n")
            self.audit_file.flush()
            self.audit_pending = None
            return result

    return AuditedGradScaler


def summarize_amp_audit(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    assert rows and [row["attempt"] for row in rows] == list(range(1, len(rows) + 1))
    actual = sum(row["optimizer_updated"] for row in rows)
    assert actual == rows[-1]["actual_updates_after"]
    return {
        "attempted_updates": len(rows),
        "actual_optimizer_updates": actual,
        "skipped_updates": len(rows) - actual,
        "scale_decreases": sum(row["scale_after"] < row["scale_before"] for row in rows),
        "first_scale": rows[0]["scale_before"],
        "last_scale": rows[-1]["scale_after"],
        "minimum_scale": min(row["scale_after"] for row in rows),
        "skipped_attempts": [row["attempt"] for row in rows if not row["optimizer_updated"]],
    }
