# Experiment Audit Report

**Date**: 2026-09-02

**Auditor**: GPT-5.5 xhigh, read-only independent review

**Project**: TriFusion RGBNT201 V16 SATR M0

## Overall Verdict: WARN

## Integrity Status: warn

V16 M0 is a valid negative train-only engineering/activity result. No fake
ground truth, self-normalized metric, dead SATR loss path, dev leakage or
official-test access was found. The warning is limited to the unbound
proposal-time threshold probe, whose positive activity coverage did not
reproduce under the formal hash-bound M0 runner.

## Checks

- **Ground-truth provenance: PASS.** Dataset identity and physical-camera
  labels are used, and complete-path fit/held-out identities are disjoint.
- **Score normalization: PASS.** M0 coverage is a raw eligible-query fraction;
  no prediction-statistic normalization is used.
- **Result existence: WARN.** Evidence, report and trackers agree, but the
  earlier threshold probe lacked sampler/tensor hash binding.
- **Executed path: PASS.** SATR hard-pair selection, two-peer teacher,
  receiver repair, protection, backward and formal activity paths executed.
- **Scope: PASS.** One seed42 train-only M0, dev0 and official0; no retrieval
  metric or promoted checkpoint.
- **Evaluation type:** `self_supervised_proxy_train_only_engineering_probe`
  with real identity/camera labels for relation construction.

## Claim impact

- Engineering feasibility and exact Signal preservation: supported.
- Three-receiver activity and identity-disjoint mutual promotion: unsupported.
- Dev, official and SOTA improvement: unsupported.

The authoritative detailed report is `EXPERIMENT_AUDIT_V16_M0.md`.
