from pathlib import Path
import json,hashlib,datetime
root=Path(__file__).parent
original=root/'independent_recompute.py';target=root/'independent_recompute_02.py'
text=original.read_text(encoding='utf-8')
old="            assert finalnorm==all_audits[f\"fold_{fold['fold']}_{endpoint}\"][-1]['actual_parameter_updates'][r]['second_norm']"
new="""            savednorm=all_audits[f"fold_{fold['fold']}_{endpoint}"][-1]['actual_parameter_updates'][r]['second_norm']
            # Cross-device double reductions are not bitwise guaranteed. Reuse the existing scalar tolerance.
            close(finalnorm,savednorm)
            maxima['final_parameter_norm_cpu_gpu_abs_error']=max(maxima['final_parameter_norm_cpu_gpu_abs_error'],abs(finalnorm-savednorm))"""
assert text.count(old)==1 and not target.exists()
target.write_text(text.replace(old,new),encoding='utf-8')
record=dict(date=datetime.datetime.now(datetime.timezone.utc).isoformat(),original_script=str(original),original_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),revised_script=str(target),revised_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),failed_attempt='independent_recompute_01.json.stderr',diagnostic='checkpoint_norm_diagnostic.json',change='Replace auditor-added GPU/CPU norm bitwise assertion with existing saved-scalar 1e-7 relative-or-absolute comparison; retain exact checkpoint/state/frozen hashes and original .005 / 1e-8 experimental gradient gates.',maximum_observed_cpu_gpu_norm_absolute_error=5.684341886080802e-14,maximum_observed_relative_error=2.0207945872648634e-16,project_files_changed=False)
(root/'audit_script_revision.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
print(json.dumps(record,indent=2))
