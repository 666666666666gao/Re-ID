from pathlib import Path

base=Path('C:/Users/gb/.codex_tmp')
source=(base/'publish_row_transport_closeout863.py').read_text(encoding='utf-8')
source=source.replace('publication863_local','publication864_local').replace('four_copy861','four_copy863')
source=source.replace('commit863','commit864').replace('target863','target864')
source=source.replace("four_copy863_2025_pending.json').write_text", "four_copy864_2025_pending.json').write_text")
source=source.replace("section='41.863'", "section='41.864'")
source=source.replace('trifusion_row_transport_closeout863_20261006.bundle','trifusion_selection_reference864_20261006.bundle')
source=source.replace('Close six row-transport experiments and fixed-best diagnosis; preserve failures and retire dominated weights',
                      'Preserve fixed-array CPU parity failure and draft matched Signal selection references')
source=source.replace("files.update(publication['files'])", "files.update(json.loads((proof/'publication863_local.json').read_bytes())['files'])\nfiles.update(publication['files'])")
source=source.replace('Six complete50 first-strict endpoints, primary0/3 and all0/15. Fixed-best6/18/24 closed after preserved cfg failure; only missing100 resumed. Five qualified own weights retired, 201slot/RAW187 retained. Goal ACTIVE/UNMET; no new experiment launched; 2025 mirror pending.',
                      'CPU saved-array additive parity failed before direct-sum calculation; original accepted studies unchanged. Nine-endpoint Signal source-reference draft/CPU component only; queue/report/real initialization/M0/formal not run. Goal ACTIVE/UNMET;2025 pending/no probe.')
lines=source.splitlines()
index=next(i for i,line in enumerate(lines) if "cmd=(directory/'cmdline')" in line)
lines[index]="  cmd=(directory/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')"
source='\n'.join(lines)+'\n'
source=source.replace("print('ALL_OWN_PRODUCERS_TERMINAL_366_AND_187_UNCHANGED_QUALIFIED_RETIREMENT_VERIFIED')",
    "assert json.loads((root/'logs/additive_geometry_fixed_arrays_launch_20261006_864/EXIT.json').read_text())['exit_code']==1\nassert not (root/'results/additive_geometry_fixed_arrays_20261006_864/SUMMARY.json').exists()\nprint('ALL_OWN_PRODUCERS_TERMINAL_366_AND_187_UNCHANGED_CPU_PARITY_FAILURE_PRESERVED')")
exec(compile(source,'publish_selection_reference864','exec'))
