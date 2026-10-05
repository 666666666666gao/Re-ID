from pathlib import Path

source=Path('C:/Users/gb/.codex_tmp/retire_dominated_closed_weights861.py').read_text(encoding='utf-8')
source=source.replace(
 " 'visual_update_control_20261001_v1_visual_update_frozen_roles_MSVR310_seed42_full',\n 'visual_update_control_20261001_v1_visual_update_low_lr_roles_MSVR310_seed42_full']",
 " 'visual_update_control_20261001_v1_visual_update_frozen_roles_MSVR310_seed42_full']")
assert "'visual_update_control_20261001_v1_visual_update_low_lr_roles_MSVR310_seed42_full'" not in source
source=source.replace('dominated_closed_weight_retirement861','nine_dominated_closed_weight_retirement861')
source=source.replace('TEN_DOMINATED_CLOSED_WEIGHTS_RETIRED','NINE_DOMINATED_CLOSED_WEIGHTS_RETIRED')
source=source.replace('Explicit ten closed','Explicit nine closed')
exec(compile(source,'retire_nine_dominated_closed_weights861.py','exec'))
