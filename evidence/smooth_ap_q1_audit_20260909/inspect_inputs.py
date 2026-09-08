from pathlib import Path
import ast, hashlib, json

OUT=Path(__file__).parent
REPO=Path(r'C:\Users\gb\.trifusion_github_publish_22c3bee')
RAW=Path(r'C:\Users\gb\.codex_tmp\smooth_ap_q1_complete_20260909')
SNAP=REPO/'evidence/smooth_ap_m0_audit_20260909/snapshots/remote/root/autodl-tmp/trifusion-v2/TriFusion-ReID'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(name,v):
    p=OUT/name
    with p.open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2)
def shape(v):
    if isinstance(v,dict):return {k:shape(x) if isinstance(x,dict) and len(x)<30 else ('dict:'+str(len(x)) if isinstance(x,dict) else 'list:'+str(len(x)) if isinstance(x,list) else x) for k,x in v.items()}
    return v
paths=set([REPO/'AGENTS.md',REPO/'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md',REPO/'refine-logs/msvr310_smooth_ap_v1/TRAINING_PLAN.md',REPO/'refine-logs/msvr310_smooth_ap_v1/EXPERIMENT_TRACKER.md'])
todo=[REPO/'configs/MSVR310/TriFusion-smooth-ap-paired-v1.json'];seen=set();pins=[]
def findrefs(obj,parent):
    if isinstance(obj,dict):
        for k,v in obj.items():
            if isinstance(v,str) and (REPO/k).is_file() and len(v)==64:
                p=REPO/k;paths.add(p);pins.append(dict(parent=str(parent),path=str(p),expected=v,actual=sha(p),match=sha(p)==v))
                if p.suffix=='.json':todo.append(p)
            if isinstance(v,str) and len(v)<500 and '\n' not in v and not v.startswith('/') and not v.startswith('http') and (REPO/v).is_file():
                p=REPO/v;paths.add(p)
                if p.suffix=='.json':todo.append(p)
            if isinstance(v,(dict,list)):findrefs(v,parent)
    elif isinstance(obj,list):
        for v in obj:findrefs(v,parent)
while todo:
    p=todo.pop()
    if p in seen:continue
    seen.add(p);paths.add(p);findrefs(json.loads(p.read_bytes()),p)
for p in SNAP.rglob('*'):
    if p.is_file() and (REPO/p.relative_to(SNAP)).is_file():paths.add(REPO/p.relative_to(SNAP))
paths.update(RAW/'q1'/f'fold_{f}_{e}'/n for f in range(3) for e in ['control','smooth_ap'] for n in ['receipt.json','training.json','rankings.json','memory_steps.jsonl'])
paths.update(RAW/n for n in ['intake_complete.json','pipeline.json','q1/summary.json','q1_cpu.json','q1_cpu.log','q1.log','remote_terminal_inventory.json'])
manifest={str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)}
write('input_hashes_initial.json',manifest);write('recursive_pin_checks.json',pins)
identity=[]
for p in sorted(paths):
    if p.is_relative_to(REPO):
        old=SNAP/p.relative_to(REPO)
        if old.is_file():identity.append(dict(path=str(p),snapshot=str(old),current_sha256=sha(p),snapshot_sha256=sha(old),match=sha(p)==sha(old)))
write('prior_snapshot_identity.json',identity)
shapes={str(p.relative_to(RAW)):shape(json.loads(p.read_bytes())) for p in [RAW/'q1/summary.json',RAW/'q1/fold_0_control/receipt.json',RAW/'q1/fold_0_control/training.json']}
shapes['audit_first']=shape(json.loads((RAW/'q1/fold_0_control/memory_steps.jsonl').read_text().splitlines()[0]))
shapes['audit_history']=shape(json.loads((RAW/'q1/fold_0_smooth_ap/memory_steps.jsonl').read_text().splitlines()[66]))
write('input_schema.json',shapes)
wrapper=ast.parse(Path(r'C:\Users\gb\.codex_tmp\role_set_remote_command_20260908.py').read_text(encoding='utf8'))
interface=[]
for node in ast.walk(wrapper):
    if isinstance(node,ast.Subscript) and ast.unparse(node.value)=='sys.argv':interface.append(ast.unparse(node))
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='add_argument':interface.append(ast.unparse(node))
print(json.dumps(dict(files=len(manifest),pins=len(pins),pin_failures=[x for x in pins if not x['match']],snapshot_compared=len(identity),snapshot_mismatches=[x for x in identity if not x['match']],bridge_interface=interface),indent=2))
