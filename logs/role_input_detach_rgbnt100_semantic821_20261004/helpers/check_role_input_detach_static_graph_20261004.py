"""Read the frozen source graph; never import or execute model code."""
from datetime import datetime
from pathlib import Path
import hashlib
import json
import paramiko

private=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
source_map=json.loads((private/'closed_m3_probe_retirement_preflight_v2_20261004/stdout.json').read_bytes())['current_source_sha256']
assert len(source_map)==322
names=[
    'modeling/trifusion/role_input_detach.py',
    'modeling/trifusion/independent_native_roles.py',
    'modeling/trifusion/evidence_author_heads.py',
    'modeling/trifusion/partitioned_evidence_clip.py',
    'modeling/trifusion/role_global_tokens.py',
    'tools/run_role_input_detach.py',
    'tools/run_native_research.py',
    'tools/run_native_partitioned.py',
    'tools/run_independent_native_evidence.py',
    'tools/run_foundation_recipe.py',
]
assert all(name in source_map for name in names)
packet=private/'role_input_detach_static_graph_20261004'
assert not packet.exists()
packet.mkdir()
client=paramiko.SSHClient()
client.load_host_keys('C:/Users/gb/.ssh/known_hosts')
client.connect('172.19.12.138',port=2026,username='gaob',key_filename='C:/Users/gb/.ssh/id_ed25519',timeout=20)
sftp=client.open_sftp()
texts={}
for name in names:
    with sftp.open('/data/gaob/Re-ID/Trifusion/'+name,'rb') as stream:
        data=stream.read()
    assert hashlib.sha256(data).hexdigest()==source_map[name],name
    path=packet/'source'/name
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(data)
    texts[name]=data.decode('utf-8')
sftp.close()
client.close()
expected={
    names[0]:['stages.detach(), context.detach(), shared_global.detach()'],
    names[1]:['raw_fused = shared_global.float() + self.readout_gain * correction.float()',
              'evidence = self.role_evidence(batch, stages, context, shared_global)'],
    names[2]:["raw = output['raw_fused']",'feature)), feature)',"features = (raw,) if self.signal.direct else raw.split(512, dim=1)"],
    names[3]:['return output + sum(deltas) / 3'],
    names[8]:['visual.requires_grad_(True)','camera.requires_grad_(True)'],
    names[9]:["for score, feature in output['heads']"],
}
evidence=[]
for name,fragments in expected.items():
    for fragment in fragments:
        assert fragment in texts[name],(name,fragment)
        lines=[index+1 for index,line in enumerate(texts[name].splitlines()) if fragment in line]
        evidence.append({'file':name,'source_sha256':source_map[name],'lines':lines,'fragment':fragment})
record={'status':'FROZEN_SOURCE_ROLE_INPUT_GRADIENT_BOUNDARY_CHECKED','at':datetime.now().astimezone().isoformat(),
    'source_files':{name:source_map[name] for name in names},'evidence':evidence,
    'facts':[
        'Detached classes stop gradients only through stages, context and shared_global when those values enter role_evidence.',
        'The original shared_global stays differentiable in raw_fused = shared_global + gain * correction.',
        'Matched author heads and their metric features use raw_fused; there is no separate global-only preservation objective in this entry.',
        'The partitioned backbone still writes the mean of three adapter deltas into the shared CLIP stream; visual and camera parameters remain trainable.',
        'Thus stopping the direct role-input derivative does not freeze global or guarantee its retrieval geometry is protected.',
    ],
    'limits':[
        'Static graph reasoning only; no new forward, gradient measurement, training, evaluation or diagnosis.',
        'At fixed role inputs the direct derivative through the role read is absent, but the fused loss derivative passed through global depends on the correction and trained heads.',
        'This does not prove gradient conflict or identify a unique cause of the observed full50 scores.',
        'Interpret the completed six endpoints and the registered fixed-best g/c/h/f diagnosis before choosing a next intervention.',
    ]}
(packet/'CHECK.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
(packet/'NOTES.md').write_text('''# 当前role-input-detach的实际边界（源码检查）

两种新类仅在进入role_evidence时对stages、context、shared_global停止梯度。最终raw_fused仍使用未detach的shared_global；作者分类头和度量损失都接收这份融合后的raw表示。分段backbone仍把三组adapter增量的均值写回共享CLIP流，视觉和camera仍参加优化。

因此，该实验隔离的是角色读取输入到共享编码器的直接反向通道，不是严格冻结或保底global。在固定参数处，若把共享参数记为θ、角色参数记为ψ，其简化关系为h=gθ+αcψ(sg(Eθ),sg(contextθ),sg(gθ),I)。共享参数仍接收Jg,θ转置乘以融合损失对h的梯度；后一个量依赖修正内容和训练后的分类头。停止一条直接导数，不等于把共享路径的学习任务变成独立global-only。

这只说明源码定义，不能据此宣布梯度冲突成立或把某一端的下降唯一归因于融合耦合。收齐六端正式50轮与首次严格评价、原一次CPU报告后，再按已登记固定best诊断比较各模型global、correction、raw_fused、fused及独立global-only。当前不改变网络、损失、学习率、gain或queue。

本检查只经SFTP读取十个冻结文本文件并对照当前322源SHA；没有导入torch、构造模型、调用GPU、进行forward/反向或新增检索分数。
''',encoding='utf-8')
print(json.dumps({'status':record['status'],'at':record['at'],'source_files':len(names),'facts':record['facts']},indent=2))
