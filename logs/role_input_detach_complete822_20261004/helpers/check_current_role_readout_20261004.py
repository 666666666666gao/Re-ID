"""Read cached frozen sources only; never import the model or contact a server."""
import ast
import hashlib
import json
from datetime import datetime
from pathlib import Path

ROOT = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
BASE = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
MANIFEST = BASE / 'role_input_detach821_rgbnt100_native_observations/142805_490290/texts/campaign_manifest.json'
OUT = BASE / 'current_role_readout_source_20261004'
assert not OUT.exists(), 'One-shot evidence packet already exists'
manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
sources = manifest['source_sha256']
assert len(sources) == 322
paths = [
    'modeling/trifusion/correspondence_roles.py',
    'modeling/trifusion/correspondence_evidence_readout.py',
    'modeling/trifusion/correspondence_context_identity.py',
    'modeling/trifusion/patch_memory_roles.py',
    'modeling/trifusion/slot_competition_roles.py',
    'modeling/trifusion/slot_competition_fp32_roles.py',
    'modeling/trifusion/role_global_tokens.py',
    'modeling/trifusion/independent_native_roles.py',
    'modeling/trifusion/image_native_evidence.py',
    'modeling/trifusion/role_input_detach.py',
    'tools/run_role_input_detach.py',
    'tools/run_native_research.py',
    'tools/run_native_partitioned.py',
    'tools/run_independent_native_evidence.py',
    'tools/run_clean_clip_joint.py',
]
texts, classes, file_records = {}, {}, []
for rel in paths:
    data = (ROOT / rel).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    assert sha == sources[rel], rel
    source = data.decode('utf-8-sig')
    texts[rel] = source
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            classes[node.name] = {
                'path': rel, 'line': node.lineno,
                'bases': [ast.unparse(x) for x in node.bases],
                'methods': {method.name: {
                    'line': method.lineno, 'end_line': method.end_lineno,
                    'source': ast.get_source_segment(source, method),
                } for method in node.body if isinstance(method, ast.FunctionDef)},
            }
    file_records.append({'path': rel, 'sha256': sha, 'bytes': len(data)})

def method(name, function):
    return classes[name]['methods'][function]['source']

def chain(name):
    result = []
    while name in classes:
        result.append(name)
        bases = classes[name]['bases']
        assert len(bases) == 1
        name = bases[0]
    return result

semantic_chain = chain('DetachedSemanticTriFusion')
native_chain = chain('DetachedNativeTriFusion')
assert semantic_chain == [
    'DetachedSemanticTriFusion', 'RawFeatureSemanticTriFusion',
    'GlobalTokenTriFusion', 'FP32SlotCompetitionTriFusion',
    'SlotCompetitionTriFusion', 'PatchMemoryTriFusion',
    'ContextIdentityTriFusion', 'EvidenceReadoutTriFusion',
    'CorrespondenceTriFusion',
]
assert native_chain == ['DetachedNativeTriFusion', 'IndependentNativeTriFusion'] + semantic_chain[1:]
assert 'structured_readout=True' in method('ContextIdentityTriFusion', '__init__')
readout = method('EvidenceReadoutTriFusion', 'read_evidence')
assert '.mean(dim=(3, 5)).reshape(batch_size, 3, 4, 128)' in readout
assert 'self.readout.weight.reshape(3, 4, 128, 384)' in readout
assert '"bmrf,mrof->bmro"' in readout
assert 'nn.Linear(3 * width, 1536, bias=False)' in method('CorrespondenceTriFusion', '__init__')
assert 'attention_normalization="independent"' in method('GlobalTokenTriFusion', '__init__')
assert 'memory_mode="full"' in method('SlotCompetitionRoles', '__init__')
attention = method('FP32SlotCompetitionRoles', 'sample_context')
assert 'positions' not in attention.split('\n', 1)[1]
assert 'local_support' not in attention
assert 'allocation_weights(scores, self.attention_normalization)' in attention
assert 'return logits.softmax(dim=-1)' in texts['modeling/trifusion/slot_competition_roles.py']
native_forward = method('IndependentNativeRoles', 'forward')
assert 'self.anchor_queries[None] + context_queries[:, 0, None]' in native_forward
assert 'detail_queries[:, None].expand(-1, 3, -1, -1)' in native_forward
assert 'self.output_norms[0](cnn + queries)' in native_forward
reader = method('ImageNativeEvidenceReader', 'forward')
assert '(q @ k.transpose(-1, -2) * 128 ** -0.5).softmax(dim=-1)' in reader
assert 'detail.shape[2] == 512' in reader
assert 'static' in method('RawFeatureSemanticTriFusion', '__init__')
assert 'token_mode=\'static\'' in texts['tools/run_clean_clip_joint.py']
assert 'stages.detach(), context.detach(), shared_global.detach()' in method('DetachedNativeTriFusion', 'role_evidence')

notes = """# 当前冻结版本的读出与槽位职责（只读源码核对）

此记录针对 role_input_detach_v1 的冻结322文件清单，15份实际源码字节与缓存清单一致。未导入 torch/模型，未读取权重、运行训练或访问服务器。不是性能实验，也不修改任何已登记实验。

1. 当前 semantic/native 的继承链到 ContextIdentityTriFusion 时明确启用 structured_readout=True。最终读出没有把全部模态、槽位平均成单个384维输入。每角色 B×3×16×128 先按4×4槽位索引分为四组，每组平均四个槽位，得到 B×3×4×128；三角色在通道维拼接成 B×3×4×384，再用12个128×384权重块各自投影并拼成1536维。
2. 虽然保存的 readout.weight 形状是1536×384，当前作用是12个不同输入块上的分组线性映射。不能仅凭这个保存形状断言最终全部修正被限制在一个全局rank128或rank384子空间。等效分块映射在维度上可以达到1536秩；这不证明实际训练权重的秩或判别性。
3. 四组的“区域”目前是学习槽位的索引分组，不能直接当作四个真实空间/语义部件。当前实际 FP32SlotCompetitionRoles.sample_context 覆盖旧采样：读取全部128个patch，采用每槽位对patch的独立softmax，不使用传入positions或local_support。offset参数冻结，返回的positions是固定参考点；旧grid_sample路径不是这个版本的实际角色读取。
4. native 细节分支读取全部512个CNN位置，具有独立key/value。细节query由anchor_queries加context_queries中的CNN角色项生成，并扩展到三模态。context_queries reshape中的3是角色索引，不能解释成三个模态各自的query。当前细节reader没有显式局部位置mask或额外位置编码；卷积网格及CLIP路径仍具有空间来源，不能泛化为整网没有空间信息。
5. 新细节在CNN出口LayerNorm之前相加，CNN证据还经桥接影响Transformer/Mamba。仅删除最终CNN读出分量，不等于删除CNN/细节路径。Transformer使用每模态一个固定上下文token加16个槽位；Mamba使用48个空间/模态token及共享参数的正逆扫描。
6. input-detach切断角色输入对共享编码器的直接反向路径；h=g+gain*c中的g路径仍训练，且梯度受融合损失影响。不能将其描述为严格保护global不变。

以上是当前代码的结构性质。它们不能唯一解释负增益、证明对应失效、槽位塌缩或指定下一项干预。等待全部六端及固定best诊断完成后，再据完整证据决定唯一下一实验。
"""
record = {
    'observed_at': datetime.now().astimezone().isoformat(),
    'status': 'FROZEN_SOURCE_READOUT_AND_SLOT_PATH_VERIFIED',
    'boundary': 'Cached local source only; no torch import, model call, server contact, training change or new scientific gate',
    'manifest_path': str(MANIFEST),
    'manifest_sha256': hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
    'manifest_source_count': len(sources),
    'verified_files': file_records,
    'semantic_class_chain': semantic_chain,
    'native_class_chain': native_chain,
    'classes': classes,
    'structured_readout_shapes': {
        'per_role': ['B', 3, 16, 128], 'per_role_grouped': ['B', 3, 4, 128],
        'concatenated_roles': ['B', 3, 4, 384],
        'weight_blocks': [3, 4, 128, 384], 'output': ['B', 1536],
        'scope': 'Dimensional structure only; no actual trained matrix rank computed',
    },
}
OUT.mkdir()
(OUT / 'CHECK.json').write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
(OUT / 'NOTES.md').write_text(notes, encoding='utf-8')
print(json.dumps({'status': record['status'], 'verified_files': len(paths), 'packet': str(OUT)}, ensure_ascii=False))
