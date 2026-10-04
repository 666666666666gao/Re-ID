from datetime import datetime
from pathlib import Path
import hashlib
import json

root=Path('C:/Users/gb/.codex_tmp/independent_evidence_draft')
paper=root/'rodi_primary_pdf_20261004'
packet=root/'rodi_method_reading_20261004'
assert not (packet/'CHECK.json').exists()
source=json.loads((paper/'DOWNLOAD.json').read_bytes())
assert hashlib.sha256((paper/'RoDI.pdf').read_bytes()).hexdigest()==source['sha256']
assert all((packet/f'page_{number}.png').is_file() for number in (4,5,6))
record={'status':'PRIMARY_RODI_METHOD_PAGES_VISUALLY_READ','at':datetime.now().astimezone().isoformat(),
    'source':source,'visual_pages':[4,5,6],
    'facts':{
        'base_fusion':'Three modality CLS queries progressively attend to modality patch keys/values; this is not CNN/Transformer/Mamba role splitting.',
        'EDFL':'Patch-level query-key similarity forms evidential opinion; belief and imbalance quantify interactions at fusion steps.',
        'MSR':'Imbalance-derived weights form a rebalanced anchor; queries rotate toward it, with a Dirichlet KL alignment term.',
        'LMD':'Belief-selected top-K patches and CLS form feature targets; cross-modality conditional diffusion predicts noise and reconstructed features, followed by belief-dependent feature combination.',
        'training_objectives':'Classification + triplet + KL + denoising/reconstruction objectives.',
        'inference_statement':'Section3.5 explicitly states that inference fusion is independent of subjective opinion metrics and modality rolling.'},
    'unverified_implementation_details':[
        'Exact nonnegative evidence conversion from cosine similarity.',
        'Concrete SVD rotation construction and numerical handling.',
        'Whether and how LMD sampling/post-fusion is used during inference.',
        'Actual output feature width and runnable evaluation implementation.',
        'Exact public weight hash, trainable visual parameter groups, total epochs and checkpoint-selection rule.',
        'Exact query/gallery membership and camera/scene filtering.'],
    'project_inferences':[
        'RoDI evaluates fusion interactions before adjusting them; this is a relevant comparison principle, not proof of our failure cause.',
        'Its patch belief is an interaction proxy, not a label-free guarantee of retrieval gain or role complementarity.',
        'Modality rolling differs from a router choosing a winning heterogeneous role; do not reclassify our old router failures as a replication of RoDI.',
        'Its reconstruction mechanism is distinct from our proposed spatial-statistic SNR restoration; both need explicit nearest-method controls if implemented.',
        'Partially matched cross-spectral messages must be compared with ordinary attention or confidence-based selection; a new name alone is insufficient.'],
    'boundary':'Primary paper reading only. No runnable implementation at previously checked repository commit; no local reproduction, new experiment, source change, GPU work, denoising/rotation module addition or protocol-equivalence claim.'}
(packet/'CHECK.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
(packet/'NOTES.md').write_text('''# RoDI方法核对与当前研究边界

实际查看固定作者PDF第4、5、6页。其起点是三个模态CLS形成的query，依次读取各模态patch；不是CNN、Transformer、Mamba三个算子的职责分工。

- EDFL：由query-key相似度形成patch层证据与融合步骤的belief/imbalance。
- MSR：根据模态间imbalance构造平衡anchor，旋转query并用Dirichlet KL约束交流。
- LMD：以belief选择局部patch，再做跨模态条件的特征去噪/重建；根据伪模态和原模态belief组合特征。

这提供的直接启发是先衡量融合交互，再控制参与交互的内容。不能把其belief当作未知身份检索增量的真值，也不能把它等同我们的旧专家赢家Router。其LMD也不同于空间统计归一化后的SNR信息恢复，不能把两者都笼统写成一个去噪模块。

论文第6页§3.5明确说推理fusion不依赖subjective opinion和modality rolling；但仅凭原文无法核定LMD推理调用和具体实现。因此不能断言测试一定运行20步扩散，也不能断言所有去噪都只在训练使用。余弦证据的非负转换、SVD旋转实现、实际输出宽度、视觉参数组及完整训练/选点/过滤仍未由执行源码确认。此前固定作者仓库只有README与论文/海报，没有可直接运行的训练评价代码。

这些是论文事实和项目推论，尚非本地复现。当前原RGBNT100队列不变；独立原生读取、部分对应或风格补偿都不能因为近邻论文存在而自动成为有效贡献。收齐当前六端和固定best诊断后再确定唯一下一项实验。

PDF、页图和全文提取仍放私有目录，只在下次正式结果更新中归档简短核对记录。
''',encoding='utf-8')
print(json.dumps({'status':record['status'],'at':record['at'],'visual_pages':record['visual_pages'],'unverified_implementation_details':record['unverified_implementation_details']},indent=2))
