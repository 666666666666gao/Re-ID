"""Record checked original-paper numbers and public-tree availability; no experiment action."""
from datetime import datetime
import hashlib
import json
from pathlib import Path

base = Path('C:/Users/gb/.codex_tmp/independent_evidence_draft/strong_reference_primary_20261004')
downloads = base / 'downloads'
packet = base / 'packet'
assert not packet.exists()
packet.mkdir()
trees = {}
for name in ('deep', 'rodi', 'pmkd'):
    data = json.loads((downloads / (name + '_main_tree.json')).read_bytes())
    receipt = json.loads((downloads / (name + '_main_tree.json.receipt.json')).read_bytes())
    assert not data['truncated']
    assert hashlib.sha256((downloads / (name + '_main_tree.json')).read_bytes()).hexdigest() == receipt['sha256']
    trees[name] = dict(tree_sha=data['sha'], paths=[entry['path'] for entry in data['tree']],
                       receipt=receipt, interpretation='Checked public main tree has no executable model/training implementation; other unpublished/private branches are outside this check.')
    (packet / (name + '_main_tree.json')).write_bytes((downloads / (name + '_main_tree.json')).read_bytes())
    (packet / (name + '_main_tree.json.receipt.json')).write_bytes((downloads / (name + '_main_tree.json.receipt.json')).read_bytes())
for name in ('rodi.pdf', 'pmkd.pdf'):
    receipt = json.loads((downloads / (name + '.receipt.json')).read_bytes())
    assert hashlib.sha256((downloads / name).read_bytes()).hexdigest() == receipt['sha256']
    (packet / (name + '.receipt.json')).write_bytes((downloads / (name + '.receipt.json')).read_bytes())

entries = [
    dict(dedup_key='doi:10.1109/TMM.2026.3660160', payload=dict(
        title='DEEP: Decoupled Semantic Prompt Learning, Guiding and Embedding for Multi-Spectral Object Re-Identification',
        authors=['Shihao Li', 'Chenglong Li', 'Aihua Zheng', 'Jin Tang', 'Bin Luo'],
        year=2026, venue='IEEE Transactions on Multimedia; accepted author version',
        primary_url='https://aihuazheng.github.io/publications/pdf/2025/2025-DEEP_Decoupled_Semantic_Prompt_Learning_Guiding_and_Embedding_for_Multi-Spectral_Object_Re-Identification.pdf',
        metrics={'RGBNT201': {'mAP':79.6, 'R1':84.2, 'R5':89.4, 'R10':91.5}, 'RGBNT100':{'mAP':88.5, 'R1':97.6}, 'MSVR310':{'mAP':66.0, 'R1':82.1}},
        locations={'RGBNT201':'Table I, PDF page 6', 'vehicles':'Table IV, PDF page 7', 'resources':'IV.B, PDF page 7', 'MSVR_filter':'IV.A, PDF page 7', 'test_prompts':'III.E, PDF page 6'},
        resources={'visual':'CLIP ViT-B/16, trainable', 'text':'Frozen CLIP text encoder; learned and image-inverted semantic prompts. Paper does not require concrete text labels; text branch/prompt resource is still present.', 'epochs':60, 'optimizer':'Adam; initial visual LR5e-6, drops at20/40', 'images':'256x128 person,128x256 vehicle', 'MSVR_filter':'Exclude same identity and time-span; no executable authors evaluator available in checked main tree.'},
        code=trees['deep'], boundary='Paper report, no local reproduction. URL folder2025 is not publication year. Main table retained; higher alternative SSE-sharp ablation is not substituted.')),
    dict(dedup_key='cvf:CVPR2026F:Li_Rolling_and_Denoising_Rethinking_Dynamic_Modal_Fusion_for_Multi-Modal_Object', payload=dict(
        title='Rolling and Denoising: Rethinking Dynamic Modal Fusion for Multi-Modal Object Re-Identification',
        authors=['Shihao Li', 'Huaibo Huang', 'Aihua Zheng', 'Jin Tang', 'Ran He'],
        year=2026, venue='CVPR Findings; author public PDF with supplementary material',
        primary_url='https://github.com/lsh-ahu/RoDI/blob/main/assets/RoDI.pdf',
        metrics={'CLIP':{'RGBNT201':{'mAP':84.1,'R1':87.2,'R5':92.0,'R10':93.2},'RGBNT100':{'mAP':88.5,'R1':97.6},'MSVR310':{'mAP':64.1,'R1':77.2}}, 'DINOv3':{'RGBNT201':{'mAP':85.3,'R1':87.9,'R5':93.0,'R10':94.8},'RGBNT100':{'mAP':89.0,'R1':99.1},'MSVR310':{'mAP':71.8,'R1':84.8}}},
        locations={'main':'Table1, PDF page6; visually checked', 'resources':'4.2, PDF page6', 'ablation':'Table2, PDF page7', 'inference':'3.5, PDF page6', 'diffusion':'3.4, PDF page5'},
        resources={'visual':'CLIP ViT-B/16 or DINOv3 distilled ViT-B/16', 'images':'CLIP256x128/128x256; DINOv3 224x224', 'batch':'B64/K8', 'optimizer':'Adam LR3.5e-4;10-epoch warmup', 'total_epochs':None, 'total_epoch_boundary':'Only10-epoch warmup stated in inspected full13-page PDF; total training epochs not established.', 'inference':'Paper says subjective-opinion metrics and modality rolling omitted. Local conditional diffusion/pseudo-feature postfusion described; exact executable inference path unavailable.'},
        code=trees['rodi'], boundary='Author PDF original source, not execution. CVF HTML403 prevented byte identity check with formal CVF edition. CLIP and DINOv3 rows remain separate.')),
    dict(dedup_key='doi:10.1609/aaai.v40i16.38338', payload=dict(
        title='Progressive Multi-modal Knowledge Distillation for Multi-spectral Object Re-identification',
        authors=['Aihua Zheng','Pengyu Li','Zi Wang','Jin Tang'],
        year=2026, venue='AAAI40(16),13351-13359',
        primary_url='https://ojs.aaai.org/index.php/AAAI/article/download/38338/42300',
        metrics={'RGBNT201':{'mAP':84.7,'R1':88.9,'R5':91.0,'R10':92.2},'RGBNT100':{'mAP':91.6,'R1':98.0},'MSVR310':None},
        locations={'main':'Tables1/2, PDF page6 (printed13356); visually checked', 'resources':'Implementation Details, PDF page5', 'stages':'Method, PDF page3', 'ablation':'Table3, PDF page6'},
        resources={'visual':'DINOv2 pretrained Transformer', 'images':'224x224 all datasets', 'batch':'B32/K8', 'optimizer':'Adam LR4.5e-5', 'epochs_statement':50, 'training':'Train shared multi-modal source; distill to fresh independent modal backbones; those teachers distill to another fresh set. Only final independent targets needed at inference.', 'cost_boundary':'50 epochs is the implementation statement; total pipeline epochs/wallclock are not resolved from the available paper/code. Do not infer total cost50 or150.', 'MSVR_boundary':'Evaluated datasets are RGBNT201,RGBNT100,WMVEID863; MSVR310 result not supplied.'},
        code=trees['pmkd'], boundary='Official original PDF, no reproduction. DINOv2 row, not CLIP. MM82.5/MS81.5 to final84.7 is an own-baseline comparison, not a10point claim.'))
]
assert len({row['dedup_key'] for row in entries}) == 3
record = dict(shard_id='root_primary_sequential_20261004', entries=entries,
              at=datetime.now().astimezone().isoformat(),
              attribution='Root original-source extraction and visual checks. Retrieval agent produced no findings; no independent review or ranking acceptance.',
              scope='Three predefined strong references; not exhaustive latest SOTA. No experiment configuration/architecture/source changes, SSH, temperature or power action.')
(packet / 'PRIMARY_ENTRIES.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
failures = dict(retrieval_agent={'task':'/root/retrieve_rodi_primary_20261004','status':'FAILED','reason':'Selected model is at capacity. No findings or artifact returned; not a reviewer rejection.'},
                second_shard={'task':'retrieve_pmkd_primary_20261004','status':'SPAWN_FAILED','reason':'Agent thread limit reached; no agent created.'},
                retrieval_failures=[{'source':'AAAI PDF webopen','reason':'Timeout fetching'}, {'source':'AAAI PDF webclick','reason':'Content length exceeds10MiB'}, {'source':'CVF RoDI HTML','reason':'403 Forbidden'}],
                source_completion='Root sequential original retrieval: official AAAI direct download successful; author RoDI PDF successful; DEEP author PDF read with web. Source failures retained, no model/backend retry.',
                unresolved='No arxiv_fetch helper resolved. Exact target filename local search found no matching target PDF. Not every local PDF inspected. Zotero/Obsidian not requested or used.')
(packet / 'RETRIEVAL_BOUNDARY.json').write_text(json.dumps(failures, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
(packet / 'REFERENCES.bib').write_text('''@article{Li2026DEEP,
  author = {Shihao Li and Chenglong Li and Aihua Zheng and Jin Tang and Bin Luo},
  title = {{DEEP}: Decoupled Semantic Prompt Learning, Guiding and Embedding for Multi-Spectral Object Re-Identification},
  journal = {IEEE Transactions on Multimedia},
  year = {2026},
  doi = {10.1109/TMM.2026.3660160}
}
@inproceedings{Li2026RoDI,
  author = {Shihao Li and Huaibo Huang and Aihua Zheng and Jin Tang and Ran He},
  title = {Rolling and Denoising: Rethinking Dynamic Modal Fusion for Multi-Modal Object Re-Identification},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Findings},
  year = {2026},
  pages = {6560--6569},
  url = {https://github.com/lsh-ahu/RoDI/blob/main/assets/RoDI.pdf}
}
@inproceedings{Zheng2026PMKD,
  author = {Aihua Zheng and Pengyu Li and Zi Wang and Jin Tang},
  title = {Progressive Multi-modal Knowledge Distillation for Multi-spectral Object Re-identification},
  booktitle = {Proceedings of the AAAI Conference on Artificial Intelligence},
  volume = {40},
  number = {16},
  year = {2026},
  pages = {13351--13359},
  doi = {10.1609/aaai.v40i16.38338}
}
''', encoding='utf-8')
(packet / 'STRONG_REFERENCE_RESOURCES.md').write_text('''# Original-source strong reference resources, 2026-10-04

Units are percentage points. These are paper reports, not matched local reproductions or an exhaustive SOTA ranking.

| Paper / pretraining | RGBNT201 mAP/R1/R5/R10 | RGBNT100 mAP/R1 | MSVR310 mAP/R1 |
|---|---|---|---|
| DEEP / CLIP visual + frozen text | 79.6/84.2/89.4/91.5 | 88.5/97.6 | 66.0/82.1 |
| RoDI / CLIP | 84.1/87.2/92.0/93.2 | 88.5/97.6 | 64.1/77.2 |
| RoDI / DINOv3 | 85.3/87.9/93.0/94.8 | 89.0/99.1 | 71.8/84.8 |
| PMKD / DINOv2 | 84.7/88.9/91.0/92.2 | 91.6/98.0 | Not reported |

DEEP: [author accepted PDF](https://aihuazheng.github.io/publications/pdf/2025/2025-DEEP_Decoupled_Semantic_Prompt_Learning_Guiding_and_Embedding_for_Multi-Spectral_Object_Re-Identification.pdf), TablesI/IV and IV.B. The frozen text branch and semantic prompts are resources; concrete annotated text labels are not required by this paper. Training60epochs; visual weights trainable. MSVR filters same-ID/same-time-span. [Checked repository](https://github.com/lsh-ahu/DEEP-ReID) currently contains README/license/assets, no training implementation.

RoDI: [author PDF](https://github.com/lsh-ahu/RoDI/blob/main/assets/RoDI.pdf), Table1 and4.2. Both backbones ViT-B/16; B64/K8, Adam3.5e-4 and10-epoch warmup. Total epochs remain unknown. Inference omits rolling/opinion metrics according to3.5; diffusion and pseudo-feature fusion are described, but no executable path is available. CLIP/DINO rows are not interchangeable. [Checked repository](https://github.com/lsh-ahu/RoDI) contains README,PDF,poster. CVF HTML retrieval403; this check establishes author-PDF tables, not byte identity with CVF edition.

PMKD: [official AAAI PDF](https://ojs.aaai.org/index.php/AAAI/article/download/38338/42300), Tables1/2, pages3/5. DINOv2,224x224,B32/K8,Adam4.5e-5; implementation states50epochs. The pipeline includes source training and two fresh-target distillation stages; total training cost is not established. MSVR is absent from the evaluated dataset list. [Checked repository](https://github.com/moonaricc/PMKD) contains only README.

The tree JSON files pin current public main content using immutable tree SHAs (not commit SHAs). No raw PDF or full paper text is published in this packet. Local previews of PMKD/RoDI tables were visually checked. Current model training and the330scientific-source seal are untouched.
''', encoding='utf-8')
files = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in packet.iterdir() if p.is_file()}
(packet / 'MANIFEST.json').write_text(json.dumps(dict(status='THREE_PRIMARY_RESOURCES_RECORDED', at=datetime.now().astimezone().isoformat(), entries=3, canonical_ids=[x['dedup_key'] for x in entries], files=files, PDFs_published=False, weights_or_data_downloaded=False, source_scope330_unchanged=True), indent=2)+'\n', encoding='utf-8')
with Path('C:/Users/gb/memory/2026-10-04.md').open('a', encoding='utf-8') as stream:
    stream.write('\n\n'+datetime.now().astimezone().isoformat()+' 原训练等待期间，root完成DEEP/RoDI/PMKD原文与作者main树资源核查，private strong_reference_primary_20261004/packet，未发布原PDF。RoDI总epoch不明、PMKD多阶段总成本不明，三repo当前主树无训练代码。只作强参照资源表、无SOTA穷尽或本机复现主张。RoDI检索agent模型容量失败，第二shard线程限额失败；parent顺序完成，非独立评审。未SSH/不查温功/科学source330不变。\n')
print(json.dumps(dict(status='THREE_PRIMARY_RESOURCES_RECORDED', files=len(files), at=datetime.now().astimezone().isoformat())))
