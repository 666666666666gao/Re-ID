from pathlib import Path
import hashlib
import json
import shutil

root = Path('C:/Users/gb/.trifusion_github_publish_22c3bee')
read = lambda p:json.loads(p.read_bytes())
sha = lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
launch = read(root/'logs/cross_depth_launch_658_20260929.json')
progress = read(root/'logs/cross_depth_progress_660_20260929.json')
manifest = read(root/'logs/cross_depth_manifest_658_20260929.json')
archive = read(root/'logs/cross_depth_m0_archive_658_20260929.json')
assert progress['campaign_status'] == 'RUNNING' and progress['controller_pid'] == launch['pid']
assert sha(root/'logs/cross_depth_manifest_658_20260929.json') == progress['manifest_sha256']
assert len(manifest['jobs']) == 9 and len(manifest['source_sha256']) == 205
active = [r for r in progress['rows'] if r['status'] == 'RUNNING']
assert len(active) == 4 and sum(r['status']=='PENDING' for r in progress['rows']) == 5
assert not any(r['status']=='FAILED' for r in progress['rows'])
for row in active:
    m0 = row['jobs'][0]
    assert m0['status']=='COMPLETE' and m0['exit_code']==0
    assert m0['m0_evidence']['trainable_parameters']==m0['m0_evidence']['nonzero_gradient_parameters']==118
    assert m0['m0_evidence']['reload_max_abs_difference']==0
    assert row['jobs'][1]['mode']=='train' and row['jobs'][1]['status']=='RUNNING'
for record in archive['archived']:
    assert all(sha(Path(record['directory'])/name)==digest for name,digest in record['files'].items())
review_dir = root/'refine-logs/cross_depth_role_state_v1'
for name in ('REVIEW_ENTRY_658_20260929','REVIEW_EVALUATION_658_20260929'):
    review = read(review_dir/f'{name}.json')
    assert review['review_independence']=='same-family' and review['acceptance_status']=='provisional'
doc = root/'docs/TRIFUSION_RGBNT201_CURRENT_COMPLETE_HANDOFF_2026-09-01.md'
assert sha(doc)=='4fffe1e7c6b56ea99000b841f73ad596f87d33ab07dddbad96e0c0982ed58003'
raw=doc.read_bytes()
key='### 0.1 历史阶段摘要'.encode()
front,history=raw[:raw.index(key)].decode(),raw[raw.index(key):]
start,end=front.index('**本页更新至'),front.index('** 历史规则')+2
front=front[:start]+'**本页更新至§41.658：context/local完整15端与能量10端已归档；下一项跨深度角色状态三条件×三集9端正式队列已启动。当前4项真实生产M0通过后进入full50训练、5项待队列接续；新结构正式终点0/9，不能填写中途best为正式指标。旧18/新9源码不变；新入口明确绑定depth_mode，205份实际源码/协议/配置冻结。完整50／单mAP-best／严格重载／完整gallery规则继续执行，研究目标ACTIVE/UNMET。**'+front[end:]
table='\n'.join(f"| {r['dataset']} | {r['variant']} | GPU{r['gpu']} | 8 / 118/118 / 0 | {r['jobs'][1].get('recorded_epochs',0)}/50 |" for r in active)
append=f'''

### 41.658 跨深度角色证据处理与持续状态对照：源码复核、真实M0及四卡正式启动（2026-09-29）

上一批context/local15/15及其完整因子分析已按§657全部归档，不能继续用其中某个正差宣称条件query或local身份监督稳定有效。新批次单独登记mixed_once / depth_mean / depth_recurrent三种结构×三数据集，共9个seed42正式端，不改写任何旧结果。

#### 只改变证据形成，保留训练目标和读出

| 条件 | 三层证据怎样处理 | 要排除的解释 |
|---|---|---|
| mixed_once | 统一三层混合后，原角色算子执行一次 | 共同结构控制 |
| depth_mean | 每一深度分别执行同一角色算子，再平均三份输出 | 更早加工／三次计算本身 |
| depth_recurrent | 第4→8→12层依次处理并传递同角色锚点状态，读出末状态 | 持续证据相对独立三次处理是否有效 |

三条件共用参数、全局条件query、采样地址和1536D区域读出；均为M1/M2开启、M3关闭、auxiliary_target=none，统一冻结layer logits为均匀值。历史query-only仍训练该logit，因此新mixed_once需重新训练，不能替换成旧context_none终点。没有增加分类/重建loss、教师、倍率扫描或新seed。CNN上一状态加在该深度卷积和采样后的锚点，Transformer/Mamba上一状态进入序列处理；这不是三套CLIP，也不是角色状态贯穿全部CLIP block。共享CLIP仍由M1适配改变输出；冻结检查只保证Signal张量不变。

新checkpoint使用trifusion-cross-depth-role-state-v1并显式记录depth_mode，加载与collector必须匹配模式。因为三条件tensor key相同，仅strict state_dict不能区分执行图。源码修正只发生于新文件：复用入口的辅助系数1.0处于停用状态且每步auxiliary_id=0；旧任务审计器不变。生产队列在M0退出后、full50前校验8步／非零梯度覆盖／冻结／重载误差／probe SHA。实际数据入口、计分器、训练配置、上游Signal以及本地模型构造依赖共205份文件纳入manifest冻结。

#### 分开记录代码、合成检查与真实生产证据

两份fresh native Codex source review请求gpt-6-astra/max/forknone，最终均WARN / source PASS_WITH_LIMITS、无当前代码阻塞；归属same-family/provisional，未独立证明底层服务身份，未执行GPU。所有报告保留原始请求／hash与单seed官方选点限制。

21:12的CPU合成检查使用已有TinySequenceMixer，旧路径与mixed_once四输出最大差0，共同地址、相同参数边界和三个深度的非零输入梯度通过。它不使用生产Mamba，不是ReID指标，也不能替代M0。

正式控制器PID{launch['pid']}于{launch['at']}启动。实际快照{progress['at']}，4 RUNNING / 5 PENDING / 0 COMPLETE / 0 FAILED：四项均已通过真实mamba_ssm.Mamba、真实训练loader的8批M0，118/118可训练张量在八次更新中观察到非零梯度、Signal张量不变、重载最大误差0；并不宣称每个张量在每批均非零。

| 数据集 | 条件 | 卡 | M0批数／梯度覆盖／重载差 | 快照已记录训练epoch |
|---|---|---|---|---|
{table}

GPU即时数据：{progress['gpu_memory'].strip().replace(chr(10),'；')}（index,MiB,util%，仅快照）。其余五端由父队列按空闲卡自动接续；poll_seconds=240，无抢占、无失败重试、无按中途成绩取消登记端。当前正式终点0/9，不报告新结构正式分数。

每端完整50轮后，以官方fused mAP选一份权重，三路完整Q/G独立重载、作者计分＋CPU复算后才能入表。205份冻结来源和三个保存纯baseline权重留存，旧发布Signal/V8/R2/V27仍为不同研究线。官方集参与epoch选择，one-seed不能支持无偏泛化或稳定性结论。该批试验只能回答跨层角色证据是否有增量；目标十点／强基线性能尚未实现。

证据：logs/cross_depth_manifest_658_20260929.json、cross_depth_launch_658_20260929.json、cross_depth_sync_658_20260929.json、cross_depth_progress_659/660_20260929.json、cross_depth_m0_archive_658_20260929.json及其8份原始M0文本，refine-logs/cross_depth_role_state_v1计划/源代码review；remote logs/cross_depth_role_state_20260929保存全部生产日志与权重。
'''
doc.write_bytes(front.encode()+history+append.encode())
shutil.copyfile(doc,Path('C:/Users/gb/Desktop/document')/doc.name)
tracker=review_dir/'EXPERIMENT_TRACKER.md'
text=tracker.read_text(encoding='utf-8')
text+=f"\n\nActual launch {launch['at']}, PID{launch['pid']}; snapshot {progress['at']}. Four real M0 passed and full50 active; five pending; formal endpoints0/9. No intermediate metrics accepted.\n"
tracker.write_text(text,encoding='utf-8')
(review_dir/'EXPERIMENT_TRACKER_20260929_2122.md').write_text(text,encoding='utf-8')
print(json.dumps({'status':'DOC658_REAL_M0_FOUR_CARD_TRAINING_UPDATED','document_sha256':sha(doc),'snapshot_at':progress['at']}))
