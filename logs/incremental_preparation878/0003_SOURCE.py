import json,os,torch
assert os.environ['CUDA_VISIBLE_DEVICES']==''
scope={};exec('"""Training-only comparisons for fused evidence; no new model parameters.\n\nThe batch-ratio control independently expresses MDReID\'s published distance\nratio idea, adapted to current global/correction/fused components. It is not\nthe author\'s full shared/private model or a copy of its unlicensed code.\nThe repair/keep comparison uses legal source-environment instance relations\nand a detached global reference. Neither objective is claimed to be novel.\n"""\nimport torch\nimport torch.nn.functional as F\n\n\ndef _distances(feature):\n    squared = feature.square().sum(dim=1)\n    return (squared[:, None] + squared[None, :] - 2 * feature @ feature.T).clamp_min(1e-12).sqrt()\n\n\ndef batch_ratio_loss(global_feature, correction, raw_fused, labels):\n    with torch.autocast(raw_fused.device.type, enabled=False):\n        same = labels[:, None].eq(labels[None, :])\n        assert bool((~same).any(dim=1).all())\n        extremes = []\n        for feature in (raw_fused.float(), global_feature.detach().float(), correction.detach().float()):\n            distance = _distances(feature)\n            positive = distance.masked_fill(~same, -torch.inf).amax()\n            negative = distance.masked_fill(same, torch.inf).amin()\n            extremes.append((positive, negative))\n        positive_ratio = extremes[0][0] / sum(pair[0] for pair in extremes)\n        negative_ratio = extremes[0][1] / sum(pair[1] for pair in extremes)\n        loss = positive_ratio + 1 - negative_ratio\n    return loss, dict(incremental_objective=\'md_batch_ratio_adaptation\',\n        positive_batch_ratio=float(positive_ratio.detach()), negative_batch_ratio=float(negative_ratio.detach()))\n\n\ndef _query_mean(values, mask):\n    counts = mask.sum(dim=(1, 2))\n    means = (values * mask).sum(dim=(1, 2)) / counts.clamp_min(1)\n    # Empty query/cell support is a measured property of the source batches.\n    # Its loss is defined as zero; ordinary author tasks remain active.\n    return means.sum() / (counts > 0).sum().clamp_min(1)\n\n\ndef repair_keep_loss(global_feature, fused, labels, environments):\n    with torch.autocast(fused.device.type, enabled=False):\n        g = F.normalize(global_feature.detach().float(), dim=1)\n        f = F.normalize(fused.float(), dim=1)\n        same = labels[:, None].eq(labels[None, :])\n        positive = same & environments[:, None].ne(environments[None, :])\n        legal = positive[:, :, None] & (~same)[:, None, :]\n        g_similarity, f_similarity = g @ g.T, f @ f.T\n        g_margin = g_similarity[:, :, None] - g_similarity[:, None, :]\n        f_margin = f_similarity[:, :, None] - f_similarity[:, None, :]\n        repair = legal & (g_margin <= 0)\n        keep = legal & (g_margin > 0)\n        # Wrong source relations must reach a positive 0.1 cosine margin;\n        # already-correct relations must retain their detached global margin.\n        repair_loss = _query_mean(F.relu(0.1 - f_margin), repair)\n        keep_loss = _query_mean(F.relu(g_margin - f_margin), keep)\n        loss = 0.5 * (repair_loss + keep_loss)\n    with torch.no_grad():\n        stats = dict(incremental_objective=\'legal_instance_global_repair_keep\',\n            legal_positive_pairs=int(positive.sum()), legal_triplets=int(legal.sum()),\n            eligible_queries=int(positive.any(dim=1).sum()),\n            multiple_positive_queries=int((positive.sum(dim=1) >= 2).sum()),\n            repair_triplets=int(repair.sum()), keep_triplets=int(keep.sum()),\n            repair_queries=int(repair.any(dim=(1, 2)).sum()), keep_queries=int(keep.any(dim=(1, 2)).sum()),\n            repair_loss=float(repair_loss), keep_loss=float(keep_loss),\n            repair_active_triplets=int((repair & (f_margin < 0.1)).sum()),\n            keep_active_triplets=int((keep & (f_margin < g_margin)).sum()))\n    return loss, stats\n',scope)
repair_keep=scope['repair_keep_loss'];ratio=scope['batch_ratio_loss']
labels=torch.tensor([0,0,1,1]);environments=torch.tensor([0,1,0,1])
g=torch.tensor([[1.,0],[1.,0],[-1.,0],[-1.,0]],requires_grad=True)
c=torch.tensor([[0.,0],[-1.,1],[0.,0],[1.,-1]],requires_grad=True)
loss,stats=repair_keep(g,g.detach()+c,labels,environments)
assert abs(float(loss)-0.75)<1e-6 and stats['legal_triplets']==8 and stats['keep_triplets']==8
gg,cg=torch.autograd.grad(loss,(g,c),allow_unused=True);assert gg is None
assert bool(torch.isfinite(cg).all()) and float(cg.norm())>0
checks=[dict(case='known_keep_degradation',loss=float(loss),expected=0.75,stats=stats,correction_gradient_norm=float(cg.norm()))]
g=g.detach().requires_grad_();c=c.detach().requires_grad_()
loss,stats=repair_keep(g,g.detach()+c,labels,torch.zeros(4,dtype=torch.long))
gg,cg=torch.autograd.grad(loss,(g,c),allow_unused=True)
assert gg is None and float(loss)==0 and stats['legal_triplets']==0 and bool((cg==0).all())
checks.append(dict(case='actual_empty_environment_support_definition',loss=float(loss),stats=stats))
g=torch.tensor([[1.,0],[0.,1],[-1.,0],[0.,-1]],requires_grad=True)
c=torch.zeros_like(g,requires_grad=True)
loss,stats=repair_keep(g,g.detach()+c,labels,environments)
gg,cg=torch.autograd.grad(loss,(g,c),allow_unused=True)
assert abs(float(loss)-0.05)<1e-6 and stats['repair_triplets']==4 and stats['keep_triplets']==4
assert gg is None and bool(torch.isfinite(cg).all()) and float(cg.norm())>0
checks.append(dict(case='known_wrong_global_margin',loss=float(loss),expected=0.05,stats=stats,correction_gradient_norm=float(cg.norm())))
g=torch.tensor([[1.,0],[1.,1],[-1.,0],[-1.,1]],requires_grad=True)
c=g.detach().clone().requires_grad_();h=(2*g.detach()).requires_grad_()
loss,stats=ratio(g,c,h,labels)
gg,cg,hg=torch.autograd.grad(loss,(g,c,h),allow_unused=True)
assert abs(float(loss)-1.0)<1e-6 and gg is None and cg is None
assert bool(torch.isfinite(hg).all()) and float(hg.norm())>0
checks.append(dict(case='analytic_distance_ratios_half',loss=float(loss),expected=1.0,stats=stats,fused_gradient_norm=float(hg.norm())))
print(json.dumps(dict(status='CPU_MATHEMATICAL_AND_GRADIENT_CHECK_PASS',checks=checks,
    torch=torch.__version__,optimizer_updates=0,real_model_forwards=0,
    boundary='Synthetic vector objective tests only. No real model/data/GPU query, M0 or retrieval evidence. Warm existing environment, no installation.')))
