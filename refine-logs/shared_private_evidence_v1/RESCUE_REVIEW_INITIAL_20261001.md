诊断结论：**未发现结构断路；FP16 私有 MLP 内部梯度丢失是待验证的首要解释。诊断脚本源码可执行。**  
`review_independence: same-family`；`acceptance_status: provisional`。

已核实：

- 九个私有适配器均连接到角色读出，无 detach 或漏掉角色索引。缺失的26项集中于 LayerNorm／下投影，上投影均曾获得非零梯度。
- 上投影零初始化，所以第一批的内部梯度数学上必为零，不能据此诊断断路。[初始化源码](/C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/correspondence_roles.py:33)
- M0在第八批后检查的是累计非零梯度集合，失败门没有误把“某一批为零”当作失败。[检查源码](/C:/Users/gb/.trifusion_github_publish_22c3bee/tools/run_visual_update_control.py:175)
- FP32参数存储不能避免 autocast 下的内部 FP16 计算及梯度下溢；局部禁用 autocast 并显式转换输入是 PyTorch 支持的处理方式。[官方 AMP 文档](https://docs.pytorch.org/docs/2.14/amp.html#gradient-scaling)
- 原始第八批日志、失败 traceback 和残留 `RUNNING` receipt 已直接读取。这里的 `RUNNING` 是终止前落盘状态。

[诊断脚本](/C:/Users/gb/.codex_tmp/diagnose_shared_private_gradient_20261001.py)通过只读审查：第八批使用七次真实更新后的相同权重，捕获三模态输入与真实上游梯度，在 optimizer step 前重放局部 FP32 导数；移除 hooks 后使用 `autograd.grad`，不会累积到原参数 `.grad`。先核对八批 loss 和缺项集合与失败运行一致，再判断原零梯度是否恢复。

若恢复，最小修复是仅替换[私有适配器计算](/C:/Users/gb/.trifusion_github_publish_22c3bee/modeling/trifusion/shared_private_evidence.py:30)：

```python
with torch.autocast(shared.device.type, enabled=False):
    private = [
        adapter(shared.float()).to(shared.dtype)
        for adapter in self.private_adapters[depth]
    ]
```

前提是诊断确认原 input/output dtype 相同。两个 roles 对照共用该类，应同时应用。随后重新通过原八批 M0；不放宽门槛。

局部重放能确认私有 MLP 的数值精度造成梯度丢失，但不能单独定位究竟是权重转半精度、反向乘积还是中间梯度舍入。当前未编辑、部署或训练，也没有修复成功或性能提升结论。
