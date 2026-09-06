# V26已结束对照端：跨版本数值轨迹边界

记录时间：2026-09-07T01:06:50.612907+08:00。
范围为已完整结束的fold0-control：V25与V26各20epoch/580更新，共9860个原损失标量。
这是训练实现的复现边界检查；V26候选仍须全六端终态，不用单折或单对照替代完整科学比较。

## 实际证据

两端完整初始model-state SHA相同，全部580步采样索引/顺序和学习率相同，前八批增强tensor收据相同。
第1步全部17个原损失数值完全相同；第2步最大差异1.648440957069397e-6，
首先出现在Mamba及融合相关项，CNN/Transformer独立项当时仍相同。
第7步最大差异已超过1e-4；第69步超过.01。整个580步最大原损失标量差异
0.2225133180618286，两个最终model-state SHA不同。
全部输出评分记录及数组文件SHA亦不相同；不能把相同seed或相同初始化写成逐位相同训练轨迹。
本检查没有模型前向、优化、图像读取或tensor加载，仅核对完整已保存文本。

这一次跨版本对照不能估计运行噪声分布，也不能用来扣除/修正V26候选增益；
当前V26两端执行相同诊断路径，科学比较仍仅使用本轮配对对照与原五项条件。
新责任loss的解析梯度及真实新增梯度证据保持，但不能将观察到的微小最终变化全归于独立稳定收益。

## 来源和原因边界

服务器mamba-ssm版本2.2.6.post3；mamba_simple.py和selective_scan_interface.py
与作者tag v2.2.6.post3、commit10b5d6358f27966f6a40e4bf0baa17a460688128逐字节相同。
项目Mamba构造使用默认fast path，安装接口调用selective_scan_cuda.bwd。
[对应作者反向核心](https://github.com/state-spaces/mamba/blob/10b5d6358f27966f6a40e4bf0baa17a460688128/csrc/selective_scan/selective_scan_bwd_kernel.cuh)
包含多个gpuAtomicAdd位置；这是需要保留的并行浮点累加来源。
未重新构建/验证已安装CUDA二进制与该tag的构建绑定，未执行隔离kernel干预，因此不能断言某一条atomic操作就是本次差异的唯一原因。

[PyTorch 2.5.1官方复现说明](https://github.com/pytorch/pytorch/blob/v2.5.1/docs/source/notes/randomness.rst)
区分随机种子、cuDNN算法选择和算子确定性；cuDNN选项并不代表所有自定义CUDA算子均得到确定性保证。
当前报告中的cudnn_deterministic=True只表示该选项，不能扩大成全模型逐位确定性证明。
这里不升级库、不改算法开关、不重训对照，也不改正在执行的V26合同。
未来主结果若达到条件，稳定性与数值敏感性需要另行验证；当前单seed限制保留。

## 完整证据

- evidence/trifusion_v26_closed_control_readonly_diagnostics_parity_20260907.json
- evidence/trifusion_v26_closed_control_first_numerical_difference_20260907.json
- evidence/trifusion_v26_installed_mamba_backward_source_20260907.json
- evidence/trifusion_v26_numerical_reproducibility_primary_sources_20260907.json

外部独立审计不可用，当前为执行器对真实记录及原始源码的核对。
