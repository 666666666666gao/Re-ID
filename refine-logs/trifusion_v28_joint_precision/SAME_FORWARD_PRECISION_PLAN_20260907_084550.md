# V28 同一次前向内的精度导数配对

2026-09-07登记的新数值诊断；不是对两次旧诊断的PASS重写。
旧两次诊断均因跨进程第2批loss不能逐bit还原而停止，各1次更新/2次source前向。
日志补全后得到actual0.6292165517807007、original0.6292153596878052，差1.1920928955078125e-6。
差值仅出现在Mamba、依赖它的fused及汇总loss，CNN/Transformer、像素SHA和风格统计均一致。
这还没有证明CUDA非确定性为根因；旧辅助回放条件失败保持原结论，不设容差放行。

本项问题改为：在一次新的真实来源forward中复现dt_proj.weight零梯度后，
固定它自己的输入、参数和真实上游梯度，仅改变joint子图执行精度，是否恢复导数？
该配对不需要声称第二步是原M0轨迹的逐bit回放。所有原loss差异继续完整保存。

仍用原fold0前两批、第一步单次更新；输入SHA和第一步全部标量须精确匹配。
第2批保留actual/expected/difference与exact布尔值，不把false打印成true。
当前完整AMP图的dt_proj.weight梯度必须为零，才算复现待诊断症状。
捕获该图的joint输入/state/输出真实scaled梯度，存远端唯一fixture。

在同一fixture做三次只读导数比较：AMP fast、FP32 fast、FP32 unfused。
FP32模式将已捕获输入转成FP32并关闭autocast，不改变其数值、权重或上游梯度。
AMP模式须逐bit复现本次完整图的joint输出和dt权重梯度；
新严格配对只发生在同一次已捕获forward内部，不设置数值容差替代它。
记录全部16参数、输入梯度和输出差异；比较期间参数SHA不变。
如果当前图不再为零、AMP不能精确复现、或FP32未恢复，都不能写成精度根因已证明。

本项最多新增1次重建optimizer更新/2次source模型前向/3次joint子图反向；
连同两项已失败诊断，诊断总更新最多3，与原M0的116步分开报告。
无Q1、dev/official、完整检索模型保存、loss scale/LR/宽度扫描。
原V28 M0_FAIL以及所有原科学/工程门保持，数值诊断不能直接赋予晋级。
