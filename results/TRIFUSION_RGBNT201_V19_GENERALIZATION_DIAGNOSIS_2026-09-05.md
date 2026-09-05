# V19完整终态的来源拟合与模态几何诊断（2026-09-05）

六个最终模型全部完成严格重建、只读前向和source/heldout诊断，耗时329.784541秒。
全部原五路heldout AP/Rank数组与Q1逐项精确相等，模型state和checkpoint文件不变；
optimizer0、checkpoint writes0、dev0、official0。源执行3edb0f9，原Q1仍为Q1_FAIL。

全部六端的七个分类头在各自全部94-source身份的干净样本上均100%准确，来源融合
检索也均100% mAP；训练已能拟合来源身份。对未见身份的Q1融合仅80.240792/
80.496828 mAP。可观察到明显的来源拟合/身份泛化落差，扩大私有尾部容量没有满足
原固定收益门，不能再把“训练没有执行或梯度完全失效”作为此次失败的解释。

## 范围与全部现有输出

每端三折累计source6252条/1142合法query、heldout3126条/571合法query；source
的每个物理身份出现在两个不同fold模型中，不能称独立6252个样本或1142新query。
每fold94-source中14跨camera身份、47-heldout中7跨camera身份，其余图库身份
保留。总共18756次triplet前向，不跨fold或source/heldout混合特征距离。

| 端点 | scope | 输出 | mAP | R1 | R5 | R10 |
|---|---|---|---:|---:|---:|---:|
| frozen_private_tail | source | baseline_only | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| frozen_private_tail | source | fused | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| frozen_private_tail | source | cnn | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| frozen_private_tail | source | transformer | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| frozen_private_tail | source | mamba | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| frozen_private_tail | heldout | baseline_only | 77.487603 | 79.334501 | 89.492119 | 93.520140 |
| frozen_private_tail | heldout | fused | 80.240792 | 83.187391 | 90.017513 | 93.870403 |
| frozen_private_tail | heldout | cnn | 79.915105 | 84.763573 | 88.966725 | 91.593695 |
| frozen_private_tail | heldout | transformer | 78.150546 | 82.136602 | 90.542907 | 92.994746 |
| frozen_private_tail | heldout | mamba | 77.801980 | 78.984238 | 89.316988 | 94.045534 |
| trained_private_tail | source | baseline_only | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| trained_private_tail | source | fused | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| trained_private_tail | source | cnn | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| trained_private_tail | source | transformer | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| trained_private_tail | source | mamba | 100.000000 | 100.000000 | 100.000000 | 100.000000 |
| trained_private_tail | heldout | baseline_only | 77.487603 | 79.334501 | 89.492119 | 93.520140 |
| trained_private_tail | heldout | fused | 80.496828 | 84.238179 | 89.842382 | 93.695271 |
| trained_private_tail | heldout | cnn | 80.054797 | 84.238179 | 90.367776 | 92.469352 |
| trained_private_tail | heldout | transformer | 79.331729 | 83.187391 | 89.141856 | 92.469352 |
| trained_private_tail | heldout | mamba | 77.379156 | 79.509632 | 89.667250 | 93.870403 |

## 来源分类拟合

分类只在来源标签上计算，不将heldout身份硬塞入94类分类头。下面对三折全部来源
记录加权，两个端各6252次来源样本评价。逐样本预测和CE保留在原始诊断JSON。

| 端点 | 分类头 | 正确/总数 | 准确率% | smoothed CE |
|---|---|---:|---:|---:|
| frozen_private_tail | fused | 6252/6252 | 100.000000 | 0.851722 |
| frozen_private_tail | cnn | 6252/6252 | 100.000000 | 0.836652 |
| frozen_private_tail | transformer | 6252/6252 | 100.000000 | 0.838168 |
| frozen_private_tail | mamba | 6252/6252 | 100.000000 | 0.835432 |
| frozen_private_tail | residual_cnn | 6252/6252 | 100.000000 | 0.850777 |
| frozen_private_tail | residual_transformer | 6252/6252 | 100.000000 | 0.846994 |
| frozen_private_tail | residual_mamba | 6252/6252 | 100.000000 | 0.847294 |
| trained_private_tail | fused | 6252/6252 | 100.000000 | 0.851222 |
| trained_private_tail | cnn | 6252/6252 | 100.000000 | 0.836807 |
| trained_private_tail | transformer | 6252/6252 | 100.000000 | 0.836646 |
| trained_private_tail | mamba | 6252/6252 | 100.000000 | 0.836359 |
| trained_private_tail | residual_cnn | 6252/6252 | 100.000000 | 0.850241 |
| trained_private_tail | residual_transformer | 6252/6252 | 100.000000 | 0.844807 |
| trained_private_tail | residual_mamba | 6252/6252 | 100.000000 | 0.848283 |

## 所有模态对几何

以下是已存在的512D单位残差向量cosine统计，不是新增单模态/跨模态检索输出，
没有模态子集实验或融合权重扫描。正例始终为同身份且跨camera，负例为异身份；
最近cosine margin=最近正例cosine-最近负例cosine。所有六个有向跨模态对和三个
同模态对均保留，按query-modality pair平均。对照来源的同模态margin为正，
heldout为负；两端三个专家的跨模态margin在source和heldout均为负。

| 端点 | scope | 专家 | 模态关系 | 同实例cos | 正例均值cos | 最近正例cos | 最近负例cos | 最近margin | 负例至少同样接近% |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| frozen_private_tail | source | cnn | same_modality | 1.000000 | 0.821023 | 0.865857 | 0.703815 | 0.162041 | 1.868068 |
| frozen_private_tail | source | cnn | different_modality | 0.289571 | 0.268804 | 0.351527 | 0.540962 | -0.189435 | 73.890835 |
| frozen_private_tail | source | transformer | same_modality | 1.000000 | 0.843354 | 0.885809 | 0.694532 | 0.191277 | 0.875657 |
| frozen_private_tail | source | transformer | different_modality | 0.259927 | 0.243841 | 0.318324 | 0.508334 | -0.190010 | 72.942207 |
| frozen_private_tail | source | mamba | same_modality | 1.000000 | 0.818856 | 0.863553 | 0.707579 | 0.155974 | 3.998832 |
| frozen_private_tail | source | mamba | different_modality | 0.273442 | 0.252096 | 0.332925 | 0.550812 | -0.217887 | 78.677758 |
| frozen_private_tail | heldout | cnn | same_modality | 1.000000 | 0.517660 | 0.621415 | 0.673746 | -0.052331 | 58.844133 |
| frozen_private_tail | heldout | cnn | different_modality | 0.242257 | 0.165064 | 0.272613 | 0.511239 | -0.238626 | 90.338587 |
| frozen_private_tail | heldout | transformer | same_modality | 1.000000 | 0.488148 | 0.598927 | 0.666625 | -0.067698 | 61.120841 |
| frozen_private_tail | heldout | transformer | different_modality | 0.222806 | 0.136542 | 0.235646 | 0.474360 | -0.238715 | 92.148278 |
| frozen_private_tail | heldout | mamba | same_modality | 1.000000 | 0.525483 | 0.628026 | 0.679142 | -0.051115 | 58.960887 |
| frozen_private_tail | heldout | mamba | different_modality | 0.222544 | 0.147219 | 0.248188 | 0.513006 | -0.264818 | 94.016346 |
| trained_private_tail | source | cnn | same_modality | 1.000000 | 0.835726 | 0.878872 | 0.709499 | 0.169373 | 1.722125 |
| trained_private_tail | source | cnn | different_modality | 0.286285 | 0.266877 | 0.348316 | 0.544885 | -0.196569 | 74.197315 |
| trained_private_tail | source | transformer | same_modality | 1.000000 | 0.851885 | 0.892388 | 0.693197 | 0.199191 | 0.904845 |
| trained_private_tail | source | transformer | different_modality | 0.256580 | 0.241895 | 0.312031 | 0.500812 | -0.188781 | 71.555750 |
| trained_private_tail | source | mamba | same_modality | 1.000000 | 0.825635 | 0.869808 | 0.701125 | 0.168683 | 2.626970 |
| trained_private_tail | source | mamba | different_modality | 0.263698 | 0.242713 | 0.322266 | 0.545172 | -0.222905 | 78.108581 |
| trained_private_tail | heldout | cnn | same_modality | 1.000000 | 0.528937 | 0.634277 | 0.680235 | -0.045958 | 58.960887 |
| trained_private_tail | heldout | cnn | different_modality | 0.239747 | 0.160428 | 0.266306 | 0.516598 | -0.250292 | 91.856392 |
| trained_private_tail | heldout | transformer | same_modality | 1.000000 | 0.489085 | 0.599688 | 0.667453 | -0.067764 | 62.521891 |
| trained_private_tail | heldout | transformer | different_modality | 0.231340 | 0.147524 | 0.242247 | 0.473694 | -0.231447 | 90.951547 |
| trained_private_tail | heldout | mamba | same_modality | 1.000000 | 0.522000 | 0.627512 | 0.678949 | -0.051436 | 59.136019 |
| trained_private_tail | heldout | mamba | different_modality | 0.223294 | 0.149974 | 0.251493 | 0.507915 | -0.256422 | 93.286632 |

完整3×3方向表（不挑RGB/TI或其他有利/不利方向）：

| 端点 | scope | 专家 | 模态对 | 同实例cos | 跨camera正例均值cos | 最近margin | 负例至少同样接近% |
|---|---|---|---|---:|---:|---:|---:|
| frozen_private_tail | source | cnn | RGB_to_RGB | 1.000000 | 0.823991 | 0.165871 | 2.189142 |
| frozen_private_tail | source | cnn | RGB_to_NI | 0.591991 | 0.538325 | -0.024608 | 43.957968 |
| frozen_private_tail | source | cnn | RGB_to_TI | 0.123050 | 0.122159 | -0.302794 | 90.893170 |
| frozen_private_tail | source | cnn | NI_to_RGB | 0.591991 | 0.535775 | -0.023496 | 40.805604 |
| frozen_private_tail | source | cnn | NI_to_NI | 1.000000 | 0.788374 | 0.145772 | 3.152364 |
| frozen_private_tail | source | cnn | NI_to_TI | 0.153671 | 0.150665 | -0.257379 | 88.003503 |
| frozen_private_tail | source | cnn | TI_to_RGB | 0.123050 | 0.118003 | -0.261596 | 89.754816 |
| frozen_private_tail | source | cnn | TI_to_NI | 0.153671 | 0.147898 | -0.266735 | 89.929947 |
| frozen_private_tail | source | cnn | TI_to_TI | 1.000000 | 0.850702 | 0.174481 | 0.262697 |
| frozen_private_tail | source | transformer | RGB_to_RGB | 1.000000 | 0.846415 | 0.195377 | 0.788091 |
| frozen_private_tail | source | transformer | RGB_to_NI | 0.550509 | 0.508908 | -0.040419 | 44.133100 |
| frozen_private_tail | source | transformer | RGB_to_TI | 0.097595 | 0.096788 | -0.288740 | 89.842382 |
| frozen_private_tail | source | transformer | NI_to_RGB | 0.550509 | 0.506680 | -0.025612 | 38.441331 |
| frozen_private_tail | source | transformer | NI_to_NI | 1.000000 | 0.815840 | 0.174582 | 1.576182 |
| frozen_private_tail | source | transformer | NI_to_TI | 0.131677 | 0.128725 | -0.256970 | 89.316988 |
| frozen_private_tail | source | transformer | TI_to_RGB | 0.097595 | 0.091630 | -0.267195 | 86.514886 |
| frozen_private_tail | source | transformer | TI_to_NI | 0.131677 | 0.130312 | -0.261123 | 89.404553 |
| frozen_private_tail | source | transformer | TI_to_TI | 1.000000 | 0.867806 | 0.203871 | 0.262697 |
| frozen_private_tail | source | mamba | RGB_to_RGB | 1.000000 | 0.824244 | 0.157628 | 3.677758 |
| frozen_private_tail | source | mamba | RGB_to_NI | 0.559581 | 0.503137 | -0.059856 | 54.028021 |
| frozen_private_tail | source | mamba | RGB_to_TI | 0.103182 | 0.100499 | -0.342133 | 91.155867 |
| frozen_private_tail | source | mamba | NI_to_RGB | 0.559581 | 0.501862 | -0.053909 | 53.064799 |
| frozen_private_tail | source | mamba | NI_to_NI | 1.000000 | 0.783494 | 0.133251 | 7.618214 |
| frozen_private_tail | source | mamba | NI_to_TI | 0.157563 | 0.152622 | -0.274843 | 92.556918 |
| frozen_private_tail | source | mamba | TI_to_RGB | 0.103182 | 0.099724 | -0.306061 | 92.031524 |
| frozen_private_tail | source | mamba | TI_to_NI | 0.157563 | 0.154730 | -0.270521 | 89.229422 |
| frozen_private_tail | source | mamba | TI_to_TI | 1.000000 | 0.848831 | 0.177043 | 0.700525 |
| frozen_private_tail | heldout | cnn | RGB_to_RGB | 1.000000 | 0.565170 | 0.000750 | 42.031524 |
| frozen_private_tail | heldout | cnn | RGB_to_NI | 0.509100 | 0.302477 | -0.191967 | 84.588441 |
| frozen_private_tail | heldout | cnn | RGB_to_TI | 0.090653 | 0.082156 | -0.284038 | 93.695271 |
| frozen_private_tail | heldout | cnn | NI_to_RGB | 0.509100 | 0.303807 | -0.185896 | 85.288967 |
| frozen_private_tail | heldout | cnn | NI_to_NI | 1.000000 | 0.463232 | -0.097001 | 70.402802 |
| frozen_private_tail | heldout | cnn | NI_to_TI | 0.127019 | 0.112885 | -0.256488 | 92.994746 |
| frozen_private_tail | heldout | cnn | TI_to_RGB | 0.090653 | 0.069121 | -0.276858 | 95.271454 |
| frozen_private_tail | heldout | cnn | TI_to_NI | 0.127019 | 0.119937 | -0.236510 | 90.192644 |
| frozen_private_tail | heldout | cnn | TI_to_TI | 1.000000 | 0.524576 | -0.060741 | 64.098074 |
| frozen_private_tail | heldout | transformer | RGB_to_RGB | 1.000000 | 0.536583 | -0.005620 | 46.584939 |
| frozen_private_tail | heldout | transformer | RGB_to_NI | 0.451166 | 0.254439 | -0.210920 | 90.893170 |
| frozen_private_tail | heldout | transformer | RGB_to_TI | 0.067783 | 0.060147 | -0.283355 | 96.672504 |
| frozen_private_tail | heldout | transformer | NI_to_RGB | 0.451166 | 0.250949 | -0.214513 | 87.040280 |
| frozen_private_tail | heldout | transformer | NI_to_NI | 1.000000 | 0.450898 | -0.105445 | 69.001751 |
| frozen_private_tail | heldout | transformer | NI_to_TI | 0.149470 | 0.099867 | -0.225537 | 88.266200 |
| frozen_private_tail | heldout | transformer | TI_to_RGB | 0.067783 | 0.042387 | -0.264813 | 98.248687 |
| frozen_private_tail | heldout | transformer | TI_to_NI | 0.149470 | 0.111461 | -0.233151 | 91.768827 |
| frozen_private_tail | heldout | transformer | TI_to_TI | 1.000000 | 0.476961 | -0.092030 | 67.775832 |
| frozen_private_tail | heldout | mamba | RGB_to_RGB | 1.000000 | 0.570720 | -0.009042 | 43.432574 |
| frozen_private_tail | heldout | mamba | RGB_to_NI | 0.481263 | 0.293784 | -0.218074 | 90.718039 |
| frozen_private_tail | heldout | mamba | RGB_to_TI | 0.051868 | 0.045269 | -0.318225 | 95.796848 |
| frozen_private_tail | heldout | mamba | NI_to_RGB | 0.481263 | 0.292852 | -0.214357 | 92.819615 |
| frozen_private_tail | heldout | mamba | NI_to_NI | 1.000000 | 0.494659 | -0.073885 | 69.702277 |
| frozen_private_tail | heldout | mamba | NI_to_TI | 0.134500 | 0.112189 | -0.254954 | 92.469352 |
| frozen_private_tail | heldout | mamba | TI_to_RGB | 0.051868 | 0.026099 | -0.333657 | 98.423818 |
| frozen_private_tail | heldout | mamba | TI_to_NI | 0.134500 | 0.113124 | -0.249643 | 93.870403 |
| frozen_private_tail | heldout | mamba | TI_to_TI | 1.000000 | 0.511069 | -0.070418 | 63.747811 |
| trained_private_tail | source | cnn | RGB_to_RGB | 1.000000 | 0.837025 | 0.168162 | 2.189142 |
| trained_private_tail | source | cnn | RGB_to_NI | 0.593514 | 0.541161 | -0.022039 | 44.395797 |
| trained_private_tail | source | cnn | RGB_to_TI | 0.122307 | 0.121855 | -0.313045 | 91.330998 |
| trained_private_tail | source | cnn | NI_to_RGB | 0.593514 | 0.539827 | -0.021216 | 39.492119 |
| trained_private_tail | source | cnn | NI_to_NI | 1.000000 | 0.803703 | 0.149383 | 2.977233 |
| trained_private_tail | source | cnn | NI_to_TI | 0.143035 | 0.140479 | -0.273752 | 87.828371 |
| trained_private_tail | source | cnn | TI_to_RGB | 0.122307 | 0.118845 | -0.276130 | 92.732049 |
| trained_private_tail | source | cnn | TI_to_NI | 0.143035 | 0.139095 | -0.273233 | 89.404553 |
| trained_private_tail | source | cnn | TI_to_TI | 1.000000 | 0.866449 | 0.190574 | 0.000000 |
| trained_private_tail | source | transformer | RGB_to_RGB | 1.000000 | 0.857222 | 0.208010 | 0.700525 |
| trained_private_tail | source | transformer | RGB_to_NI | 0.554690 | 0.518453 | -0.035339 | 46.234676 |
| trained_private_tail | source | transformer | RGB_to_TI | 0.103229 | 0.102997 | -0.291165 | 89.842382 |
| trained_private_tail | source | transformer | NI_to_RGB | 0.554690 | 0.513561 | -0.019876 | 37.915937 |
| trained_private_tail | source | transformer | NI_to_NI | 1.000000 | 0.824329 | 0.180653 | 1.663748 |
| trained_private_tail | source | transformer | NI_to_TI | 0.111822 | 0.106583 | -0.263448 | 85.113835 |
| trained_private_tail | source | transformer | TI_to_RGB | 0.103229 | 0.098819 | -0.253702 | 84.063047 |
| trained_private_tail | source | transformer | TI_to_NI | 0.111822 | 0.110957 | -0.269156 | 86.164623 |
| trained_private_tail | source | transformer | TI_to_TI | 1.000000 | 0.874103 | 0.208910 | 0.350263 |
| trained_private_tail | source | mamba | RGB_to_RGB | 1.000000 | 0.827166 | 0.168746 | 2.276708 |
| trained_private_tail | source | mamba | RGB_to_NI | 0.561948 | 0.507949 | -0.051902 | 50.788091 |
| trained_private_tail | source | mamba | RGB_to_TI | 0.096215 | 0.093328 | -0.338327 | 91.243433 |
| trained_private_tail | source | mamba | NI_to_RGB | 0.561948 | 0.505706 | -0.053939 | 51.225919 |
| trained_private_tail | source | mamba | NI_to_NI | 1.000000 | 0.790773 | 0.144536 | 5.341506 |
| trained_private_tail | source | mamba | NI_to_TI | 0.132930 | 0.128040 | -0.297255 | 94.395797 |
| trained_private_tail | source | mamba | TI_to_RGB | 0.096215 | 0.093793 | -0.303600 | 89.579685 |
| trained_private_tail | source | mamba | TI_to_NI | 0.132930 | 0.127460 | -0.292408 | 91.418564 |
| trained_private_tail | source | mamba | TI_to_TI | 1.000000 | 0.858967 | 0.192766 | 0.262697 |
| trained_private_tail | heldout | cnn | RGB_to_RGB | 1.000000 | 0.575370 | 0.008104 | 43.432574 |
| trained_private_tail | heldout | cnn | RGB_to_NI | 0.510177 | 0.306827 | -0.197873 | 85.464098 |
| trained_private_tail | heldout | cnn | RGB_to_TI | 0.084888 | 0.070856 | -0.303979 | 98.073555 |
| trained_private_tail | heldout | cnn | NI_to_RGB | 0.510177 | 0.305729 | -0.189810 | 85.113835 |
| trained_private_tail | heldout | cnn | NI_to_NI | 1.000000 | 0.485256 | -0.085441 | 66.900175 |
| trained_private_tail | heldout | cnn | NI_to_TI | 0.124175 | 0.102487 | -0.266825 | 92.644483 |
| trained_private_tail | heldout | cnn | TI_to_RGB | 0.084888 | 0.061135 | -0.299548 | 96.672504 |
| trained_private_tail | heldout | cnn | TI_to_NI | 0.124175 | 0.115537 | -0.243719 | 93.169877 |
| trained_private_tail | heldout | cnn | TI_to_TI | 1.000000 | 0.526186 | -0.060538 | 66.549912 |
| trained_private_tail | heldout | transformer | RGB_to_RGB | 1.000000 | 0.545620 | -0.006298 | 46.409807 |
| trained_private_tail | heldout | transformer | RGB_to_NI | 0.462065 | 0.263411 | -0.217472 | 88.091068 |
| trained_private_tail | heldout | transformer | RGB_to_TI | 0.090513 | 0.069618 | -0.268192 | 94.220665 |
| trained_private_tail | heldout | transformer | NI_to_RGB | 0.462065 | 0.260143 | -0.206906 | 90.367776 |
| trained_private_tail | heldout | transformer | NI_to_NI | 1.000000 | 0.445635 | -0.101787 | 70.227671 |
| trained_private_tail | heldout | transformer | NI_to_TI | 0.141443 | 0.113920 | -0.202724 | 85.288967 |
| trained_private_tail | heldout | transformer | TI_to_RGB | 0.090513 | 0.057003 | -0.264750 | 95.096322 |
| trained_private_tail | heldout | transformer | TI_to_NI | 0.141443 | 0.121050 | -0.228638 | 92.644483 |
| trained_private_tail | heldout | transformer | TI_to_TI | 1.000000 | 0.476002 | -0.095209 | 70.928196 |
| trained_private_tail | heldout | mamba | RGB_to_RGB | 1.000000 | 0.557593 | -0.009526 | 46.935201 |
| trained_private_tail | heldout | mamba | RGB_to_NI | 0.495732 | 0.303141 | -0.202712 | 89.141856 |
| trained_private_tail | heldout | mamba | RGB_to_TI | 0.056468 | 0.050744 | -0.311065 | 95.796848 |
| trained_private_tail | heldout | mamba | NI_to_RGB | 0.495732 | 0.300291 | -0.206570 | 91.243433 |
| trained_private_tail | heldout | mamba | NI_to_NI | 1.000000 | 0.492000 | -0.079754 | 69.527145 |
| trained_private_tail | heldout | mamba | NI_to_TI | 0.117680 | 0.105649 | -0.252066 | 93.520140 |
| trained_private_tail | heldout | mamba | TI_to_RGB | 0.056468 | 0.031758 | -0.318425 | 98.073555 |
| trained_private_tail | heldout | mamba | TI_to_NI | 0.117680 | 0.108258 | -0.247695 | 91.943958 |
| trained_private_tail | heldout | mamba | TI_to_TI | 1.000000 | 0.516408 | -0.065029 | 60.945709 |

## 全部heldout查询的实际部署距离变化

仍是原五路输出，训练尾部减冻结尾部，按全部571 query平均。正距离更小表示
正例更接近，负距离更大表示负例更远；最终mAP以原Q1为准。

| 输出 | 最近正例距离变化 | 最近负例距离变化 | 最近margin变化 |
|---|---:|---:|---:|
| baseline_only | 0.000000000 | 0.000000000 | 0.000000000 |
| fused | -0.002814117 | -0.000990033 | 0.001824084 |
| cnn | -0.008615169 | -0.005311347 | 0.003303822 |
| transformer | 0.000143808 | 0.001325416 | 0.001181608 |
| mamba | 0.000291435 | 0.002920868 | 0.002629433 |

## 支持与不支持的解释

1. 完整来源样本能被既有分类头区分，来源完整融合也饱和；在真实未见身份上却
   未取得稳定配对收益。接下来需要检验泛化约束，而不是继续增加私有尾部容量。
2. 拼接ID/triplet监督未要求专家内部各模态具有共同身份方向。source跨模态
   最近负例更接近的比例仍约71.6%–78.7%，说明这种共同几何并未自动形成。
3. 但现有concat距离只使用相同模态块之间的内积。对各模态施加不同正交旋转
   能改变跨模态cosine而保持原拼接距离不变。因此“跨模态方向不一致”本身不能
   被宣称为融合失败的已证因果机制，也不能证明强制对齐必定改善多模态检索。
4. 由此形成的下一假设仅是：把真实身份感知的跨模态监督作为每个专家内部的
   训练约束，可能改善未见身份泛化。不同专家无需相互对齐；采用前须单独固定
   一项完整配对训练、明确loss温度/权重/预算/下界及原五路科学条件。不得通过
   挑选身份、方向、专家、层数或epoch来“修复”V19失败。

这不是独立新验证，也不支持dev/official/SOTA。V19 Q1独立审计已完成，
engineering PASS、integrity WARN、scientific FAIL，见EXPERIMENT_AUDIT_V19_Q1。
该审计的范围是Q1原始数组与源码；本节后续诊断尚待独立审计，不冒称已被包含。

原始诊断：`evidence/trifusion_v19_generalization_geometry_20260905.json`，47990970
字节，SHA `0e40093688ed568b7e0584672e4a74098c5fba4e57df06fba4bab1b6405adbe6`。
固定范围：`docs/V19_GENERALIZATION_DIAGNOSIS_PROTOCOL_2026-09-05.md`；源码：
`tools/diagnose_v19_generalization_geometry.py`；所有原始数组的算术汇总：
`tools/summarize_v19_generalization_geometry.py`与同名evidence summary JSON。
原始日志、启动回执和传输SHA回执均归档。大权重/图像仍只在服务器，独立审计员
若未实际读取其字节，仍须保留receipt-bound限制。
