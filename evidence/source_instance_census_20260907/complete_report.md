# 完整来源实例—原型普查（2026-09-07）

这是来源模型在已见训练身份上的只读机制诊断。没有新训练或未知身份检索资格结果。
每折使用原V12合法初始化，覆盖全部2126/2075/2051记录；共6252记录—模型配对。
干净/固定增强两视图各完整一次，全部14输出、两关系协议、200前向batch。
全部350112输出行和168聚合项已经CPU独立实现复算；执行器核验不是外部独立审计。

identity_exclude_record排除本记录正例；cross_camera只接受同身份异相机正例。
所有来源记录始终留在干净图库。跨相机无正例的query不计AP，但完整输出且仍能作为其他身份的负例。
真身份原型仅由query的合法干净正例构造，负身份原型由各自全部干净实例构造。
单实例为保存的FP32归一化特征，转float64求点积；prototype求和后归一化。

## clean / identity_exclude_record：全部三折按合法query数加权

以下数量包含不同fold坐标内的重复来源身份与图片，不能作为独立身份样本量。

| 输出 | 全部query | 合法query | source mAP | source R1 | 原型正确 | 原型正确但R1错 | 原型正确但AP<1 | 负原型隐藏首位竞争 | 非正关系/全部关系 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_only | 6252 | 6252 | 100.000000 | 100.000000 | 6252 | 0 | 0 | 0 | 0/286626942 |
| fused | 6252 | 6252 | 100.000000 | 100.000000 | 6252 | 0 | 0 | 0 | 0/286626942 |
| cnn | 6252 | 6252 | 100.000000 | 100.000000 | 6252 | 0 | 0 | 0 | 0/286626942 |
| transformer | 6252 | 6252 | 100.000000 | 100.000000 | 6252 | 0 | 0 | 0 | 0/286626942 |
| mamba | 6252 | 6252 | 99.999973 | 100.000000 | 6252 | 0 | 1 | 0 | 1/286626942 |
| cnn_RGB_residual | 6252 | 6252 | 98.695930 | 99.632118 | 6232 | 13 | 1444 | 10 | 62703/286626942 |
| cnn_NI_residual | 6252 | 6252 | 97.832800 | 99.424184 | 6217 | 18 | 1934 | 11 | 131913/286626942 |
| cnn_TI_residual | 6252 | 6252 | 99.219668 | 99.968010 | 6248 | 2 | 1343 | 2 | 34602/286626942 |
| transformer_RGB_residual | 6252 | 6252 | 99.240273 | 99.488164 | 6228 | 14 | 748 | 15 | 32924/286626942 |
| transformer_NI_residual | 6252 | 6252 | 98.675505 | 99.440179 | 6223 | 17 | 1291 | 8 | 64272/286626942 |
| transformer_TI_residual | 6252 | 6252 | 99.467727 | 99.968010 | 6252 | 2 | 875 | 2 | 23394/286626942 |
| mamba_RGB_residual | 6252 | 6252 | 98.625266 | 99.408189 | 6232 | 21 | 1407 | 15 | 68160/286626942 |
| mamba_NI_residual | 6252 | 6252 | 97.401962 | 99.456174 | 6217 | 12 | 2030 | 12 | 162323/286626942 |
| mamba_TI_residual | 6252 | 6252 | 99.097263 | 99.952015 | 6249 | 2 | 1329 | 2 | 43287/286626942 |

## clean / cross_camera：全部三折按合法query数加权

以下数量包含不同fold坐标内的重复来源身份与图片，不能作为独立身份样本量。

| 输出 | 全部query | 合法query | source mAP | source R1 | 原型正确 | 原型正确但R1错 | 原型正确但AP<1 | 负原型隐藏首位竞争 | 非正关系/全部关系 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_only | 6252 | 1142 | 100.000000 | 100.000000 | 1142 | 0 | 0 | 0 | 0/30550172 |
| fused | 6252 | 1142 | 100.000000 | 100.000000 | 1142 | 0 | 0 | 0 | 0/30550172 |
| cnn | 6252 | 1142 | 100.000000 | 100.000000 | 1142 | 0 | 0 | 0 | 0/30550172 |
| transformer | 6252 | 1142 | 100.000000 | 100.000000 | 1142 | 0 | 0 | 0 | 0/30550172 |
| mamba | 6252 | 1142 | 100.000000 | 100.000000 | 1142 | 0 | 0 | 0 | 0/30550172 |
| cnn_RGB_residual | 6252 | 1142 | 94.919561 | 95.709282 | 1115 | 25 | 420 | 29 | 17671/30550172 |
| cnn_NI_residual | 6252 | 1142 | 91.188190 | 94.395797 | 1111 | 38 | 554 | 49 | 35542/30550172 |
| cnn_TI_residual | 6252 | 1142 | 97.119745 | 99.474606 | 1133 | 2 | 369 | 2 | 9793/30550172 |
| transformer_RGB_residual | 6252 | 1142 | 97.668140 | 98.861646 | 1138 | 9 | 274 | 9 | 7536/30550172 |
| transformer_NI_residual | 6252 | 1142 | 95.953457 | 97.110333 | 1128 | 21 | 374 | 26 | 12411/30550172 |
| transformer_TI_residual | 6252 | 1142 | 98.765829 | 99.649737 | 1138 | 2 | 157 | 3 | 3741/30550172 |
| mamba_RGB_residual | 6252 | 1142 | 93.381665 | 94.395797 | 1115 | 39 | 428 | 41 | 23455/30550172 |
| mamba_NI_residual | 6252 | 1142 | 86.654136 | 88.528897 | 1087 | 78 | 554 | 94 | 50902/30550172 |
| mamba_TI_residual | 6252 | 1142 | 95.529514 | 97.985989 | 1123 | 5 | 341 | 10 | 15528/30550172 |

## augmented / identity_exclude_record：全部三折按合法query数加权

以下数量包含不同fold坐标内的重复来源身份与图片，不能作为独立身份样本量。

| 输出 | 全部query | 合法query | source mAP | source R1 | 原型正确 | 原型正确但R1错 | 原型正确但AP<1 | 负原型隐藏首位竞争 | 非正关系/全部关系 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_only | 6252 | 6252 | 99.999855 | 100.000000 | 6252 | 0 | 1 | 0 | 5/286626942 |
| fused | 6252 | 6252 | 99.999973 | 100.000000 | 6252 | 0 | 1 | 0 | 1/286626942 |
| cnn | 6252 | 6252 | 99.997919 | 100.000000 | 6252 | 0 | 2 | 0 | 85/286626942 |
| transformer | 6252 | 6252 | 100.000000 | 100.000000 | 6252 | 0 | 0 | 0 | 0/286626942 |
| mamba | 6252 | 6252 | 99.997550 | 100.000000 | 6252 | 0 | 3 | 0 | 103/286626942 |
| cnn_RGB_residual | 6252 | 6252 | 97.129731 | 98.816379 | 6186 | 26 | 1926 | 37 | 166939/286626942 |
| cnn_NI_residual | 6252 | 6252 | 95.411976 | 98.432502 | 6127 | 26 | 2424 | 53 | 313720/286626942 |
| cnn_TI_residual | 6252 | 6252 | 97.221110 | 98.896353 | 6174 | 20 | 1841 | 38 | 166360/286626942 |
| transformer_RGB_residual | 6252 | 6252 | 98.052761 | 98.784389 | 6183 | 27 | 1071 | 31 | 141242/286626942 |
| transformer_NI_residual | 6252 | 6252 | 96.928002 | 98.608445 | 6147 | 23 | 1729 | 47 | 190087/286626942 |
| transformer_TI_residual | 6252 | 6252 | 97.979184 | 99.296225 | 6184 | 5 | 1346 | 18 | 126624/286626942 |
| mamba_RGB_residual | 6252 | 6252 | 96.929543 | 98.608445 | 6168 | 30 | 1884 | 44 | 196940/286626942 |
| mamba_NI_residual | 6252 | 6252 | 94.948542 | 98.464491 | 6121 | 24 | 2530 | 49 | 343848/286626942 |
| mamba_TI_residual | 6252 | 6252 | 97.138130 | 98.992322 | 6171 | 11 | 1819 | 33 | 162673/286626942 |

## augmented / cross_camera：全部三折按合法query数加权

以下数量包含不同fold坐标内的重复来源身份与图片，不能作为独立身份样本量。

| 输出 | 全部query | 合法query | source mAP | source R1 | 原型正确 | 原型正确但R1错 | 原型正确但AP<1 | 负原型隐藏首位竞争 | 非正关系/全部关系 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_only | 6252 | 1142 | 100.000000 | 100.000000 | 1142 | 0 | 0 | 0 | 0/30550172 |
| fused | 6252 | 1142 | 100.000000 | 100.000000 | 1142 | 0 | 0 | 0 | 0/30550172 |
| cnn | 6252 | 1142 | 99.996285 | 100.000000 | 1142 | 0 | 1 | 0 | 4/30550172 |
| transformer | 6252 | 1142 | 100.000000 | 100.000000 | 1142 | 0 | 0 | 0 | 0/30550172 |
| mamba | 6252 | 1142 | 99.999027 | 100.000000 | 1142 | 0 | 1 | 0 | 1/30550172 |
| cnn_RGB_residual | 6252 | 1142 | 93.497113 | 96.322242 | 1106 | 12 | 450 | 24 | 24137/30550172 |
| cnn_NI_residual | 6252 | 1142 | 88.635496 | 92.206655 | 1086 | 41 | 567 | 55 | 58205/30550172 |
| cnn_TI_residual | 6252 | 1142 | 91.964360 | 93.695271 | 1093 | 29 | 421 | 43 | 36830/30550172 |
| transformer_RGB_residual | 6252 | 1142 | 95.815907 | 97.110333 | 1119 | 13 | 317 | 20 | 18997/30550172 |
| transformer_NI_residual | 6252 | 1142 | 92.610954 | 94.483363 | 1110 | 35 | 434 | 39 | 30385/30550172 |
| transformer_TI_residual | 6252 | 1142 | 94.979715 | 96.322242 | 1104 | 13 | 245 | 22 | 22920/30550172 |
| mamba_RGB_residual | 6252 | 1142 | 91.408889 | 94.220665 | 1093 | 29 | 457 | 42 | 41470/30550172 |
| mamba_NI_residual | 6252 | 1142 | 84.512909 | 88.441331 | 1061 | 58 | 586 | 89 | 69957/30550172 |
| mamba_TI_residual | 6252 | 1142 | 91.042222 | 94.308231 | 1086 | 21 | 405 | 38 | 38261/30550172 |

## 全部三折分项

| fold | view | protocol | output | 合法query | mAP | R1 | 原型正确但R1错 | 原型正确但AP<1 |
|---|---|---|---|---:|---:|---:|---:|---:|
| 0 | clean | identity_exclude_record | baseline_only | 2126 | 100.000000 | 100.000000 | 0 | 0 |
| 0 | clean | identity_exclude_record | fused | 2126 | 100.000000 | 100.000000 | 0 | 0 |
| 0 | clean | identity_exclude_record | cnn | 2126 | 100.000000 | 100.000000 | 0 | 0 |
| 0 | clean | identity_exclude_record | transformer | 2126 | 100.000000 | 100.000000 | 0 | 0 |
| 0 | clean | identity_exclude_record | mamba | 2126 | 100.000000 | 100.000000 | 0 | 0 |
| 0 | clean | identity_exclude_record | cnn_RGB_residual | 2126 | 98.851033 | 99.952963 | 1 | 571 |
| 0 | clean | identity_exclude_record | cnn_NI_residual | 2126 | 97.794322 | 100.000000 | 0 | 760 |
| 0 | clean | identity_exclude_record | cnn_TI_residual | 2126 | 98.929155 | 99.952963 | 1 | 559 |
| 0 | clean | identity_exclude_record | transformer_RGB_residual | 2126 | 99.650163 | 99.952963 | 0 | 244 |
| 0 | clean | identity_exclude_record | transformer_NI_residual | 2126 | 99.181533 | 99.952963 | 1 | 524 |
| 0 | clean | identity_exclude_record | transformer_TI_residual | 2126 | 99.197200 | 99.952963 | 1 | 382 |
| 0 | clean | identity_exclude_record | mamba_RGB_residual | 2126 | 98.555902 | 99.952963 | 0 | 593 |
| 0 | clean | identity_exclude_record | mamba_NI_residual | 2126 | 97.530828 | 100.000000 | 0 | 791 |
| 0 | clean | identity_exclude_record | mamba_TI_residual | 2126 | 98.536724 | 99.858890 | 2 | 567 |
| 0 | clean | cross_camera | baseline_only | 381 | 100.000000 | 100.000000 | 0 | 0 |
| 0 | clean | cross_camera | fused | 381 | 100.000000 | 100.000000 | 0 | 0 |
| 0 | clean | cross_camera | cnn | 381 | 100.000000 | 100.000000 | 0 | 0 |
| 0 | clean | cross_camera | transformer | 381 | 100.000000 | 100.000000 | 0 | 0 |
| 0 | clean | cross_camera | mamba | 381 | 100.000000 | 100.000000 | 0 | 0 |
| 0 | clean | cross_camera | cnn_RGB_residual | 381 | 92.053383 | 93.175853 | 10 | 163 |
| 0 | clean | cross_camera | cnn_NI_residual | 381 | 90.700601 | 94.225722 | 12 | 195 |
| 0 | clean | cross_camera | cnn_TI_residual | 381 | 95.298363 | 98.425197 | 2 | 121 |
| 0 | clean | cross_camera | transformer_RGB_residual | 381 | 97.469563 | 98.687664 | 4 | 98 |
| 0 | clean | cross_camera | transformer_NI_residual | 381 | 97.724460 | 98.687664 | 3 | 128 |
| 0 | clean | cross_camera | transformer_TI_residual | 381 | 98.227650 | 99.212598 | 1 | 56 |
| 0 | clean | cross_camera | mamba_RGB_residual | 381 | 88.023954 | 88.713911 | 24 | 183 |
| 0 | clean | cross_camera | mamba_NI_residual | 381 | 85.901927 | 90.026247 | 24 | 207 |
| 0 | clean | cross_camera | mamba_TI_residual | 381 | 92.753350 | 95.013123 | 3 | 109 |
| 0 | augmented | identity_exclude_record | baseline_only | 2126 | 100.000000 | 100.000000 | 0 | 0 |
| 0 | augmented | identity_exclude_record | fused | 2126 | 100.000000 | 100.000000 | 0 | 0 |
| 0 | augmented | identity_exclude_record | cnn | 2126 | 99.999562 | 100.000000 | 0 | 1 |
| 0 | augmented | identity_exclude_record | transformer | 2126 | 100.000000 | 100.000000 | 0 | 0 |
| 0 | augmented | identity_exclude_record | mamba | 2126 | 99.999888 | 100.000000 | 0 | 1 |
| 0 | augmented | identity_exclude_record | cnn_RGB_residual | 2126 | 96.858123 | 99.012230 | 4 | 715 |
| 0 | augmented | identity_exclude_record | cnn_NI_residual | 2126 | 94.886392 | 98.494826 | 7 | 935 |
| 0 | augmented | identity_exclude_record | cnn_TI_residual | 2126 | 96.523612 | 98.494826 | 8 | 713 |
| 0 | augmented | identity_exclude_record | transformer_RGB_residual | 2126 | 98.312969 | 99.341486 | 2 | 386 |
| 0 | augmented | identity_exclude_record | transformer_NI_residual | 2126 | 97.199030 | 99.059266 | 7 | 669 |
| 0 | augmented | identity_exclude_record | transformer_TI_residual | 2126 | 97.508884 | 99.200376 | 2 | 540 |
| 0 | augmented | identity_exclude_record | mamba_RGB_residual | 2126 | 96.403299 | 98.824083 | 3 | 740 |
| 0 | augmented | identity_exclude_record | mamba_NI_residual | 2126 | 94.516738 | 98.730009 | 7 | 963 |
| 0 | augmented | identity_exclude_record | mamba_TI_residual | 2126 | 96.267409 | 98.871119 | 4 | 690 |
| 0 | augmented | cross_camera | baseline_only | 381 | 100.000000 | 100.000000 | 0 | 0 |
| 0 | augmented | cross_camera | fused | 381 | 100.000000 | 100.000000 | 0 | 0 |
| 0 | augmented | cross_camera | cnn | 381 | 99.988865 | 100.000000 | 0 | 1 |
| 0 | augmented | cross_camera | transformer | 381 | 100.000000 | 100.000000 | 0 | 0 |
| 0 | augmented | cross_camera | mamba | 381 | 99.997084 | 100.000000 | 0 | 1 |
| 0 | augmented | cross_camera | cnn_RGB_residual | 381 | 89.757216 | 93.700787 | 6 | 165 |
| 0 | augmented | cross_camera | cnn_NI_residual | 381 | 86.636326 | 92.125984 | 9 | 198 |
| 0 | augmented | cross_camera | cnn_TI_residual | 381 | 88.855421 | 90.288714 | 16 | 138 |
| 0 | augmented | cross_camera | transformer_RGB_residual | 381 | 94.819847 | 96.062992 | 8 | 114 |
| 0 | augmented | cross_camera | transformer_NI_residual | 381 | 93.390186 | 95.800525 | 6 | 148 |
| 0 | augmented | cross_camera | transformer_TI_residual | 381 | 94.177127 | 95.800525 | 5 | 68 |
| 0 | augmented | cross_camera | mamba_RGB_residual | 381 | 84.937290 | 88.713911 | 17 | 173 |
| 0 | augmented | cross_camera | mamba_NI_residual | 381 | 82.189951 | 86.351706 | 19 | 206 |
| 0 | augmented | cross_camera | mamba_TI_residual | 381 | 88.247816 | 91.601050 | 11 | 124 |
| 1 | clean | identity_exclude_record | baseline_only | 2075 | 100.000000 | 100.000000 | 0 | 0 |
| 1 | clean | identity_exclude_record | fused | 2075 | 100.000000 | 100.000000 | 0 | 0 |
| 1 | clean | identity_exclude_record | cnn | 2075 | 100.000000 | 100.000000 | 0 | 0 |
| 1 | clean | identity_exclude_record | transformer | 2075 | 100.000000 | 100.000000 | 0 | 0 |
| 1 | clean | identity_exclude_record | mamba | 2075 | 99.999920 | 100.000000 | 0 | 1 |
| 1 | clean | identity_exclude_record | cnn_RGB_residual | 2075 | 98.033770 | 98.939759 | 12 | 444 |
| 1 | clean | identity_exclude_record | cnn_NI_residual | 2075 | 97.412998 | 98.265060 | 18 | 580 |
| 1 | clean | identity_exclude_record | cnn_TI_residual | 2075 | 99.328291 | 99.951807 | 1 | 379 |
| 1 | clean | identity_exclude_record | transformer_RGB_residual | 2075 | 98.356515 | 98.506024 | 14 | 293 |
| 1 | clean | identity_exclude_record | transformer_NI_residual | 2075 | 97.825011 | 98.361446 | 16 | 414 |
| 1 | clean | identity_exclude_record | transformer_TI_residual | 2075 | 99.677653 | 100.000000 | 0 | 198 |
| 1 | clean | identity_exclude_record | mamba_RGB_residual | 2075 | 98.017355 | 98.313253 | 20 | 432 |
| 1 | clean | identity_exclude_record | mamba_NI_residual | 2075 | 96.972976 | 98.361446 | 12 | 566 |
| 1 | clean | identity_exclude_record | mamba_TI_residual | 2075 | 99.430686 | 100.000000 | 0 | 312 |
| 1 | clean | cross_camera | baseline_only | 392 | 100.000000 | 100.000000 | 0 | 0 |
| 1 | clean | cross_camera | fused | 392 | 100.000000 | 100.000000 | 0 | 0 |
| 1 | clean | cross_camera | cnn | 392 | 100.000000 | 100.000000 | 0 | 0 |
| 1 | clean | cross_camera | transformer | 392 | 100.000000 | 100.000000 | 0 | 0 |
| 1 | clean | cross_camera | mamba | 392 | 100.000000 | 100.000000 | 0 | 0 |
| 1 | clean | cross_camera | cnn_RGB_residual | 392 | 96.213252 | 96.173469 | 10 | 121 |
| 1 | clean | cross_camera | cnn_NI_residual | 392 | 91.993323 | 97.193878 | 5 | 177 |
| 1 | clean | cross_camera | cnn_TI_residual | 392 | 98.371331 | 100.000000 | 0 | 121 |
| 1 | clean | cross_camera | transformer_RGB_residual | 392 | 97.099979 | 98.724490 | 4 | 99 |
| 1 | clean | cross_camera | transformer_NI_residual | 392 | 95.387495 | 96.428571 | 8 | 119 |
| 1 | clean | cross_camera | transformer_TI_residual | 392 | 99.575709 | 100.000000 | 0 | 26 |
| 1 | clean | cross_camera | mamba_RGB_residual | 392 | 94.731187 | 95.918367 | 12 | 136 |
| 1 | clean | cross_camera | mamba_NI_residual | 392 | 87.149435 | 88.520408 | 28 | 156 |
| 1 | clean | cross_camera | mamba_TI_residual | 392 | 98.101903 | 100.000000 | 0 | 77 |
| 1 | augmented | identity_exclude_record | baseline_only | 2075 | 100.000000 | 100.000000 | 0 | 0 |
| 1 | augmented | identity_exclude_record | fused | 2075 | 100.000000 | 100.000000 | 0 | 0 |
| 1 | augmented | identity_exclude_record | cnn | 2075 | 100.000000 | 100.000000 | 0 | 0 |
| 1 | augmented | identity_exclude_record | transformer | 2075 | 100.000000 | 100.000000 | 0 | 0 |
| 1 | augmented | identity_exclude_record | mamba | 2075 | 100.000000 | 100.000000 | 0 | 0 |
| 1 | augmented | identity_exclude_record | cnn_RGB_residual | 2075 | 96.858441 | 98.168675 | 16 | 603 |
| 1 | augmented | identity_exclude_record | cnn_NI_residual | 2075 | 95.570924 | 98.072289 | 15 | 739 |
| 1 | augmented | identity_exclude_record | cnn_TI_residual | 2075 | 97.455387 | 99.277108 | 3 | 552 |
| 1 | augmented | identity_exclude_record | transformer_RGB_residual | 2075 | 97.041576 | 97.638554 | 22 | 381 |
| 1 | augmented | identity_exclude_record | transformer_NI_residual | 2075 | 96.428190 | 97.734940 | 13 | 529 |
| 1 | augmented | identity_exclude_record | transformer_TI_residual | 2075 | 98.557213 | 99.807229 | 0 | 347 |
| 1 | augmented | identity_exclude_record | mamba_RGB_residual | 2075 | 96.665600 | 98.024096 | 21 | 599 |
| 1 | augmented | identity_exclude_record | mamba_NI_residual | 2075 | 95.344393 | 97.975904 | 11 | 707 |
| 1 | augmented | identity_exclude_record | mamba_TI_residual | 2075 | 97.623940 | 99.228916 | 3 | 541 |
| 1 | augmented | cross_camera | baseline_only | 392 | 100.000000 | 100.000000 | 0 | 0 |
| 1 | augmented | cross_camera | fused | 392 | 100.000000 | 100.000000 | 0 | 0 |
| 1 | augmented | cross_camera | cnn | 392 | 100.000000 | 100.000000 | 0 | 0 |
| 1 | augmented | cross_camera | transformer | 392 | 100.000000 | 100.000000 | 0 | 0 |
| 1 | augmented | cross_camera | mamba | 392 | 100.000000 | 100.000000 | 0 | 0 |
| 1 | augmented | cross_camera | cnn_RGB_residual | 392 | 95.772623 | 97.959184 | 3 | 134 |
| 1 | augmented | cross_camera | cnn_NI_residual | 392 | 90.502332 | 93.622449 | 14 | 192 |
| 1 | augmented | cross_camera | cnn_TI_residual | 392 | 93.635251 | 95.153061 | 6 | 131 |
| 1 | augmented | cross_camera | transformer_RGB_residual | 392 | 95.383411 | 96.938776 | 4 | 105 |
| 1 | augmented | cross_camera | transformer_NI_residual | 392 | 93.117563 | 94.897959 | 16 | 146 |
| 1 | augmented | cross_camera | transformer_TI_residual | 392 | 96.417618 | 97.704082 | 1 | 67 |
| 1 | augmented | cross_camera | mamba_RGB_residual | 392 | 93.970841 | 96.683673 | 7 | 153 |
| 1 | augmented | cross_camera | mamba_NI_residual | 392 | 86.480705 | 89.795918 | 25 | 180 |
| 1 | augmented | cross_camera | mamba_TI_residual | 392 | 93.346723 | 96.173469 | 5 | 113 |
| 2 | clean | identity_exclude_record | baseline_only | 2051 | 100.000000 | 100.000000 | 0 | 0 |
| 2 | clean | identity_exclude_record | fused | 2051 | 100.000000 | 100.000000 | 0 | 0 |
| 2 | clean | identity_exclude_record | cnn | 2051 | 100.000000 | 100.000000 | 0 | 0 |
| 2 | clean | identity_exclude_record | transformer | 2051 | 100.000000 | 100.000000 | 0 | 0 |
| 2 | clean | identity_exclude_record | mamba | 2051 | 100.000000 | 100.000000 | 0 | 0 |
| 2 | clean | identity_exclude_record | cnn_RGB_residual | 2051 | 99.205063 | 100.000000 | 0 | 429 |
| 2 | clean | identity_exclude_record | cnn_NI_residual | 2051 | 98.297398 | 100.000000 | 0 | 594 |
| 2 | clean | identity_exclude_record | cnn_TI_residual | 2051 | 99.410911 | 100.000000 | 0 | 405 |
| 2 | clean | identity_exclude_record | transformer_RGB_residual | 2051 | 99.709494 | 100.000000 | 0 | 211 |
| 2 | clean | identity_exclude_record | transformer_NI_residual | 2051 | 99.011418 | 100.000000 | 0 | 353 |
| 2 | clean | identity_exclude_record | transformer_TI_residual | 2051 | 99.535763 | 99.951243 | 1 | 295 |
| 2 | clean | identity_exclude_record | mamba_RGB_residual | 2051 | 99.312190 | 99.951243 | 1 | 382 |
| 2 | clean | identity_exclude_record | mamba_NI_residual | 2051 | 97.702390 | 100.000000 | 0 | 673 |
| 2 | clean | identity_exclude_record | mamba_TI_residual | 2051 | 99.340974 | 100.000000 | 0 | 450 |
| 2 | clean | cross_camera | baseline_only | 369 | 100.000000 | 100.000000 | 0 | 0 |
| 2 | clean | cross_camera | fused | 369 | 100.000000 | 100.000000 | 0 | 0 |
| 2 | clean | cross_camera | cnn | 369 | 100.000000 | 100.000000 | 0 | 0 |
| 2 | clean | cross_camera | transformer | 369 | 100.000000 | 100.000000 | 0 | 0 |
| 2 | clean | cross_camera | mamba | 369 | 100.000000 | 100.000000 | 0 | 0 |
| 2 | clean | cross_camera | cnn_RGB_residual | 369 | 96.504621 | 97.831978 | 5 | 136 |
| 2 | clean | cross_camera | cnn_NI_residual | 369 | 90.836320 | 91.598916 | 21 | 182 |
| 2 | clean | cross_camera | cnn_TI_residual | 369 | 97.670760 | 100.000000 | 0 | 127 |
| 2 | clean | cross_camera | transformer_RGB_residual | 369 | 98.476749 | 99.186992 | 1 | 77 |
| 2 | clean | cross_camera | transformer_NI_residual | 369 | 94.726098 | 96.205962 | 10 | 127 |
| 2 | clean | cross_camera | transformer_TI_residual | 369 | 98.461150 | 99.728997 | 1 | 75 |
| 2 | clean | cross_camera | mamba_RGB_residual | 369 | 97.479972 | 98.644986 | 3 | 109 |
| 2 | clean | cross_camera | mamba_NI_residual | 369 | 86.904634 | 86.991870 | 26 | 191 |
| 2 | clean | cross_camera | mamba_TI_residual | 369 | 95.663233 | 98.915989 | 2 | 155 |
| 2 | augmented | identity_exclude_record | baseline_only | 2051 | 99.999557 | 100.000000 | 0 | 1 |
| 2 | augmented | identity_exclude_record | fused | 2051 | 99.999919 | 100.000000 | 0 | 1 |
| 2 | augmented | identity_exclude_record | cnn | 2051 | 99.994111 | 100.000000 | 0 | 1 |
| 2 | augmented | identity_exclude_record | transformer | 2051 | 100.000000 | 100.000000 | 0 | 0 |
| 2 | augmented | identity_exclude_record | mamba | 2051 | 99.992649 | 100.000000 | 0 | 2 |
| 2 | augmented | identity_exclude_record | cnn_RGB_residual | 2051 | 97.685737 | 99.268649 | 6 | 608 |
| 2 | augmented | identity_exclude_record | cnn_NI_residual | 2051 | 95.795971 | 98.732326 | 4 | 750 |
| 2 | augmented | identity_exclude_record | cnn_TI_residual | 2051 | 97.707094 | 98.927353 | 9 | 576 |
| 2 | augmented | identity_exclude_record | transformer_RGB_residual | 2051 | 98.806054 | 99.366163 | 3 | 304 |
| 2 | augmented | identity_exclude_record | transformer_NI_residual | 2051 | 97.152725 | 99.024866 | 3 | 531 |
| 2 | augmented | identity_exclude_record | transformer_TI_residual | 2051 | 97.881889 | 98.878596 | 3 | 459 |
| 2 | augmented | identity_exclude_record | mamba_RGB_residual | 2051 | 97.742063 | 98.976109 | 6 | 545 |
| 2 | augmented | identity_exclude_record | mamba_NI_residual | 2051 | 94.995653 | 98.683569 | 6 | 860 |
| 2 | augmented | identity_exclude_record | mamba_TI_residual | 2051 | 97.549195 | 98.878596 | 4 | 588 |
| 2 | augmented | cross_camera | baseline_only | 369 | 100.000000 | 100.000000 | 0 | 0 |
| 2 | augmented | cross_camera | fused | 369 | 100.000000 | 100.000000 | 0 | 0 |
| 2 | augmented | cross_camera | cnn | 369 | 100.000000 | 100.000000 | 0 | 0 |
| 2 | augmented | cross_camera | transformer | 369 | 100.000000 | 100.000000 | 0 | 0 |
| 2 | augmented | cross_camera | mamba | 369 | 100.000000 | 100.000000 | 0 | 0 |
| 2 | augmented | cross_camera | cnn_RGB_residual | 369 | 94.941289 | 97.289973 | 3 | 151 |
| 2 | augmented | cross_camera | cnn_NI_residual | 369 | 88.716484 | 90.785908 | 18 | 177 |
| 2 | augmented | cross_camera | cnn_TI_residual | 369 | 93.399365 | 95.663957 | 7 | 152 |
| 2 | augmented | cross_camera | transformer_RGB_residual | 369 | 97.303813 | 98.373984 | 1 | 98 |
| 2 | augmented | cross_camera | transformer_NI_residual | 369 | 91.268196 | 92.682927 | 13 | 140 |
| 2 | augmented | cross_camera | transformer_TI_residual | 369 | 94.280876 | 95.392954 | 7 | 110 |
| 2 | augmented | cross_camera | mamba_RGB_residual | 369 | 95.369306 | 97.289973 | 5 | 131 |
| 2 | augmented | cross_camera | mamba_NI_residual | 369 | 84.820961 | 89.159892 | 14 | 200 |
| 2 | augmented | cross_camera | mamba_TI_residual | 369 | 91.479361 | 95.121951 | 5 | 168 |

## 工件与限制

执行commit：99e2a4cf9772737ac3daa14a83a0b4abb3e2d915。
完整来源摘要SHA：b1a1068fa1841e962979986f12438696e28063838dcd36f92f0aa5e24b0a3031。
全量核验SHA：fd4f0f068fd9bce9b812271789cb807b984498b58c1ced2099d83165648a90bb。
数值核验最大误差：0。
提取与统计253.239秒，CPU核验113.656秒。
保存三份完整特征与12份全行记录，未保存模型权重或大型相似度矩阵。
一次固定增强普查不能代表所有增强或训练模式分布，也没有测缓存年龄与特征漂移。
非正关系按negative>=positive计；stable排名的同分次序按原记录索引保留。
原型正确但实例误排的数量只说明存在性和范围，不证明增加记忆、XBM或排序loss能改善新身份。
不得将这里的source mAP与OOF、30-dev、官方测试或论文数值直接相减。
