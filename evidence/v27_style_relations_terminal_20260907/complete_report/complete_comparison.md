# V27 来源统计扰动：完整关系支持诊断

全部三折、固定初始化/对照终点/统计扰动终点、两种输入和两种关系协议已完成完整数组复算。
这不是 Q1 检索或新训练结果；V27 原 Q1_FAIL（4/5）保持。下表分母是重复来源关系曝光，不能当作独立图片或身份。
零余弦间隔非正与欧氏0.3间隔未满足分别报告；后者不是原batch-hard Triplet的同一统计。

## identity / 计划all：全部模型和输入

| 模型状态 | 输入 | 表示 | 关系曝光 | 非正关系 | 非正% | 0.3未满足% | 平均hinge |
|---|---|---|---:|---:|---:|---:|---:|
| initial | original | baseline_only | 42147840 | 16 | 0.00003796 | 0.47564952 | 0.0001956213451 |
| initial | original | fused | 42147840 | 4 | 0.00000949 | 0.09676652 | 3.826825296e-05 |
| initial | original | cnn | 42147840 | 5 | 0.00001186 | 0.15938658 | 6.661220574e-05 |
| initial | original | transformer | 42147840 | 5 | 0.00001186 | 0.06413140 | 2.54737225e-05 |
| initial | original | mamba | 42147840 | 4 | 0.00000949 | 0.17017479 | 7.245542379e-05 |
| initial | original | cnn_RGB_residual | 42147840 | 56473 | 0.13398789 | 2.38246136 | 0.00260858241 |
| initial | original | cnn_NI_residual | 42147840 | 85554 | 0.20298549 | 3.62573740 | 0.004011168469 |
| initial | original | cnn_TI_residual | 42147840 | 61360 | 0.14558279 | 2.56535803 | 0.002875956218 |
| initial | original | transformer_RGB_residual | 42147840 | 35164 | 0.08343014 | 1.48223966 | 0.001588717619 |
| initial | original | transformer_NI_residual | 42147840 | 61712 | 0.14641794 | 2.53104548 | 0.002777362741 |
| initial | original | transformer_TI_residual | 42147840 | 42728 | 0.10137649 | 1.76045558 | 0.001951379553 |
| initial | original | mamba_RGB_residual | 42147840 | 59819 | 0.14192661 | 2.45312453 | 0.002717429725 |
| initial | original | mamba_NI_residual | 42147840 | 96056 | 0.22790254 | 3.74972952 | 0.004240483769 |
| initial | original | mamba_TI_residual | 42147840 | 62330 | 0.14788421 | 2.45903467 | 0.002799802122 |
| initial | original | pure_bank | 42147840 | 2 | 0.00000475 | 0.04412088 | 1.875798618e-05 |
| initial | original | pure_cnn | 42147840 | 28 | 0.00006643 | 0.13283955 | 6.421492117e-05 |
| initial | original | pure_transformer | 42147840 | 5 | 0.00001186 | 0.03024354 | 1.313710119e-05 |
| initial | original | pure_mamba | 42147840 | 65 | 0.00015422 | 0.13885409 | 7.08679703e-05 |
| initial | registered_style | baseline_only | 42147840 | 16 | 0.00003796 | 0.47564952 | 0.0001956213451 |
| initial | registered_style | fused | 42147840 | 46 | 0.00010914 | 0.33714420 | 0.000151594967 |
| initial | registered_style | cnn | 42147840 | 68 | 0.00016134 | 0.40618214 | 0.0001903961637 |
| initial | registered_style | transformer | 42147840 | 62 | 0.00014710 | 0.38715626 | 0.0001780939672 |
| initial | registered_style | mamba | 42147840 | 106 | 0.00025150 | 0.44220771 | 0.0002101598993 |
| initial | registered_style | cnn_RGB_residual | 42147840 | 219539 | 0.52087841 | 5.20100674 | 0.006970283866 |
| initial | registered_style | cnn_NI_residual | 42147840 | 291495 | 0.69160128 | 6.98701760 | 0.009367481864 |
| initial | registered_style | cnn_TI_residual | 42147840 | 119308 | 0.28307026 | 3.77167134 | 0.004609332921 |
| initial | registered_style | transformer_RGB_residual | 42147840 | 307280 | 0.72905278 | 5.92966804 | 0.008679985939 |
| initial | registered_style | transformer_NI_residual | 42147840 | 365527 | 0.86724966 | 7.60065759 | 0.01082418314 |
| initial | registered_style | transformer_TI_residual | 42147840 | 110980 | 0.26331124 | 3.31421492 | 0.004087745839 |
| initial | registered_style | mamba_RGB_residual | 42147840 | 242601 | 0.57559533 | 5.46398107 | 0.007482311344 |
| initial | registered_style | mamba_NI_residual | 42147840 | 331482 | 0.78647447 | 7.38641411 | 0.01018514056 |
| initial | registered_style | mamba_TI_residual | 42147840 | 122531 | 0.29071715 | 3.71750723 | 0.004591049692 |
| initial | registered_style | pure_bank | 42147840 | 1445 | 0.00342841 | 0.74973474 | 0.0004734848592 |
| initial | registered_style | pure_cnn | 42147840 | 2535 | 0.00601454 | 0.87849816 | 0.0005881861861 |
| initial | registered_style | pure_transformer | 42147840 | 3180 | 0.00754487 | 1.19607553 | 0.0008186895861 |
| initial | registered_style | pure_mamba | 42147840 | 2935 | 0.00696358 | 0.96628439 | 0.0006601383976 |
| control_final | original | baseline_only | 42147840 | 16 | 0.00003796 | 0.47564952 | 0.0001956213451 |
| control_final | original | fused | 42147840 | 0 | 0.00000000 | 0.02899318 | 9.502784895e-06 |
| control_final | original | cnn | 42147840 | 0 | 0.00000000 | 0.04158220 | 1.3283457e-05 |
| control_final | original | transformer | 42147840 | 0 | 0.00000000 | 0.02302372 | 7.825407214e-06 |
| control_final | original | mamba | 42147840 | 2 | 0.00000475 | 0.06177066 | 2.166028298e-05 |
| control_final | original | cnn_RGB_residual | 42147840 | 29636 | 0.07031440 | 1.73982344 | 0.001719129986 |
| control_final | original | cnn_NI_residual | 42147840 | 44977 | 0.10671247 | 2.74322955 | 0.002728037538 |
| control_final | original | cnn_TI_residual | 42147840 | 29848 | 0.07081739 | 1.75399261 | 0.001785513213 |
| control_final | original | transformer_RGB_residual | 42147840 | 31497 | 0.07472981 | 1.23658057 | 0.001355401279 |
| control_final | original | transformer_NI_residual | 42147840 | 54484 | 0.12926878 | 2.14566393 | 0.002383784674 |
| control_final | original | transformer_TI_residual | 42147840 | 31616 | 0.07501215 | 1.38583852 | 0.001505121283 |
| control_final | original | mamba_RGB_residual | 42147840 | 40159 | 0.09528128 | 1.91660830 | 0.002015869416 |
| control_final | original | mamba_NI_residual | 42147840 | 63335 | 0.15026867 | 3.02460340 | 0.003209586795 |
| control_final | original | mamba_TI_residual | 42147840 | 39492 | 0.09369875 | 1.90752836 | 0.002036881828 |
| control_final | original | pure_bank | 42147840 | 0 | 0.00000000 | 0.00536445 | 1.92518767e-06 |
| control_final | original | pure_cnn | 42147840 | 0 | 0.00000000 | 0.01614792 | 5.769268201e-06 |
| control_final | original | pure_transformer | 42147840 | 0 | 0.00000000 | 0.00644873 | 2.363511524e-06 |
| control_final | original | pure_mamba | 42147840 | 0 | 0.00000000 | 0.02711408 | 1.025681641e-05 |
| control_final | registered_style | baseline_only | 42147840 | 16 | 0.00003796 | 0.47564952 | 0.0001956213451 |
| control_final | registered_style | fused | 42147840 | 8 | 0.00001898 | 0.20435685 | 8.404096104e-05 |
| control_final | registered_style | cnn | 42147840 | 4 | 0.00000949 | 0.21394928 | 8.571876123e-05 |
| control_final | registered_style | transformer | 42147840 | 51 | 0.00012100 | 0.28357088 | 0.0001282026985 |
| control_final | registered_style | mamba | 42147840 | 26 | 0.00006169 | 0.27079917 | 0.0001174112378 |
| control_final | registered_style | cnn_RGB_residual | 42147840 | 158534 | 0.37613790 | 4.42146501 | 0.005569455933 |
| control_final | registered_style | cnn_NI_residual | 42147840 | 203275 | 0.48229043 | 5.93554972 | 0.007396153637 |
| control_final | registered_style | cnn_TI_residual | 42147840 | 69419 | 0.16470358 | 2.84542695 | 0.003193444129 |
| control_final | registered_style | transformer_RGB_residual | 42147840 | 297877 | 0.70674322 | 5.22467106 | 0.007929060492 |
| control_final | registered_style | transformer_NI_residual | 42147840 | 348279 | 0.82632704 | 6.59455384 | 0.009726876381 |
| control_final | registered_style | transformer_TI_residual | 42147840 | 94222 | 0.22355120 | 2.78296112 | 0.00343976296 |
| control_final | registered_style | mamba_RGB_residual | 42147840 | 197467 | 0.46851037 | 4.70799927 | 0.006279380255 |
| control_final | registered_style | mamba_NI_residual | 42147840 | 270697 | 0.64225593 | 6.46910731 | 0.00864829721 |
| control_final | registered_style | mamba_TI_residual | 42147840 | 95056 | 0.22552994 | 3.14980554 | 0.003754245691 |
| control_final | registered_style | pure_bank | 42147840 | 815 | 0.00193367 | 0.52520366 | 0.000316026311 |
| control_final | registered_style | pure_cnn | 42147840 | 782 | 0.00185537 | 0.54464950 | 0.0003251492934 |
| control_final | registered_style | pure_transformer | 42147840 | 3149 | 0.00747132 | 0.92083960 | 0.0006491652615 |
| control_final | registered_style | pure_mamba | 42147840 | 1644 | 0.00390056 | 0.67593262 | 0.0004349092116 |
| style_final | original | baseline_only | 42147840 | 16 | 0.00003796 | 0.47564952 | 0.0001956213451 |
| style_final | original | fused | 42147840 | 0 | 0.00000000 | 0.03621063 | 1.288905104e-05 |
| style_final | original | cnn | 42147840 | 0 | 0.00000000 | 0.05331234 | 1.860046833e-05 |
| style_final | original | transformer | 42147840 | 0 | 0.00000000 | 0.02584237 | 9.355193991e-06 |
| style_final | original | mamba | 42147840 | 2 | 0.00000475 | 0.07096686 | 2.706750593e-05 |
| style_final | original | cnn_RGB_residual | 42147840 | 34989 | 0.08301493 | 1.85493966 | 0.001900611657 |
| style_final | original | cnn_NI_residual | 42147840 | 52680 | 0.12498861 | 2.88201720 | 0.002953204517 |
| style_final | original | cnn_TI_residual | 42147840 | 32005 | 0.07593509 | 1.69743693 | 0.001775170504 |
| style_final | original | transformer_RGB_residual | 42147840 | 34020 | 0.08071588 | 1.30848224 | 0.001454468717 |
| style_final | original | transformer_NI_residual | 42147840 | 57655 | 0.13679230 | 2.23836382 | 0.002503039823 |
| style_final | original | transformer_TI_residual | 42147840 | 28085 | 0.06663449 | 1.21022335 | 0.001336347433 |
| style_final | original | mamba_RGB_residual | 42147840 | 42765 | 0.10146427 | 1.88369321 | 0.002030083207 |
| style_final | original | mamba_NI_residual | 42147840 | 62098 | 0.14733377 | 2.95093177 | 0.003144915586 |
| style_final | original | mamba_TI_residual | 42147840 | 38822 | 0.09210911 | 1.74741339 | 0.001909404206 |
| style_final | original | pure_bank | 42147840 | 0 | 0.00000000 | 0.00878574 | 3.390638489e-06 |
| style_final | original | pure_cnn | 42147840 | 0 | 0.00000000 | 0.02402970 | 9.432274572e-06 |
| style_final | original | pure_transformer | 42147840 | 0 | 0.00000000 | 0.00768485 | 3.034994303e-06 |
| style_final | original | pure_mamba | 42147840 | 6 | 0.00001424 | 0.03644078 | 1.582387399e-05 |
| style_final | registered_style | baseline_only | 42147840 | 16 | 0.00003796 | 0.47564952 | 0.0001956213451 |
| style_final | registered_style | fused | 42147840 | 1 | 0.00000237 | 0.05946212 | 2.091714211e-05 |
| style_final | registered_style | cnn | 42147840 | 1 | 0.00000237 | 0.07867070 | 2.744309255e-05 |
| style_final | registered_style | transformer | 42147840 | 2 | 0.00000475 | 0.04904166 | 1.78515779e-05 |
| style_final | registered_style | mamba | 42147840 | 7 | 0.00001661 | 0.10871969 | 4.145993015e-05 |
| style_final | registered_style | cnn_RGB_residual | 42147840 | 59439 | 0.14102502 | 2.74960235 | 0.002951087222 |
| style_final | registered_style | cnn_NI_residual | 42147840 | 86379 | 0.20494289 | 4.09819578 | 0.004406974488 |
| style_final | registered_style | cnn_TI_residual | 42147840 | 43476 | 0.10315119 | 2.13672871 | 0.002289643884 |
| style_final | registered_style | transformer_RGB_residual | 42147840 | 66902 | 0.15873174 | 2.27686638 | 0.002660793547 |
| style_final | registered_style | transformer_NI_residual | 42147840 | 104490 | 0.24791306 | 3.60182396 | 0.004229047487 |
| style_final | registered_style | transformer_TI_residual | 42147840 | 41910 | 0.09943570 | 1.64930635 | 0.001868484322 |
| style_final | registered_style | mamba_RGB_residual | 42147840 | 71845 | 0.17045951 | 2.78336446 | 0.003149574581 |
| style_final | registered_style | mamba_NI_residual | 42147840 | 103409 | 0.24534828 | 4.20618471 | 0.00473087289 |
| style_final | registered_style | mamba_TI_residual | 42147840 | 54487 | 0.12927590 | 2.23199101 | 0.002504816462 |
| style_final | registered_style | pure_bank | 42147840 | 8 | 0.00001898 | 0.04087042 | 1.699162896e-05 |
| style_final | registered_style | pure_cnn | 42147840 | 21 | 0.00004982 | 0.07442849 | 3.127716781e-05 |
| style_final | registered_style | pure_transformer | 42147840 | 14 | 0.00003322 | 0.04626809 | 2.01066343e-05 |
| style_final | registered_style | pure_mamba | 42147840 | 43 | 0.00010202 | 0.11005783 | 5.143849882e-05 |

| 模型状态 | 输入 | 关系曝光 | V26形式辅助损失 | 同形目标间隔导数范数比（批均值） |
|---|---|---:|---:|---:|
| initial | original | 42147840 | 6.294726945e-06 | 0.0170004051 |
| initial | registered_style | 42147840 | 2.656404599e-05 | 0.02267087263 |
| control_final | original | 42147840 | 3.112514181e-06 | 0.01132949994 |
| control_final | registered_style | 42147840 | 1.878784304e-05 | 0.0175720478 |
| style_final | original | 42147840 | 3.536627538e-06 | 0.01203196586 |
| style_final | registered_style | 42147840 | 5.554370616e-06 | 0.01326590327 |

## identity / 计划active：全部模型和输入

| 模型状态 | 输入 | 表示 | 关系曝光 | 非正关系 | 非正% | 0.3未满足% | 平均hinge |
|---|---|---|---:|---:|---:|---:|---:|
| initial | original | baseline_only | 20547072 | 5 | 0.00002433 | 0.49134495 | 0.0002007678352 |
| initial | original | fused | 20547072 | 0 | 0.00000000 | 0.10195613 | 4.08161032e-05 |
| initial | original | cnn | 20547072 | 1 | 0.00000487 | 0.16630593 | 7.003639646e-05 |
| initial | original | transformer | 20547072 | 0 | 0.00000000 | 0.06873485 | 2.782803796e-05 |
| initial | original | mamba | 20547072 | 1 | 0.00000487 | 0.17682811 | 7.547746258e-05 |
| initial | original | cnn_RGB_residual | 20547072 | 26508 | 0.12901108 | 2.37411442 | 0.002582286503 |
| initial | original | cnn_NI_residual | 20547072 | 42084 | 0.20481750 | 3.67380326 | 0.004060516933 |
| initial | original | cnn_TI_residual | 20547072 | 32569 | 0.15850920 | 2.64711196 | 0.003017649076 |
| initial | original | transformer_RGB_residual | 20547072 | 16455 | 0.08008440 | 1.47847343 | 0.001576770838 |
| initial | original | transformer_NI_residual | 20547072 | 30246 | 0.14720346 | 2.59364449 | 0.002831302943 |
| initial | original | transformer_TI_residual | 20547072 | 22481 | 0.10941218 | 1.81020926 | 0.002041764166 |
| initial | original | mamba_RGB_residual | 20547072 | 28212 | 0.13730423 | 2.44102907 | 0.002685516206 |
| initial | original | mamba_NI_residual | 20547072 | 47135 | 0.22940008 | 3.82240350 | 0.004307199893 |
| initial | original | mamba_TI_residual | 20547072 | 32613 | 0.15872335 | 2.53916957 | 0.002935903649 |
| initial | original | pure_bank | 20547072 | 0 | 0.00000000 | 0.04807011 | 2.084264662e-05 |
| initial | original | pure_cnn | 20547072 | 17 | 0.00008274 | 0.13966467 | 6.907004152e-05 |
| initial | original | pure_transformer | 20547072 | 1 | 0.00000487 | 0.03346949 | 1.472693518e-05 |
| initial | original | pure_mamba | 20547072 | 24 | 0.00011680 | 0.14444881 | 7.320054409e-05 |
| initial | registered_style | baseline_only | 20547072 | 5 | 0.00002433 | 0.49134495 | 0.0002007678352 |
| initial | registered_style | fused | 20547072 | 42 | 0.00020441 | 0.59503855 | 0.0002732811577 |
| initial | registered_style | cnn | 20547072 | 64 | 0.00031148 | 0.67255325 | 0.0003239522077 |
| initial | registered_style | transformer | 20547072 | 57 | 0.00027741 | 0.73134995 | 0.0003408952065 |
| initial | registered_style | mamba | 20547072 | 103 | 0.00050129 | 0.73484436 | 0.0003579481817 |
| initial | registered_style | cnn_RGB_residual | 20547072 | 189574 | 0.92263267 | 8.15574599 | 0.01152936641 |
| initial | registered_style | cnn_NI_residual | 20547072 | 248025 | 1.20710630 | 10.56873700 | 0.01504782646 |
| initial | registered_style | cnn_TI_residual | 20547072 | 90517 | 0.44053479 | 5.12160078 | 0.006573293594 |
| initial | registered_style | transformer_RGB_residual | 20547072 | 288571 | 1.40443855 | 10.60140345 | 0.01612296226 |
| initial | registered_style | transformer_NI_residual | 20547072 | 334061 | 1.62583262 | 12.99284881 | 0.0193376012 |
| initial | registered_style | transformer_TI_residual | 20547072 | 90733 | 0.44158603 | 4.99740790 | 0.006424053982 |
| initial | registered_style | mamba_RGB_residual | 20547072 | 210994 | 1.02688111 | 8.61714506 | 0.01245963235 |
| initial | registered_style | mamba_NI_residual | 20547072 | 282561 | 1.37518864 | 11.28226932 | 0.01650136767 |
| initial | registered_style | mamba_TI_residual | 20547072 | 92814 | 0.45171400 | 5.12065174 | 0.006610257638 |
| initial | registered_style | pure_bank | 20547072 | 1443 | 0.00702290 | 1.49548315 | 0.0009536157195 |
| initial | registered_style | pure_cnn | 20547072 | 2524 | 0.01228399 | 1.66922080 | 0.001143882893 |
| initial | registered_style | pure_transformer | 20547072 | 3176 | 0.01545719 | 2.42491972 | 0.001667142289 |
| initial | registered_style | pure_mamba | 20547072 | 2894 | 0.01408473 | 1.84174173 | 0.001281960395 |
| control_final | original | baseline_only | 20547072 | 5 | 0.00002433 | 0.49134495 | 0.0002007678352 |
| control_final | original | fused | 20547072 | 0 | 0.00000000 | 0.03045203 | 9.999765976e-06 |
| control_final | original | cnn | 20547072 | 0 | 0.00000000 | 0.04144143 | 1.305277662e-05 |
| control_final | original | transformer | 20547072 | 0 | 0.00000000 | 0.02593070 | 8.79853299e-06 |
| control_final | original | mamba | 20547072 | 0 | 0.00000000 | 0.06304548 | 2.199834992e-05 |
| control_final | original | cnn_RGB_residual | 20547072 | 13928 | 0.06778581 | 1.71002954 | 0.00167956684 |
| control_final | original | cnn_NI_residual | 20547072 | 21359 | 0.10395155 | 2.77126590 | 0.002732554013 |
| control_final | original | cnn_TI_residual | 20547072 | 15302 | 0.07447290 | 1.79007987 | 0.001843673855 |
| control_final | original | transformer_RGB_residual | 20547072 | 14944 | 0.07273056 | 1.21697145 | 0.001330253647 |
| control_final | original | transformer_NI_residual | 20547072 | 27047 | 0.13163433 | 2.19380649 | 0.002435166484 |
| control_final | original | transformer_TI_residual | 20547072 | 16637 | 0.08097017 | 1.42164295 | 0.001571762357 |
| control_final | original | mamba_RGB_residual | 20547072 | 19034 | 0.09263607 | 1.88988485 | 0.001978576263 |
| control_final | original | mamba_NI_residual | 20547072 | 30222 | 0.14708665 | 3.07055429 | 0.003238179804 |
| control_final | original | mamba_TI_residual | 20547072 | 20272 | 0.09866126 | 1.96024524 | 0.00211905007 |
| control_final | original | pure_bank | 20547072 | 0 | 0.00000000 | 0.00539736 | 1.969081349e-06 |
| control_final | original | pure_cnn | 20547072 | 0 | 0.00000000 | 0.01518951 | 5.251915281e-06 |
| control_final | original | pure_transformer | 20547072 | 0 | 0.00000000 | 0.00751445 | 2.628810187e-06 |
| control_final | original | pure_mamba | 20547072 | 0 | 0.00000000 | 0.02613024 | 9.483382928e-06 |
| control_final | registered_style | baseline_only | 20547072 | 5 | 0.00002433 | 0.49134495 | 0.0002007678352 |
| control_final | registered_style | fused | 20547072 | 8 | 0.00003893 | 0.39017238 | 0.0001628985888 |
| control_final | registered_style | cnn | 20547072 | 4 | 0.00001947 | 0.39501492 | 0.0001616380161 |
| control_final | registered_style | transformer | 20547072 | 51 | 0.00024821 | 0.56038641 | 0.0002557263101 |
| control_final | registered_style | mamba | 20547072 | 24 | 0.00011680 | 0.49182190 | 0.0002184105649 |
| control_final | registered_style | cnn_RGB_residual | 20547072 | 142826 | 0.69511607 | 7.21083276 | 0.009577671347 |
| control_final | registered_style | cnn_NI_residual | 20547072 | 179657 | 0.87436789 | 9.31961498 | 0.01230817678 |
| control_final | registered_style | cnn_TI_residual | 20547072 | 54873 | 0.26705995 | 4.02891955 | 0.004731737272 |
| control_final | registered_style | transformer_RGB_residual | 20547072 | 281324 | 1.36916832 | 9.39766990 | 0.0148146828 |
| control_final | registered_style | transformer_NI_residual | 20547072 | 320842 | 1.56149742 | 11.31973451 | 0.0174979187 |
| control_final | registered_style | transformer_TI_residual | 20547072 | 79243 | 0.38566566 | 4.28753547 | 0.005540258105 |
| control_final | registered_style | mamba_RGB_residual | 20547072 | 176342 | 0.85823420 | 7.61581504 | 0.01072423952 |
| control_final | registered_style | mamba_NI_residual | 20547072 | 237584 | 1.15629127 | 10.13620335 | 0.01439450886 |
| control_final | registered_style | mamba_TI_residual | 20547072 | 75836 | 0.36908422 | 4.50850613 | 0.005641847737 |
| control_final | registered_style | pure_bank | 20547072 | 815 | 0.00396650 | 1.07173421 | 0.0006462790778 |
| control_final | registered_style | pure_cnn | 20547072 | 782 | 0.00380590 | 1.09929532 | 0.0006603904284 |
| control_final | registered_style | pure_transformer | 20547072 | 3149 | 0.01532578 | 1.88318803 | 0.001329401631 |
| control_final | registered_style | pure_mamba | 20547072 | 1644 | 0.00800114 | 1.35704007 | 0.0008805652192 |
| style_final | original | baseline_only | 20547072 | 5 | 0.00002433 | 0.49134495 | 0.0002007678352 |
| style_final | original | fused | 20547072 | 0 | 0.00000000 | 0.03877438 | 1.444360623e-05 |
| style_final | original | cnn | 20547072 | 0 | 0.00000000 | 0.05534609 | 1.984631127e-05 |
| style_final | original | transformer | 20547072 | 0 | 0.00000000 | 0.02881189 | 1.088699242e-05 |
| style_final | original | mamba | 20547072 | 1 | 0.00000487 | 0.07447777 | 2.913604373e-05 |
| style_final | original | cnn_RGB_residual | 20547072 | 15834 | 0.07706207 | 1.80572200 | 0.00183445167 |
| style_final | original | cnn_NI_residual | 20547072 | 25080 | 0.12206119 | 2.85272763 | 0.002910649363 |
| style_final | original | cnn_TI_residual | 20547072 | 16428 | 0.07995300 | 1.71069143 | 0.001808229656 |
| style_final | original | transformer_RGB_residual | 20547072 | 16378 | 0.07970965 | 1.29594621 | 0.001440877175 |
| style_final | original | transformer_NI_residual | 20547072 | 27583 | 0.13424297 | 2.24000772 | 0.002490947043 |
| style_final | original | transformer_TI_residual | 20547072 | 14948 | 0.07275002 | 1.21857265 | 0.001373693764 |
| style_final | original | mamba_RGB_residual | 20547072 | 19715 | 0.09595041 | 1.83725934 | 0.001963716491 |
| style_final | original | mamba_NI_residual | 20547072 | 29622 | 0.14416653 | 2.95876220 | 0.003138720392 |
| style_final | original | mamba_TI_residual | 20547072 | 20247 | 0.09853959 | 1.78956398 | 0.00198794557 |
| style_final | original | pure_bank | 20547072 | 0 | 0.00000000 | 0.01031290 | 4.19545035e-06 |
| style_final | original | pure_cnn | 20547072 | 0 | 0.00000000 | 0.02540508 | 1.000586287e-05 |
| style_final | original | pure_transformer | 20547072 | 0 | 0.00000000 | 0.00925679 | 3.844504286e-06 |
| style_final | original | pure_mamba | 20547072 | 3 | 0.00001460 | 0.03876465 | 1.720789749e-05 |
| style_final | registered_style | baseline_only | 20547072 | 5 | 0.00002433 | 0.49134495 | 0.0002007678352 |
| style_final | registered_style | fused | 20547072 | 1 | 0.00000487 | 0.08646974 | 3.091148536e-05 |
| style_final | registered_style | cnn | 20547072 | 1 | 0.00000487 | 0.10736323 | 3.798502762e-05 |
| style_final | registered_style | transformer | 20547072 | 2 | 0.00000973 | 0.07640018 | 2.831547222e-05 |
| style_final | registered_style | mamba | 20547072 | 6 | 0.00002920 | 0.15191946 | 5.865896521e-05 |
| style_final | registered_style | cnn_RGB_residual | 20547072 | 40284 | 0.19605713 | 3.64092752 | 0.003989273342 |
| style_final | registered_style | cnn_NI_residual | 20547072 | 58779 | 0.28606996 | 5.34745291 | 0.005892741613 |
| style_final | registered_style | cnn_TI_residual | 20547072 | 27899 | 0.13578090 | 2.61180279 | 0.002863559665 |
| style_final | registered_style | transformer_RGB_residual | 20547072 | 49260 | 0.23974219 | 3.28237522 | 0.003915389647 |
| style_final | registered_style | transformer_NI_residual | 20547072 | 74418 | 0.36218299 | 5.03684905 | 0.006031475585 |
| style_final | registered_style | transformer_TI_residual | 20547072 | 28773 | 0.14003455 | 2.11925573 | 0.002465256613 |
| style_final | registered_style | mamba_RGB_residual | 20547072 | 48795 | 0.23747909 | 3.68273884 | 0.004260109052 |
| style_final | registered_style | mamba_NI_residual | 20547072 | 70933 | 0.34522194 | 5.53364002 | 0.006391966144 |
| style_final | registered_style | mamba_TI_residual | 20547072 | 35912 | 0.17477916 | 2.78356936 | 0.003209304044 |
| style_final | registered_style | pure_bank | 20547072 | 8 | 0.00003893 | 0.07612764 | 3.209491798e-05 |
| style_final | registered_style | pure_cnn | 20547072 | 21 | 0.00010220 | 0.12878721 | 5.481590029e-05 |
| style_final | registered_style | pure_transformer | 20547072 | 14 | 0.00006814 | 0.08840189 | 3.886325299e-05 |
| style_final | registered_style | pure_mamba | 20547072 | 40 | 0.00019467 | 0.18977400 | 9.026353817e-05 |

| 模型状态 | 输入 | 关系曝光 | V26形式辅助损失 | 同形目标间隔导数范数比（批均值） |
|---|---|---:|---:|---:|
| initial | original | 20547072 | 6.4243391e-06 | 0.01751809243 |
| initial | registered_style | 20547072 | 4.800242946e-05 | 0.02914982069 |
| control_final | original | 20547072 | 3.114708719e-06 | 0.01161327512 |
| control_final | registered_style | 20547072 | 3.526922946e-05 | 0.0244185015 |
| style_final | original | 20547072 | 3.531988911e-06 | 0.0124112841 |
| style_final | registered_style | 20547072 | 7.670949071e-06 | 0.01494243776 |

## identity / 计划inactive：全部模型和输入

| 模型状态 | 输入 | 表示 | 关系曝光 | 非正关系 | 非正% | 0.3未满足% | 平均hinge |
|---|---|---|---:|---:|---:|---:|---:|
| initial | original | baseline_only | 21600768 | 11 | 0.00005092 | 0.46071973 | 0.0001907259033 |
| initial | original | fused | 21600768 | 4 | 0.00001852 | 0.09183007 | 3.584468809e-05 |
| initial | original | cnn | 21600768 | 4 | 0.00001852 | 0.15280475 | 6.335504872e-05 |
| initial | original | transformer | 21600768 | 5 | 0.00002315 | 0.05975251 | 2.323425169e-05 |
| initial | original | mamba | 21600768 | 3 | 0.00001389 | 0.16384603 | 6.958080151e-05 |
| initial | original | cnn_RGB_residual | 21600768 | 29965 | 0.13872192 | 2.39040112 | 0.002633595589 |
| initial | original | cnn_NI_residual | 21600768 | 43470 | 0.20124284 | 3.58001623 | 0.003964227247 |
| initial | original | cnn_TI_residual | 21600768 | 28791 | 0.13328693 | 2.48759211 | 0.002741175207 |
| initial | original | transformer_RGB_residual | 21600768 | 18709 | 0.08661266 | 1.48582217 | 0.001600081631 |
| initial | original | transformer_NI_residual | 21600768 | 31466 | 0.14567075 | 2.47150009 | 0.002726053769 |
| initial | original | transformer_TI_residual | 21600768 | 20247 | 0.09373278 | 1.71312890 | 0.001865403947 |
| initial | original | mamba_RGB_residual | 21600768 | 31607 | 0.14632350 | 2.46462996 | 0.002747786487 |
| initial | original | mamba_NI_residual | 21600768 | 48921 | 0.22647806 | 3.68060062 | 0.00417702209 |
| initial | original | mamba_TI_residual | 21600768 | 29717 | 0.13757381 | 2.38280880 | 0.002670339694 |
| initial | original | pure_bank | 21600768 | 2 | 0.00000926 | 0.04036431 | 1.67750165e-05 |
| initial | original | pure_cnn | 21600768 | 11 | 0.00005092 | 0.12634736 | 5.959663595e-05 |
| initial | original | pure_transformer | 21600768 | 4 | 0.00001852 | 0.02717496 | 1.162482008e-05 |
| initial | original | pure_mamba | 21600768 | 41 | 0.00018981 | 0.13353229 | 6.864918061e-05 |
| initial | registered_style | baseline_only | 21600768 | 11 | 0.00005092 | 0.46071973 | 0.0001907259033 |
| initial | registered_style | fused | 21600768 | 4 | 0.00001852 | 0.09183007 | 3.584468809e-05 |
| initial | registered_style | cnn | 21600768 | 4 | 0.00001852 | 0.15280475 | 6.335504872e-05 |
| initial | registered_style | transformer | 21600768 | 5 | 0.00002315 | 0.05975251 | 2.323425169e-05 |
| initial | registered_style | mamba | 21600768 | 3 | 0.00001389 | 0.16384603 | 6.958080151e-05 |
| initial | registered_style | cnn_RGB_residual | 21600768 | 29965 | 0.13872192 | 2.39040112 | 0.002633595589 |
| initial | registered_style | cnn_NI_residual | 21600768 | 43470 | 0.20124284 | 3.58001623 | 0.003964227247 |
| initial | registered_style | cnn_TI_residual | 21600768 | 28791 | 0.13328693 | 2.48759211 | 0.002741175207 |
| initial | registered_style | transformer_RGB_residual | 21600768 | 18709 | 0.08661266 | 1.48582217 | 0.001600081631 |
| initial | registered_style | transformer_NI_residual | 21600768 | 31466 | 0.14567075 | 2.47150009 | 0.002726053769 |
| initial | registered_style | transformer_TI_residual | 21600768 | 20247 | 0.09373278 | 1.71312890 | 0.001865403947 |
| initial | registered_style | mamba_RGB_residual | 21600768 | 31607 | 0.14632350 | 2.46462996 | 0.002747786487 |
| initial | registered_style | mamba_NI_residual | 21600768 | 48921 | 0.22647806 | 3.68060062 | 0.00417702209 |
| initial | registered_style | mamba_TI_residual | 21600768 | 29717 | 0.13757381 | 2.38280880 | 0.002670339694 |
| initial | registered_style | pure_bank | 21600768 | 2 | 0.00000926 | 0.04036431 | 1.67750165e-05 |
| initial | registered_style | pure_cnn | 21600768 | 11 | 0.00005092 | 0.12634736 | 5.959663595e-05 |
| initial | registered_style | pure_transformer | 21600768 | 4 | 0.00001852 | 0.02717496 | 1.162482008e-05 |
| initial | registered_style | pure_mamba | 21600768 | 41 | 0.00018981 | 0.13353229 | 6.864918061e-05 |
| control_final | original | baseline_only | 21600768 | 11 | 0.00005092 | 0.46071973 | 0.0001907259033 |
| control_final | original | fused | 21600768 | 0 | 0.00000000 | 0.02760550 | 9.030046793e-06 |
| control_final | original | cnn | 21600768 | 0 | 0.00000000 | 0.04171611 | 1.350288469e-05 |
| control_final | original | transformer | 21600768 | 0 | 0.00000000 | 0.02025854 | 6.899750989e-06 |
| control_final | original | mamba | 21600768 | 2 | 0.00000926 | 0.06055803 | 2.133870711e-05 |
| control_final | original | cnn_RGB_residual | 21600768 | 15708 | 0.07271964 | 1.76816398 | 0.001756763222 |
| control_final | original | cnn_NI_residual | 21600768 | 23618 | 0.10933870 | 2.71656082 | 0.00272374138 |
| control_final | original | cnn_TI_residual | 21600768 | 14546 | 0.06734020 | 1.71966571 | 0.001730189676 |
| control_final | original | transformer_RGB_residual | 21600768 | 16553 | 0.07663153 | 1.25523315 | 0.001379322198 |
| control_final | original | transformer_NI_residual | 21600768 | 27437 | 0.12701863 | 2.09986978 | 0.002334909295 |
| control_final | original | transformer_TI_residual | 21600768 | 14979 | 0.06934476 | 1.35178064 | 0.001441730993 |
| control_final | original | mamba_RGB_residual | 21600768 | 21125 | 0.09779745 | 1.94202817 | 0.00205134339 |
| control_final | original | mamba_NI_residual | 21600768 | 33113 | 0.15329548 | 2.98089401 | 0.003182388567 |
| control_final | original | mamba_TI_residual | 21600768 | 19220 | 0.08897832 | 1.85738303 | 0.001958721794 |
| control_final | original | pure_bank | 21600768 | 0 | 0.00000000 | 0.00533314 | 1.883435146e-06 |
| control_final | original | pure_cnn | 21600768 | 0 | 0.00000000 | 0.01705958 | 6.261384394e-06 |
| control_final | original | pure_transformer | 21600768 | 0 | 0.00000000 | 0.00543499 | 2.11115426e-06 |
| control_final | original | pure_mamba | 21600768 | 0 | 0.00000000 | 0.02804993 | 1.099252143e-05 |
| control_final | registered_style | baseline_only | 21600768 | 11 | 0.00005092 | 0.46071973 | 0.0001907259033 |
| control_final | registered_style | fused | 21600768 | 0 | 0.00000000 | 0.02760550 | 9.030046793e-06 |
| control_final | registered_style | cnn | 21600768 | 0 | 0.00000000 | 0.04171611 | 1.350288469e-05 |
| control_final | registered_style | transformer | 21600768 | 0 | 0.00000000 | 0.02025854 | 6.899750989e-06 |
| control_final | registered_style | mamba | 21600768 | 2 | 0.00000926 | 0.06055803 | 2.133870711e-05 |
| control_final | registered_style | cnn_RGB_residual | 21600768 | 15708 | 0.07271964 | 1.76816398 | 0.001756763222 |
| control_final | registered_style | cnn_NI_residual | 21600768 | 23618 | 0.10933870 | 2.71656082 | 0.00272374138 |
| control_final | registered_style | cnn_TI_residual | 21600768 | 14546 | 0.06734020 | 1.71966571 | 0.001730189676 |
| control_final | registered_style | transformer_RGB_residual | 21600768 | 16553 | 0.07663153 | 1.25523315 | 0.001379322198 |
| control_final | registered_style | transformer_NI_residual | 21600768 | 27437 | 0.12701863 | 2.09986978 | 0.002334909295 |
| control_final | registered_style | transformer_TI_residual | 21600768 | 14979 | 0.06934476 | 1.35178064 | 0.001441730993 |
| control_final | registered_style | mamba_RGB_residual | 21600768 | 21125 | 0.09779745 | 1.94202817 | 0.00205134339 |
| control_final | registered_style | mamba_NI_residual | 21600768 | 33113 | 0.15329548 | 2.98089401 | 0.003182388567 |
| control_final | registered_style | mamba_TI_residual | 21600768 | 19220 | 0.08897832 | 1.85738303 | 0.001958721794 |
| control_final | registered_style | pure_bank | 21600768 | 0 | 0.00000000 | 0.00533314 | 1.883435146e-06 |
| control_final | registered_style | pure_cnn | 21600768 | 0 | 0.00000000 | 0.01705958 | 6.261384394e-06 |
| control_final | registered_style | pure_transformer | 21600768 | 0 | 0.00000000 | 0.00543499 | 2.11115426e-06 |
| control_final | registered_style | pure_mamba | 21600768 | 0 | 0.00000000 | 0.02804993 | 1.099252143e-05 |
| style_final | original | baseline_only | 21600768 | 11 | 0.00005092 | 0.46071973 | 0.0001907259033 |
| style_final | original | fused | 21600768 | 0 | 0.00000000 | 0.03377195 | 1.14103278e-05 |
| style_final | original | cnn | 21600768 | 0 | 0.00000000 | 0.05137780 | 1.741539822e-05 |
| style_final | original | transformer | 21600768 | 0 | 0.00000000 | 0.02301770 | 7.898117441e-06 |
| style_final | original | mamba | 21600768 | 1 | 0.00000463 | 0.06762723 | 2.509987242e-05 |
| style_final | original | cnn_RGB_residual | 21600768 | 19155 | 0.08867740 | 1.90175646 | 0.001963544327 |
| style_final | original | cnn_NI_residual | 21600768 | 27600 | 0.12777323 | 2.90987802 | 0.002993683809 |
| style_final | original | cnn_TI_residual | 21600768 | 15577 | 0.07211318 | 1.68482898 | 0.001743723994 |
| style_final | original | transformer_RGB_residual | 21600768 | 17642 | 0.08167302 | 1.32040676 | 0.001467397257 |
| style_final | original | transformer_NI_residual | 21600768 | 30072 | 0.13921727 | 2.23680010 | 0.002514542711 |
| style_final | original | transformer_TI_residual | 21600768 | 13137 | 0.06081728 | 1.20228133 | 0.001300822874 |
| style_final | original | mamba_RGB_residual | 21600768 | 23050 | 0.10670917 | 1.92786201 | 0.002093212523 |
| style_final | original | mamba_NI_residual | 21600768 | 32476 | 0.15034651 | 2.94348331 | 0.003150808575 |
| style_final | original | mamba_TI_residual | 21600768 | 18575 | 0.08599231 | 1.70731892 | 0.001834694128 |
| style_final | original | pure_bank | 21600768 | 0 | 0.00000000 | 0.00733307 | 2.625085744e-06 |
| style_final | original | pure_cnn | 21600768 | 0 | 0.00000000 | 0.02272141 | 8.886666191e-06 |
| style_final | original | pure_transformer | 21600768 | 0 | 0.00000000 | 0.00618959 | 2.264972613e-06 |
| style_final | original | pure_mamba | 21600768 | 3 | 0.00001389 | 0.03423026 | 1.450736383e-05 |
| style_final | registered_style | baseline_only | 21600768 | 11 | 0.00005092 | 0.46071973 | 0.0001907259033 |
| style_final | registered_style | fused | 21600768 | 0 | 0.00000000 | 0.03377195 | 1.14103278e-05 |
| style_final | registered_style | cnn | 21600768 | 0 | 0.00000000 | 0.05137780 | 1.741539822e-05 |
| style_final | registered_style | transformer | 21600768 | 0 | 0.00000000 | 0.02301770 | 7.898117441e-06 |
| style_final | registered_style | mamba | 21600768 | 1 | 0.00000463 | 0.06762723 | 2.509987242e-05 |
| style_final | registered_style | cnn_RGB_residual | 21600768 | 19155 | 0.08867740 | 1.90175646 | 0.001963544327 |
| style_final | registered_style | cnn_NI_residual | 21600768 | 27600 | 0.12777323 | 2.90987802 | 0.002993683809 |
| style_final | registered_style | cnn_TI_residual | 21600768 | 15577 | 0.07211318 | 1.68482898 | 0.001743723994 |
| style_final | registered_style | transformer_RGB_residual | 21600768 | 17642 | 0.08167302 | 1.32040676 | 0.001467397257 |
| style_final | registered_style | transformer_NI_residual | 21600768 | 30072 | 0.13921727 | 2.23680010 | 0.002514542711 |
| style_final | registered_style | transformer_TI_residual | 21600768 | 13137 | 0.06081728 | 1.20228133 | 0.001300822874 |
| style_final | registered_style | mamba_RGB_residual | 21600768 | 23050 | 0.10670917 | 1.92786201 | 0.002093212523 |
| style_final | registered_style | mamba_NI_residual | 21600768 | 32476 | 0.15034651 | 2.94348331 | 0.003150808575 |
| style_final | registered_style | mamba_TI_residual | 21600768 | 18575 | 0.08599231 | 1.70731892 | 0.001834694128 |
| style_final | registered_style | pure_bank | 21600768 | 0 | 0.00000000 | 0.00733307 | 2.625085744e-06 |
| style_final | registered_style | pure_cnn | 21600768 | 0 | 0.00000000 | 0.02272141 | 8.886666191e-06 |
| style_final | registered_style | pure_transformer | 21600768 | 0 | 0.00000000 | 0.00618959 | 2.264972613e-06 |
| style_final | registered_style | pure_mamba | 21600768 | 3 | 0.00001389 | 0.03423026 | 1.450736383e-05 |

| 模型状态 | 输入 | 关系曝光 | V26形式辅助损失 | 同形目标间隔导数范数比（批均值） |
|---|---|---:|---:|---:|
| initial | original | 21600768 | 6.171437334e-06 | 0.01650797081 |
| initial | registered_style | 21600768 | 6.171437334e-06 | 0.01650797081 |
| control_final | original | 21600768 | 3.110426694e-06 | 0.01105956746 |
| control_final | registered_style | 21600768 | 3.110426694e-06 | 0.01105956746 |
| style_final | original | 21600768 | 3.54103989e-06 | 0.01167115095 |
| style_final | registered_style | 21600768 | 3.54103989e-06 | 0.01167115095 |

## cross_camera / 计划all：全部模型和输入

| 模型状态 | 输入 | 表示 | 关系曝光 | 非正关系 | 非正% | 0.3未满足% | 平均hinge |
|---|---|---|---:|---:|---:|---:|---:|
| initial | original | baseline_only | 3401664 | 7 | 0.00020578 | 1.43159348 | 0.000594234097 |
| initial | original | fused | 3401664 | 4 | 0.00011759 | 0.26645783 | 0.0001098760781 |
| initial | original | cnn | 3401664 | 4 | 0.00011759 | 0.42752606 | 0.000184278965 |
| initial | original | transformer | 3401664 | 5 | 0.00014699 | 0.17432645 | 7.060845551e-05 |
| initial | original | mamba | 3401664 | 3 | 0.00008819 | 0.45151432 | 0.0001959011597 |
| initial | original | cnn_RGB_residual | 3401664 | 6537 | 0.19217066 | 3.58595088 | 0.003945198704 |
| initial | original | cnn_NI_residual | 3401664 | 12456 | 0.36617373 | 6.32172960 | 0.00722536939 |
| initial | original | cnn_TI_residual | 3401664 | 10873 | 0.31963768 | 4.67141963 | 0.005598331304 |
| initial | original | transformer_RGB_residual | 3401664 | 4172 | 0.12264586 | 2.26615562 | 0.002456579559 |
| initial | original | transformer_NI_residual | 3401664 | 9584 | 0.28174446 | 4.36668642 | 0.004948754398 |
| initial | original | transformer_TI_residual | 3401664 | 6565 | 0.19299378 | 3.12229544 | 0.003546349836 |
| initial | original | mamba_RGB_residual | 3401664 | 7505 | 0.22062732 | 3.89706332 | 0.004352020369 |
| initial | original | mamba_NI_residual | 3401664 | 15255 | 0.44845699 | 6.72950062 | 0.007971582012 |
| initial | original | mamba_TI_residual | 3401664 | 10445 | 0.30705561 | 4.43136065 | 0.005335920509 |
| initial | original | pure_bank | 3401664 | 2 | 0.00005879 | 0.11406182 | 5.297507862e-05 |
| initial | original | pure_cnn | 3401664 | 8 | 0.00023518 | 0.31073028 | 0.0001580428103 |
| initial | original | pure_transformer | 3401664 | 4 | 0.00011759 | 0.07707992 | 3.366682821e-05 |
| initial | original | pure_mamba | 3401664 | 11 | 0.00032337 | 0.31017173 | 0.0001545936208 |
| initial | registered_style | baseline_only | 3401664 | 7 | 0.00020578 | 1.43159348 | 0.000594234097 |
| initial | registered_style | fused | 3401664 | 16 | 0.00047036 | 0.91164207 | 0.0004269603316 |
| initial | registered_style | cnn | 3401664 | 20 | 0.00058795 | 1.10998617 | 0.0005482455321 |
| initial | registered_style | transformer | 3401664 | 19 | 0.00055855 | 0.89979492 | 0.0004198316387 |
| initial | registered_style | mamba | 3401664 | 27 | 0.00079373 | 1.20129443 | 0.0005884542928 |
| initial | registered_style | cnn_RGB_residual | 3401664 | 25046 | 0.73628671 | 7.16205363 | 0.009765561836 |
| initial | registered_style | cnn_NI_residual | 3401664 | 48141 | 1.41521914 | 11.46444799 | 0.01683066371 |
| initial | registered_style | cnn_TI_residual | 3401664 | 16948 | 0.49822675 | 6.18911803 | 0.00784503258 |
| initial | registered_style | transformer_RGB_residual | 3401664 | 28270 | 0.83106386 | 7.19277389 | 0.01026842195 |
| initial | registered_style | transformer_NI_residual | 3401664 | 48737 | 1.43273998 | 10.80015545 | 0.0163196631 |
| initial | registered_style | transformer_TI_residual | 3401664 | 13516 | 0.39733495 | 4.89019492 | 0.006127101028 |
| initial | registered_style | mamba_RGB_residual | 3401664 | 27715 | 0.81474831 | 7.63764440 | 0.01059109281 |
| initial | registered_style | mamba_NI_residual | 3401664 | 53602 | 1.57575822 | 12.25950006 | 0.01830359441 |
| initial | registered_style | mamba_TI_residual | 3401664 | 16700 | 0.49093620 | 5.98639372 | 0.007612291685 |
| initial | registered_style | pure_bank | 3401664 | 292 | 0.00858403 | 1.51505263 | 0.001017566429 |
| initial | registered_style | pure_cnn | 3401664 | 608 | 0.01787361 | 1.81008471 | 0.001308210695 |
| initial | registered_style | pure_transformer | 3401664 | 425 | 0.01249389 | 1.90139297 | 0.001328613156 |
| initial | registered_style | pure_mamba | 3401664 | 572 | 0.01681530 | 1.98655717 | 0.001419194506 |
| control_final | original | baseline_only | 3401664 | 7 | 0.00020578 | 1.43159348 | 0.000594234097 |
| control_final | original | fused | 3401664 | 0 | 0.00000000 | 0.08325337 | 2.997163903e-05 |
| control_final | original | cnn | 3401664 | 0 | 0.00000000 | 0.12576198 | 4.413141303e-05 |
| control_final | original | transformer | 3401664 | 0 | 0.00000000 | 0.05902993 | 2.085157528e-05 |
| control_final | original | mamba | 3401664 | 2 | 0.00005879 | 0.16900552 | 6.345330466e-05 |
| control_final | original | cnn_RGB_residual | 3401664 | 3491 | 0.10262624 | 2.63641559 | 0.002614409881 |
| control_final | original | cnn_NI_residual | 3401664 | 6190 | 0.18196977 | 4.61021430 | 0.004691112055 |
| control_final | original | cnn_TI_residual | 3401664 | 5061 | 0.14878013 | 3.14407890 | 0.003368735418 |
| control_final | original | transformer_RGB_residual | 3401664 | 3068 | 0.09019115 | 1.78012878 | 0.001898508474 |
| control_final | original | transformer_NI_residual | 3401664 | 6975 | 0.20504671 | 3.48673473 | 0.003898563692 |
| control_final | original | transformer_TI_residual | 3401664 | 3928 | 0.11547290 | 2.18939907 | 0.00238651355 |
| control_final | original | mamba_RGB_residual | 3401664 | 4924 | 0.14475269 | 2.90948783 | 0.003116840673 |
| control_final | original | mamba_NI_residual | 3401664 | 10134 | 0.29791302 | 5.31948482 | 0.005983331355 |
| control_final | original | mamba_TI_residual | 3401664 | 6238 | 0.18338084 | 3.32260917 | 0.003711364272 |
| control_final | original | pure_bank | 3401664 | 0 | 0.00000000 | 0.01684470 | 6.661014634e-06 |
| control_final | original | pure_cnn | 3401664 | 0 | 0.00000000 | 0.04683002 | 1.921024534e-05 |
| control_final | original | pure_transformer | 3401664 | 0 | 0.00000000 | 0.01170016 | 3.944466032e-06 |
| control_final | original | pure_mamba | 3401664 | 0 | 0.00000000 | 0.06640868 | 2.786682998e-05 |
| control_final | registered_style | baseline_only | 3401664 | 7 | 0.00020578 | 1.43159348 | 0.000594234097 |
| control_final | registered_style | fused | 3401664 | 1 | 0.00002940 | 0.58944681 | 0.0002608092319 |
| control_final | registered_style | cnn | 3401664 | 1 | 0.00002940 | 0.63295493 | 0.0002739092074 |
| control_final | registered_style | transformer | 3401664 | 5 | 0.00014699 | 0.70600741 | 0.0003412284762 |
| control_final | registered_style | mamba | 3401664 | 8 | 0.00023518 | 0.76756552 | 0.0003465535179 |
| control_final | registered_style | cnn_RGB_residual | 3401664 | 18208 | 0.53526745 | 6.13567360 | 0.007855277714 |
| control_final | registered_style | cnn_NI_residual | 3401664 | 35080 | 1.03126000 | 9.75743048 | 0.01340985306 |
| control_final | registered_style | cnn_TI_residual | 3401664 | 9603 | 0.28230301 | 4.50744106 | 0.005239576694 |
| control_final | registered_style | transformer_RGB_residual | 3401664 | 29122 | 0.85611042 | 6.47747691 | 0.009739934055 |
| control_final | registered_style | transformer_NI_residual | 3401664 | 49507 | 1.45537596 | 9.68508354 | 0.0154390308 |
| control_final | registered_style | transformer_TI_residual | 3401664 | 8911 | 0.26196003 | 3.60379508 | 0.004356294347 |
| control_final | registered_style | mamba_RGB_residual | 3401664 | 22527 | 0.66223472 | 6.33578155 | 0.008642285545 |
| control_final | registered_style | mamba_NI_residual | 3401664 | 45730 | 1.34434206 | 10.71137537 | 0.0157865355 |
| control_final | registered_style | mamba_TI_residual | 3401664 | 11877 | 0.34915265 | 4.83237028 | 0.005845969178 |
| control_final | registered_style | pure_bank | 3401664 | 218 | 0.00640863 | 1.08346974 | 0.0007166255173 |
| control_final | registered_style | pure_cnn | 3401664 | 243 | 0.00714356 | 1.15752173 | 0.0007512695946 |
| control_final | registered_style | pure_transformer | 3401664 | 616 | 0.01810878 | 1.52634123 | 0.001151288731 |
| control_final | registered_style | pure_mamba | 3401664 | 341 | 0.01002451 | 1.39472917 | 0.0009612074519 |
| style_final | original | baseline_only | 3401664 | 7 | 0.00020578 | 1.43159348 | 0.000594234097 |
| style_final | original | fused | 3401664 | 0 | 0.00000000 | 0.09589424 | 3.701939373e-05 |
| style_final | original | cnn | 3401664 | 0 | 0.00000000 | 0.14513485 | 5.481868695e-05 |
| style_final | original | transformer | 3401664 | 0 | 0.00000000 | 0.06579133 | 2.580088163e-05 |
| style_final | original | mamba | 3401664 | 1 | 0.00002940 | 0.18696732 | 7.408767302e-05 |
| style_final | original | cnn_RGB_residual | 3401664 | 3922 | 0.11529651 | 2.94999741 | 0.003003636826 |
| style_final | original | cnn_NI_residual | 3401664 | 7559 | 0.22221477 | 4.97824006 | 0.005248763401 |
| style_final | original | cnn_TI_residual | 3401664 | 5630 | 0.16550723 | 3.03804256 | 0.003375175142 |
| style_final | original | transformer_RGB_residual | 3401664 | 3902 | 0.11470857 | 2.02145185 | 0.002248777875 |
| style_final | original | transformer_NI_residual | 3401664 | 8327 | 0.24479196 | 3.75821951 | 0.004340294082 |
| style_final | original | transformer_TI_residual | 3401664 | 3435 | 0.10097999 | 1.96227493 | 0.002158189681 |
| style_final | original | mamba_RGB_residual | 3401664 | 5346 | 0.15715838 | 3.01896366 | 0.003308255706 |
| style_final | original | mamba_NI_residual | 3401664 | 9114 | 0.26792770 | 5.19583945 | 0.005722411308 |
| style_final | original | mamba_TI_residual | 3401664 | 6026 | 0.17714860 | 2.91210419 | 0.003352640142 |
| style_final | original | pure_bank | 3401664 | 0 | 0.00000000 | 0.02292995 | 9.770512911e-06 |
| style_final | original | pure_cnn | 3401664 | 0 | 0.00000000 | 0.06002944 | 2.562604531e-05 |
| style_final | original | pure_transformer | 3401664 | 0 | 0.00000000 | 0.01707988 | 6.209930143e-06 |
| style_final | original | pure_mamba | 3401664 | 0 | 0.00000000 | 0.08125435 | 3.710342648e-05 |
| style_final | registered_style | baseline_only | 3401664 | 7 | 0.00020578 | 1.43159348 | 0.000594234097 |
| style_final | registered_style | fused | 3401664 | 0 | 0.00000000 | 0.15501237 | 5.438375066e-05 |
| style_final | registered_style | cnn | 3401664 | 0 | 0.00000000 | 0.20475273 | 7.330935879e-05 |
| style_final | registered_style | transformer | 3401664 | 0 | 0.00000000 | 0.12852533 | 4.666749757e-05 |
| style_final | registered_style | mamba | 3401664 | 1 | 0.00002940 | 0.28547793 | 0.0001073385214 |
| style_final | registered_style | cnn_RGB_residual | 3401664 | 6980 | 0.20519369 | 4.20673529 | 0.004553922737 |
| style_final | registered_style | cnn_NI_residual | 3401664 | 14379 | 0.42270489 | 6.99745771 | 0.00801394451 |
| style_final | registered_style | cnn_TI_residual | 3401664 | 6513 | 0.19146512 | 3.52656817 | 0.003960821915 |
| style_final | registered_style | transformer_RGB_residual | 3401664 | 8598 | 0.25275865 | 3.47676902 | 0.004178784152 |
| style_final | registered_style | transformer_NI_residual | 3401664 | 16424 | 0.48282252 | 5.89681991 | 0.00738463854 |
| style_final | registered_style | transformer_TI_residual | 3401664 | 4680 | 0.13757973 | 2.40523461 | 0.0027125577 |
| style_final | registered_style | mamba_RGB_residual | 3401664 | 8988 | 0.26422363 | 4.24145359 | 0.004910290753 |
| style_final | registered_style | mamba_NI_residual | 3401664 | 17329 | 0.50942715 | 7.40034877 | 0.008844141258 |
| style_final | registered_style | mamba_TI_residual | 3401664 | 7537 | 0.22156803 | 3.46941967 | 0.004058796363 |
| style_final | registered_style | pure_bank | 3401664 | 0 | 0.00000000 | 0.09042633 | 3.656953974e-05 |
| style_final | registered_style | pure_cnn | 3401664 | 1 | 0.00002940 | 0.15498297 | 6.528857203e-05 |
| style_final | registered_style | pure_transformer | 3401664 | 3 | 0.00008819 | 0.10756500 | 4.809161616e-05 |
| style_final | registered_style | pure_mamba | 3401664 | 3 | 0.00008819 | 0.23870670 | 0.0001069666337 |

| 模型状态 | 输入 | 关系曝光 | V26形式辅助损失 | 同形目标间隔导数范数比（批均值） |
|---|---|---:|---:|---:|
| initial | original | 3401664 | 1.292598006e-05 | 0.009655055779 |
| initial | registered_style | 3401664 | 5.637043899e-05 | 0.01561363451 |
| control_final | original | 3401664 | 5.838193835e-06 | 0.006613960288 |
| control_final | registered_style | 3401664 | 4.200718932e-05 | 0.01268027218 |
| style_final | original | 3401664 | 6.58464191e-06 | 0.006905220383 |
| style_final | registered_style | 3401664 | 1.085060243e-05 | 0.008610897275 |

## cross_camera / 计划active：全部模型和输入

| 模型状态 | 输入 | 表示 | 关系曝光 | 非正关系 | 非正% | 0.3未满足% | 平均hinge |
|---|---|---|---:|---:|---:|---:|---:|
| initial | original | baseline_only | 1698704 | 0 | 0.00000000 | 1.47065057 | 0.0005921282834 |
| initial | original | fused | 1698704 | 0 | 0.00000000 | 0.25566550 | 9.934570293e-05 |
| initial | original | cnn | 1698704 | 0 | 0.00000000 | 0.42791446 | 0.000174467153 |
| initial | original | transformer | 1698704 | 0 | 0.00000000 | 0.16359531 | 6.195136883e-05 |
| initial | original | mamba | 1698704 | 0 | 0.00000000 | 0.44186627 | 0.0001829433187 |
| initial | original | cnn_RGB_residual | 1698704 | 3312 | 0.19497217 | 3.70123341 | 0.004070131992 |
| initial | original | cnn_NI_residual | 1698704 | 6451 | 0.37976010 | 6.59956061 | 0.007551712633 |
| initial | original | cnn_TI_residual | 1698704 | 5205 | 0.30641006 | 4.64913252 | 0.00554796937 |
| initial | original | transformer_RGB_residual | 1698704 | 2208 | 0.12998144 | 2.30976085 | 0.00253249096 |
| initial | original | transformer_NI_residual | 1698704 | 5058 | 0.29775641 | 4.54293391 | 0.005160885933 |
| initial | original | transformer_TI_residual | 1698704 | 2719 | 0.16006320 | 3.01718251 | 0.003330909281 |
| initial | original | mamba_RGB_residual | 1698704 | 3873 | 0.22799734 | 3.96926127 | 0.004452761403 |
| initial | original | mamba_NI_residual | 1698704 | 7856 | 0.46247021 | 6.91179864 | 0.008199084221 |
| initial | original | mamba_TI_residual | 1698704 | 4917 | 0.28945596 | 4.44680180 | 0.005288208204 |
| initial | original | pure_bank | 1698704 | 0 | 0.00000000 | 0.10690503 | 4.514316023e-05 |
| initial | original | pure_cnn | 1698704 | 1 | 0.00005887 | 0.31777167 | 0.0001564494002 |
| initial | original | pure_transformer | 1698704 | 0 | 0.00000000 | 0.06522620 | 2.650187931e-05 |
| initial | original | pure_mamba | 1698704 | 3 | 0.00017661 | 0.30223041 | 0.0001444523688 |
| initial | registered_style | baseline_only | 1698704 | 0 | 0.00000000 | 1.47065057 | 0.0005921282834 |
| initial | registered_style | fused | 1698704 | 12 | 0.00070642 | 1.54765044 | 0.0007343086453 |
| initial | registered_style | cnn | 1698704 | 16 | 0.00094189 | 1.79454455 | 0.0009033121833 |
| initial | registered_style | transformer | 1698704 | 14 | 0.00082416 | 1.61634988 | 0.0007612726928 |
| initial | registered_style | mamba | 1698704 | 24 | 0.00141284 | 1.94330501 | 0.0009690331029 |
| initial | registered_style | cnn_RGB_residual | 1698704 | 21821 | 1.28456753 | 10.86239863 | 0.01572544082 |
| initial | registered_style | cnn_NI_residual | 1698704 | 42136 | 2.48047924 | 16.89788215 | 0.02678636676 |
| initial | registered_style | cnn_TI_residual | 1698704 | 11280 | 0.66403564 | 7.68833181 | 0.0100470009 |
| initial | registered_style | transformer_RGB_residual | 1698704 | 26306 | 1.54859234 | 12.17534073 | 0.01817574784 |
| initial | registered_style | transformer_NI_residual | 1698704 | 44211 | 2.60263118 | 17.42599064 | 0.02793119245 |
| initial | registered_style | transformer_TI_residual | 1698704 | 9670 | 0.56925750 | 6.55741083 | 0.008498877581 |
| initial | registered_style | mamba_RGB_residual | 1698704 | 24083 | 1.41772787 | 11.45979523 | 0.01694653789 |
| initial | registered_style | mamba_NI_residual | 1698704 | 46203 | 2.71989705 | 17.98565259 | 0.02888899524 |
| initial | registered_style | mamba_TI_residual | 1698704 | 11172 | 0.65767785 | 7.56076397 | 0.009846653865 |
| initial | registered_style | pure_bank | 1698704 | 290 | 0.01707184 | 2.91239674 | 0.001976742587 |
| initial | registered_style | pure_cnn | 1698704 | 601 | 0.03537991 | 3.32023707 | 0.002459666846 |
| initial | registered_style | pure_transformer | 1698704 | 421 | 0.02478360 | 3.71842299 | 0.002619638944 |
| initial | registered_style | pure_mamba | 1698704 | 564 | 0.03320178 | 3.65920137 | 0.00267682252 |
| control_final | original | baseline_only | 1698704 | 0 | 0.00000000 | 1.47065057 | 0.0005921282834 |
| control_final | original | fused | 1698704 | 0 | 0.00000000 | 0.07040662 | 2.274208609e-05 |
| control_final | original | cnn | 1698704 | 0 | 0.00000000 | 0.10672842 | 3.543640155e-05 |
| control_final | original | transformer | 1698704 | 0 | 0.00000000 | 0.05492422 | 1.71890677e-05 |
| control_final | original | mamba | 1698704 | 0 | 0.00000000 | 0.14140192 | 4.852637852e-05 |
| control_final | original | cnn_RGB_residual | 1698704 | 1737 | 0.10225442 | 2.66397206 | 0.002638654397 |
| control_final | original | cnn_NI_residual | 1698704 | 3054 | 0.17978412 | 4.75856889 | 0.004828503429 |
| control_final | original | cnn_TI_residual | 1698704 | 2266 | 0.13339581 | 3.09847978 | 0.003296332091 |
| control_final | original | transformer_RGB_residual | 1698704 | 1608 | 0.09466040 | 1.80608275 | 0.001957221901 |
| control_final | original | transformer_NI_residual | 1698704 | 3626 | 0.21345685 | 3.65372661 | 0.004092491586 |
| control_final | original | transformer_TI_residual | 1698704 | 1627 | 0.09577890 | 2.07381627 | 0.002202747098 |
| control_final | original | mamba_RGB_residual | 1698704 | 2631 | 0.15488278 | 2.89026222 | 0.003155583021 |
| control_final | original | mamba_NI_residual | 1698704 | 5070 | 0.29846283 | 5.37397922 | 0.00600472297 |
| control_final | original | mamba_TI_residual | 1698704 | 2675 | 0.15747299 | 3.23511336 | 0.003498080664 |
| control_final | original | pure_bank | 1698704 | 0 | 0.00000000 | 0.01236237 | 3.835026825e-06 |
| control_final | original | pure_cnn | 1698704 | 0 | 0.00000000 | 0.04085468 | 1.74590238e-05 |
| control_final | original | pure_transformer | 1698704 | 0 | 0.00000000 | 0.01094952 | 3.271298529e-06 |
| control_final | original | pure_mamba | 1698704 | 0 | 0.00000000 | 0.04786002 | 1.743107933e-05 |
| control_final | registered_style | baseline_only | 1698704 | 0 | 0.00000000 | 1.47065057 | 0.0005921282834 |
| control_final | registered_style | fused | 1698704 | 1 | 0.00005887 | 1.08406173 | 0.0004849956215 |
| control_final | registered_style | cnn | 1698704 | 1 | 0.00005887 | 1.12238507 | 0.0004955676847 |
| control_final | registered_style | transformer | 1698704 | 5 | 0.00029434 | 1.35050015 | 0.0006587455544 |
| control_final | registered_style | mamba | 1698704 | 6 | 0.00035321 | 1.34002157 | 0.0006154360954 |
| control_final | registered_style | cnn_RGB_residual | 1698704 | 16454 | 0.96862078 | 9.67125526 | 0.01313352074 |
| control_final | registered_style | cnn_NI_residual | 1698704 | 31944 | 1.88049242 | 15.06589730 | 0.02228782971 |
| control_final | registered_style | cnn_TI_residual | 1698704 | 6808 | 0.40077612 | 5.82861994 | 0.007042701922 |
| control_final | registered_style | transformer_RGB_residual | 1698704 | 27662 | 1.62841790 | 11.21254792 | 0.01765971928 |
| control_final | registered_style | transformer_NI_residual | 1698704 | 46158 | 2.71724797 | 16.06595381 | 0.02720233975 |
| control_final | registered_style | transformer_TI_residual | 1698704 | 6610 | 0.38912018 | 4.90615198 | 0.006147243858 |
| control_final | registered_style | mamba_RGB_residual | 1698704 | 20234 | 1.19114337 | 9.75143403 | 0.01422031643 |
| control_final | registered_style | mamba_NI_residual | 1698704 | 40666 | 2.39394268 | 16.17126939 | 0.02563569258 |
| control_final | registered_style | mamba_TI_residual | 1698704 | 8314 | 0.48943194 | 6.25841818 | 0.0077726386 |
| control_final | registered_style | pure_bank | 1698704 | 218 | 0.01283331 | 2.14828481 | 0.001425542805 |
| control_final | registered_style | pure_cnn | 1698704 | 243 | 0.01430502 | 2.26502086 | 0.001483411853 |
| control_final | registered_style | pure_transformer | 1698704 | 616 | 0.03626294 | 3.04402650 | 0.002300834429 |
| control_final | registered_style | pure_mamba | 1698704 | 341 | 0.02007413 | 2.70782903 | 0.001886450752 |
| style_final | original | baseline_only | 1698704 | 0 | 0.00000000 | 1.47065057 | 0.0005921282834 |
| style_final | original | fused | 1698704 | 0 | 0.00000000 | 0.08494711 | 3.179531793e-05 |
| style_final | original | cnn | 1698704 | 0 | 0.00000000 | 0.13310147 | 4.999570772e-05 |
| style_final | original | transformer | 1698704 | 0 | 0.00000000 | 0.05869180 | 2.054672751e-05 |
| style_final | original | mamba | 1698704 | 0 | 0.00000000 | 0.16818704 | 6.478200477e-05 |
| style_final | original | cnn_RGB_residual | 1698704 | 1772 | 0.10431482 | 2.89444188 | 0.002920286532 |
| style_final | original | cnn_NI_residual | 1698704 | 3638 | 0.21416327 | 5.04302103 | 0.005284534796 |
| style_final | original | cnn_TI_residual | 1698704 | 2668 | 0.15706091 | 2.94913063 | 0.003275196758 |
| style_final | original | transformer_RGB_residual | 1698704 | 1943 | 0.11438132 | 2.04508849 | 0.002282745325 |
| style_final | original | transformer_NI_residual | 1698704 | 4362 | 0.25678400 | 3.77446571 | 0.004404128294 |
| style_final | original | transformer_TI_residual | 1698704 | 1457 | 0.08577127 | 1.81061562 | 0.001943614364 |
| style_final | original | mamba_RGB_residual | 1698704 | 2572 | 0.15140955 | 2.95796089 | 0.003248933276 |
| style_final | original | mamba_NI_residual | 1698704 | 4457 | 0.26237649 | 5.23681583 | 0.005742214407 |
| style_final | original | mamba_TI_residual | 1698704 | 2757 | 0.16230020 | 2.83074626 | 0.003210558485 |
| style_final | original | pure_bank | 1698704 | 0 | 0.00000000 | 0.01942657 | 7.890973645e-06 |
| style_final | original | pure_cnn | 1698704 | 0 | 0.00000000 | 0.05880954 | 2.677468543e-05 |
| style_final | original | pure_transformer | 1698704 | 0 | 0.00000000 | 0.01183255 | 3.12762764e-06 |
| style_final | original | pure_mamba | 1698704 | 0 | 0.00000000 | 0.07087756 | 3.105416475e-05 |
| style_final | registered_style | baseline_only | 1698704 | 0 | 0.00000000 | 1.47065057 | 0.0005921282834 |
| style_final | registered_style | fused | 1698704 | 0 | 0.00000000 | 0.20333148 | 6.656753714e-05 |
| style_final | registered_style | cnn | 1698704 | 0 | 0.00000000 | 0.25248660 | 8.702337866e-05 |
| style_final | registered_style | transformer | 1698704 | 0 | 0.00000000 | 0.18431699 | 6.233223943e-05 |
| style_final | registered_style | mamba | 1698704 | 0 | 0.00000000 | 0.36545508 | 0.0001313670095 |
| style_final | registered_style | cnn_RGB_residual | 1698704 | 4830 | 0.28433441 | 5.41106632 | 0.006024742502 |
| style_final | registered_style | cnn_NI_residual | 1698704 | 10458 | 0.61564581 | 9.08651537 | 0.01082182501 |
| style_final | registered_style | cnn_TI_residual | 1698704 | 3551 | 0.20904172 | 3.92740583 | 0.004447957607 |
| style_final | registered_style | transformer_RGB_residual | 1698704 | 6639 | 0.39082736 | 4.95936902 | 0.006147593393 |
| style_final | registered_style | transformer_NI_residual | 1698704 | 12459 | 0.73344149 | 8.05702465 | 0.01050044463 |
| style_final | registered_style | transformer_TI_residual | 1698704 | 2702 | 0.15906244 | 2.69764479 | 0.003053739337 |
| style_final | registered_style | mamba_RGB_residual | 1698704 | 6214 | 0.36580829 | 5.40600364 | 0.006457017172 |
| style_final | registered_style | mamba_NI_residual | 1698704 | 12672 | 0.74598047 | 9.65135774 | 0.01199349561 |
| style_final | registered_style | mamba_TI_residual | 1698704 | 4268 | 0.25125036 | 3.94677354 | 0.004624640158 |
| style_final | registered_style | pure_bank | 1698704 | 0 | 0.00000000 | 0.15458844 | 6.155617065e-05 |
| style_final | registered_style | pure_cnn | 1698704 | 1 | 0.00005887 | 0.24895450 | 0.0001061991109 |
| style_final | registered_style | pure_transformer | 1698704 | 3 | 0.00017661 | 0.19302951 | 8.699593171e-05 |
| style_final | registered_style | pure_mamba | 1698704 | 3 | 0.00017661 | 0.38617676 | 0.0001709556172 |

| 模型状态 | 输入 | 关系曝光 | V26形式辅助损失 | 同形目标间隔导数范数比（批均值） |
|---|---|---:|---:|---:|
| initial | original | 1698704 | 1.225113228e-05 | 0.009474164459 |
| initial | registered_style | 1698704 | 9.92488976e-05 | 0.02169689006 |
| control_final | original | 1698704 | 5.463393696e-06 | 0.006494675759 |
| control_final | registered_style | 1698704 | 7.789200389e-05 | 0.01893839246 |
| style_final | original | 1698704 | 6.183296181e-06 | 0.006786275515 |
| style_final | registered_style | 1698704 | 1.472590532e-05 | 0.01028509991 |

## cross_camera / 计划inactive：全部模型和输入

| 模型状态 | 输入 | 表示 | 关系曝光 | 非正关系 | 非正% | 0.3未满足% | 平均hinge |
|---|---|---|---:|---:|---:|---:|---:|
| initial | original | baseline_only | 1702960 | 7 | 0.00041105 | 1.39263400 | 0.0005963346478 |
| initial | original | fused | 1702960 | 4 | 0.00023489 | 0.27722319 | 0.000120380136 |
| initial | original | cnn | 1702960 | 4 | 0.00023489 | 0.42713863 | 0.0001940662556 |
| initial | original | transformer | 1702960 | 5 | 0.00029361 | 0.18503077 | 7.924390659e-05 |
| initial | original | mamba | 1702960 | 3 | 0.00017616 | 0.46113825 | 0.0002088266168 |
| initial | original | cnn_RGB_residual | 1702960 | 3225 | 0.18937615 | 3.47095645 | 0.003820577647 |
| initial | original | cnn_NI_residual | 1702960 | 6005 | 0.35262132 | 6.04459294 | 0.006899841737 |
| initial | original | cnn_TI_residual | 1702960 | 5668 | 0.33283225 | 4.69365105 | 0.005648567374 |
| initial | original | transformer_RGB_residual | 1702960 | 1964 | 0.11532860 | 2.22265937 | 0.002380857875 |
| initial | original | transformer_NI_residual | 1702960 | 4526 | 0.26577254 | 4.19087941 | 0.004737153017 |
| initial | original | transformer_TI_residual | 1702960 | 3846 | 0.22584206 | 3.22714568 | 0.003761251967 |
| initial | original | mamba_RGB_residual | 1702960 | 3632 | 0.21327571 | 3.82504580 | 0.004251531106 |
| initial | original | mamba_NI_residual | 1702960 | 7399 | 0.43447879 | 6.54765820 | 0.007744648372 |
| initial | original | mamba_TI_residual | 1702960 | 5528 | 0.32461127 | 4.41595810 | 0.005383513573 |
| initial | original | pure_bank | 1702960 | 2 | 0.00011744 | 0.12120073 | 6.078742365e-05 |
| initial | original | pure_cnn | 1702960 | 7 | 0.00041105 | 0.30370649 | 0.0001596322383 |
| initial | original | pure_transformer | 1702960 | 4 | 0.00023489 | 0.08890403 | 4.081387063e-05 |
| initial | original | pure_mamba | 1702960 | 8 | 0.00046977 | 0.31809320 | 0.0001647095281 |
| initial | registered_style | baseline_only | 1702960 | 7 | 0.00041105 | 1.39263400 | 0.0005963346478 |
| initial | registered_style | fused | 1702960 | 4 | 0.00023489 | 0.27722319 | 0.000120380136 |
| initial | registered_style | cnn | 1702960 | 4 | 0.00023489 | 0.42713863 | 0.0001940662556 |
| initial | registered_style | transformer | 1702960 | 5 | 0.00029361 | 0.18503077 | 7.924390659e-05 |
| initial | registered_style | mamba | 1702960 | 3 | 0.00017616 | 0.46113825 | 0.0002088266168 |
| initial | registered_style | cnn_RGB_residual | 1702960 | 3225 | 0.18937615 | 3.47095645 | 0.003820577647 |
| initial | registered_style | cnn_NI_residual | 1702960 | 6005 | 0.35262132 | 6.04459294 | 0.006899841737 |
| initial | registered_style | cnn_TI_residual | 1702960 | 5668 | 0.33283225 | 4.69365105 | 0.005648567374 |
| initial | registered_style | transformer_RGB_residual | 1702960 | 1964 | 0.11532860 | 2.22265937 | 0.002380857875 |
| initial | registered_style | transformer_NI_residual | 1702960 | 4526 | 0.26577254 | 4.19087941 | 0.004737153017 |
| initial | registered_style | transformer_TI_residual | 1702960 | 3846 | 0.22584206 | 3.22714568 | 0.003761251967 |
| initial | registered_style | mamba_RGB_residual | 1702960 | 3632 | 0.21327571 | 3.82504580 | 0.004251531106 |
| initial | registered_style | mamba_NI_residual | 1702960 | 7399 | 0.43447879 | 6.54765820 | 0.007744648372 |
| initial | registered_style | mamba_TI_residual | 1702960 | 5528 | 0.32461127 | 4.41595810 | 0.005383513573 |
| initial | registered_style | pure_bank | 1702960 | 2 | 0.00011744 | 0.12120073 | 6.078742365e-05 |
| initial | registered_style | pure_cnn | 1702960 | 7 | 0.00041105 | 0.30370649 | 0.0001596322383 |
| initial | registered_style | pure_transformer | 1702960 | 4 | 0.00023489 | 0.08890403 | 4.081387063e-05 |
| initial | registered_style | pure_mamba | 1702960 | 8 | 0.00046977 | 0.31809320 | 0.0001647095281 |
| control_final | original | baseline_only | 1702960 | 7 | 0.00041105 | 1.39263400 | 0.0005963346478 |
| control_final | original | fused | 1702960 | 0 | 0.00000000 | 0.09606802 | 3.718312403e-05 |
| control_final | original | cnn | 1702960 | 0 | 0.00000000 | 0.14474797 | 5.280469412e-05 |
| control_final | original | transformer | 1702960 | 0 | 0.00000000 | 0.06312538 | 2.45049296e-05 |
| control_final | original | mamba | 1702960 | 2 | 0.00011744 | 0.19654014 | 7.834292575e-05 |
| control_final | original | cnn_RGB_residual | 1702960 | 1754 | 0.10299713 | 2.60892798 | 0.002590225956 |
| control_final | original | cnn_NI_residual | 1702960 | 3136 | 0.18414995 | 4.46223047 | 0.004554064046 |
| control_final | original | cnn_TI_residual | 1702960 | 2795 | 0.16412599 | 3.18956405 | 0.003440957797 |
| control_final | original | transformer_RGB_residual | 1702960 | 1460 | 0.08573308 | 1.75423968 | 0.001839941781 |
| control_final | original | transformer_NI_residual | 1702960 | 3349 | 0.19665758 | 3.32016019 | 0.003705120458 |
| control_final | original | transformer_TI_residual | 1702960 | 2301 | 0.13511768 | 2.30469301 | 0.002569820736 |
| control_final | original | mamba_RGB_residual | 1702960 | 2293 | 0.13464791 | 2.92866538 | 0.003078195149 |
| control_final | original | mamba_NI_residual | 1702960 | 5064 | 0.29736459 | 5.26512660 | 0.005961993202 |
| control_final | original | mamba_TI_residual | 1702960 | 3563 | 0.20922394 | 3.40988632 | 0.003924114846 |
| control_final | original | pure_bank | 1702960 | 0 | 0.00000000 | 0.02131583 | 9.479939797e-06 |
| control_final | original | pure_cnn | 1702960 | 0 | 0.00000000 | 0.05279044 | 2.095709027e-05 |
| control_final | original | pure_transformer | 1702960 | 0 | 0.00000000 | 0.01244891 | 4.61595117e-06 |
| control_final | original | pure_mamba | 1702960 | 0 | 0.00000000 | 0.08491098 | 3.827649983e-05 |
| control_final | registered_style | baseline_only | 1702960 | 7 | 0.00041105 | 1.39263400 | 0.0005963346478 |
| control_final | registered_style | fused | 1702960 | 0 | 0.00000000 | 0.09606802 | 3.718312403e-05 |
| control_final | registered_style | cnn | 1702960 | 0 | 0.00000000 | 0.14474797 | 5.280469412e-05 |
| control_final | registered_style | transformer | 1702960 | 0 | 0.00000000 | 0.06312538 | 2.45049296e-05 |
| control_final | registered_style | mamba | 1702960 | 2 | 0.00011744 | 0.19654014 | 7.834292575e-05 |
| control_final | registered_style | cnn_RGB_residual | 1702960 | 1754 | 0.10299713 | 2.60892798 | 0.002590225956 |
| control_final | registered_style | cnn_NI_residual | 1702960 | 3136 | 0.18414995 | 4.46223047 | 0.004554064046 |
| control_final | registered_style | cnn_TI_residual | 1702960 | 2795 | 0.16412599 | 3.18956405 | 0.003440957797 |
| control_final | registered_style | transformer_RGB_residual | 1702960 | 1460 | 0.08573308 | 1.75423968 | 0.001839941781 |
| control_final | registered_style | transformer_NI_residual | 1702960 | 3349 | 0.19665758 | 3.32016019 | 0.003705120458 |
| control_final | registered_style | transformer_TI_residual | 1702960 | 2301 | 0.13511768 | 2.30469301 | 0.002569820736 |
| control_final | registered_style | mamba_RGB_residual | 1702960 | 2293 | 0.13464791 | 2.92866538 | 0.003078195149 |
| control_final | registered_style | mamba_NI_residual | 1702960 | 5064 | 0.29736459 | 5.26512660 | 0.005961993202 |
| control_final | registered_style | mamba_TI_residual | 1702960 | 3563 | 0.20922394 | 3.40988632 | 0.003924114846 |
| control_final | registered_style | pure_bank | 1702960 | 0 | 0.00000000 | 0.02131583 | 9.479939797e-06 |
| control_final | registered_style | pure_cnn | 1702960 | 0 | 0.00000000 | 0.05279044 | 2.095709027e-05 |
| control_final | registered_style | pure_transformer | 1702960 | 0 | 0.00000000 | 0.01244891 | 4.61595117e-06 |
| control_final | registered_style | pure_mamba | 1702960 | 0 | 0.00000000 | 0.08491098 | 3.827649983e-05 |
| style_final | original | baseline_only | 1702960 | 7 | 0.00041105 | 1.39263400 | 0.0005963346478 |
| style_final | original | fused | 1702960 | 0 | 0.00000000 | 0.10681402 | 4.223041363e-05 |
| style_final | original | cnn | 1702960 | 0 | 0.00000000 | 0.15713816 | 5.96296127e-05 |
| style_final | original | transformer | 1702960 | 0 | 0.00000000 | 0.07287312 | 3.104190469e-05 |
| style_final | original | mamba | 1702960 | 1 | 0.00005872 | 0.20570066 | 8.337008475e-05 |
| style_final | original | cnn_RGB_residual | 1702960 | 2150 | 0.12625076 | 3.00541410 | 0.003086778813 |
| style_final | original | cnn_NI_residual | 1702960 | 3921 | 0.23024616 | 4.91362099 | 0.005213081406 |
| style_final | original | cnn_TI_residual | 1702960 | 2962 | 0.17393245 | 3.12673228 | 0.003474903661 |
| style_final | original | transformer_RGB_residual | 1702960 | 1959 | 0.11503500 | 1.99787429 | 0.002214895316 |
| style_final | original | transformer_NI_residual | 1702960 | 3965 | 0.23282990 | 3.74201391 | 0.004276619402 |
| style_final | original | transformer_TI_residual | 1702960 | 1978 | 0.11615070 | 2.11355522 | 0.002372228736 |
| style_final | original | mamba_RGB_residual | 1702960 | 2774 | 0.16289285 | 3.07981397 | 0.003367429879 |
| style_final | original | mamba_NI_residual | 1702960 | 4657 | 0.27346503 | 5.15496547 | 0.005702657701 |
| style_final | original | mamba_TI_residual | 1702960 | 3269 | 0.19195988 | 2.99325880 | 0.003494366712 |
| style_final | original | pure_bank | 1702960 | 0 | 0.00000000 | 0.02642458 | 1.164535487e-05 |
| style_final | original | pure_cnn | 1702960 | 0 | 0.00000000 | 0.06124630 | 2.448027585e-05 |
| style_final | original | pure_transformer | 1702960 | 0 | 0.00000000 | 0.02231409 | 9.284529425e-06 |
| style_final | original | pure_mamba | 1702960 | 0 | 0.00000000 | 0.09160521 | 4.313757003e-05 |
| style_final | registered_style | baseline_only | 1702960 | 7 | 0.00041105 | 1.39263400 | 0.0005963346478 |
| style_final | registered_style | fused | 1702960 | 0 | 0.00000000 | 0.10681402 | 4.223041363e-05 |
| style_final | registered_style | cnn | 1702960 | 0 | 0.00000000 | 0.15713816 | 5.96296127e-05 |
| style_final | registered_style | transformer | 1702960 | 0 | 0.00000000 | 0.07287312 | 3.104190469e-05 |
| style_final | registered_style | mamba | 1702960 | 1 | 0.00005872 | 0.20570066 | 8.337008475e-05 |
| style_final | registered_style | cnn_RGB_residual | 1702960 | 2150 | 0.12625076 | 3.00541410 | 0.003086778813 |
| style_final | registered_style | cnn_NI_residual | 1702960 | 3921 | 0.23024616 | 4.91362099 | 0.005213081406 |
| style_final | registered_style | cnn_TI_residual | 1702960 | 2962 | 0.17393245 | 3.12673228 | 0.003474903661 |
| style_final | registered_style | transformer_RGB_residual | 1702960 | 1959 | 0.11503500 | 1.99787429 | 0.002214895316 |
| style_final | registered_style | transformer_NI_residual | 1702960 | 3965 | 0.23282990 | 3.74201391 | 0.004276619402 |
| style_final | registered_style | transformer_TI_residual | 1702960 | 1978 | 0.11615070 | 2.11355522 | 0.002372228736 |
| style_final | registered_style | mamba_RGB_residual | 1702960 | 2774 | 0.16289285 | 3.07981397 | 0.003367429879 |
| style_final | registered_style | mamba_NI_residual | 1702960 | 4657 | 0.27346503 | 5.15496547 | 0.005702657701 |
| style_final | registered_style | mamba_TI_residual | 1702960 | 3269 | 0.19195988 | 2.99325880 | 0.003494366712 |
| style_final | registered_style | pure_bank | 1702960 | 0 | 0.00000000 | 0.02642458 | 1.164535487e-05 |
| style_final | registered_style | pure_cnn | 1702960 | 0 | 0.00000000 | 0.06124630 | 2.448027585e-05 |
| style_final | registered_style | pure_transformer | 1702960 | 0 | 0.00000000 | 0.02231409 | 9.284529425e-06 |
| style_final | registered_style | pure_mamba | 1702960 | 0 | 0.00000000 | 0.09160521 | 4.313757003e-05 |

| 模型状态 | 输入 | 关系曝光 | V26形式辅助损失 | 同形目标间隔导数范数比（批均值） |
|---|---|---:|---:|---:|
| initial | original | 1702960 | 1.359914128e-05 | 0.009827123132 |
| initial | registered_style | 1702960 | 1.359914128e-05 | 0.009827123132 |
| control_final | original | 1702960 | 6.212057283e-06 | 0.00672742606 |
| control_final | registered_style | 1702960 | 6.212057283e-06 | 0.00672742606 |
| style_final | original | 1702960 | 6.984984605e-06 | 0.007018363063 |
| style_final | registered_style | 1702960 | 6.984984605e-06 | 0.007018363063 |

## 完整输出和解释边界

全部72个原始条件、各折及三折汇总、active/inactive及其完整合计均保存，CSV包含零计数；没有筛选有利batch或身份。
all_source_identities.csv保留72×94=6768行，包括没有该协议正例的来源身份；其关系分母0，百分比为空而不是伪造0%性能。
槽位支持表保留全部16格：bit8/4/2/1依次表示Signal、纯银行、同角色纯残差、fused间隔非正。计数单位为槽位-关系曝光，跨槽位可能重复。
间隔导数比只比较责任加权与未加权同形目标，未经过encoder Jacobian；不能称为模型参数梯度、总损失梯度或已证明的梯度冲突。
固定eval状态与原训练逐步变化的参数/BN状态不同。原V27只保存前8批像素摘要，不能声称逐像素复现全部优化轨迹。
本诊断只有批内候选，不能证明完整来源图库的实例难例覆盖、缓存慢漂移、未知身份泛化或官方测试提升。
报告只整理预先固定的完整统计，不增加责任loss、记忆库、参数更新或调参晋级。

源摘要SHA256：1b4ef5ec2c8b9a07202287d2bbe21da0a19715d27784d385831dadaca51315ff；完整数组核验SHA256：7e90cc716e2cfe26b6aebc0d1107172133d9c9e2d87cebec9ae57774ae48e11f。
