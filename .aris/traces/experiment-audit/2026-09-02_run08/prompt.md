请对已完成的 V15 Q1 做只读独立实验完整性审计，不改文件、不跑实验。

目标证据：`evidence/trifusion_v15_q1_seed42_71152d3.json`、同名
`console.log`；启动失败的 optimizer0 日志
`evidence/trifusion_v15_q1_seed42_4cdc1ec_optimizer0.console.log`；代码 commit
`71152d3848c05177da0af30b0b921c6a3aa9942a`；M0 receipt
`evidence/trifusion_v15_m0_seed42_1f2de44.json`。

核查 A-F：GT provenance、normalization、JSON/console/hash 数值一致性、
executed/dead path、fold isolation/same-fold comparator/访问边界、eval 类型；
特别确认 `status=PASS` 只是完成而 `gate.passed=false`、
`next_phase_authorized=false`，D1 未执行，dev0/official0。

给 Overall PASS/WARN/FAIL、integrity_status、scientific verdict、D1
authorization、claim boundary。
