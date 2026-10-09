# 冻结文本包必要性对照

| 数据集 | 文本包 | best轮 | mAP | R1 | own-global mAP | 末轮mAP | 正式更新 |
|---|---|---:|---:|---:|---:|---:|---:|

主要配对过线 0/3，可比较配对 0/3；正式端完成 0/6。

Conditional pretrained versus one fixed random text-package contrast. Same original raw tasks/1536 deployment, new592000 trainable parameters, frozen38M text body/template. No independent image identity information, word/layer-specific attribution, novel prompt claim, full-pipeline seeds, unconsumed test or SOTA. Missing endpoints have no substituted score.

缺失：RGBNT201/pretrained，停止阶段 accept-m0，无替代分数。

缺失：RGBNT201/random，停止阶段 accept-m0，无替代分数。

缺失：MSVR310/pretrained，停止阶段 accept-m0，无替代分数。

缺失：MSVR310/random，停止阶段 accept-m0，无替代分数。

缺失：RGBNT100/pretrained，停止阶段 accept-m0，无替代分数。

缺失：RGBNT100/random，停止阶段 accept-m0，无替代分数。
