# 路线图第2步：48局专家数据试点

采集完成：48局；数据质量门槛：通过。本轮未训练。

|划分|真实终局|终局组|过滤后价值样本|终局前16手样本|
|---|---:|---:|---:|---:|
|train|32|16|2471|489|
|development|16|8|1218|245|

原始轨迹跨组几何轨道0；冲突/历史状态在双方监督过滤，完整棋谱不删除。策略探针有效48/48，最终导出48。

{"all_48": true, "train_terminal_groups": true, "development_terminal_groups": true, "train_value": true, "development_value": true, "train_endgame": true, "development_endgame": true, "valid_probes": true, "bucket_coverage": true, "player_coverage": true}

引擎均固定depth5，交换配置所在阵营，P1先手未变；派生搜索并非原版Clysto。所有胜负由严格规则真实终局给出，max_ply价值权重0，真实和棋0权重1。没有旋转换色胜负增强；没有按输赢补局或改split。