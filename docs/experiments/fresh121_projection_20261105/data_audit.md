# 全新64根开发集独立审计

**所有数据检查通过。** 64个根局面与32个六手开局锚点在完整镜像＋旋转换色归一后，均与281785个历史排除轨道无重合；与693个原开局书样本亦无等价轨道重合。64根、32锚点各自无重复。

双方各32例，己方第7/8/9/10步各16例。64个完整前缀逐手严格合法、根非终局，decision_id/player/ply/orbit与重放一致。全部64例规范编码可正确解码回原始棋盘＋走棋方。冻结教师路径64/64合法、深度均5，root分数与强制首选分数逐值一致，value_weight均为0。本审计未重新查询教师、未运行任何模型开发评测。

schedule与development标签逐字段64/64对应；两文件SHA256与development_freeze.json逐值匹配。开发标签未进入训练标志为false。128次教师查询为64个根查询及64个首选强制分数检查，不是重复模型开发评测。

历史清单包含1134个JSON/JSONL/PT来源，417个PT用weights_only=True和有限数值构造器安全解析，包含31680个数值样本。原20项末局缺前缀已按curriculum121.py的准确构造恢复：teacher_games里test真胜局前10条、prefix=opening+paths[:-8]、prior/candidate顺序，核对group/side/start_ply及全部路径合法；原缺口证据与三个来源哈希保存在resolved，新增180历史轨道，现incomplete=0。

另有418个.pkl/.npz归属20261022—28旧81格实验，按主线程对其配置/源代码的审查排除，不读取、不混用。独立目录计数与原清单一致：search_budget300、value_ablation_80 80、longrun18、value_ablation8、jumpstar_distill4，其余四个selfplay/followup目录各2。JSON记录完整目录分布、历史覆盖计数与冻结文件SHA256。

隔离声明针对开发根及六手分支锚点；从标准初始局面出发的早期前缀自然可能经过共同状态，不把这些共同开局状态称为全新。结论只涉及数据合法性、冻结和历史隔离；棋力或泛化收益等待主线程固定最终模型评测。
