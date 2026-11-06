# 新接力机制与真实续弈偏好：专项评测协议

当前路线图：步骤3专项，先冻结机制根和真实续弈标签，再按训练覆盖门槛决定是否训练。无论专项结果如何，本轮不登记冠军，不宣称整局棋力提高。

## 目标与机制

在同一根上比较“最后方棋子先一步进入新接力链入口”与“先兑现前锋已有长跳”。rear/front只是几何身份，不预设rear优。

新根生成、规模、split和候选路径先按collector固定配置冻结schedule及哈希，再走分支，不按NN错误或续弈标签补根。根与强制子局面按完整对称轨道排除历史；跨split输入root/child碰另一split续弈时双方整组mask，保留原始四局及raw_label。续弈之间/历史状态的共享单独统计，不把输入隔离写成所有未来轨迹都独立。

机制至少要求：rear是最后方棋子的一步相邻前进，走后该子有前进至少两跳，且出现原根该子任何合法前进走法（邻步、单跳、连跳均计）都不能到达的新终点；front是非最后子已有的前进连跳。保存完整mechanism_witness，prepare_rows只读重算接力证据，并核验rootprefix/board/player/ply。上述证据是“当前占位下出现新跳链机会”，不是承诺对手应对后一定保留。

## 分支schema与真实终局

每根四分支：depth3和depth4两套固定续弈配置，各自强制rear/front一步后，由双方同配置派生引擎继续到规则终局或限步。

统一game格式：opening是未走forced的rootprefix；start_ply=root_ply；paths包含forced首步和全部后续；search_info与paths等长，首项明确forced、不伪装搜索，后续逐步记录请求/实际深度及aborted；plies为终局绝对手数。根方own_moves=(plies-start_ply+1)//2，包含forced一次。

prepare_rows重放每个有效根四条归档路径，检查首步、逐步走棋方/ply、配置深度完成、终局winner/reason和own_moves。它不重新搜索。任一配置超时/未完成不能冒充完整续弈。

## 偏好标签

对根方计终局points：胜1、严格规则真和0.5、负0。max_ply是截断，不能当真和；任一分支max_ply/未完成，该配置不给偏好。

每个配置按以下顺序比较：

1. points不同，按points较高方给point偏好，无论获胜耗时多久。
2. points相同且两分支都为根方胜，比较从forced起的根方实际落子数；差至少2手，才给较快方tempo偏好。
3. 同负、同真和、同胜但仅差0或1手，均无偏好；不把更快输或更快和当优。

depth3和depth4配置都给非零且同号偏好，才有稳定label：+1 rear优、−1 front优。两配置可分别由point与tempo产生同号偏好，kind记mixed，来源仍逐配置保留。方向不一致或任一未给偏好则label0。

所有根value_weight=0。本监督是“固定两套续弈策略下的相对收益/胜局落子效率”，不是普适状态价值z、实际胜率或全合法动作最优证明。真实终局只保证分支结果真实，不保证后续策略最优。

## 训练与开发覆盖分开

- 训练eligible至少8根，两方向各至少2，才可启动专项拟合。
- 开发eligible至少8根，两方向各至少2，才具备正式验收coverage。
- 开发覆盖不足不阻止已经具备训练覆盖的拟合诊断；但开发gate必须失败，不把训练记住当泛化通过。
- 不追加根、重分split、复制稀少方向或改阈值凑覆盖。

开发基线仅v3一次，候选仅固定最终检查点一次，不在中途看开发择点。中途训练指标使用prepared cache，不重复搜索或续弈。完整collector预算、优化器、更新步数和采样序列由runner配置在开跑前锁定。

## 主要pair指标

g=logit(rear)−logit(front)，y为稳定label。

- 准确：y×g严格大于0；logit tie算未辨明，不算正确。
- rear优/front优各报n、准确率；balanced accuracy为两方向准确率等权平均。缺方向则为null。
- pair CE=softplus(−y×g)，数值稳定计算。
- 每个depth分别报告point_regret：预测选错或tie时取该配置两points的绝对差，选对为0。同points时真实point regret恒为0，包括tempo根，不得造搜索分差。
- 每个depth分别报告winning_tempo_regret：仅该配置两分支都胜且构成tempo偏好的适用集合，选错或tie计两根方落子数绝对差，选对为0。适用n与均值单列，不把它与points或搜索分数混成一个量。
- ties在points/tempo regret中采用保守的未判对惩罚；这是pair决策诊断，不代表模型实际全合法argmax的真实续弈损失。
- 实际合法argmax是否落入候选pair、是否落到稳定偏好方仅辅助报告。pair排对不等价为真正选择了该招；候选外动作没有新续弈标签，不补造它的质量分。

评测统一eval与requires_grad状态并在结束恢复，保留逐例logit gap、margin、方向/类型、两配置regret和双方分层。

## 正式开发gate

1. 开发覆盖至少8根、每方向至少2。
2. balanced accuracy至少0.75，且比v3至少增加0.125。
3. rear优/front优准确率分别至少0.60。
4. pair CE不高于v3。
5. depth3和depth4的point_regret均值各不高于v3，按全部有效pair计算。
6. depth3和depth4各自适用的winning_tempo_regret均值不高于v3。
7. runner合并旧653策略CE、旧653价值MSE、普通353策略CE各不超v3+0.05，以及value参数哈希不变。

若某配置没有适用的both-win tempo pair，则该项n=0、mean=None，gate记N/A（布尔检查可为True但必须注明范围）；不因缺测阻断点数目标验收，也绝不能声称已验证tempo能力。只要n>0就必须与相同适用集合的v3比较，不能把缺测填成0。

value参数冻结不代表共享CNN更新后value输出冻结，因此旧价值MSE仍独立保护。无开发或缺方向时relay_gate安全失败，不运行训练以外的新评价来补齐。

通过只能准备后续NN/MCTS实际选择验证，不能直接晋升。最终整局棋力仍需另行固定候选、全新隔离开局和预声明独立实战验收。失败不在本开发上补步数、挑检查点或扫描系数重考。

## 实现与测试

relay121_targets.compare_returns与stable_preference负责points优先、胜局tempo、无效截断和跨配置稳定性。relay121_metrics.prepare_rows负责机制和归档重放，relay_metrics与relay_gate负责固定指标及门槛。没有score4/5字段依赖，不将搜索分数塞进真实收益指标。

单元测试覆盖截断/未完、points优先、两手tempo阈值、forced计数、同负/真和无tempo、跨配置同号/翻转、points与tempo分离、N/A范围、空开发、安全gate、实际argmax与pair不同、状态恢复和新接力终点证据。
