# AlphaZero 路线审计（2026-10-22）

> 后续更新：本报告以下为修复前的历史审计快照。已完成 MCTS/评测修复与首轮自对弈训练，当前结果见 [SELFPLAY_TRIAL.md](SELFPLAY_TRIAL.md)。旧增强缺陷与历史数据没有全面修复，本轮使用独立新数据。

结论：已复现搜索统计和旧数据生成中的实现错误，不能把先前结果解释为“自对弈路线已证伪”。尚未证明修复后能胜 clysto；没有重训，没有修改训练/MCTS生产代码或模型文件。本次只增加只读诊断脚本与结果。

## 1. 两套 MCTS 重复计算访问次数（当前代码仍存在）

位置：alphazero_nn.py:349、self_play_resnet_mcts.py:123。

一次回溯同时 `child.N += 1` 和 `parent.N += 1`。中间节点既是子又是父，在同一次模拟中增加两次；W 却只积累一次，随后 Q=W/N 因而被扭曲，根访问分布也不是实际模拟计数。

复现：固定合成二叉树、零网络值、均匀 prior、32 次模拟。NumPy 和 ResNet 两版均为 root.N=32、根孩子 N=[33,29]，合计62；将重复计数在内存中修正后，两版均为 [17,15]、合计32。内存试验不会改写生产文件。叶子耗尽时的无走法值是 -1，所以合成树 W 不是全零，这不影响计数守恒结论。

建议：每条选中边的 N/W/Q 更新一次，仅根节点额外增加一次访问计数，添加“根孩子访问和等于模拟次数”的回归检查。先修这个，再比较策略与搜索增益。

## 2. 早期训练中混入未完赛的伪胜负

位置：alphazero_nn.py:425-430；self_play_v2.py 也有同类截断后按规则估分判胜逻辑。

对本地 replay.pkl 的50局全部按规则重放：只有12局真正结束，38局未结束却有 winner=1/2。expand_game_records 再把该 winner 转成所有步的 ±1。因此76%的局不是实际胜负监督。未结束按启发式赋标签可以作为另一种有明确含义的训练目标，但不能与真实胜负混为一谈、据此验证 AlphaZero 路线。

较新的 az_selfplay.py 已把超步数设 z=0，但 nn_torch.load_samples 会把所有 z=0 的 vw 设0，包括自对弈截断样本。抽查 az_r3_0.pkl 与 az_r3_1.pkl 共30939个样本，21400个（约69.2%）为z=0，这部分不会训练 value。它们不应直接等同于规则规定的和棋。需区分“真实和棋、截断未知、开局专家样本”来源，分别制定目标和权重。

## 3. 旧的双方交换增强产生错误棋谱

位置：self_play_v2.py:152 的 augment_with_swap，仅翻 step.player/winner，却保留起始棋盘和全部路径。expand_game_records 总从原始 init_board 重放，所以第一手标成P2，却仍移动P1起始区的棋子。这不是有效对称增强。

本地 replay_v2_aug.pkl：400局、57412步中，有28706步的玩家轮次与原始开局不符；每20步抽查落子，发现1529次当前方非法落子。该文件已实际存在，不能只视为潜在代码问题。

另一个实验函数 nn_torch.augment_rot 旋转棋盘却保留玩家身份平面，也不符合双方目标区的对称变换；不过训练CLI的 --rot 默认0，因此不能断言所有训练都受影响。

建议：旧增强文件先隔离。若做180度旋转+颜色交换，应同步变换棋盘、玩家身份、所有起终点/路径和winner，并按新先手信息重放；或者直接在已编码样本上生成严格等价的对称样本。不要只翻 player 标签。

## 4. 评测配置存在误导

- evaluate_nn_vs_opponent.py:78 的 torch 分支始终创建 NNEasyEngineTorch，未传递或实现 --use_mcts / --n_simulations，但仍打印用户传入的 use_mcts。故 torch 模式即使指定 --use_mcts 1，实际仍是纯策略选点。
- GUI 的 NN 选项同样是纯 policy，不是 ResNet+MCTS；它的实战成绩不能代表完整自对弈搜索系统。
- NumPy NNEasyEngine 的 MCTS 路径调用带根 Dirichlet 噪声的 mcts_think；没有评测关闭噪声的开关。构造参数 seed 只控制 self.rng，不控制 MCTS 内全局 np.random 的噪声，导致指定seed仍未完全复现。
- evaluate_nn_vs_opponent 的步数上限仍用 evaluate_v2 强判胜负；nightly_az 以对固定 v0d4 的20局结果选冠军，不是候选对当前冠军的直接对战。不能据此保证自对弈模型持续进步，也不能把所有判胜都算真实到家。

建议：分开 pure policy / NN+MCTS 两套评测；评测关闭探索噪声；固定合法开局成对交换先后手，实际胜负、截断、重复局面分别统计。保留独立测试开局，避免反复选最大造成乐观估计。

## 5. 已排除或缩小的疑点

- 合成“立即获胜/对手反击”小树测试：两套 MCTS 在测试中都能选择立即获胜，备份视角符号在该测试下正确。主要已复现错误是计数，不能宣称所有 value 都反了。
- 随机小网络、单局面上的 NumPy手写梯度 vs PyTorch autograd：loss差约5.4e-7、最大梯度差约9.54e-7；该冒烟检查未发现明显反传错误，不代表全面梯度验证。
- 较新 replay_nn_mcts_b.pkl 的147局、19853步，全都真正完赛，winner均匹配规则结果，pi归一化无异常，每20步抽查合法性无异常。不能把早期伪标签问题套到这个文件。
- replay_clysto_merged.pkl 的326局、32297步全部完赛，重放结果与winner匹配；replay_v2_argmax_230.pkl 的230局也全部完赛。抽查均未发现当前方非法落子。
- 未找到旧 /tmp/night 日志。缺少每个模型的精确数据清单、命令与评测日志，所以不能严格分解“某一冠军模型失败有多少来自上述各项”。

## 推荐处理顺序

1. 修两套 MCTS 的计数，补计数守恒、获胜和对手反击测试；评测禁用根噪声。
2. 隔离旧错误增强/伪胜负数据，保留来源与终局原因；优先使用已经重放验证的真实完赛数据。
3. 在同一个现有模型上比较修前/修后 MCTS，确认搜索能提升策略；先不要跑通宵训练。
4. 接通真实 ResNet+MCTS 评测，再做少量可复现的自对弈训练、候选对冠军对战。通过后才放大规模。

可复现命令（项目目录）：

```sh
/Users/liuzirui/miniconda3/envs/checkers/bin/python diagnostics/audit_alphazero.py
/Users/liuzirui/miniconda3/envs/checkers/bin/python diagnostics/check_az_components.py
```

结果：diagnostics/alphazero_audit.json、diagnostics/az_components.json。所有读取均为项目内本地已有pickle数据，未下载外部模型或启动训练。
