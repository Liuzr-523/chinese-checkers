# 中国跳棋 AI 项目 · 完整交接文档

> 用途：换设备 / 换 AI 无缝衔接。另一个 AI 读完这份文档后，应当能完全接管项目，无需重新向用户追问背景。
> 最后更新：2026-10-05

---

## 0. 一分钟速览

- **项目**：中国跳棋（Chinese Checkers）AI，Python 实现，从零手写。
- **目标**：做一个能战胜启发式基线、并最终用神经网络下棋的 AI，作为用户的深度学习学习项目。
- **当前状态**：启发式引擎 V2 已可用；手写 numpy 的 CNN 能与 V0 打平；但与 GitHub 上的 C++ 强引擎 clysto 对战 0:10 惨败。
- **下一步**：① 搜索性能优化（评估函数查表化 + 走法排序改造 + 置换表）；② 手写 numpy 迁移到 PyTorch。
- **代码位置**：`/Users/liuzirui/Desktop/py/chinese-checkers/`（已 `git init`，首次 commit 完成）。

---

## 1. 用户画像与项目初衷

### 用户是谁
- AI 专业大一新生，暑假预科阶段。
- Python 熟练；已自学 AI 概念：评估函数、CNN、多臂老虎机（UCB1）、神经网络训练（前向/反向传播、梯度下降、损失函数、softmax 温度参数）。
- 有在个人电脑部署 KataGo（围棋 AI）的经验。
- **核心诉求**：不是做产品，是把整个 AI 训练过程完整、详细地学会，未来能**给老师完整讲出来**——每个专业名词都要能拆解、被追问时答得上细节。

### 学习路径（项目就是教材）
1. 手写棋盘规则与走法生成（几何/图论）
2. 手写搜索（Minimax → Alpha-Beta → 迭代加深）
3. 手写评估函数（启发式，逐特征 AB 测试）
4. 手写神经网络（纯 numpy，不依赖框架，前向+反向传播都自己实现）
5. 行为克隆（BC）+ 温度采样 + MCTS + AlphaZero 自对弈
6. （计划中）迁移到 PyTorch，对比手写 vs 框架

### 沟通风格（重要，接手后要遵守）
- 用户要求：**非任务时不闲聊，回复尽量短、抓重点**。
- 技术决策喜欢先讨论清楚再动手，尤其大改动（如 PyTorch 迁移）要先商量。
- 喜欢"先试最小方案、验证了再扩展"的节奏。

---

## 2. 技术栈与运行环境

| 项 | 值 |
|---|---|
| 语言 | Python 3（系统解释器 `/usr/bin/python3`，3.9.6） |
| 第三方库 | 目前**只依赖 pygame**（GUI）。numpy 手写实现中，训练脚本用的是纯 Python list（无 numpy 依赖） |
| 棋盘编号 | 0–80，星形 17 行（1,2,...,9,...,2,1 个点），共 81 格 |
| 坐标 | `POS_TO_XY`：81 个 (x,y) 轴坐标；注意是**菱形/斜轴坐标，不是直角方格** |
| 对战对手 | clysto/chinese-checkers（C++，已编译为 `cc_cli`，在 `clysto-cc/` 目录） |
| 版本控制 | git（本地已 init + 首次 commit，主分支 `main`） |

---

## 3. 完整演进史（时间线，接手前必读）

这段是"项目怎么一步步来的"，用户希望给老师讲的就是这条线。

### 阶段一：棋盘规则（踩坑最多）
- 确立 0–80 编号 + 星形棋盘。
- **核心规则**：跳棋 = 三点共线 + 中点有子 + 跳过的子**不被吃掉**；允许连续跳；连跳是递归闭包。
- **最大坑**：用直角 x/y 判断共线是错的。菱形棋盘必须用**方向向量 + 中点几何中心 + 拓扑相邻**判断。曾因此反复出错（79→74→63 被误判成 79→74→64），用户多次指出"你算坐标的逻辑错了，邻接表一直是对的"。
- 最终方案：`get_jump_moves` 用 `pos_to_xy` 求方向向量 (dx,dy)，中点 (x2,y2)、目标 (x2+dx, y2+dy)，再 `xy_to_pos` 反查。
- `NEIGHBORS_81` 邻接表是权威正确数据，**不要瞎改**。

### 阶段二：搜索引擎演进
- `RandomEngine`（无方向，500 步全平）
- `GreedyEngine`（单步评估，对 Random 60% 平局）
- `MinimaxEngine`
- `AlphaBetaEngine`
- `AlphaBetaIDSEngine`（迭代加深 + 限时 + 节点计数 + 超时中断）—— 当前主引擎

### 阶段三：评估函数迭代（v0→v7，逐特征 AB 测试）
- `evaluate_v0`：`最落后棋子进度 * 100 + 总进度`（min×100+sum）。这是基线，保留至今。
- `evaluate_v1`：v0 + 连跳机会×W（W 调大后 AI 只顾局部跳、0:20 惨败，已弃）
- `evaluate_v2`：v0 + 后排启动惩罚(W=3) + 对手进度差(W=0.05) —— **当前最佳**，10:0 胜 v0，平均 80 步
- `evaluate_v3~v7`：各种实验特征，多数已弃或未优于 v2
- **教训**：一次加多个特征无法归因，必须**一次加一个 + 小权重 + 单测**。

### 阶段四：GUI 与交互
- `gui.py`（pygame）：支持双方 AI/真人自由选择、拖拽棋子、AI 自对弈。
- `play_vs_ai.py` / `play_turn.py`：终端交互。
- 乱码问题（字体）+ 双 AI 报错已解决。

### 阶段五：神经网络（手写 numpy CNN）
- 结构：输入 4 通道（我方/对方/空位/轮次），3×3 卷积（32 滤波器），policy head（162 维 logits：src81+dst81）+ value head。
- 全部手写：`_conv3x3`、`_relu`、`_softmax`、`forward`、`backward`（含反卷积）、`train_step_sgd`、`AdamOptimizer`。
- 训练方式：**行为克隆（BC）**——让网络模仿 V2 的走法。
- 关键发现（很重要，见 §5 教训）：
  - `val_loss` 低 ≠ 实战强。
  - 温度 T 控制采样多样性：T=1.0（探索）比 T=0.3/0.6（贪心）泛化更好。
  - V2 的 argmax（最优）走法数据能让 NN 10:0 胜 V0；但 V2 的 T=1.0 采样（探索）走法数据让 NN 0:10 输。**数据质量决定一切**。

### 阶段六：方案 1 与方案 3（突破瓶颈的两次尝试）
- **方案 1**（8 步随机开局增加数据多样性）：失败，T=0.5 泛化 0:6 全负，数据量(80局)和随机性超出 NN 容量。
- **方案 3**（AlphaZero 式 MCTS 自对弈）：改造 `train_iteration` 支持 Adam + warm start + 逐迭代保存 ckpt。5 迭代跑通，但最佳 ckpt(it004) 仅与 BC 持平（10:10 vs V0），T=0.5 泛化仍 0:6 全负。**未超过 BC**。

### 阶段七：引入外部强引擎 clysto（当前最新进展）
- 用户找到 GitHub 项目 `clysto/chinese-checkers`（C++），要求对比实力。
- 我做的工作：
  1. 克隆到 `/Users/liuzirui/Desktop/py/clysto-cc/`，用假 spdlog 头绕过依赖，`clang++` 直接编译出 `cc_cli`。
  2. 发现两边棋盘**编号不同**（他是 9×9 轴坐标菱形，我们是 17 行星形），写了坐标映射并**验证 81 点邻接完全同构 + 双方初始三角区对齐**。
  3. 写桥接对战脚本 `arena_vs_clysto.py`，他每步走法经我们规则合法性校验（全程零失败，规则一致）。
- **结果：V2 vs clysto = 0:10 全败**，先后手都输。
- **根因**：他是 C++ + MTD(f) + 置换表 + 开局库 + 空步剪枝，1 秒搜十几层；我们是 Python + 基础 Alpha-Beta，5 秒只到深度 4（见 §6 性能分析）。
- **价值**：白捡一个强力陪练；他的棋谱可当 NN 的高质量训练数据（"拜师"路线）。

---

## 4. 代码文件结构（15 个源文件）

| 文件 | 作用 | 是否核心 |
|---|---|---|
| `Chinese Checker.py` | 棋盘核心：`POS_TO_XY`/`XY_TO_POS`、`NEIGHBORS_81`、走法生成（`get_all_moves`/`get_all_jump_paths`）、`apply_move`、`is_game_over`、`init_board` | ★★★ |
| `engine.py` | 搜索与评估：`Engine` 基类、`evaluate_v0~v7`、`Random/Greedy/Minimax/AlphaBeta/AlphaBetaIDS` 引擎、`order_moves`、`_alphabeta_count` | ★★★ |
| `alphazero_nn.py` | 手写 numpy CNN + BC 训练 + MCTS + AlphaZero 训练循环（`NetParams`/`forward`/`backward`/`AdamOptimizer`/`train_iteration`/`train_iteration_bc`/`mcts_think`/`NNEasyEngine`） | ★★★ |
| `gui.py` | pygame 图形界面（AI/真人自由组合） | ★★ |
| `arena.py` | 对战框架（两引擎对弈、记录棋谱） | ★★ |
| `arena_vs_clysto.py` | 与 clysto 引擎对战桥接（含坐标映射 + CLI 子进程适配器） | ★★ |
| `self_play_v2.py` | V2 自对弈数据生成 | ★ |
| `gen_argmax_data.py` | 生成 V2 argmax 走法数据 | ★ |
| `merge_replay.py` | 合并棋谱 pkl | ★ |
| `eval_ckpts.py` | 批量评估 checkpoint | ★ |
| `evaluate_nn_vs_opponent.py` | NN vs 对手评估（支持 `--ckpt`） | ★ |
| `play_vs_ai.py` / `play_turn.py` | 终端交互 | ★ |
| `test_logic.py` | 核心逻辑随机种子测试（20+ seed × 500 步 × 7 断言） | ★ |

### 关键接口约定（改动时务必保持）
- `Engine.think(board, player)` 返回 `(path, info)`；`path` 是走法路径 list（`[起点, 中间点..., 终点]`），单步为 `[a, b]`。
- 所有走法函数返回一致格式，否则下游报错。
- `board` 是长度 81 的 list：`0`=空、`1`=P1、`2`=P2。
- 文件导入用 `__file__` 绝对路径 + `sys.path.insert`，保证任意目录可运行。

### 数据文件（被 `.gitignore` 排除，不入库）
- `nn_checkpoints/`：`params_bc_chosen_by_eval.pkl`（BC 最佳，10:0 版本）、`params_az_it_004.pkl`（AlphaZero 最佳）、`params_latest.pkl`、`params_bc_latest.pkl`
- 训练棋谱：`replay.pkl`（AlphaZero 自对弈 50 局）、`replay_v2_argmax*.pkl`（BC 数据源）
- `clysto-cc/cc_cli`：编译好的 clysto 引擎二进制

---

## 5. 已确认的关键事实与教训（避免重蹈覆辙）

1. **val_loss 不是实战强度的可靠指标**：NN 可能学到低 loss 的死板策略，泛化差。
2. **温度 T**：T=1.0（高探索）比 T=0.3/0.6 策略更稳健；但用 T=1.0 的探索数据训练 NN 无法战胜 V0。**训练数据必须用近最优（低温/argmax）走法**。
3. **数据质量 > 数据量 > 模型结构**：V2 argmax 数据 → NN 10:0 胜 V0；V2 采样数据 → 0:10 输。
4. **一次只加一个特征、用小权重**，否则无法归因。
5. **连跳 DFS 的棋盘拷贝不是瓶颈**（实测仅 2%），别在这上面浪费优化。
6. **评估函数是最大瓶颈**（实测 68%），且 `order_moves` 用完整评估做排序是隐性浪费。
7. **坐标映射**：菱形棋盘不能用直角坐标判断共线；邻接表是权威数据别乱改。
8. 跳过的棋子不被吃掉；连跳禁止回跳（否则 A→C→A→C 无限递归）。

---

## 6. 性能分析与优化路线（当前待办主线）

### 实测数据（V2 搜索 5 秒，深度 4，69,274 节点）
| 开销来源 | 占比 |
|---|---|
| `evaluate_v2` + `evaluate_v0`（被调 31 万次） | ~68% |
| 连跳 DFS 全链路 | ~14% |
| `is_game_over`（每节点扫两遍全盘） | ~7% |
| `list.copy`（DFS + apply_move） | ~2% |
| 其他 | 余量 |

**关键发现**：5 秒只到深度 4 就超时（`aborted: True`），这是 0:10 输 clysto 的直接原因。

### 优化路线（已与用户讨论，按 ROI 排序）
1. **评估函数查表化**（最高优先级，预期 5–10 倍）：预计算 81 格位置分表，评估 = 10 次查表加法，消灭 81 格大循环。**注意**：同时要改 `order_moves`，别再用完整评估做排序（换成轻量信号，如纯查表快评）。
2. **Zobrist 置换表**（2–3 倍）：跳棋换位局面极多，TT 直接省 30–70% 节点。
3. **Killer/History 走法排序**：当前 prune_rate 仅 6%，排序差，潜力大。
4. **MTD(f)**：收益最低、强依赖 TT，可最后做或不做。

### 另一 AI（豆包）给的四点建议评估（用户让点评过）
豆包四点（评估查表化 / Zobrist TT / Killer / MTD(f)）方向都对，但：
- 漏了 `is_game_over` 7% 和 `order_moves` 隐性浪费两个盲区。
- "优化完能赢 clysto 3–4 局"是拍脑袋，别当预期，**以搜索深度提升为硬指标**。

---

## 7. 待办清单（接手后从这里继续）

### 近期主线
- [ ] 搜索优化第 1 步：评估查表化 + `order_moves` 换轻量排序（改完测同 5 秒的深度，验收）
- [ ] 搜索优化第 2 步：Zobrist TT + `is_game_over` 移出内层
- [ ] 视效果决定第 3 步 Killer/History

### 中期
- [ ] **PyTorch 迁移**（用户明确想做，动手前先商量）：重写 `alphazero_nn.py` 网络部分，复用 pkl 棋谱数据，BC 重训后评估；对比手写 numpy vs PyTorch
- [ ] 用 clysto 生成高质量棋谱给 NN 做 BC（"拜师"路线）

### 文档与版本
- [ ] 每次大版本变更后写讲解文档（结构见 §8）
- [ ] 用户会自己用 GitHub Desktop 推送；每次改完提醒 commit

---

## 8. 讲解文档模板（用户要讲给老师听）

每次大变更后，为以下内容写一份文档：

1. **这版做了什么、为什么**（动机→方案→结果）
2. **怎么讲给老师听**（讲稿要点，从直观到严谨）
3. **专业名词拆解**（每个词：定义 / 作用 / 为什么需要 / 可能被追问的点）

### 需要重点拆解的专业名词清单（累计）
- 搜索类：Minimax、Alpha-Beta 剪枝、迭代加深、置换表、Zobrist 哈希、MTD(f)、走法排序、Killer/History 启发、剪枝率
- 评估类：启发式评估函数、位置分表、min×100+sum
- 神经网络类：CNN、卷积核、通道、前向/反向传播、梯度、梯度裁剪、L2/权重衰减、损失函数、softmax、温度、argmax、采样、val_loss vs 实战
- 训练类：行为克隆(BC)、自对弈、MCTS、UCB1/PUCT、探索-利用、replay buffer、warm start、checkpoint
- 优化器类：SGD、Adam、学习率、动量

---

## 9. 给接手 AI 的协作建议

1. **先读 `Chinese Checker.py` + `engine.py`**，再读 `alphazero_nn.py`，这是项目三大核心。
2. **改代码前先 Read 原文件**，不要凭记忆；邻接表、坐标映射是雷区。
3. **大改动先跟用户商量**（尤其 PyTorch 迁移、改评估函数、删文件）。
4. **验证优先**：改完跑 `test_logic.py`（核心逻辑）和实战胜率，别只看 val_loss。
5. **遵守用户的"少废话"规则**：非任务时回复短、抓重点；任务执行时给阶段性简报。
6. **GitHub 工作流**：改完提醒用户 commit；用户自己在 GitHub Desktop 里 push。
7. clysto 引擎在 `clysto-cc/`，跑 `arena_vs_clysto.py` 前确认 `cc_cli` 存在；若换设备需重新编译（`clang++ -std=c++14 -O2 -Isrc -Ivendor src/cli.cpp src/game.cpp src/transtable.cpp -o cc_cli`）。

---

## 10. 已知的运行命令备忘

```bash
# 终端交互对战
cd /Users/liuzirui/Desktop/py/chinese-checkers && python3 play_vs_ai.py

# GUI（pygame）
cd /Users/liuzirui/Desktop/py/chinese-checkers && python3 gui.py

# 与 clysto 对战（10 局，各 1 秒/步）
cd /Users/liuzirui/Desktop/py/chinese-checkers && python3 arena_vs_clysto.py --n_games 10

# 核心逻辑测试
cd /Users/liuzirui/Desktop/py/chinese-checkers && python3 test_logic.py
```
