# 中国跳棋 AI

从零手写、不依赖任何 AI 框架的中国跳棋 AI 学习项目：棋盘规则、搜索与评估、神经网络（纯 numpy）、数据合成、对战外部强引擎。

## 运行

```bash
pip3 install pygame        # 仅 GUI 需要

python3 gui.py             # 图形界面（人 vs AI / AI vs AI）
python3 play_vs_ai.py      # 终端人机对战
python3 test_logic.py      # 核心逻辑回归测试
```

## 对战结果

- 启发式 v2 vs 基线 v0：**10:0**
- 神经网络（行为克隆）vs v0：10:0
- 神经网络 vs 外部 C++ 引擎 clysto：0:10（搜索深度差距）

## TODO

- [ ] 评估函数查表化 + 走法排序优化
- [ ] Zobrist 置换表
- [ ] PyTorch 迁移（手写 numpy → 框架）
- [ ] 用 clysto 棋谱训练神经网络（"拜师"路线）

## 说明

`arena_vs_clysto.py` 需自行编译外部引擎 [clysto/chinese-checkers](https://github.com/clysto/chinese-checkers)，不配置不影响其他功能。

License: MIT