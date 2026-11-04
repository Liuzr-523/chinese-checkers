# 实验工具与本机数据

## 运行依赖

- GUI：`pip install pygame`，然后`python gui.py`。
- CNN训练与评测：另安装`numpy torch`；分析绘图另安装`matplotlib`和简体中文字体。
- 121格C-PVS：`python build_ccore121.py`，使用同一个Python运行测试和GUI。
- 冻结版派生专家：`python diagnostics/build121_analysis_expert.py`。此命令从仓库已有C源码编译，不需要原版Clysto工程。
- `build_clysto121.py`用于从外部Clysto源码重新生成派生评估，需将`CLYSTO_DIR`指向相应工程；它与编译冻结源码是两个用途。

## 脚本分类

- `az121.py`、`train121_trial.py`、`selfplay121_*.py`：121格CNN/MCTS与自对弈试验。
- `curriculum121.py`、`imitate_clysto121.py`、`corrected121_decisions.py`等：专家蒸馏、针对性数据和旧能力保留试验。
- `neural121_registry.py`、`checkpoint121_save.py`：独立验收、冠军登记与本机存档。
- `confirm121_*.py`、`arena121.py`、`iterate121.py`：固定预算对局与搜索对照。
- `diagnostics/`：误差诊断、受控对照和中文绘图。部分脚本固定读取历史实验目录，用于复核对应实验。
- `engine_jumpstar.py`、`arena_vs_jumpstar.py`：历史联网试验桥接。严格规则不兼容时停止；不在普通121格GUI中提供。

模型权重、冠军登记、第三方裁判二进制和大规模训练数据不随仓库发布，缺少这些输入时对应历史试验不会自动启动。新克隆仍可运行人类对战、参考搜索、C-PVS及不依赖训练权重的单测。公开实验数据与复现指令位于`docs/experiments/`。

## 测试

```sh
python build_ccore121.py
python diagnostics/build121_analysis_expert.py
python -m unittest discover -s tests -q
```

历史数据样本缺失时，相关可选测试会明确跳过；JumpStar本地裁判缺失时跳过其外部集成测试。测试不会调用联网下棋服务。

81格历史入口和模型继续沿用旧规则；不得用其成绩充当121格成绩。实验代码入库代表流程可复现，不代表其中每个候选通过了棋力验收。
