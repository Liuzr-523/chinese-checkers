# 模型与数据目录

- `models/champions/`：GUI/评测选中的冠军模型（4个）
- `models/final/`：各次训练最终模型（171个）
- `models/checkpoints/`：训练中间快照（1154个）
- `data/replays/`：棋谱和 replay 数据（53个）
- `data/selfplay/`：AlphaZero 自对弈训练样本（18个）
- `data/positions/`：待标注局面（16个）
- `data/dagger/`：DAgger 标注样本（48个）

以上是本次迁移数量；包含782个 .pkl 与682个 .pt。文件内容未改，配套权重放在同一分类。

项目脚本仍接受原来的文件名参数，例如 `--ckpt params_resnet_champ.pt`、`--data 'az_r3_*.pkl'`，由 checkpoint_paths.py 自动定位。GUI 和新产出的模型/数据也使用分类目录。显式指定其他目录的路径保持原样。

layout_manifest.json 记录每个文件的原路径、新路径、字节数与迁移前 inode；迁移通过同盘 rename 完成，逐项检查字节数和 inode，未复制、删除或重新保存模型。
