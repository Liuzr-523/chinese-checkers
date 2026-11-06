# 接力机制与真实续弈收益专项：采集异常停止

路线图第3步，2026-11-06。单次执行既定入口后，候选推进搜索未完成预声明深度，收集器拒绝继续。**未训练、无候选、无开发验收结果，现役v3保留**。本轮没有棋力升级证据。

![采集状态](采集状态.png)

## 执行与结果

接手时main工作区干净，HEAD为497cfec；无本项目采集/训练/竞技进程，目标实验目录不存在。预检记录的9个源码SHA全部一致，冠军SHA为`a0a40208061aaaa45931131f1760bb20ae114388c6ae6ad29181b204565395d9`。交接提交已经成功push。

按[原协议](protocol.md)仅启动一次train121_relay_returns.py，主PID26557、collector PID26562。未另起collector，未改CNN、strict121规则或C内核。父进程记录212.445秒，收集器记录211.124秒；这些是各自time.monotonic差值，不据此反推墙钟运行时长。

|项目|实际结果|
|---|---|
|历史排除轨道|288729|
|已接收机制根|22，均为train；ply40/41各6根，56/57各5根|
|48根schedule冻结|未完成，只有schedule_partial.json|
|完整续弈|0 / 192|
|dataset、标签、覆盖判定|未生成|
|训练、开发基线、候选评测|均未开始|
|历史7例模型回归|无最终候选，不运行|

失败现场为P2、ply35，requested_depth=3、depth_reached=2、aborted=true、nodes=5057。ConsistentExpert抛出fixed-depth incomplete，父进程随后以CalledProcessError终止。原生搜索有单次120秒截止保护；现有归档足以确认深度未完成，**不足以判定具体为何触发中止**，不将它归因于标签、模型或数据覆盖失败。原始[失败局面](artifacts/fixed_depth_failure.json)、[收集状态](artifacts/collection_status.json)、[主状态](artifacts/status.json)保留。

没有跳过失败候选、补根、重分split、放宽深度或重复启动。本轮在训练覆盖判断之前结束，因此不能写成“覆盖不足未训练”。下一步仍是步骤3：先针对归档现场诊断搜索中止来源，再明确新的执行/恢复边界；现有目录不得直接重跑或覆盖。

## 独立复核

独立审计脚本不调用生产prepare_rows或stable_preference。对已接受22根逐一重放完整prefix，检查root/child/anchor轨道、前进邻步后原合法走法不可达的新多跳末点、前锋连跳、历史及跨组输入排重，全部通过；输入历史/跨组重叠为0。[独立审计结果](artifacts/independent_data_audit.json)。

```sh
/Users/liuzirui/miniconda3/envs/checkers/bin/python diagnostics/audit_relay121_returns.py --experiment nn_checkpoints/experiments/relay121_returns_20261106 --out nn_checkpoints/experiments/relay121_returns_20261106/independent_data_audit.json --partial
```

未产生192局，故终局标签、逐步深度、跨split续弈碰撞和训练抽样审计均不适用，不宣称它们通过。独立脚本完整数据分支尚无本轮数据可验证。

## 执行后修复与工件

只读审查另外发现runner调用old_metrics却漏导入，若未来进入基线评测会NameError；此次实际失败发生得更早。执行后补充`from unfreeze121_trial import metrics as old_metrics`，名称解析检查通过，不改预算、训练目标或门槛。未用修后源码重新运行本轮。原执行源码保留在[executed_source](artifacts/executed_source/train121_relay_returns.py)，与当前修复版本明确区分。

[事后失败摘要](artifacts/failure_summary.json)不是runner生成的result.json（后者不存在）。[完整本机工件清单](artifacts/artifact_manifest.json)记录执行源码、历史排除、部分根、原生依赖与错误归档SHA；大历史轨道、模型和原生二进制留本机。公共工件保留部分根、配置、执行源码及错误证据。现役registry与checkpoint最终再核验不变，无实验后台进程。

修复后验证：211项本机unittest全部通过（4.137秒）；旧评测导入身份检查通过，独立训练审查确认配置与门槛吻合协议。测试结果不代表训练完成或棋力升级。
