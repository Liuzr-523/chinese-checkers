# 路线图第3步：训练工件独立审计

结论：存在未通过检查，见JSON。 两组未选出候选，不进入第4步。

本轮只读现有源码快照、冻结数据、检查点、抽样轨迹和已存评测结果；没有训练、教师查询、对局或重新开发评测。发现保存的logits SHA不一致后，仅以固定seed的合成随机输入进行数值见证，不重算开发局面。

|检查|结果|
|---|---|
|source_snapshots|通过|
|current_source|执行后声明修复：2份不同|
|data_unchanged|通过|
|v3_unchanged|通过|
|registry_unchanged|通过|
|initial_checkpoints_exact_v3|通过|
|head_nonvalue_unchanged_all_checkpoints|通过|
|head_development_policy_logits_sha_equal|未通过|
|common_trace_2000|通过|
|sampling_audit_matches|通过|
|status_complete|通过|
|no_candidate_selected|通过|
|not_ready_for_step4|通过|

配置固定了20份源码与8份数据哈希；20份执行源码快照全部与记录一致。执行后训练器与其回归测试已修复，当前这两份源码SHA与旧config不同，其余18份当前源码仍一致。v3与冠军注册表未变化。

head_only与joint均保存0/1000/2000步检查点；两组0步全部state_dict张量与v3逐值一致。head_only三个检查点所有非value参数及位置buffer均逐值不变，value参数确有更新。joint允许共享CNN与策略参数变化，未误用head_only冻结方式。

从common_sample_sequence.json重新按训练器相同JSON序列化方式计算SHA，与两组sampling_audit一致。2000步全部ID属于各自训练池，未采样开发探针；每步前8个value样本均为真实终局前16手。每组价值/策略各32000次抽样，组曝光与重复次数逐项重算一致，value批内重复175次、policy重复198次。

最终检查点SHA与sampling_audit记录一致；head_only已有开发合法logits SHA与v3不一致；这是本次审计中唯一未通过项。CE与accuracy相同，所有非value参数逐值相同，但不能将已存SHA写成通过，也不修改原门槛或重算开发结果。

|最终检查点|SHA256|
|---|---|
|head_only|4a621d5245419c91ced7ec4c25a55d848399d67df351a9ddf05ddf79ad322500|
|joint|d43328b3bcb291e48db15c67e3778f946caa9afb27b616c09008474d851e37d1|

上述审计证明训练工件、参数冻结与抽样记录一致，不把任何训练损失下降当作棋力提升。门槛结果由主报告解释；本轮selected=null、ready_for_step4=false。


## 合成输入的数值见证

为定位已存policy SHA不一致，使用torch单线程、固定seed111079，生成batch=1/2/32的`torch.randn(N,6,17,17)`；没有读取开发局面、没有重新选择模型。

原v3与head_only若同时全部requires_grad=False，或同时全部requires_grad=True，三种batch的全部策略logits均torch.equal，最大差0。若模拟原评测路径：baseline全部冻结、加载后的head_only默认全部可训练，则batch1仍相同，batch2出现最大1.90735e-6差异（63个元素），batch32出现最大3.81470e-6差异（587个元素）；所有argmax仍相同。全部测试都在eval/no_grad下。

此见证把微差关联到两模型不同requires_grad状态下的数值执行路径；未用profiler进一步归因具体底层kernel。独立价值头更新没有改变共享或策略参数。当前结果和严格SHA门槛保持原样；后续新实验应在基线和候选统一参数requires_grad状态后做可重复的策略指纹比较，不能事后改本轮门槛。


## 执行后修正的边界

主代理已在执行结束后修复policy_metrics，统一requires_grad=False计算指纹，并在finally恢复每个参数原requires_grad与模型training状态，已补合成回归测试。它只修复审计数值路径，不改变本轮训练、检查点、基线、开发结果及任何gates；本轮不会重训或重新开发评测。原执行源码快照与config哈希对应，后续当前工作区源码可以是修复版本，应分别标注。

已存policy SHA不一致不能误写成head_only实际改变策略。两组均失败开发MSE增益、常数0及末16手双方价值门槛，因此即使排除这个审计路径问题，仍无合格候选。


执行后修复文件SHA：

- `train121_dual_value.py`：`36e2575fac5047e26871c946c5dd21bd233811b69eceb5911c3534dd3651d441`（旧执行SHA仍保存在JSON及原source快照）。
- `tests/test_train121_dual_value.py`：`f7ba44bffa9c33947a1d8376429f25d75717a5e05e20e06fac738b28ebd814fe`（旧执行SHA仍保存在JSON及原source快照）。

主代理确认3项训练器回归测试通过；本审计保留原开发logits指纹不一致及原gates，没有重算开发结果。
