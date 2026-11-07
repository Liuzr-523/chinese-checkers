# CNN胜／和／负闭环：受控首轮

路线图第3步，2026-11-07。用户已授权开始预测—完整对局—终局纠错—复核，并新增top3至5候选走法胜率及搜索量展示。无自动晋升，现役v3保留；首轮原型不是第6步生产迭代已达成。

## 模型、显示与搜索

CNN与策略完整继承v3；旧scalar value参数冻结且不参与WDL forward；新增pooled width→32→3胜/和/负头，最后层零初始化，初始三概率1/3。新格式wdl121-v1与旧冠军互拒。MCTS值pwin-ploss，真实终局优先±1/0。

GUI独立后台模型，显示P1胜/和/P2胜，及固定128模拟分析的前5候选（按visits排序）、起终点/完整路径、当前方胜和负概率与访问数。候选由root child回传三概率均值给出，不把Q直接换算为胜率；未访问不填概率，max240叶不伪造真和WDL。分析不修改棋手/冠军，不将搜索限制为5招；PUCT同时考虑Q、先验、探索。显示实验候选/未校准，模型不可用则无伪数字。

## 预冻结数据

torch单线程，初始化seed111120。完整历史覆盖预检；保守镜像/旋转换色orbit仅用于排重，不增强标签。56组先固定：0..31 train、32..47 development、48..55 probe。seed111121，每组最多128候选；随机合法prefix目标8/9/16/17（group%4），root及8手anchor排历史及他组；root=anchor允许组内同一。prefix仅溯源不训练。56根schedule/hash全部冻结后才对局。

前48组由固定初始WDL0双方32模拟自弈，seed111122+group。每步现有Dirichlet root噪声，前12手按visit分布温度1采样，之后argmax；训练策略目标始终root visits。每步记录当前方/ply/WDL预测/path/完整paths/visits/priors/Q。整盘固定模型。真实终局按当前方class 0胜1和2负；max_ply/未完仅policy，无WDL监督。

逐步重放；跨train/dev轨迹orbit相撞双方sample排除，再排历史输入，原件保留。不按胜负补盘/重分。probe root/anchor若碰训练轨迹，该probe整对标无隔离资格，保留并分列，不能冒称全轨迹独立。

## 固定训练

train至少16盘有保留真实终局标签才训练。自然胜者分布保留，不按赢家重采样；报告阵营先验。dev有效完整局>=8只影响验收，不决定训练。

单配置2000更新，AdamW lr1e-4 decay1e-4 clip1，采样seed新111123/旧111124。batch新16按有效train game均匀再组内sample均匀；旧train8+普通train8仅合法策略KL（fresh121_projection_20261105/data_sources），validation不入训练。loss=新soft-policy CE+新WDL CE+.1旧合法policy KL(v3)。旧scalar value参数冻结，其余CNN/策略/WDL可训。完整trace，保存0/1000/2000；每400仅train。新dev仅初始WDL0和固定final各一次，不择点/扫描。标签只来自本轮真实终局。

## 概率验证和行为复核

报告三类Brier平方和、NLL、expectedpoint MSE、10箱maxconf ECE、按player/末16手/game macro。对照uniform和train-only玩家先验；先验按实际game均匀sample目标计算，三类各加1伪计数固定平滑，不用dev估prior。概率依赖本轮自弈策略，不称对任意对手的绝对胜率。

dev gate：有效完整局>=8；game macro NLL同时比uniform/训练prior至少低.02；macro Brier比uniform至少低.02且不高于prior；两玩家Brier各不劣uniform；末16手Brier不劣uniform（无适用N/A）；新policy CE<=WDL0+.05；旧653/普通353 policy CE各<=v3+.05；旧value参数逐值不变。ECE仅诊断，小样本始终calibrated=false。

candidate存在才做8 probe根×阵营交换×两个固定对手（WDL0、现役v3）共32局诊断，双方32模拟，无噪声/额外书库。无论dev过否按固定方案报告，不自动追加。真实终局/max240分别列，不能用它替代未来独立整局晋升。冠军不覆盖。

## 预算与原件

采集2400秒、训练1200秒、评测1200秒、总5400秒硬预算；新目录/锁/单入口防重复，caffeinate防空闲睡眠不阻止合盖。源码/native/历史/父模型/旧数据/样本hash归档，独立重放标签/轨道/预测/trace/参数/gate/probe，中文图报告，main commit/push。GUI候选指针仅在final评测后写，role=experimental、calibrated=false、development_pass如实，不切换棋手。

执行前接口修正记录：首入口在旧policy-only样本兼容检查结果返回前被启动，已在历史预检约22秒时停止，未产生schedule/对局/训练。原目录保留stopped_preflight，旧终局字段不进入新WDL标签。新输出目录wdl121_loop_20261107_run1；保守扣除60秒预检预算，不重置总预算。最终GUI默认指针仍写原目录新文件，指向run1真实候选且明确实验未校准。新增旧数据适配与game-uniform prior回归测试；258项测试通过。

第二项预检修正：run1历史检查拒绝了旧搜索/叶值诊断日志，原因是info.paths候选集合被旧读谱器误当连续棋谱。已按完整候选元数据结构分类（真正轨迹仍重放，非法元数据仍阻断），同时修直接121根目录分类假阴性；聚焦四份历史均incomplete=0，证据在run1/history_search_metadata_diagnosis.json。修后从新目录wdl121_loop_20261107_run2执行；保守累计扣600秒预检预算，原两目录不覆盖，无先前开局/自弈/训练样本需重采。首轮数据/训练/验收方案不变。
