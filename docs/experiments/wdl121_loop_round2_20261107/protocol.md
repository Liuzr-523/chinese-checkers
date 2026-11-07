# WDL第二轮：继承首轮，用新完整对局继续纠错

路线图第3步，2026-11-07，用户授权。CNN/strict121，直接main，无自动晋升。父为首轮run2 candidate_step2000.pt，SHA 0e9d850853771c0b0d0e58d6bcddc7e972378548ea9c8bb4461b6a01cae8c79d。完整继承CNN/策略/WDL全部参数，零步逐值验证，不重置胜率头；AdamW状态新建，明确是参数续训，不冒称完整优化器断点恢复。

iteration2，seed-base112120（初始化112120、schedule112121、自弈112122+group、新采样112123/旧112124）。新目录wdl121_loop_round2_20261107。全历史（含首轮自弈全部中间态、样本、probe）覆盖无缺口后，新56根事前冻结：32train/16development/8probe，prefix随机合法8/9/16/17，root/anchor保守orbit排历史及跨组，不做身份标签对称增强。训练只用本轮新48盘，旧train8/普通train8仅policy KL(v3)，不把首轮dev当本轮train。

父模型固定双方32模拟采48完整自弈，root噪声和前12步按visits采样同首轮。真实终局才赋WDL胜/和/负；限步无value标签。每步保存当时预测与搜索分布。跨split orbit双方样本屏蔽、历史排除，prefix不训练。train至少16个有保留真终局标签的完整game才训练，不按结果补根或换split。

固定2000更新，lr1e-4 AdamW decay1e-4 clip1。新16按game均匀再组内sample，旧8/普通8只保持KL，loss=new softpolicyCE+WDLCE+.1旧KL。旧scalar value参数冻结，其余CNN/策略/WDL可训。保存0/1000/2000，每400仅train；新开发父基线一次/final一次，不择点/扫参。

原三类开发gate、policy保持与参数检查沿用首轮；额外胜负区分gate在数据产生前固定：仅真胜负样本，以pwin/(pwin+ploss)计算binary Brier，按game macro，候选同时比父与train-only玩家先验至少低.005；两玩家binary Brier各不劣父；无真胜负覆盖安全失败。train先验仍按实际game均匀目标，各类加1伪计数，不用开发估prior。三类改进不能仅靠压低和棋概率过关。ECE/每类校准及胜负概率范围另报告；无和棋开发覆盖不得声称和棋校准。

候选存在才按8probe根×2阵营×父WDL和现役v3两对手，共32局固定诊断，双方32模拟无噪声/书库；输入隔离与截断单列，不据小样本晋升。冠军v3不变；GUI独立实验分析仅在开发全gate通过时更新至第二轮，否则保留首轮指针，不用较差候选覆盖。

预算采集2400/训练1200/评测1200/总5400秒，单入口/目录锁/完整hash和trace，失败保存原因不自动重启。中文图报告，独立重放、继承参数、种子/trace、标签和gate复算，成果commit/push。首轮模型及原件不覆盖。

执行前覆盖定义补充：胜负区分正式验收至少需要开发集真实P1胜及P2胜各2个独立game；不能以一盘交替视角天然产生胜/负两种label充当双方向覆盖。不足时gate失败，但不阻止按train就绪门槛完成既定训练，不追加盘数。
