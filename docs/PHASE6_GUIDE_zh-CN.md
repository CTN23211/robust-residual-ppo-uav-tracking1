# Phase 6 使用说明：从完整实验到最终论文级证据链

## 一、现在不再训练模型

Phase 6 的任务是**冻结已有实验结果并重组证据**。当前数据已经覆盖：

- Nominal：PD+FF / Direct PPO / Residual PPO
- Phase 5.1：velocity-estimation / feedforward fault
- Phase 5.2：position / velocity observation noise
- Phase 5.3：observation delay
- Phase 5.4：temporary vision loss
- Phase 5.5：speed step / lateral sine / constant turn
- Phase 5B-1：targeted robust training，训练 `alpha ~ U(0, 0.75)`，测试 held-out `alpha=1.0`
- Phase 5B-2：cross-robustness validation

Phase 6 不修改任何 Phase 5 CSV，不重新选择 severity，也不针对已看到的测试结果继续调模型。

## 二、一条命令完成 Phase 6

在 PowerShell 中：

```powershell
conda activate uav_rl
cd robust-residual-ppo-uav-tracking
python -m analysis.run_phase6
```

程序会自动：

1. 检查所有 Phase 5 CSV 是否齐全；
2. 检查行数是否和既有实验设计一致；
3. 检查多个 evaluator 的 nominal 结果是否一致；
4. 检查 `alpha=1.0` 是否仍然只属于 held-out severity；
5. 建立统一 master results；
6. 生成 6 张核心图；
7. 生成 3 张逻辑核心表（Table 3 分 A/B 两部分）；
8. 自动生成 Phase 6 核心结论说明。

## 三、最终核心文件

### 1. 数据总表

```text
results/phase6/phase6_master_results.csv
```

统一字段包括：phase、family、condition、severity、method、success、RMSE、condition-specific primary error、stable ratio、capture/recovery、overshoot、physical command smoothness 等。

### 2. 校验报告

```text
results/phase6/phase6_validation_report.txt
```

如果这里出现 `FAIL`，不要继续使用最终图表。

### 3. 自动结果说明

```text
results/phase6/PHASE6_KEY_FINDINGS.md
```

这是以后写 README、研究报告和论文 Results 部分的最短证据链。

## 四、6 张核心图分别回答什么问题

### Figure 1 — `fig1_method_overview`

回答：**为什么研究 Residual PPO，以及 Robust Residual PPO 到底改了什么？**

图中明确区分：

- PD+FF：`u_base = v_hat_UGV + Kp e_p + Kd e_v`
- Direct PPO：策略直接输出完整 velocity command
- Residual PPO：`u = u_base + Delta u_RL`
- residual authority：每轴 `±0.15 m/s`
- fault model：`v_hat_UGV = v_UGV + alpha(v_UAV - v_UGV)`
- robust training：`alpha ~ U(0, 0.75)`
- held-out severity：`alpha=1.0`

### Figure 2 — `fig2_nominal_tradeoff`

回答：**三种 architecture 在正常情况下分别是什么性格？**

核心结论：

- Direct PPO：最快、精度最高，但控制动作最活跃；
- PD+FF：最平滑，但捕获最慢、稳态误差最大；
- Residual PPO：形成 accuracy / capture / smoothness 中间 Pareto compromise。

### Figure 3 — `fig3_zero_shot_robustness_map`

回答：**训练完成后直接面对各类 distribution shift，谁最容易失效？**

左侧看 Success Rate；右侧看相对于 matched nominal 的 condition-specific tracking error degradation。

注意：velocity fault 是 controller-path feedforward fault，因此 Direct PPO 对应格为 `N/A`，不能人为给它构造一个“不等价”的故障。

### Figure 4 — `fig4_velocity_fault_sweep`

回答：**最初真实工程问题对应的 velocity-estimation contamination 如何随着 alpha 增强而恶化？**

核心看：

- success；
- forward overshoot；
- steady RMSE。

这是“工程问题 → 数学 fault model → 仿真 failure reproduction”的核心图。

### Figure 5 — `fig5_targeted_robust_training`

回答：**只使用 `alpha<=0.75` 训练，能否泛化到没训练过的 `alpha=1.0` severity？**

headline：

- success：58% → 95%；
- steady RMSE：降低约 41.6%；
- forward overshoot：降低约 34.3%。

图中 `alpha=0.75` 是 training-severity boundary，`alpha=1.0` 应写作 held-out severity / severity extrapolation。

### Figure 6 — `fig6_cross_robustness_tradeoffs`

回答：**解决 velocity fault 后，有没有把其他能力练坏？**

当前结果：

- 8 个 cross-validation condition 中，7 个 primary tracking error 下降；
- high velocity noise success 99% → 97%，是明确的小幅 reliability regression；
- constant turn primary error +8.1%，是主要 accuracy trade-off；
- severe position noise command smoothness cost +118.6%，说明 Robust PPO 在强 noisy observation 下控制更活跃。

所以结论应该是：**broad positive transfer with identifiable trade-offs**，而不是“所有条件都全面提高”。

## 五、3 张核心表

### Table 1 — Nominal Controller Comparison

位置：

```text
tables/phase6/table1_nominal_controller_comparison.*
```

回答三种 controller 的正常性能和 smoothness trade-off。

### Table 2 — Severe-endpoint Zero-shot Robustness

位置：

```text
tables/phase6/table2_zero_shot_severe_robustness.*
```

只保留 Phase 5A 每个扰动族预先定义的 severe endpoint，避免正文塞入几十个 severity 点。

### Table 3 — Robust Training + Cross Robustness

位置：

```text
tables/phase6/table3a_targeted_robust_training.*
tables/phase6/table3b_cross_robustness.*
tables/phase6/table3_robust_training_and_cross_robustness.md
```

Table 3A 回答 targeted training 是否有效；Table 3B 回答是否产生 cross-perturbation regression。

## 六、正文 Results 建议结构

不要按 Phase 1、Phase 2、Phase 3 流水账写。推荐：

```text
4.1 Nominal Tracking Performance
4.2 Zero-shot Robustness under Perception and Control Perturbations
4.3 Generalization to Unseen Target Motion
4.4 Failure Analysis under Velocity-Estimation Corruption
4.5 Targeted Robust Training and Cross-Robustness
```

核心故事是：

```text
真实工程前冲问题
    ↓
三种 architecture 对比
    ↓
Residual PPO 获得较好的 robustness-smoothness compromise
    ↓
系统性 zero-shot stress test
    ↓
锁定 severe velocity-estimation fault 为主要 failure mode
    ↓
targeted domain randomization: alpha ∈ [0, 0.75]
    ↓
held-out alpha=1.0: success 58% → 95%
    ↓
cross-robustness 检查
    ↓
大部分正迁移 + 明确报告 observation-noise / constant-turn trade-off
```

## 七、必须保留的方法学边界

1. **Phase 5.1 不是完整 sensor corruption。** 它只污染显式 feedforward estimate，Residual PPO 的 observation 在该实验中仍是真实状态。
2. **`alpha=1.0` 不是 completely unseen fault。** 它是相同 fault family 的 held-out severity。
3. **Vision loss 是 last-valid-target hold。** UGV 在当前测试中仍为 constant velocity，因此不能直接声称覆盖了“目标在视觉丢失时突然机动”的全部情况。
4. **不要写 delay improves performance。** 某些 delay 下 RMSE 下降只说明当前闭环组合产生了特定响应。
5. **不要只报告 Full RMSE。** Full RMSE 会被初始追赶 transient 主导，应同时报告 condition-specific error、success、stable ratio、forward overshoot 和 command smoothness。
6. **Phase 6 后冻结结果。** 不再根据 held-out test 表现调整 alpha 上限、severity 或模型权重，否则会破坏 held-out evaluation 的完整性。

## 八、如果以后正式投稿，再考虑的 Supplementary 内容

主文只保留 6 图 3 表。已有 Phase 5 原始 sweep 图可以进入 Supplementary：

- full observation-noise curves；
- full delay curves；
- full temporary-vision-loss curves；
- complete unseen-trajectory metrics；
- raw/paired seed-level results；
- training TensorBoard curves。

当前阶段不建议为了增加实验数量再训练新的 multi-disturbance policy。
