# 面向速度估计故障的 UAV 跟踪 Robust Residual PPO

[English Version](README.md)

这是一个**仿真型强化学习控制研究项目**：面向无人机对移动地面目标的跟踪任务，对比经典 **PD+前馈**、**Direct PPO**、**Residual PPO**，并进一步通过针对速度估计故障的 domain randomization 训练 **Robust Residual PPO**。

核心问题是：

> **能否保留经典控制器的结构与平滑性，同时让受限的 Residual RL 学到足够的修正能力，使系统在前馈速度估计被污染时仍保持较强鲁棒性？**

<p align="center">
  <img src="plots/phase6/fig1_method_overview.png" width="900" alt="方法总览">
</p>

---

## 核心结果

将显式前馈路径中的目标速度估计污染建模为：

`v_hat_UGV = v_UGV + alpha (v_UAV - v_UGV)`

Robust Residual PPO 从头训练，每个 episode 随机采样：

`alpha ~ Uniform(0, 0.75)`

并将 `alpha=1.0` 完全留作训练范围之外的 held-out severe severity。

在 `alpha=1.0` 下：

| 指标 | Nominal Residual PPO | Robust Residual PPO | 变化 |
|---|---:|---:|---:|
| 成功率 | 58% | **95%** | **+37 个百分点** |
| 稳态 RMSE | 0.0650 m | **0.0379 m** | **−41.6%** |
| 前向超调 | 0.187 m | **0.123 m** | **−34.3%** |

<p align="center">
  <img src="plots/phase6/fig5_targeted_robust_training.png" width="900" alt="鲁棒训练结果">
</p>

---

## 我在这个项目中完成的工作

- 构建 **2D UAV–UGV 跟踪仿真器** 和 Gymnasium 环境，并加入初始状态 / 目标速度随机化。
- 实现 **PD+前馈、Direct PPO、Residual PPO** 三种控制架构。
- 设计 Residual PPO：以结构化 PD+前馈为主控制路径，每个轴限制 **±0.15 m/s** 的学习修正权限。
- 将工程中“速度估计污染导致前馈补偿异常”的问题抽象为可控的 **velocity-coupling fault model**。
- 从头训练 Robust Residual PPO，训练故障范围为 `alpha ~ U(0,0.75)`，并严格保留 `alpha=1.0` 作为 held-out severity。
- 实现位置/速度噪声、观测延迟、临时视觉丢失、未见目标运动等 robustness stress tests。
- 构建固定 seed 的 **100-episode 配对评估流程**，保留 episode-level 原始 CSV 结果。
- 构建最终 Phase 6 分析层：一致性校验、统一 master results、7 张核心图、论文级表格和证据链报告。

更详细的“个人工作 vs 第三方库”边界见 [`docs/CONTRIBUTIONS.md`](docs/CONTRIBUTIONS.md)。

---

## 正常工况下三种架构的权衡

| 方法 | 成功率 | 稳态 RMSE | 捕获时间 | Command smoothness |
|---|---:|---:|---:|---:|
| PD+FF | 100% | 0.0558 m | 5.569 s | **0.000010** |
| Direct PPO | 100% | **0.0063 m** | **3.179 s** | 0.002325 |
| Residual PPO | 100% | 0.0173 m | 3.884 s | 0.000036 |

<p align="center">
  <img src="plots/phase6/fig2_nominal_tradeoff.png" width="900" alt="正常工况对比">
</p>

Direct PPO 在正常工况下最快、误差最低，但动作明显更活跃；Residual PPO 形成了更居中的 **精度–捕获速度–平滑性折中**。

---

## Zero-shot 鲁棒性评估

<p align="center">
  <img src="plots/phase6/fig3_zero_shot_robustness_map.png" width="900" alt="Zero-shot 鲁棒性图">
</p>

严重工况代表性结果：

| 工况 | Direct PPO 成功率 | Residual PPO 成功率 |
|---|---:|---:|
| 位置噪声 `sigma_p=0.10 m` | 86% | **100%** |
| 速度噪声 `sigma_v=0.10 m/s` | 79% | **99%** |
| 观测延迟 `400 ms` | 0% | **100%** |
| Constant turn `0.25 rad/s` | 53% | **100%** |

速度估计故障仅作用于显式 feedforward path，因此 Direct PPO 没有等价故障通路，在对应实验中记为 `N/A`。

---

## 故障强度扫描

<p align="center">
  <img src="plots/phase6/fig4_velocity_fault_sweep.png" width="900" alt="故障强度扫描">
</p>

随着 `alpha` 增强，目标速度前馈估计越来越被 UAV 自身速度污染，结构化基线 / nominal residual controller 出现明显前向超调并最终失效。

---

## 连续 3D 鲁棒性 Landscape

基于 `alpha` sweep 的配对 episode-level 数据，可以进一步把 Robust Residual PPO 相对 Nominal Residual PPO 的收益表示为一个连续三维曲面：

<p align="center">
  <img src="plots/phase6/fig7_3d_robustness_landscape.png" width="900" alt="连续 3D 鲁棒性 landscape">
</p>

三维坐标含义为：

- X：velocity-coupling fault severity `alpha`
- Y：随机化后的 UGV target speed
- Z：`Nominal Residual PPO steady RMSE − Robust Residual PPO steady RMSE`

因此 Z 轴为正表示 robust policy 的稳态误差更低。该图直接使用 `robust_residual_alpha_sweep_raw.csv` 中的配对原始评估数据：每个测试 `alpha ∈ {0, 0.25, 0.50, 0.75, 1.0}` 均为固定的 100 个 matched seeds。每个 alpha 下按照目标速度划分为 8 个区间并计算配对均值，再进行**仅用于可视化的线性插值**，并不代表额外进行了未记录的连续网格实验。

40 个实际 alpha×speed 单元的平均 steady-RMSE reduction 均为正，且最大的收益主要出现在 held-out `alpha=1.0` 附近。图中使用的 40 个经验单元均值同时保存于 `results/phase6/fig7_robustness_landscape_cells.csv`。

---

## Targeted Robust Training 的跨扰动结果

<p align="center">
  <img src="plots/phase6/fig6_cross_robustness_tradeoffs.png" width="900" alt="跨扰动验证">
</p>

Targeted training 在 8 个 cross-validation condition 中有 7 个 primary tracking error 改善，但仍存在可识别 trade-off：

- 高速度噪声：成功率 `99% → 97%`
- constant-turn：primary error `+8.1%`
- 严重位置噪声：command smoothness cost `+118.6%`

因此合理结论是：**broad positive transfer with identifiable trade-offs**，而不是“所有扰动下全面提升”。

---

## 复现

安装：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

不重新训练、直接基于冻结结果重建最终分析：

```bash
python -m analysis.run_phase6
```

训练 Robust Residual PPO：

```bash
python -m train.train_robust_residual_ppo
```

评估 nominal vs robust residual PPO：

```bash
python -m evaluation.evaluate_robust_residual_ppo
```

---

## 项目结构

```text
robust-residual-ppo-uav-tracking/
├── controllers/
├── envs/
├── train/
├── evaluation/
├── analysis/
├── models/
├── results/
├── plots/phase6/
├── tables/phase6/
├── docs/
├── requirements.txt
├── NOTICE.md
└── README.md
```

---

## 研究边界

这是一个**仿真控制研究**，不直接声称学习策略已经完成 sim-to-real 实机部署。

必须保留的解释边界包括：

- velocity fault 只污染显式 feedforward velocity estimate；该实验中 PPO observation 仍保持真实状态。
- `alpha=1.0` 是同一 fault family 的 held-out severity，不是完全未见过的 fault type。
- temporary vision loss 使用 last-valid-target hold，且当前目标仍为恒速。
- 个别 delay 下 RMSE 下降不能解释为“延迟有益”。
- 不应只报告 Full RMSE；需要同时结合 success、condition-specific error、stable ratio、overshoot 和 command smoothness。

详细内容见 [`docs/RESULTS.md`](docs/RESULTS.md)、[`docs/EXPERIMENT_PROTOCOL.md`](docs/EXPERIMENT_PROTOCOL.md) 和 [`docs/PHASE6_ANALYSIS.md`](docs/PHASE6_ANALYSIS.md)。
