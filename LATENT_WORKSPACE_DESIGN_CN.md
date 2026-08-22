# 持久 Latent Workspace v1

## 定位

这不是 MLA，也不是挂在最终 2048 维特征后的 attention neck。它是与 ResNeSt 正常视觉流并行的跨阶段隐式工作区：低层提供观察，latent 在多个尺度间保留状态，高层既继续读取当前视觉特征，也把修正写回 CNN 主流。

## 冻结结构

```text
image → stem → C2 ───────────────→ C3' ─────────→ C4' ─────────→ C5' → GeM/BN → 2048-D
                  │                ▲  │           ▲  │           ▲
                  │ read-only      │  │ read      │  │ read      │
                  ▼                │  ▼           │  ▼           │
              [8 persistent latent slots, 192-D, one shared Read–Mix–Write cell]
                                   │ write        │ write        │ write
                                   └──────────────┴──────────────┘
```

- C2：平均池化 2 倍后读入，不写回；
- C3/C4/C5：各自 1×1 投影到 192 维，执行 latent cross-attention、latent self-attention/FFN，再由视觉 token 查询 latent 并投影写回；
- 一个共享 cell 在四个阶段循环复用，latent 状态持续传递；四个可学习 stage embedding 标识尺度；
- 8 slots、192 dim、8 heads、MLP ratio 4；
- C3/C4/C5 各有独立标量门，`g=0.05+0.95*sigmoid(a)`，初始化为 0.10；
- CNN 残差主路始终存在，最终 head、描述符维度、推理匹配和三项原损失不变。

## 训练边界

`workspace` 是 meta architecture 的顶层子模块，不放进 `backbone`。现有 `FREEZE_LAYERS: [backbone]` 只冻结 ResNeSt；workspace 与 heads 一样从 step 0 使用 `BASE_LR=3.5e-4` 和原 warmup。v1 不降低 latent 学习率，不使用辅助 loss，也不改变 identity sampler。

健康指标每 200 iter 写入 `metrics.json`：

- `latent/c3_gate`、`c4_gate`、`c5_gate`；
- 三阶段 `write_ratio`；
- `grad_norm`、梯度非零元素比例、有限梯度比例、参数梯度覆盖率；
- `slot_variance`、`slot_cosine`。

早期 slot cosine 很高并不单独等于死亡；只有它与 slot variance≈0、梯度覆盖/非零比例≈0 或 write ratio≈0 同时出现时才判定结构失活。先监控，不用额外多样性损失强行制造差异。

## 实现与入口

- 核心：`fastreid/modeling/meta_arch/latent_workspace.py`；
- 分阶段 ResNeSt：`fastreid/modeling/backbones/resnest.py`；
- 健康钩子：`pet_id/latent_hooks.py`；
- 正式配置：`configs/modern_latent_workspace_s101_224.yaml`；
- 回归测试：`tests/test_latent_workspace.py`。

```powershell
.\scripts\run_modern_pipeline.ps1 -Mode LatentSmoke
.\scripts\run_modern_pipeline.ps1 -Mode LatentTrain
.\scripts\run_modern_pipeline.ps1 -Mode LatentResume

D:\CondaData\envs\torch312\python.exe .\scripts\benchmark_latent_workspace.py
```

## 已完成验收

- 原 ResNeSt 普通 forward 与分阶段 forward 逐元素一致，最大绝对误差 0；
- 所有 workspace 参数均能获得有限非零梯度；三个门初值均为 0.10；
- workspace 在 backbone freeze 期间属于 normal 参数组，学习率与 heads 相同；
- checkpoint 可完整恢复 latent，真实 trainer resume 可恢复 optimizer/schedulers/scaler 并从下一 epoch 开始；
- dim-192 真实 smoke 完成 AMP 训练、CE/Triplet/Circle、验证、best/final checkpoint；
- RTX 4060 Laptop batch-28 A/B：baseline 3,836 MiB / 0.331s，latent 4,215 MiB / 0.369s。

尚未完成的是科学效果验收：必须跑完相同 5,400/600 身份划分、seed、batch 和 35 epoch 的 baseline/latent 对照后，才能判断 AUC 是否有增益。
