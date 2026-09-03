# 版本与角色命名

本项目同时存在模型代际、部署角色、实验编号、接口版本和文件格式版本。它们不能只用
“V1/V2/V3/V4”互相替代。当前状态固定表述如下：

| 名称 | 当前指向 | 含义 |
|---|---|---|
| 当前研发主线 | `UnifiedPetReID V4` / `unified_pet_reid_v4_v1` | 已完成验证的高分辨率候选；是当前模型代际，但尚未执行生产激活 |
| 生产基线 | `UnifiedPetReID V3` / `unified_pet_reid_v3_v1` | 当前默认服务实际加载的冻结模型 |
| 回滚模型 | `UnifiedPetReID V3` / `unified_pet_reid_v3_v1` | V4 正式激活前后的安全回滚点 |
| 旧兼容家族 | `Semantic V3` / `dogfacenet_semantic_v3_v1` | 旧多分支模型家族，不代表 UnifiedPetReID 的当前代际 |
| 研究路径 | BIFOR、Agent experiment、Controller experiment | 独立实验，不参与 UnifiedPetReID 的连续代际编号 |

因此，“当前是 V4”和“生产默认仍是 V3”可以同时成立：前者描述研发代际，后者描述
已激活的部署指针。没有迁移图库、写入激活记录并复跑生产验收前，不应仅通过改文案把
V4 静默变成生产默认。

## 公共命令使用角色名

`scripts/pet-reid-stack.ps1` 与 `scripts/test-live-stack.ps1` 的公共选择名为：

| `-Model` 值 | 实际运行 |
|---|---|
| `production` | UnifiedPetReID V3 生产基线 |
| `candidate` | UnifiedPetReID V4 当前研发候选 |
| `rollback` | UnifiedPetReID V3 回滚模型 |
| `legacy-semantic` | Semantic V3 兼容模式 |
| `research-bifor` | Semantic V3 + BIFOR 研究路径 |
| `research-agent` | 多专家 Agent experiment |

旧值 `unified-v3`、`unified-v4`、`semantic-v3`、`semantic-v3-bifor`、`agent-v1`
继续作为兼容别名，供历史命令和自动化使用；新文档不再用它们表达部署角色。

## 不属于模型代际的编号

- HTTP 路径中的 `/v1` 是 API 契约版本。
- JSON 中的 `schema_version` 是序列化格式版本。
- 包名末尾的 `_v1` 是该模型包的修订号，例如 `unified_pet_reid_v4_v1` 表示
  “UnifiedPetReID V4 的第 1 个锁定包”，不是把模型叫成 V1。
- `Agent V1/V2`、`Controller V1/V2` 是历史实验编号。新增公共文案统一写成
  `Agent experiment` 或 `Controller experiment`，需要复现实验时再附原始编号。
- `blind_v3.json`、`candidate_lock_v4.json` 等名称属于已消费协议的历史证据，必须保留。

## V4 正式激活边界

V4 的 `deployment_record.json` 记录的是当次验收事实，其中
`validated_optional_candidate` 和 `default_changed=false` 不得回写或改名。正式切换应另建
激活记录，完成 V4 图库迁移或重录、更新 `models/registry.json` 的
`default_deployment`、切换生产入口并复跑 CPU/CUDA 与真实 HTTP 全链路验收。
