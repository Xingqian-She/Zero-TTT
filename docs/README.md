# Zero-TTT 文档

当前系统文档记录已经实现的能力；开发规划单独列出待开发范围和阶段验收要求。
设计演变由 Git 保存，规划中的目标不代表当前系统已具备对应功能。

## 当前系统

- [系统架构](architecture/overview.md)
- [公共契约](architecture/contracts.md)
- [三条有限流程](workflows/finite-workflows.md)
- [Docker 运维](operations/docker.md)
- [本地不可变对象存储与 SQLite](decisions/0001-local-artifacts-and-sqlite.md)
- [KataGo 输入](integrations/katago.md)

OpenAPI 与 JSON Schema 由代码生成，不在文档中维护重复字段表：

```powershell
docker compose run --rm --no-deps test python scripts/generate_contracts.py --check
```

## 开发规划

- [分阶段开发规划：从可视化训练到局域网教师闭环](roadmap/staged-development.md)

规划按 S01–S28 给出前置条件、必做范围、接口与产物、验收步骤和交接要求。
每次仅实施指定阶段；已实现能力与运行验收结果以当前源码及阶段交接记录为准。
