# Zero-TTT 当前文档

这里仅记录当前已经实现的系统，不承担历史档案职责；设计演变由 Git 保存。

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
