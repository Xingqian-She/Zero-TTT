# 公共契约

公共调用固定在 `/api/v1`，Worker 租约协议固定在 `/internal/v1`。契约模型使用严格校验，
未知字段会被拒绝；当前信封版本为 v1。权威描述由以下命令从 Pydantic 与 FastAPI 代码生成：

```powershell
docker compose run --rm dev python scripts/generate_contracts.py
```

输出位于 `generated/contracts/openapi.json` 与 `generated/contracts/schemas/*.json`。

服务间只传 `ArtifactRef`。URI 必须使用 `artifact://`，引用同时包含逻辑类型、稳定 ID、
格式版本、SHA-256 和字节数。消费者在打开内容前验证大小与哈希。

Worker 通过能力注册和长轮询领取 `JobEnvelope`。信封包含尝试次数、幂等键、输入引用、
租约令牌和到期时间。完成、失败、续租与事件追加都必须携带该令牌。Control 提供至少一次
执行语义；业务服务依靠内容寻址、作业临时目录和原子提交实现幂等。

工作流使用相同幂等键和相同内容并发提交时返回同一工作流；相同键对应不同模板、Run 或
参数时返回 422。既有 404、409、422 错误保留 JSON `detail`，意外异常返回通用 500 响应，
具体异常仅写入 Control 日志。数据库与产物格式版本保持不变。

事件序号由 Control SQLite 单调分配。客户端可以用分页接口断点读取，也可以通过 SSE 流
持续消费；UI 重启后使用最后持久序号重建视图。
