# Docker 运维

先设置业务数据根目录；默认值为 `D:/datasets/Zero-TTT`：

```powershell
$env:ZERO_TTT_DATA_ROOT = 'D:/datasets/Zero-TTT'
docker compose build
docker compose up -d control data-worker trainer-worker selfplay-worker ui tensorboard
docker compose ps
```

端口仅绑定本机：UI `8080`、Control API `8090`、TensorBoard `6006`。

权威质量门禁：

`test` 复用开发镜像，以只读方式挂载当前源码，不挂载业务数据、不申请 GPU，并禁用网络。
测试缓存写入容器临时目录。开发镜像通过 editable 安装加载工作区所有包；生产镜像使用普通
安装。开发依赖约束与 `uv.lock` 一致，Pyright 所需 Node 已在镜像构建阶段安装。

```powershell
docker compose --profile dev build
docker compose run --rm --no-deps test python -m ruff check .
docker compose run --rm --no-deps test python -m ruff format --check .
docker compose run --rm --no-deps test pyright
docker compose run --rm --no-deps test python -m pytest -q
docker compose run --rm --no-deps test python scripts/check_docs.py
docker compose run --rm --no-deps test python scripts/generate_contracts.py --check
docker compose config --quiet
docker compose run --rm --no-deps test git -c safe.directory=/workspace -c core.autocrlf=true -c core.safecrlf=false diff --check
```

隔离的 Compose 端到端验收使用项目专属命名卷，不挂载正式业务目录、不发布宿主端口。
运行前确认该项目没有需要保留的既有测试数据，并确保正式 GPU 作业没有运行。
先由测试驱动器领取一份 60 秒租约，重启 Control/UI 后验证租约和事件仍然存在，再等待租约
过期、启动真正的 Worker 接管作业。恢复检查需在领取租约后的 60 秒内开始。

```powershell
docker compose -f compose.yaml -f compose.e2e.yaml --project-name zero-ttt-e2e --profile dev run --rm --no-deps dev python scripts/compose_e2e_test.py prepare /datasets
docker compose -f compose.yaml -f compose.e2e.yaml --project-name zero-ttt-e2e up -d --wait control ui
docker compose -f compose.yaml -f compose.e2e.yaml --project-name zero-ttt-e2e --profile dev run --rm --no-deps dev python scripts/compose_e2e_test.py run recovery-start
docker compose -f compose.yaml -f compose.e2e.yaml --project-name zero-ttt-e2e restart control ui
docker compose -f compose.yaml -f compose.e2e.yaml --project-name zero-ttt-e2e up -d --wait control ui
docker compose -f compose.yaml -f compose.e2e.yaml --project-name zero-ttt-e2e --profile dev run --rm --no-deps dev python scripts/compose_e2e_test.py run recovery-check
docker compose -f compose.yaml -f compose.e2e.yaml --project-name zero-ttt-e2e up -d --wait
docker compose -f compose.yaml -f compose.e2e.yaml --project-name zero-ttt-e2e --profile dev run --rm --no-deps dev python scripts/compose_e2e_test.py run bootstrap
docker compose -f compose.yaml -f compose.e2e.yaml --project-name zero-ttt-e2e restart control ui
docker compose -f compose.yaml -f compose.e2e.yaml --project-name zero-ttt-e2e up -d --wait control ui
docker compose -f compose.yaml -f compose.e2e.yaml --project-name zero-ttt-e2e --profile dev run --rm --no-deps dev python scripts/compose_e2e_test.py run alpha
docker compose -f compose.yaml -f compose.e2e.yaml --project-name zero-ttt-e2e --profile dev down -v
```

GPU 验收分别执行驱动/严格 FP32、完整 optimizer step 和并发 MCTS 自博弈：

```powershell
docker compose run --rm dev python scripts/docker_smoke_test.py
docker compose run --rm dev python scripts/model_smoke_test.py --configs configs/profiles/rtx4090l.toml --default-optimizer-steps 1 --accumulation-steps 1 --disable-compile
docker compose run --rm dev python scripts/selfplay_gpu_smoke_test.py
```

Control 重启后会从 `control.sqlite` 恢复作业；过期租约重新排队。Data 与各 GPU Worker 启动
时会重用已提交的内容寻址产物，临时文件不会被当作完成结果。Trainer 与 Self-play 共同竞争
`gpu-exclusive` 租约，因此单 GPU 主机上不会并发运行。

三个 Worker 收到 SIGTERM/SIGINT 后停止发起新的领取请求，继续续租并完成当前有限作业，
包括停止信号到来时已在途领取的作业，然后退出。Docker 停止宽限期统一为 30 分钟；超时被
终止的作业由租约恢复机制接管。用户取消和续租失败仍通过处理器已有的安全边界协作停止。
单纯的事件上报失败不会阻断作业失败终态上报，通信异常可在 Worker 容器日志中查看。

业务目录：

```text
raw/                         用户输入，只读
work/                        可清理的作业临时文件
artifacts/data/              Data 唯一写
artifacts/models/            Trainer 唯一写
artifacts/selfplay/          Self-play 唯一写
state/control/control.sqlite Control 独占
state/data/data.sqlite       Data 独占
```
