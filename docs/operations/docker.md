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

S01 的完整验收入口在仓库根目录执行，支持 Windows PowerShell 5.1 和 PowerShell 7：

```powershell
.\scripts\accept_s01.ps1 -Stage all
```

也可分别指定 `cpu`、`gpu`、`e2e`；分阶段运行只报告本阶段结果，不自动合并成完整 S01
通过。脚本按权限／引擎、镜像与依赖、源码、CPU、GPU、E2E 顺序诊断，并使用构建缓存
重建所需镜像。构建需要网络；CPU 检查与独立 GPU smoke 的运行容器禁网。

若 Docker Hub 连接超时，先确认本机已有代理能访问 Docker Hub，再只给当前 PowerShell
设置代理。2026-09-23 本机验收使用如下地址；其他机器按实际监听端口填写，无需修改全局设置：

```powershell
$env:HTTP_PROXY = 'http://127.0.0.1:7897'
$env:HTTPS_PROXY = 'http://127.0.0.1:7897'
.\scripts\accept_s01.ps1 -Stage all
```

每次证据保存到 `tmp/acceptance/s01/<运行编号>/`：`report.json` 记录阶段状态、命令、
退出码、耗时和 pytest 实际结果；`environment.json` 记录源码来源、文件哈希、依赖和
生效配置；其他日志记录 Git 差异、镜像 ID、设备、Compose 和实际容器挂载。
状态区分 `passed`、`failed`、`blocked`、`not_run`（运行中为 `running`）。
Docker 权限、引擎、构建或设备占用问题属于环境诊断，不得据此宣称业务测试失败或通过。

GPU 使用前检查容器预留、计算进程、利用率和显存；Windows 的空闲 C+G 桌面上下文允许
保留，但利用率超过 5% 或显存占用超过 512 MiB 时停止验收。脚本不会停止其他容器或进程。
E2E 每次创建独立项目和专属卷，
不挂正式业务目录、不发布端口。成功后清理本次项目和卷；失败时保存日志、停止本次服务，
保留测试卷，并在 `report.json` 中给出限定到该项目的清理命令。

固定配置为 `configs/acceptance/s01.toml`：19×19、严格 FP32、64 维／2 层小模型、
CUDA 模型及 CPU EMA、batch 2、累积 1、关闭 compile；4 局自博弈、每局最多 2 手、
每手最多 2 次模拟。模型 smoke 执行 1 次预热更新和 1 次测量更新；E2E 冷启动与混合
训练各执行 1 步，使用确定性的 32 局原始 fixture，保证 train／validation 均非空。
这是链路验收预算，不能用于正式吞吐或棋力结论。生产 profile 和 `configs/test.toml` 保持独立。

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

隔离 E2E 入口会由无 GPU 的 `e2e-driver` 准备数据并领取 60 秒租约，重启 Control/UI 后
验证租约和事件仍然存在，等待租约过期后启动 Worker 接管。随后完成 data-bootstrap、
cold-start、alpha-zero-round，并检查本轮训练进度、产物身份、文件大小和哈希。
两次训练实际使用 CUDA，且检查自博弈的 GPU 分配证据。单独复跑命令如下：

```powershell
.\scripts\accept_s01.ps1 -Stage e2e
```

GPU 验收分别执行驱动/严格 FP32、完整 optimizer step 和并发 MCTS 自博弈：

```powershell
.\scripts\accept_s01.ps1 -Stage gpu

# 以下是同一预算的底层命令；执行前同样需要确认 GPU 空闲。
docker compose run --rm --no-deps -T gpu-smoke python scripts/docker_smoke_test.py
docker compose run --rm --no-deps -T gpu-smoke python scripts/model_smoke_test.py --configs configs/acceptance/s01.toml --default-optimizer-steps 1 --baseline-optimizer-steps 1
docker compose run --rm --no-deps -T gpu-smoke python scripts/selfplay_gpu_smoke_test.py --config configs/acceptance/s01.toml
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
