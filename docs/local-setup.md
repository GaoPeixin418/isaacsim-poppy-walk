# 本地运行手册：在你的 5070 Laptop 上看 Poppy 走路

> 目标：本地只做**可视化/调试**（这正是最初计划里"本地 8GB 只跑 viewer"的定位），
> 训练仍在云端 3090 上做。本手册从零开始，假设你只有 Isaac Sim（或什么都还没装）。
>
> 写给 Windows + Git Bash 环境（仓库里的 `.sh` 脚本都用 Git Bash 跑）。

## 0. 版本对照表（先对齐，否则检查点加载不上）

| 组件 | 云端（训练机） | 本地需要 |
|---|---|---|
| Isaac Sim | 4.5.0 | **4.5.x**（你已装的话先确认版本） |
| Isaac Lab | v2.1.0 | **v2.1.0**（必须与云端同版，见下） |
| Python | 3.10 | 用 Isaac Sim 自带的，不用另装 |

**为什么 Isaac Lab 必须同版**：检查点（`.pt`）里存的是网络权重，网络结构由
观测/动作维度决定；观测维度由 Isaac Lab 的代码决定。版本不一致会出现
"加载了但维度对不上"的隐晦报错。同版 = 排除一整类问题。

确认 Isaac Sim 版本：打开 Isaac Sim，左下角或 `About` 里看版本号；
或命令行 `"<Isaac Sim 目录>\python.bat" -c "import omni.isaac.version"`。

## 1. 装 Isaac Lab v2.1.0（如果还没装）

```bash
# 找个放代码的地方（不要放在有中文/空格的路径！）
cd D:\
git clone --branch v2.1.0 https://github.com/isaac-sim/IsaacLab.git
cd IsaacLab

# Windows：用官方安装脚本把 Isaac Lab 装进 Isaac Sim 自带的 python
# （脚本会自动找 Omniverse 的 Isaac Sim；找不到时用 --isaac_path 指给它）
.\isaaclab.bat -i
```

装完自检（Git Bash 里）：

```bash
cd /d/IsaacLab
./isaaclab.sh -p -c "import isaaclab; print(isaaclab.__version__)"
```

## 2. 克隆项目仓库 + 装我们的扩展包

```bash
cd /d
git clone https://github.com/GaoPeixin418/isaacsim-poppy-walk.git
cd isaacsim-poppy-walk

# 用 Isaac Lab 的 python 把 poppy_walk 装成开发模式（-e：改代码立即生效）
/d/IsaacLab/isaaclab.sh -p -m pip install -e source/poppy_walk
```

## 3. 重建 USD 资产（仓库里 USD 不进 git，必须本地生成一次）

```bash
cd /d/isaacsim-poppy-walk

# POPPY_PYTHON = Isaac Lab 的 python 入口；ISAACLAB_ROOT = IsaacLab 仓库位置
export POPPY_PYTHON=/d/IsaacLab/isaaclab.sh   # 见下方"坑 1"
export ISAACLAB_ROOT=/d/IsaacLab

bash scripts/rebuild_asset.sh
```

跑完最后会打印 `ASSET_REBUILD_DONE`，且 `assets/poppy/poppy.usd` 存在。
（首次启动 URDF importer 会下载依赖、较慢，几分钟属正常。）

> **坑 1（Windows 特有）**：`run.sh` 里是 `exec "$POPPY_PYTHON" "$@"`，
> 而 `isaaclab.sh` 是个 wrapper（转发到 Isaac Sim 的 python）。
> 如果 `exec` 走不通，改用直接指定 Isaac Sim 自带 python：
> ```bash
> export POPPY_PYTHON="/c/Users/<你>/AppData/Local/ov/pkg/isaac-sim-4.5.0/python.bat"
> ```
> （Omniverse Launcher 的默认安装位置；`python.bat` 虽是 bat，Git Bash 能 exec 它。）

## 4. 放入训练好的检查点

训练产物不进 git，从云端拷。我已把两个里程碑检查点下载到了
`D:\workbuddy\2026-09-14-16-30-34\out\checkpoints\`，拷进你本地仓库：

```bash
cd /d/isaacsim-poppy-walk
mkdir -p logs/rsl_rl/poppy_walk/2026-09-16_d4v2 logs/rsl_rl/poppy_stand/2026-09-16_d3
cp /d/workbuddy/2026-09-14-16-30-34/out/checkpoints/poppy_walk/model_2999.pt \
   logs/rsl_rl/poppy_walk/2026-09-16_d4v2/
cp /d/workbuddy/2026-09-14-16-30-34/out/checkpoints/poppy_stand/model_1199.pt \
   logs/rsl_rl/poppy_stand/2026-09-16_d3/
```

- `poppy_stand/model_1199.pt` = D3 站立（1200 轮训练完成）
- `poppy_walk/model_2999.pt` = D4 v2 行走（**就是"单腿跛行"那一版**，正好用来对照看）

## 5. 播放（这一步会开出真 3D 窗口，可以拖视角）

```bash
cd /d/isaacsim-poppy-walk
export POPPY_PYTHON=...   # 同第 3 步
export ISAACLAB_ROOT=/d/IsaacLab

# 看行走（跛行版）：4 个机器人并排走，窗口里鼠标拖动转视角
bash scripts/run.sh scripts/play_poppy.py --task Poppy-Walk-Play-v0 --num_envs 4

# 看站立：
bash scripts/run.sh scripts/play_poppy.py --task Poppy-Stand-Play-v0 --num_envs 8
```

看的时候重点确认两件事（对应我数据里的结论）：
1. **站立**：晃不晃、会不会慢慢往下蹲（数据说它 87% 时间稳站）
2. **行走（跛行版）**：注意看**左脚是不是全程没落地**、右脚单脚支撑着往前蹭
   —— 这就是奖励漏洞的实物证据，D5 修的就是它

## 6. 常见坑速查

| 症状 | 原因 / 解法 |
|---|---|
| 首次启动卡在 logo 几分钟 | 正常：在编译 shader 缓存，第二次起就快 |
| `No module named 'carb'` | 用了系统 python 而不是 Isaac Sim 的 —— 检查 `POPPY_PYTHON` |
| 找不到 play.py | `ISAACLAB_ROOT` 没设或指错 |
| 窗口开了但机器人不动 | 检查点没放对位置（`logs/rsl_rl/<实验名>/<任意目录>/model_*.pt`），脚本按**轮数最大**自动挑 |
| 显存不够（8GB） | GUI 模式 `--num_envs` 压到 2~4；viewer 本身吃 2GB+ |
| EULA 提示 | `run.sh` 已预置 `OMNI_KIT_ACCEPT_EULA=YES`；不经 run.sh 跑时手动 export |

任何一步报错：把**最后 30 行输出**发我（或存成文件给我路径），我来诊断。
