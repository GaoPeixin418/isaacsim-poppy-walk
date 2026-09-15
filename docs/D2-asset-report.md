# D2 · Poppy 资产修正报告

> 执行时间：2026-09-15
> 目标：把官方 URDF 变成物理上正确的 Isaac Sim 资产，并定位当年"反关节站不稳"的根因
> 验收标准：手动摆直立姿态能静止不发散（待目视标定，见文末）

---

## 一、结论摘要（先看这个）

| 检查项 | 结果 |
|---|---|
| 网格路径改写（`package://` → 相对路径） | ✅ 52 处 |
| 网格文件名大小写修正 | ✅ 10 处（**这是当年失败的头号嫌疑，见 §3.1**） |
| 碰撞网格可用性 | ✅ 26/26 刚体都有碰撞几何体（convexHull） |
| 关节限位是否被 importer 带进 USD | ✅ 25/25 逐项一致 |
| 膝盖单向限位是否生效 | ✅ 反向余量仅 3.5° |
| 上身锁定 | ✅ 15 关节 revolute → fixed，articulation 25 → **10 DOF** |
| 力矩上限是否被带入 | ✅ 但值是 URDF 数据手册峰值，**必须覆盖**（见 §3.3） |
| 驱动增益单位 | ⚠️ 存在一个未定量的单位边界，**D3 必须实测**（见 §3.2） |

**当年"反关节"的结论更新**：URDF 侧限位完全正确，importer 也正确带入了限位。所以反关节**不是限位丢失**，而是 **PD 增益量级错误**（当年抄大机器人的 kp，对 MX-28 是 50 倍过量）叠加 **Linux 下资产导入失败/几何缺失** 的综合结果。详见 §3。

---

## 二、执行过程与产物

### 2.1 拉取网格（`scripts/d2_fetch_meshes.py`）

官方 URDF 引用的网格写的是 `package://meshes/X.STL` —— `package://` 是 **ROS 的包解析协议**，Isaac Sim 不认。不改写会丢掉全部几何体。

从 `poppy-project/poppy-humanoid` 仓库拉取（不走 git clone，用 GitHub API 列清单 + raw 下载，避免拉下整个 CAD 仓库）：

```
仓库文件总数: 142    STL: 52（26 视觉 *_visual.STL + 26 碰撞 *_respondable.STL）
总大小: 6.13 MB    → assets/poppy/urdf/meshes/
```

### 2.2 改写 URDF（`scripts/d2_patch_urdf.py`）

生成两个版本：

| 文件 | DOF | 用途 |
|---|---|---|
| `Poppy_Humanoid.urdf` → `poppy_full.usd` | 25 | **诊断用**：完整保留所有关节，用来核对限位是否被导入 |
| `Poppy_Humanoid_locked.urdf` → `poppy.usd` | 10 | **训练用**：上身 15 关节改为 fixed 并合并 |

为什么锁定上身：Poppy 的腰腹 5 自由度多关节躯干是其特色，但（a）7 天冲刺不做加法，（b）25 DOF 的 articulation 求解更慢更不稳，（c）不锁的话策略可能学到"扭腰乱甩"的退化解。**代价明确记录在此：放弃躯干自由度。**

### 2.3 转换（`scripts/d2_convert.sh`）

```bash
python scripts/tools/convert_urdf.py <urdf> <out.usd> \
  --merge-joints \
  --joint-stiffness 8.0 --joint-damping 0.3 --joint-target-type position \
  --headless
```

**`--joint-stiffness/--joint-damping` 绝对不能沿用默认的 100 / 1.0** —— 那是给几十上百 N·m 的大机器人用的量级，对堵转 2.5 N·m 的 MX-28 是 40 倍过量，数值直接爆炸。这里取按重力力矩估的保守值（hip_y 未来还需在环境配置里单独提到 20）。

### 2.4 核对（`scripts/d2_inspect_usd.py`）

用 `usd-core` 直接读 USD，把限位、驱动增益、碰撞几何体逐项打印并与 URDF 侧对比。**不需要启动 Kit**（比走 `isaaclab.sh` 快得多）。

---

## 三、四个值得记住的发现

### 3.1 ⭐ 文件名大小写 —— 在 Linux 上直接导致导入失败

官方仓库里 **10 个网格的扩展名是小写 `.stl`**，其余 42 个是大写 `.STL`；而 URDF 里**一律**引用 `.STL`。

```
abs_motors / bust_motors / head / l_foot / l_forearm /
l_hip_motor / l_hip / l_shin / l_shoulder_motor / l_shoulder   ← 这 10 个是小写 .stl
```

Windows / macOS 的文件系统不区分大小写，**这个问题在本地永远不会暴露**；但 Linux 区分大小写，这 10 个文件"不存在"。

**对照实验（实测证据）**：把大小写改回原样再转换一次，结果不是"静默丢几何体"而是**导入直接崩溃**：

```
[Error] Failed to execute a command: URDFImportRobot.
<RuntimeError> Used null prim
→ 产出的 USD：0 个关节、0 个刚体（空壳）
```

所以修好这一步之前，Linux 上根本拿不到能用的资产。**这解释了为什么"用别人成功过的框架能导入成功"却在服务器上跑不出结果** —— 那套框架的资产是在大小写不敏感的环境里验证的。

> 注：最初推测的失效方式是"左腿静默缺碰撞体"，实测失效方式更早、更彻底（直接崩溃）。以实测为准。

修法：不改仓库文件（保持上游一致），而是在 URDF 侧把引用改成磁盘上的实际文件名。

### 3.2 ⭐ 单位边界：USD 存"度"，URDF/Isaac Lab 存"弧度"

核对时第一版脚本报了 **25 处"限位不符"** —— 这是**我脚本自己的假警报**。看比值就清楚：

```
URDF -0.7854 rad  →  USD -45.0000       比值 57.296 = 180/π
URDF -2.3387 rad  →  USD -134.0000      比值 57.296
URDF -0.01745 rad →  USD -1.0000        比值 57.296
```

**这是正确的换算**：`UsdPhysics` 规范规定 revolute 关节的 `lower/upper` 单位是**度**，而 URDF 和 Isaac Lab 内部都用**弧度**。

Isaac Lab 源码里的证据：

```python
# isaaclab/sim/schemas/schemas.py:614
if cfg["stiffness"] is not None:
    # N-m/rad --> N-m/deg
    cfg["stiffness"] = cfg["stiffness"] * math.pi / 180.0

# test/sim/test_schemas.py
# for angular drives, we expect user to set in radians
# the values reported by USD are in degrees
```

**这条边界是一类经典 bug 的温床**：谁把 USD 里读出来的 `-134`（度）当成弧度用，膝盖的限位就从"只能单向弯 134°"变成"能反向弯 134 rad" —— 等于**没有限位**，RL 立刻会找到"反着弯膝盖"这条作弊解。这正是"反关节"最可能的机制之一。

### 3.3 ⚠️ 驱动增益：存在一个尚未定量的单位路径，D3 必须实测

已确认的事实：

- USD 里存的驱动增益是**每度**单位。我们传 `--joint-stiffness 8.0`，USD 里落成 `0.139626`（= 8 × π/180），换算回每弧度正好 8.0 —— **转换器把入参当 N·m/rad 处理，这是对的**。
- Isaac Lab 的 `modify_joint_drive_properties`（USD schema 层）**明确**接受 N·m/rad 并自动转成 N·m/deg。
- 但底层 `Articulation.write_joint_stiffness_to_sim()` **只是把值原样写进 PhysX**（`set_dof_stiffnesses`），源码里没有换算；而 `ImplicitActuator` 正是通过这条路径写入配置里的 `stiffness`。

**这意味着：如果环境配置里写 `stiffness=8.0`，实际写进物理引擎的可能是 8.0（N·m/deg，等效 458 N·m/rad，比 MX-28 强 180 倍 → 必爆）。**

**D3 的第一件事就是用一个 10 分钟实验把这件事钉死**，不要靠推断：

```
实验设计：
1. 用我们的 USD 建 articulation，读 root_physx_view.get_dof_stiffnesses()
   → 期望读到 0.139626（USD 原值）
2. 在 env cfg 里把 actuator stiffness 设为 8.0，创建后读回 robot.data.joint_stiffness
   → 若返回 8.0 说明是"原样写入"，cfg 单位 = N·m/deg
   → 若返回 0.1396 说明有换算，cfg 单位 = N·m/rad
3. 决定性验证：锁住其余关节、关闭重力，给 knee 一个固定的位置误差 δ，
   测量稳态保持力矩 τ，用 kp_eff = τ/δ 反推。
   与 8 N·m/rad 和 458 N·m/rad 对照，落在哪个附近就是哪个。
```

在得到结论前，**不要开始站立训练** —— 这是当年失败的直接原因。

### 3.4 碰撞几何：26/26 全部就位

沿用官方 URDF 自带的 `*_respondable.STL` 作为碰撞网格（不需要自己简化），近似方式全部是 `convexHull`。26 个刚体每个都有且只有 1 个碰撞体，质量也正确带入（例如 `l_foot 0.0468 kg`、`pelvis 0.1852 kg`、`l_thigh 0.1149 kg`）。

---

## 四、产物清单

```
assets/poppy/
  urdf/
    Poppy_Humanoid_orig.urdf        官方原版（未修改，留档）
    Poppy_Humanoid.urdf             修正版（25 DOF）
    Poppy_Humanoid_locked.urdf      修正版 + 锁上身（10 DOF）← 训练用
    Poppy_Humanoid_joints.tsv       URDF 侧关节表（核对基准）
    Poppy_Humanoid_locked_joints.tsv
    meshes/                         52 个 STL（6.13 MB）
  poppy.usd                         训练用资产（gitignore，可重建）
  poppy_full.usd                    诊断用资产（gitignore）
  configuration/                    转换器拆分出的 base/physics/sensor 层
scripts/
  d2_fetch_meshes.py                拉网格
  d2_patch_urdf.py                  改路径 + 修大小写 + 锁上身
  d2_convert.sh                     转换
  d2_inspect_usd.py                 核对限位/增益/碰撞
```

**为什么 USD 不进 git**：它是 URDF + mesh 的函数，可一条命令重建；二进制大文件进版本库会让仓库变重，也不利于 diff。仓库里保留的是"源"（URDF + 脚本 + mesh），复现方式写在 README（D7 补）。

---

## 五、待完成（D2 收尾）

1. **驱动增益单位实测**（§3.3）—— D3 第一步
2. **rest pose 目视标定**（需你确认，我看不到画面）：
   - 屈膝直到脚跟微抬（右膝往负方向、左膝往正方向）
   - 调 hip_y 前倾补偿，再调 ankle_y 让脚掌贴平
   - 确认：双脚平放、躯干竖直、重心投影落在脚掌内
   - **数值必须目视标定，不要纸上推算**（关节轴向约定要看 mesh 才能确定）
3. 在 viewer 里逐关节手动给目标角，确认方向正确、膝盖不能反弯

标定完成后 D2 才算真正结束，进入 D3 站立训练。

---

## 六、复现命令

```bash
# 1. 拉网格
python scripts/d2_fetch_meshes.py

# 2. 改写 URDF（两个版本）
python scripts/d2_patch_urdf.py --mode full
python scripts/d2_patch_urdf.py --mode lockupper

# 3. 转换（需要 Isaac Sim 环境，设 OMNI_KIT_ACCEPT_EULA=YES）
bash scripts/d2_convert.sh

# 4. 核对
python scripts/d2_inspect_usd.py assets/poppy/poppy.usd \
  --ref assets/poppy/urdf/Poppy_Humanoid_locked_joints.tsv
```
