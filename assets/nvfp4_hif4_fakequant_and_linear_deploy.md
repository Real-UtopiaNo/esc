# NVFP4 / HiF4 Fake Quant 实现与线性层实装指南

本文说明本仓库中两种 4-bit 块浮点格式的 **伪量化（fake quant）** 实现细节；用仅含线性层的小模型实操一遍 NVFP4 / HiF4；并说明在 **内网仅有 CPU、无 Blackwell GPU** 时如何在线性层落地可交付产物。

## 内网复现：需要安装的库（演示脚本）

只复现 **`scripts/demo_linear_block_quant.py`**（CPU 伪量化演示）时，**不需要 GPU / CUDA / TensorRT**。除本仓库源码外，Python 环境最少安装：

| 包 | 用途 | 说明 |
|----|------|------|
| `torch`（CPU 版即可） | `nn.Linear`、`F.linear`、张量运算 | 建议 ≥ 2.1；需支持 `torch.float8_e4m3fn`（NVFP4 块 scale 用）。内网无 GPU 时装 CPU wheel 即可 |
| `numpy` | `pytorch_quantization.tensor_quant` 导入依赖 | |
| `PyYAML`（`pyyaml`） | 同上 | `pip` 包名 `pyyaml`，`import yaml` |
| `absl-py` | `pytorch_quantization` 包初始化会 `import absl` | 即便演示只调 `fake_*_quant`，走包导入仍需要 |

示例（按内网镜像源改 URL）：

```bash
# CPU 版 PyTorch：以 https://pytorch.org 当时提供的 CPU 命令为准，例如：
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy pyyaml absl-py
```

仓库侧：

```bash
# 把仓库放到内网机器后，在仓库根目录：
export PYTHONPATH=/path/to/esc   # Windows: set PYTHONPATH=D:\path\to\esc
python scripts/demo_linear_block_quant.py
```

需要带上的源码目录（不必装整个语音评测栈）：至少  
`pytorch_quantization/tensor_quant.py`（及该包的 `__init__.py` 等导入链）、`scripts/demo_linear_block_quant.py`。

**不需要**为跑通演示而安装：`torchaudio`、`librosa`、`jiwer`、`cma`、`pesq`、`torch_tensorrt` 等（那些是 Conformer / ESC 评测用的）。

本机容器内已跑通示例：

```bash
# 容器 mpsenet，工作目录 /workspace/esc
PYTHONPATH=/workspace/esc python scripts/demo_linear_block_quant.py
# exit 0；打印 FP32 / 动态 NVFP4 / HiF4 的输出与误差
```

相关代码：

| 文件 | 作用 |
|------|------|
| `pytorch_quantization/tensor_quant.py` | `fake_block_quant` / `fake_nvfp4_quant` / `fake_hif4_quant` |
| `pytorch_quantization/nn/modules/tensor_quantizer.py` | `use_block_quant` / `use_hif4` / `_quant_forward` |
| `utils/quant_utils.py` | `set_block_quant`、`_is_gemm`、分块轴选择 |
| `inference.py` | `--quant_format {nvfp4,hif4}`、`--coverage {all,gemm}` |
| `tests/test_nvfp4.py`、`tests/test_hif4.py` | 与 ModelOpt / Algorithm 1 对齐的单元测试 |
| `scripts/demo_linear_block_quant.py` | 仅 Linear 小模型上的 NVFP4 / HiF4 伪量化演示（CPU） |

**重要前提：** 本仓库全程是 **quantize → dequantize → 仍用 FP32/TF32 GEMM**。没有调用 Tensor Core 的 NVFP4 / HiF4 算子。精度数字是“格式误差上界估计”，不是硬件吞吐结果。

---

## 1. 公共流水线（两种格式共用）

### 1.1 如何挂到模型上

1. `quant_modules.initialize()` 把 `nn.Linear` / `Conv*` / `LayerNorm` 等替换成带 `input_quantizer` / `weight_quantizer` 的量化模块。
2. `set_block_quant(model, coverage, element)` 遍历带量化器的模块：
   - `coverage="gemm"`：非 GEMM 层调用 `quantizer.disable()`（直通 FP32）。
   - `element="e2m1"` → NVFP4；`"hif4"` → HiF4；`"int4"` → 同结构但元素为 INT4（对照实验用）。
3. 每个量化层：**先 fake-quant 输入，再 fake-quant 权重，再做普通 matmul/conv**；输出本身不量化，由下一层输入量化器负责。

### 1.2 GEMM 判定与分块轴

```text
_is_gemm:
  QuantLinear                         → True
  QuantConv1d/2d 且 groups==1
    且 in_channels>1                  → True
  其余（LN / GLU / BN / depthwise /
        首层 in_channels=1 Conv2d）   → False
```

分块轴（`_block_input_axis` / 权重）：

| 张量 | 典型形状 | `block_axis` | 含义 |
|------|----------|--------------|------|
| Linear 激活 | `[..., in]` | `-1` | 沿最后一维（与 K 同向）每 16/64 个一组 |
| Linear 权重 | `[out, in]` | `-1` | 沿 `in` 分块 |
| 稠密 Conv 激活 | `[B,C,...]` | `1` | 沿 channel |
| Conv 权重 | `[out, in, k...]` | `None` | 展平 `in×k…` 后再按块量化 |

不足一块的尾部 **零填充**，算完再裁回原长度。

### 1.3 校准差异（入口行为）

| 格式 | 全局 / 外层 scale | 本仓库校准 |
|------|-------------------|------------|
| NVFP4 | 需要 per-tensor `global_amax`（静态存进 quantizer） | `--calib max`，校准后写入 `scale/*.pkl` |
| HiF4 | **无**静态全局 amax；三级 scale 全从当前张量算 | `inference.py` **跳过** `calibrate_model` |

CLI 示例：

```bash
# NVFP4，仅 GEMM
python inference.py --model_name conformer --quant_format nvfp4 --coverage gemm --calib max

# HiF4，仅 GEMM（无需 calib 参数起作用）
python inference.py --model_name conformer --quant_format hif4 --coverage gemm
```

---

## 2. NVFP4 Fake Quant 细节

实现：`fake_block_quant(..., element="e2m1")`，别名 `fake_nvfp4_quant`。  
与 NVIDIA ModelOpt `NVFP4QTensor` 在单元测试中 **逐元素一致**。

### 2.1 格式组成

| 层级 | 位宽 / 类型 | 粒度 | 作用 |
|------|-------------|------|------|
| 元素 | FP4 **E2M1** | 每元素 | 数值网格 |
| 块 scale \(s_b\) | FP8 **E4M3** | 每 **16** 元素 | 块内归一化 |
| 全局 scale \(s_g\) | FP32 | 每张量一个 | 由校准 `global_amax`（记为 \(\beta\)）推出 |

E2M1 取值（含符号）：

\[
\{0,\ \pm0.5,\ \pm1,\ \pm1.5,\ \pm2,\ \pm3,\ \pm4,\ \pm6\}
\]

最大幅度 `E2M1_MAX = 6`；E4M3 最大有限值 `E4M3_MAX = 448`。

### 2.2 计算公式（本仓库实现）

1. 全局 scale：

\[
s_g = \frac{\beta}{6 \times 448}
\]

其中 \(\beta =\) `global_amax`（权重 / 激活各自一份；激活默认静态 Max，也可改成动态 per-tensor）。

2. 沿 `block_axis` 切成 16 元块，块内绝对值最大为 \(a_b\)：

\[
s_b = \mathrm{cast}_{\mathrm{E4M3}}\Big(\mathrm{clamp}\big(\tfrac{a_b}{6\cdot s_g},\ 2^{-9},\ 448\big)\Big)
\]

全零块时先把未量化的 \(s_b\) 置 1，再进 E4M3。

3. Fake quant（量化再反量化）：

\[
\hat x = \mathrm{round}_{\mathrm{E2M1,RNE}}\Big(\frac{x}{s_b s_g}\Big)\cdot s_b s_g
\]

`_round_to_e2m1`：按边界表就近映射，中点 **round-half-to-even**（对若干“向上”的中点单独处理），超过 6 饱和到 6。

### 2.3 代码路径

```text
TensorQuantizer.use_block_quant(axis, "e2m1")
  → _quant_format = "block"
  → MaxCalibrator（axis=None，收全局 amax）

forward:
  amax = _get_amax(inputs)          # 校准后的静态值，或动态
  fake_block_quant(inputs, amax, "e2m1", block_axis)
```

### 2.4 与真实硬件的对应关系

Blackwell Tensor Core / cuBLASLt / TensorRT 的 **NVFP4 block-scaled GEMM** 使用同一套“E2M1 + 每 16 元 E4M3 scale +（可选）张量级 scale”语义。本仓库的 fake quant 用来估精度；真正加速需要导出 packed 权重与 scale，并调用对应 GEMM API（见第 5 节）。

---

## 3. HiF4 Fake Quant 细节

实现：`fake_hif4_quant`，对应论文 [arXiv:2602.11287](https://arxiv.org/abs/2602.11287) **Algorithm 1**。单元测试与手写参考实现对齐。

### 3.1 格式组成（相对 NVFP4 的关键差异）

| 项目 | NVFP4 | HiF4 |
|------|-------|------|
| 块大小 | 16 | **64** |
| 元素 | E2M1（最大 ±6） | **S1P2**（≡ E1M2，步进 0.25，最大 **±1.75**） |
| Scale 结构 | 全局 FP32 + 块 E4M3 | **三级全动态**：E6M2（块）+ E1×8 + E1×16 |
| 静态校准 | 需要 `global_amax` | **不需要** |

层次含义（每个 64 元块）：

1. \(v_{16}\)：每 4 元一组取 max → 16 个  
2. \(v_8\)：再两两合并 → 8 个  
3. \(v_{\max}\)：块内总 max  
4. 块 scale（unsigned **E6M2**）：\(s_f = v_{\max}/7\)，再 `_to_e6m2`（无次正规；最大有限约 \(1.5\cdot 2^{15}\)）  
5. **E1_8**：`v8 / s ≥ 4` → 0/1，控制 \(2^{\mathrm{E1\_8}}\)  
6. **E1_16**：在 E1_8 展开后，`v16` 再判 `≥ 2` → 0/1  
7. 元素：\(x' = x / (s\cdot 2^{\mathrm{E1\_8}+\mathrm{E1\_16}})\)，再 `_round_to_s1p2`  
8. 重建：\(\hat x = s\cdot 2^{\mathrm{E1\_8}+\mathrm{E1\_16}}\cdot \mathrm{S1P2}(x')\)；全零块输出 0  

块内可表示峰值与论文一致：\(2^{1+1}\times 1.75 = 7\)（`HIF4_INTRAGROUP_MAX`）。

### 3.2 代码路径

```text
TensorQuantizer.use_hif4(axis)
  → _quant_format = "hif4", _calibrator = None

forward:
  fake_hif4_quant(inputs, block_axis)   # 不读 amax
```

### 3.3 部署注意

HiF4 面向 **昇腾等** 侧的专用格式与算子；**不是** Blackwell 原生 NVFP4。在 NVIDIA 卡上只能继续用本仓库这类 fake quant 做精度摸底，不能指望 TensorRT 的 NVFP4 GEMM 直接吃 HiF4 权重。

---

## 4. 演示：仅线性层小模型上的 NVFP4 / HiF4 伪量化

脚本：[`scripts/demo_linear_block_quant.py`](../scripts/demo_linear_block_quant.py)（**纯 CPU**，不依赖 GPU）。

**本演示全部动态，不做静态校准。** NVFP4 的 `global_amax` 在每次调用时取当前张量的 `abs().amax()`；HiF4 本身无全局 amax。这对应仓库里 `--act_dynamic_tensor` 对激活全局 scale 的做法，权重侧演示里也同样在线取 max（部署时可改回离线静态权重 amax）。

### 4.1 模型

三层 MLP，只有 `nn.Linear` + `ReLU`（无卷积 / Norm / 注意力）：

```text
x[B,64] → Linear 64→128 → ReLU → Linear 128→128 → ReLU → Linear 128→16 → y[B,16]
```

伪量化只作用在每层的 **输入激活** 与 **权重** 上；`bias` 与 `ReLU` 保持 FP32；矩阵乘仍是 `F.linear`（FP32）。

### 4.2 跑法

```bash
# 仓库根目录；容器内示例：
PYTHONPATH=. python scripts/demo_linear_block_quant.py
# 可选: --seed 0 --batch 8 --d-in 64 --d-hidden 128 --d-out 16
```

### 4.3 流程对照（脚本 Step 1–3）

| 步骤 | NVFP4（动态） | HiF4（动态） |
|------|---------------|--------------|
| 校准 | **无**；`β = absmax(当前张量)` | **无** |
| 单层 | `fake_nvfp4(x, absmax(x))`，`fake_nvfp4(W, absmax(W))`，再 `F.linear` | `fake_hif4(x)` / `fake_hif4(W)`，再 `F.linear` |
| 整网 | 每层前向时现场算 amax | 每层在线三级 scale |
| 误差 | 相对 FP32 的 MSE / max_abs / rel_L2 | 同左 |

核心调用：

```python
from pytorch_quantization.tensor_quant import fake_nvfp4_quant, fake_hif4_quant

# NVFP4：动态 per-tensor global_amax
y = F.linear(
    fake_nvfp4_quant(x, x.abs().amax()),
    fake_nvfp4_quant(W, W.abs().amax()),
    bias,
)

# HiF4：无 amax
y = F.linear(fake_hif4_quant(x), fake_hif4_quant(W), bias)
```

### 4.4 一次实测输出摘要（`seed=0`，CPU，全动态）

`fc1` 上动态 amax 示例：`amax_x≈2.90`，`amax_w≈0.125`（均来自当前 eval batch / 权重，非校准缓存）。

| 路径 | 整网输出 vs FP32（MSE / max_abs / rel_L2） |
|------|---------------------------------------------|
| NVFP4 W4A4（动态） | ≈ 4.1e-4 / 0.063 / 0.185 |
| HiF4 W4A4 | ≈ 3.1e-4 / 0.043 / 0.161 |

数字随随机种子变化；演示目的是走通 **动态** 伪量化流程，不是刷榜。

### 4.5 从演示到内网模型的对应关系

| 演示里做的事 | 内网 Linear 上对应 |
|--------------|-------------------|
| `LinearMLP` | 只替换 / 只量化 `nn.Linear` |
| 动态 `absmax` / HiF4 | 无需 calib 集；或权重仍可离线静态 amax |
| `linear_nvfp4` / `linear_hif4` | 自定义 `forward` 或包装模块 |
| Step 3 端到端误差 | 任务指标验收前的数值冒烟 |
| （未做）打包 E2M1 码 | 见第 5 节阶段 B |

---

## 5. 内网模型：仅 CPU 环境下线性层实装步骤

**约束：** 内网机器 **只有 CPU，没有 Blackwell（也没有可用 GPU）**。因此：

- **跑不了** Tensor Core / cuBLASLt / TensorRT 的 NVFP4 GEMM，也谈不上 GPU 侧吞吐收益。
- **能做的** 是：在 CPU 上完成格式正确的量化、打包、以及「解包 → FP32 matmul」的参考实现；用任务指标验收精度；把产物交给有 GPU 的环境再换真 kernel。
- 下文以 **NVFP4 + 仅 Linear** 为主；HiF4 步骤平行（换 `fake_hif4_quant`、块长 64、无全局 amax），但真加速仍取决于昇腾等侧，同样不在本机 CPU 上完成。

### 阶段 A：CPU 上精度摸底（fake quant）

1. **锁定范围**  
   只动 `nn.Linear`（`Conv1d(k=1)` 可先改写成 `Linear` 再量化）。空间卷积、depthwise、Norm、非线性保持 FP32。不要在 CPU 上追求“全模型 4-bit kernel”。

2. **接入本仓库伪量化（CPU 可跑）**  
   - 拷贝或依赖 `fake_nvfp4_quant` / `TensorQuantizer.use_block_quant`（纯 PyTorch，无 CUDA 也可）。  
   - 或自写等价逻辑，但必须通过与 `tests/test_nvfp4.py` 同网格的单测（E2M1 舍入、E4M3 scale、尾块 pad）。

3. **校准（CPU）**  
   用一小段代表性数据前向，对权重 / 激活各收 per-tensor Max \(\beta\)（`global_amax`）。激活也可推理时动态算 \(\beta\)；本仓库经验块级已够细时两者差距很小。HiF4 跳过这一步。

4. **端到端验收**  
   内网验证集上对比：**FP32** vs **Linear-only W4A4 fake quant**（权重离线量化或每步 fake-quant，激活每步 fake-quant，matmul 仍用 `torch.matmul` / `F.linear`）。  
   不过关：跳过敏感层、改 W4A16（只 quant 权重）、或收窄覆盖——**先把精度定住，再谈打包。**

5. **单层自洽检查**  
   对若干 Linear，固定输入 \(A\)、权重 \(W\)：  
   `Y_ref = fake_nvfp4(A) @ fake_nvfp4(W).T`（+ bias）  
   应与替换后的模块输出 **逐元素一致（或仅有浮点累加噪声）**。这是 CPU 上唯一可靠的“对齐金标准”。

### 阶段 B：CPU 上“实装”线性层（参考路径，非 Tensor Core）

在 CPU 上所谓实装，推荐做成 **显式编解码 + FP32 GEMM**，而不是假装有 NVFP4 硬件：

6. **离线量化并打包权重（每层）**  
   用校准 \(\beta_W\) 计算：
   - 每元素 E2M1 码（可存成 `uint8`，每 byte 塞 2 个 FP4，或先存 int 索引再打包）  
   - 每 16 个 `in_features` 一个 E4M3 block scale（`float8_e4m3fn` 或 `uint8` 位型）  
   - 张量级 `global_scale` \(s_g = \beta_W / (6\times 448)\)  
   持久化到自定义 checkpoint（`state_dict` 扩展字段或并列 `.npz`），**与层名一一对应**。

7. **实现 CPU `Nvfp4Linear`（参考算子）**  
   ```text
   forward(x):
     # 激活：在线 block quant（算 block amax → E4M3），得到 x_hat（FP32 反量化值）
     # 或：保持“量化码 + scale”，在 matmul 前反量化
     w_hat = dequant_nvfp4(packed_w, block_scales, global_scale)  # → FP32 [out, in]
     y = F.linear(x_hat, w_hat, bias)   # 标准 CPU BLAS / PyTorch
   ```  
   语义必须与阶段 A 的 `fake_nvfp4` **一致**；速度通常 **慢于** 原 FP32 权重（多一次 dequant），目标是 **行为正确 + 可交付格式**，不是 CPU 加速。

8. **激活两种实现选一（写进接口约定）**  
   - **简单（推荐先做）：** 激活走与权重相同的 fake quant，得到 FP32 `x_hat` 再乘。  
   - **严格打包：** 激活也产出 E2M1 + E4M3 scales，再写 CPU 反量化；便于以后和 GPU kernel 入参对齐。  
   文档里固定：`block_axis=-1`、`block_size=16`、尾部零 pad 规则。

9. **模块替换与回归**  
   用 `Nvfp4Linear` 替换目标 `nn.Linear`；跑单层对齐 + 全链路任务指标；敏感层可配置 fallback 回 FP32。

### 阶段 C：为“将来有 GPU”准备的交付物（本机仍不跑 TRT）

内网 CPU **不要**强行装 TensorRT / 编 Blackwell engine。只需准备好外发包，到有卡机器再编译：

10. **交付清单**  
    - [ ] 校准后的 \(\beta\) / `global_scale`（每层）  
    - [ ] packed 权重 + block scales（布局说明：K 维分块、是否 row-major、endian）  
    - [ ] 替换层名单与跳过名单  
    - [ ] CPU 参考实现输出的若干层 golden（固定 seed 的 `A → Y`）  
    - [ ] 端到端指标（fake quant / CPU dequant 路径）

11. **有卡侧再做的事（非本机）**  
    ModelOpt / TensorRT / TRT-LLM / CUTLASS NVFP4 GEMM：用同一套 scale 语义加载，**先对 golden 对齐，再换真 GEMM**。卷积仍建议 FP16/FP8；NVFP4 Conv 官方未覆盖。

12. **验收清单（CPU 环境）**  
    - [ ] 单测：E2M1 / E4M3 / pad 与本仓库测试一致  
    - [ ] 单层：`Nvfp4Linear` ≡ `fake_nvfp4` 路径  
    - [ ] 端到端任务指标可接受  
    - [ ] 打包文件可被独立脚本解码回 FP32 权重  
    - [ ] （可选）文档写明：性能数字仅在 GPU 环境有效  

### 阶段 D：和本仓库 Conformer 的对应关系（参考）

| 层 | CPU 内网可做 | 将来 GPU |
|----|--------------|----------|
| 146× Linear | fake quant / pack / dequant+FP32 matmul | NVFP4 GEMM |
| 32× pointwise Conv1d `k=1` | 改写成 Linear 后同上 | 同 Linear |
| 1× Conv2d `3×3 s=2` | 保持 FP32 | FP16/FP8，勿指望 NVFP4 Conv |
| depthwise / LN / GLU / BN | 保持 FP32 | 通常不量化 |

---

## 6. 一页对照表

| | NVFP4（本仓库） | HiF4（本仓库） |
|--|-----------------|----------------|
| 核心函数 | `fake_block_quant(..., "e2m1")` | `fake_hif4_quant` |
| 元素 | E2M1 | S1P2 |
| 块长 | 16 | 64 |
| Scale | \(s_g\) FP32（静/动）+ \(s_b\) E4M3 | E6M2 + E1_8 + E1_16（全动态） |
| 校准 | Max `global_amax` | 无 |
| 计算 | fake quant 后仍 FP32 matmul | 同左 |
| 内网仅 CPU | 可 pack + dequant 参考实现；**无** Tensor Core 加速 | 同左；真加速看昇腾等 |
| 有 GPU 时 | Blackwell NVFP4 GEMM（Linear） | 厂商 HiF4 算子 |
| 卷积 | 硬件 NVFP4 Conv 基本未覆盖 | 视厂商；本仓库未接 |

---

## 7. 建议阅读顺序

1. `pytorch_quantization/tensor_quant.py`：`fake_block_quant`、`fake_hif4_quant`  
2. `utils/quant_utils.py`：`set_block_quant`  
3. `tests/test_nvfp4.py`、`tests/test_hif4.py`  
4. **本文第 4 节 + `scripts/demo_linear_block_quant.py`**（线性层伪量化实操）  
5. `assets/experiments_summary.md` 第 4、5、8 节（Conformer 上的精度结论）  
6. 本文第 5 节（CPU 实装与交付物）；有卡后再读 ModelOpt / TensorRT-LLM NVFP4 文档
