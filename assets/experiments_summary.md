# ESC 复现与 Conformer 4-bit 量化实验总结

论文：*Evolution Strategy–Based Calibration for Low-Bit Quantization of Speech Models*（[arXiv 2603.08173](https://arxiv.org/abs/2603.08173)）

本文汇总在本仓库中做过的全部实验：MP-SENet 与 Conformer 的全精度基线、Conformer INT4 的论文复现、NVFP4 量化，以及用来解释 NVFP4 收益来源的对照实验。每组实验都写明了权重（W）和激活（A）具体是怎么量化的。

---

## 0. 结论速览

1. **复现可信。** Conformer FP32 与论文完全一致（WER 15.94 / CER 4.84）。INT4 的 4 种校准基线与论文的差距都在 ±1.6 个 WER 点以内。从头跑的 ESC 得到 38.29 / 13.51，论文是 38.49 / 13.50。
2. **论文的 INT4 方案在 Conformer 上损失很大。** 它的权重按通道量化，激活是静态逐张量。只用 Max 校准时模型完全崩溃（WER 145%），ESC 把 WER 降到 38.29%，但仍比 FP32 高约 22 个点。
3. **用 16 元素块两级 scale 的 4-bit 方案，只用 Max 校准，W4A4 的 WER 就能降到 19.5–22%。** 这组方案包括 NVFP4，以及同样结构但换成 INT4 元素的 INT4-block16。它们远好于 INT4 + ESC 的 38.29%。
4. **收益来自 scale 粒度，而不是 FP4 元素格式。** 两级 scale 结构相同时，INT4 元素不比 E2M1 差：W4A4 全部层 19.53 对 21.92，只量化权重 17.46 对 18.86。
5. **激活量化需要块级的细粒度 scale。** 激活改成每帧一个动态 scale 只能降到 53.10%，改成 16 元素块才降到 19.53%。
6. **非 GEMM 层的量化损失不小。** NVFP4 只量化 GEMM 层时 WER 为 17.15%，量化全部层为 21.92%。
7. **块缩放方案里，激活的全局 scale 不需要精细校准。** 把它从静态 Max 改成按每个输入动态计算，NVFP4 和 INT4-block16 的 WER 变化都不超过 0.09。
8. **GEMM 层输出误差高度相似，且随深度累积。** FP32 vs NVFP4-GEMM 与 FP32 vs INT4-block16-GEMM 的逐层 mean MSE 几乎落在同一条对角线上；误差最大的是 `decoder.linear`、深层 FFN 第一层线性（`ff*.module.1`）和 `conv_subsample.module.2`。
9. **普通 INT4 上，激活动态 per-tensor 几乎无用。** 权重仍 per-channel Max 时，激活从静态逐张量改成动态逐张量，全部层 WER 145→127，只量化 GEMM 层仍有 96.8；远差于动态 per-frame（全部层 53.1 / 仅 GEMM 38.3）和块缩放（~19）。
10. **ESC 静态 scale 可以优于朴素的动态 per-frame（全部层）。** ESC 全部层 38.29，动态 per-frame 全部层 53.10；但动态 per-frame 只量化 GEMM 时为 38.27，与 ESC 几乎持平。说明 ESC 搜到的是「对任务友好的截断」，而全部层上的 per-frame Max 仍会量化有害的非 GEMM 层，且对每帧取 max 未必最优。
11. **HiF4（arXiv:2602.11287）在本模型上略优于 NVFP4。** 全部层 WER 18.90 vs 21.92；仅 GEMM 16.91 vs 17.15（更接近 FP32 15.94）。HiF4 无需静态全局校准，三级 scale 全动态。
12. **HiF8 Cast（arXiv:2409.16626，无 PTS）接近 FP32。** 全部层 WER **16.34 / 4.96**，仅 GEMM **16.03 / 4.86**（FP32 为 15.94 / 4.84）。这是 8-bit 元素直接 TA cast，不是 HiF4 那种 4-bit 块浮点。

---

## 1. 实验环境与评测设置

| 项目 | 设置 |
|---|---|
| 运行环境 | WSL2 Ubuntu-26.04 中的 Docker 容器 `mpsenet`（`nvcr.io/nvidia/pytorch:26.09-py3`），RTX 5060 Ti（SM 12.0），torch 2.14 / CUDA 13.4 |
| 量化实现 | 仓库自带、经作者修改的 NVIDIA `pytorch_quantization`，全部是**伪量化**：在 FP32 中先量化再反量化，没有真实的低比特 kernel |
| GEMM 精度 | 量化后的 \(\hat A\)、\(\hat W\) 在高精度下做矩阵乘。容器默认开启 TF32（`matmul.allow_tf32 = cudnn.allow_tf32 = True`），FP32 基线也一样 |
| Conformer 数据 | 校准用 LibriSpeech train-clean-100 的前 **101** 条（校准循环在 `i >= 100` 时才停止）；评测用 test-clean 全部 **2620** 条；贪心 CTC 解码，按句平均 WER / CER |
| MP-SENet 数据 | VoiceBank+DEMAND 测试集（824 条） |
| 可重复性 | 除 ESC（CMA-ES 没有固定随机种子）外，所有实验都是确定性的；每组只跑一次 |

### 1.1 Conformer 中的量化位置（所有量化实验共用）

`quant_modules.initialize()` 在构建模型前，把 `torch.nn` 里的层替换成带量化器的版本：

| 层 | 个数 | 输入（激活）量化器 | 权重量化器 |
|---|---|---|---|
| Linear（FFN、注意力 Q/K/V/Pos/Out、`linear_proj`、CTC 输出层） | 146 | 有 | 有 |
| Conv1d（每个 block：pointwise ×2、depthwise ×1） | 48 | 有 | 有 |
| Conv2d（下采样） | 2 | 有 | 有 |
| LayerNorm | 80 | 有 | 有（\(\gamma\)） |
| GLU | 16 | 有 | 无 |
| BatchNorm1d | 16 | 有 | 无 |
| **合计** | | **308** | **276** |

- 量化的权重共 8,636,752 个，占总参数 9,295,901 的 **92.91%**。其中 Linear 占 79.29%，Conv1d 占 11.48%，Conv2d 占 2.02%，LayerNorm 占 0.12%。
- 不量化的部分：
  - 解码器 LSTM（596,480 个参数，占 6.42%），作者在替换列表里把它注释掉了。
  - 所有 bias（53,453 个），BatchNorm 的仿射参数，相对位置参数 u/v。
  - 注意力里的 \(QK^\top\) 和 attn·V 两个矩阵乘法，以及 softmax（用的是函数 `F.softmax`）。
  - SiLU、ReLU、残差加法、梅尔特征提取。
- 每个量化层的输出都不量化，由下一层的输入量化器再量化。

---

## 2. 全精度（FP32）基线

### 2.1 MP-SENet（语音增强）

| 指标 | 本地，ESC 的评测方式 | ESC 论文 | 本地，标准方式（干净语音作参考） | MP-SENet 原论文 |
|---|---|---|---|---|
| PESQ_WB | 1.970 | 2.12 | 3.496 | 3.50 |
| STOI (%) | 94.04 | 93.67 | 96.02 | 96 |
| PESQ_NB | 2.831 | – | – | – |
| SI-SDR (dB) | 9.448 | – | 19.94 | – |

- **做了什么：** 用户下载的 checkpoint（`g_best_vb`）是 MP-SENet 2025 年的长版本（TS-Transformer、SPConv），和 ESC 代码里的 Interspeech 2023 版结构不同，加载不了。我从 MP-SENet 仓库 commit `f5ec75f57f` 取回了 Interspeech 版的 `g_best` 和 `config.json`，这个权重可以严格无误地加载。
- **结论：**
  - 模型本身是对的：用标准方式打分，结果和 MP-SENet 原论文一致。
  - ESC 代码打分时把**增强后的输出当作参考信号，把带噪输入当作被测信号**，所以这个 "PESQ" 衡量的是增强结果和带噪输入有多像，而不是语音质量。按这种算法，PESQ 越高说明去噪越少。因此 ESC 论文里 MP-SENet 的 INT4 结果（2.51）比 FP32（2.12）还高，不能理解成量化后质量更好。
  - 本地 PESQ 比 ESC 论文低 0.15，原因不明。可能是数据预处理（重采样）或 checkpoint 不同。

### 2.2 Conformer（语音识别）

| | WER (%) | CER (%) |
|---|---|---|
| 本地 FP32 | 15.94 | 4.84 |
| ESC 论文 | 15.94 | 4.84 |

**结论：** 和论文完全一致。后面所有 Conformer 实验都以它为基线。

---

## 3. Conformer INT4 论文复现（W4A4）

### 3.1 量化方式

公式（`pytorch_quantization/tensor_quant.py` 中的 `_tensor_quant`）：

\[
Q(x)=\mathrm{clamp}\big(\mathrm{round}(x/s),-8,7\big),\quad s=\beta/7,\quad \hat x = s\cdot Q(x),\quad Z=0
\]

- 舍入是 `torch.round`（逢中点取偶）。\(s\) 以 FP32 存储，本身不再量化。码点范围是非对称的 \([-8,7]\)（`narrow_range = False`）。
- **W：**
  - Linear 和 Conv 的权重（196 个量化器）每个输出通道一个 \(\beta\)，取该通道的 \(\max|w|\)。
  - LayerNorm 的 \(\gamma\)（80 个量化器）整个张量一个 \(\beta\)。
  - 权重**只用 Max** 校准，因为 `set_calibrators` 只替换激活的校准器。
- **A：** 每个量化点整个张量一个 \(\beta\)，静态，校准后固定。\(\beta\) 由 101 条校准语音按下表的方法确定（直方图共 2048 个 bin）：

| 方法 | 激活 \(\beta\) 怎么确定 |
|---|---|
| Max | 校准数据上的 \(\max|a|\) |
| Percentile p | 直方图上 \(|a|\) 的第 p 百分位数 |
| MSE | 在直方图上搜索，使量化前后的均方误差最小 |
| Entropy | 在直方图上搜索，使量化前后分布的 KL 散度最小 |
| ESC | 先用 MSE 得到初值 \(\beta_i\)，再用 CMA-ES 给 308 个 \(\beta_i\) 各乘一个系数 \(\theta_i\)。初值 \(\theta=1\)，\(\sigma_0=0.1\)，预算 100 次评测（种群 21，实际 5 代共 105 次）。目标函数是 train-clean-100 前 100 条的平均 CER，最终取搜索分布的均值 `xfavorite` |
| ESC（仓库自带 scale） | 直接加载 `scale/conformer_b4_esc.pkl` 里的激活 \(\beta\) |

### 3.2 结果

| # | 方法 | 本地 WER / CER (%) | 论文 WER / CER (%) | Δ WER |
|---|---|---|---|---|
| – | FP32 | 15.94 / 4.84 | 15.94 / 4.84 | 0.00 |
| 1 | ESC（仓库自带 scale） | 40.06 / 14.12 | 38.49 / 13.50 | +1.57 |
| 2 | Max | 145.48 / 84.90 | 144.14 / 84.81 | +1.34 |
| 3 | Percentile 99.9 | 50.14 / 18.54 | 50.83 / 18.60 | −0.69 |
| 4 | Percentile 99.99 | 102.89 / 65.75 | – | – |
| 5 | Percentile 99.999 | 106.76 / 82.34 | – | – |
| 6a | MSE（ESC 的初始化） | 41.49 / 14.60 | 41.22 / 14.61 | +0.27 |
| 6b | **ESC（从头跑）** | **38.29 / 13.51** | **38.49 / 13.50** | **−0.20** |
| 7 | Entropy | 51.59 / 19.10 | 50.83 / 18.76 | +0.76 |

6a 和 6b 出自同一次运行。

### 3.3 结论

- 复现与论文吻合，所有可比的行都在 ±1.6 WER 以内。
- 修复了一个 bug：原代码下 `--calib max` 会报错，因为 `HistogramCalibrator` 不支持 `"max"`。现在只有当 `--calib` 不是 max 时才替换成直方图校准器。
- 静态逐张量的 INT4 激活对校准方法极其敏感：WER 从 Max 的 145% 到 ESC 的 38%。CMA-ES 在 MSE 初值的基础上又降了 3.2 个 WER 点，但仍比 FP32 高约 22 个点。
- 论文正文说 Percentile 取 99.99 / 99.999 / 99.9999 中最好的，但只有 99.9 能对上论文的数值，而且百分位越高结果越差。
- 仓库自带的 ESC scale 比从头跑差 1.6 个 WER 点。CMA-ES 没有固定种子，每次运行的结果会有波动。

---

## 4. Conformer NVFP4（W4A4 / W4A16）

### 4.1 量化方式

两级 scale：全张量一个 FP32 全局 scale \(s_g\)，每 16 个元素一个 FP8 E4M3 块 scale \(s_b\)，元素是 E2M1。

\[
s_g=\frac{\beta}{6\times448},\quad
s_b=\mathrm{E4M3}\Big(\mathrm{clamp}\big(\tfrac{\max_{b}|x|}{6\,s_g},2^{-9},448\big)\Big),\quad
\hat x=\mathrm{E2M1}_{\mathrm{RNE}}\Big(\tfrac{x}{s_b s_g}\Big)\cdot s_b s_g
\]

- E2M1 的取值是 \(\{0,\pm0.5,\pm1,\pm1.5,\pm2,\pm3,\pm4,\pm6\}\)，就近舍入、逢中点取偶，超过 6 的饱和到 6。全零块的 \(s_b\) 取 1。不足 16 的块补零。
- 实现（`fake_block_quant` / `fake_nvfp4_quant`）和 NVIDIA ModelOpt 的 `NVFP4QTensor` 逐元素完全一致，有单元测试验证。
- 等效位宽：每元素 4 + 8/16 = **4.5 bit**。
- **W：** \(\beta_W=\max|W|\)，整个张量一个，静态。分块沿 GEMM 的累加维 K：Linear 沿 `in_features`；卷积把 \(C_{in}\times k\) 展平后分块；LayerNorm 的 \(\gamma\) 沿自身。
- **A：** 全局 \(\beta_A\) 是静态的，由 101 条校准语音用 Max 得到，每个量化点一个。块 scale \(s_b\) 在推理时按当前 16 个激活的实际最大值动态计算。分块方向：
  - Linear 和 LayerNorm 的输入沿最后一维。
  - pointwise Conv1d 和第二个 Conv2d 的输入沿通道维。
  - depthwise Conv1d 和第一个 Conv2d（只有 1 个输入通道）沿最后一维（时间或频率）。
  - GLU 和 BatchNorm1d 的输入沿通道维。
  - 测试时超过校准最大值的激活会被截到 \(\beta_A\)。
- **量化范围：**
  - `all`：和 INT4 实验相同，584 个量化器全部量化。
  - `gemm`：只量化 146 个 Linear、32 个 pointwise Conv1d 和第二个 Conv2d 的输入与权重，共 358 个量化器。其余 226 个保持 FP32，包括 LayerNorm、GLU、BatchNorm1d、depthwise 卷积和第一个 Conv2d。
  - `--weight_only`：关掉全部 308 个激活量化器。
- 只用 Max 校准，没有跑 ESC。

### 4.2 结果

| 配置 | W | A | WER / CER (%) |
|---|---|---|---|
| NVFP4 W4A4，全部层（主结果） | NVFP4 | NVFP4 | 21.92 / 6.82 |
| NVFP4 W4A4，只量化 GEMM 层 | NVFP4（仅 GEMM 层） | NVFP4（仅 GEMM 层） | 17.15 / 5.19 |
| NVFP4 W4A16，全部层 | NVFP4 | FP32 | 18.86 / 5.80 |

### 4.3 结论

- 只用 Max 校准，NVFP4 W4A4 的 WER（21.92%）就远低于 INT4 + ESC（38.29%）；只量化 GEMM 层时只比 FP32 高 1.2 个 WER 点。
- 只量化权重（全部层，18.86%）反而比只在 GEMM 层量化权重和激活（17.15%）还差。说明非 GEMM 层**权重**的量化损失不小，这部分权重包括 LayerNorm 的 \(\gamma\)、depthwise 卷积核和第一个 Conv2d。这一点还没有逐层做消融确认。

---

## 5. 对照实验：NVFP4 的收益从哪里来

NVFP4 和论文的 INT4 同时在 4 个方面不同：元素格式（E2M1 对 INT4）、权重 scale 粒度（每 16 元素对每通道）、激活 scale（块级动态对静态逐张量）、位宽（4.5 对约 4）。下面的对照实验每次只改变其中一个因素。

### 5.1 量化方式

| 实验 | W | A |
|---|---|---|
| **INT4-block16** | 和 NVFP4 完全相同的两级 scale 和分块方向，元素换成 INT4：\(Q=\mathrm{clamp}(\mathrm{round}(x/(s_b s_g)),-8,7)\)，\(s_g=\beta/(7\times448)\)，\(s_b\) 用 E4M3 存储，每元素等效 4.5 bit | 同 NVFP4：全局 \(\beta_A\) 静态 Max，块 scale 动态，INT4 元素 |
| **INT4 W4A16** | 同论文：每输出通道 Max（LayerNorm 整个张量） | 不量化（FP32） |
| **INT4 激活每帧动态** | 同论文：每输出通道 Max（LayerNorm 整个张量） | INT4，\(s=\mathrm{amax}/7\)。amax 在推理时**按每一帧**现场计算，不参与校准：Linear 和 LayerNorm 的输入对最后一维取最大值；Conv1d、GLU、BatchNorm1d 的输入对通道维取最大值；Conv2d 的输入对通道和频率两维取最大值 |

### 5.2 结果

**相同两级块 scale 下只换元素格式（每元素都是 4.5 bit）：**

| 量化范围 | NVFP4（E2M1 元素） | INT4-block16（INT4 元素） |
|---|---|---|
| W4A4，全部层 | 21.92 / 6.82 | **19.53 / 6.01** |
| W4A4，只量化 GEMM 层 | **17.15 / 5.19** | 17.40 / 5.31 |
| W4A16，全部层 | 18.86 / 5.80 | **17.46 / 5.28** |

**相同 INT4 元素下改变 scale 结构（全部层）：**

| 权重 scale | 激活 scale | 校准方法 | WER / CER (%) |
|---|---|---|---|
| 每输出通道 | 不量化激活 | Max | 23.34 / 7.75 |
| 16 元素块（两级） | 不量化激活 | Max | 17.46 / 5.28 |
| 每输出通道 | 静态，逐张量 | Max | 145.48 / 84.90 |
| 每输出通道 | 静态，逐张量 | ESC | 38.29 / 13.51 |
| 每输出通道 | 动态，逐张量 | Max（只用于权重） | 127.45 / 76.69 |
| 每输出通道（仅 GEMM 层） | 动态，逐张量 | Max（只用于权重） | 96.77 / 69.01 |
| 每输出通道 | 动态，每帧一个 | Max（只用于权重） | 53.10 / 22.06 |
| 每输出通道（仅 GEMM 层） | 动态，每帧一个 | Max（只用于权重） | 38.27 / 15.26 |
| 16 元素块（两级） | 16 元素块（两级，块 scale 动态） | Max | **19.53 / 6.01** |

### 5.3 结论

1. **元素格式不是 NVFP4 胜出的原因。** 两级 scale 相同时，INT4 元素在"全部层"的两组实验里比 E2M1 好 1.4–2.4 个 WER 点；只量化 GEMM 层时两者基本持平，0.25 的差距不能算显著。一个可能的解释是：16 元素块已经把异常值隔离开，块内分布较平坦，INT4 的 16 个均匀电平比 E2M1 顶端稀疏的电平（4 和 6 之间没有值）更合适。这一点只是推测，没有专门验证。
2. **权重的 scale 粒度很重要。** 只量化权重时，每输出通道改成每 16 元素一块，WER 从 23.34 降到 17.46。
3. **激活的 scale 粒度贡献最大。** 静态逐张量（Max）的 WER 是 145%；改成**动态逐张量**只降到 127%（仅 GEMM 仍有 97%），几乎仍崩溃；**每帧动态**降到 53%，仍不如 ESC 静态（38%）；**16 元素块**才降到 19.5%。关键不是「动不动」，而是 scale 够不够细。
4. **对比的公平性。** "NVFP4 远好于论文的 INT4"这个说法只在方案整体对比的意义上成立；它不能说明 FP4 元素优于 INT4 元素。另外，块缩放方案每元素多用 0.5 bit。
5. **部署角度。** Blackwell 张量核心原生支持的是 NVFP4，没有对应的块缩放 INT4 格式。所以 INT4-block16 只用来做精度归因，并不是现成可部署的替代方案。

### 5.4 补充实验：激活的全局 scale 改为动态（只量化 GEMM 层）

**量化方式：**

- W：不变，即两级块 scale，权重全局 \(\beta_W=\max|W|\)。
- A：块 scale 和元素格式不变；全局 per-tensor \(\beta_A\) 不再由校准数据的 Max 得到，而是在推理时取**当前输入张量**的 \(\max|a|\)。评测时 batch size 为 1，所以相当于每条语音、每一层各算一次。激活不参与校准，也不会因为超过校准最大值而被截断。
- 量化范围：179 个 GEMM 层的输入为动态 per-tensor，它们的权重仍是静态全局 scale，其余 226 个量化器保持 FP32。

| 量化范围 | 激活全局 scale | NVFP4 | INT4-block16 |
|---|---|---|---|
| W4A4，只量化 GEMM 层 | 静态（Max 校准） | 17.15 / 5.19 | 17.40 / 5.31 |
| W4A4，只量化 GEMM 层 | **动态（每个输入）** | 17.24 / 5.21 | 17.38 / 5.35 |

**结论：**

- 全局 scale 是静态还是动态，对两种元素格式都几乎没有影响（WER 差 ≤ 0.09，在单次运行的波动范围内）。
- 原因是全局 scale 只是一个粗粒度的基准：真正决定量化精度的是每 16 个元素的块 scale，而 E4M3 块 scale 的取值范围（\(2^{-9}\) 到 448）足够宽，能吸收全局 scale 的差异。
- 所以对 NVFP4 的激活来说，用 Max 做静态校准已经足够，不需要更复杂的校准方法，也不需要在推理时额外遍历一遍激活来求全局最大值。后者在真实 kernel 里会增加一次规约开销。

---

## 6. 全部 Conformer 实验汇总（test-clean，WER / CER %）

| 实验 | W | A | 校准方法 | WER | CER |
|---|---|---|---|---|---|
| FP32 | – | – | – | 15.94 | 4.84 |
| INT4 Max | INT4 每通道 | INT4 静态逐张量 | Max | 145.48 | 84.90 |
| INT4 Percentile 99.9 | INT4 每通道 | INT4 静态逐张量 | Percentile | 50.14 | 18.54 |
| INT4 Percentile 99.99 | INT4 每通道 | INT4 静态逐张量 | Percentile | 102.89 | 65.75 |
| INT4 Percentile 99.999 | INT4 每通道 | INT4 静态逐张量 | Percentile | 106.76 | 82.34 |
| INT4 Entropy | INT4 每通道 | INT4 静态逐张量 | Entropy | 51.59 | 19.10 |
| INT4 MSE | INT4 每通道 | INT4 静态逐张量 | MSE | 41.49 | 14.60 |
| INT4 ESC（从头跑） | INT4 每通道 | INT4 静态逐张量 | MSE + CMA-ES | 38.29 | 13.51 |
| INT4 ESC（仓库自带 scale） | INT4 每通道 | INT4 静态逐张量 | 预计算 | 40.06 | 14.12 |
| INT4 W4A16 | INT4 每通道 | FP32 | Max | 23.34 | 7.75 |
| INT4 激活动态 per-tensor，全部层 | INT4 每通道 | INT4 动态逐张量 | Max（只用于权重） | 127.45 | 76.69 |
| INT4 激活动态 per-tensor，只量化 GEMM 层 | INT4 每通道 | INT4 动态逐张量 | Max（只用于权重） | 96.77 | 69.01 |
| INT4 激活每帧动态，全部层 | INT4 每通道 | INT4 每帧动态 | Max（只用于权重） | 53.10 | 22.06 |
| INT4 激活每帧动态，只量化 GEMM 层 | INT4 每通道 | INT4 每帧动态 | Max（只用于权重） | 38.27 | 15.26 |
| NVFP4 W4A4，全部层 | NVFP4 | NVFP4 | Max（全局） | 21.92 | 6.82 |
| NVFP4 W4A4，只量化 GEMM 层 | NVFP4 | NVFP4 | Max（全局） | 17.15 | 5.19 |
| NVFP4 W4A4，只量化 GEMM 层，激活全局 scale 动态 | NVFP4 | NVFP4，全局 scale 动态 | Max（只用于权重） | 17.24 | 5.21 |
| NVFP4 W4A16 | NVFP4 | FP32 | Max（全局） | 18.86 | 5.80 |
| HiF4 W4A4，全部层 | HiF4 | HiF4 | 无（全动态） | 18.90 | 5.71 |
| HiF4 W4A4，只量化 GEMM 层 | HiF4 | HiF4 | 无（全动态） | 16.91 | 5.09 |
| INT4-block16 W4A4，全部层 | INT4 + 两级块 scale | INT4 + 两级块 scale | Max（全局） | 19.53 | 6.01 |
| INT4-block16 W4A4，只量化 GEMM 层 | INT4 + 两级块 scale | INT4 + 两级块 scale | Max（全局） | 17.40 | 5.31 |
| INT4-block16 W4A4，只量化 GEMM 层，激活全局 scale 动态 | INT4 + 两级块 scale | INT4 + 两级块 scale，全局 scale 动态 | Max（只用于权重） | 17.38 | 5.35 |
| INT4-block16 W4A16 | INT4 + 两级块 scale | FP32 | Max（全局） | 17.46 | 5.28 |

---

## 7. GEMM 层输出 MSE（method A）

对 NVFP4 W4A4（GEMM）和 INT4-block16 W4A4（GEMM），在 test-clean 全部 2620 条上，各自与 FP32 做完整前向，对齐 179 个 GEMM 层的输出，计算逐层 mean MSE（先对每条 utterance 做元素平均 MSE，再对 2620 条取平均）。激活全局 scale 使用此前评测保存的 Max 校准结果（`scale/conformer_nvfp4_gemm_max.pkl`、`scale/conformer_int4_block_gemm_max.pkl`）。

### 7.1 结果摘要

| 汇总 | FP32 vs NVFP4 GEMM | FP32 vs INT4-block16 GEMM |
|---|---:|---:|
| 179 层 mean MSE 的平均值 | 0.02179 | 0.02245 |
| 该格式 MSE 更小的层数 | 113 | 66 |

误差最大的层类型（block 内类型对 16 个 block 取平均；单例层单独列出）：

| 层类型 | NVFP4 | INT4-block16 |
|---|---:|---:|
| `decoder.linear` | 7.59e-01 | 7.46e-01 |
| `conv_subsample.module.2` | 1.04e-01 | 9.86e-02 |
| `ff2.module.1` | 4.98e-02 | 5.06e-02 |
| `ff1.module.1` | 4.34e-02 | 4.50e-02 |
| `attention.W_k` | 2.13e-02 | 2.33e-02 |
| `attention.W_out`（最小一类） | 1.71e-03 | 1.83e-03 |

### 7.2 结论

- 两种 4-bit 块缩放格式的逐层误差几乎重合（散点沿 \(y=x\)），与二者 WER 接近（17.15 vs 17.40）一致。
- 绝对 MSE 随 Conformer block 深度明显增大（热力图中 `ff1.1` / `ff2.1` 从浅到深变亮），符合 method A 的误差累积。
- `decoder.linear`（CTC 输出投影）绝对 MSE 远高于其余层；`W_out` / `W_pos` / `pw6` 相对很小。

### 7.3 产物

| 文件 | 内容 |
|---|---|
| [`layer_mse_gemm_table.md`](layer_mse_gemm_table.md) | 179 行全表：行=量化层，列=FP32 vs NVFP4 / FP32 vs INT4-block16 |
| [`layer_mse_gemm.csv`](layer_mse_gemm.csv) | 同上，另含 RelMSE |
| [`layer_mse_gemm_heatmap.png`](layer_mse_gemm_heatmap.png) | block × 层类型热力图（两格式并排） |
| [`layer_mse_gemm_scatter.png`](layer_mse_gemm_scatter.png) | 逐层 NVFP4 MSE vs INT4-block16 MSE |
| [`layer_mse_gemm_by_type.png`](layer_mse_gemm_by_type.png) | 按层类型汇总柱图 |
| [`layer_mse_gemm_topk.md`](layer_mse_gemm_topk.md) | 两格式各自 Top-10 |
| `scripts/layer_mse_gemm.py` | 复现脚本 |

---

## 8. Conformer HiFloat4（W4A4）

依据原论文 [arXiv:2602.11287](https://arxiv.org/abs/2602.11287) Algorithm 1 实现伪量化：

- 块大小 64；三级 scale：unsigned E6M2（块）+ E1_8（每 8 元）+ E1_16（每 4 元）；元素 S1P2（≡ E1M2，最大 ±1.75）
- 无 per-tensor 静态 scale；权重与激活的全部 scale 由当前张量现场计算
- 分块轴与 NVFP4 相同；`coverage=all|gemm` 同前
- 实现：`fake_hif4_quant`（`tensor_quant.py`），CLI `--quant_format hif4`

| 覆盖 | HiF4 WER / CER | NVFP4 WER / CER | FP32 |
|---|---:|---:|---:|
| 全部层 | **18.90 / 5.71** | 21.92 / 6.82 | 15.94 / 4.84 |
| 只量化 GEMM | **16.91 / 5.09** | 17.15 / 5.19 | 15.94 / 4.84 |

在相同 W4A4、相同层覆盖下，HiF4 略好于 NVFP4；GEMM-only 仅比 FP32 高约 1.0 WER。

---

## 9. Conformer HiFloat8 Cast（W8A8，无 PTS）

依据 [arXiv:2409.16626](https://arxiv.org/abs/2409.16626) 的 **HiF8 元素格式** + 前向 **round-half-away-from-zero（TA）** 做伪量化；**不做**论文 Algorithm 1 的 per-tensor 幂次缩放（PTS），即 HiF8_Cast。

- 每元素独立 cast 到 HiF8 网格再反量化；无块 scale、无全局 amax、无校准
- CLI：`--quant_format hif8 --coverage {all,gemm}`
- 实现：`fake_hif8_quant` / `use_hif8`；单测 `tests/test_hif8.py`（含「冲刷到 0 的负数不得写成 0x80/NaN」）

| 覆盖 | HiF8 Cast WER / CER | HiF4 | NVFP4 | FP32 |
|---|---:|---:|---:|---:|
| 全部层 | **16.34 / 4.96** | 18.90 / 5.71 | 21.92 / 6.82 | 15.94 / 4.84 |
| 只量化 GEMM | **16.03 / 4.86** | 16.91 / 5.09 | 17.15 / 5.19 | 15.94 / 4.84 |

相对 FP32，全部层约 +0.40 WER，GEMM-only 约 +0.09 WER；明显好于同覆盖下的 HiF4 / NVFP4（后两者是 4-bit 方案，比特预算不同，仅作对照）。

---

## 10. 局限

- 全部是伪量化，没有用真实低比特 kernel 测精度和速度。GEMM 实际以 TF32 计算。
- 每组实验只跑一次，只用 test-clean 评测，没有给出置信区间。
- NVFP4 和 INT4-block16 只用了 Max 校准，没有在这两种格式上跑 ESC。
- "每帧动态"和非 GEMM 层（例如 depthwise 卷积）的块量化是伪量化里的定义，不一定能在真实 kernel 里同样实现。
- 不量化的部分（LSTM、注意力矩阵乘、softmax 等）在所有实验中都一致，所以不影响横向对比，但说明这些方案都不是端到端的纯 4-bit 推理。

---

## 11. 复现命令（容器内，工作目录 `/workspace/esc`）

```bash
# FP32
python inference.py --model mpnet --fp32_only
python inference.py --model conformer --fp32_only

# INT4 论文基线（W4A4）
python inference.py --model conformer --num_bits 4 --calib max
python inference.py --model conformer --num_bits 4 --calib percentile --percentile 99.9
python inference.py --model conformer --num_bits 4 --calib entropy
python inference.py --model conformer --num_bits 4 --calib mse --esc          # 先输出 MSE 结果，再输出 ESC 结果
python inference.py --model conformer --num_bits 4 --precompute_scale scale/conformer_b4_esc.pkl

# NVFP4
python inference.py --model conformer --quant_format nvfp4 --coverage all --calib max
python inference.py --model conformer --quant_format nvfp4 --coverage gemm --calib max
python inference.py --model conformer --quant_format nvfp4 --coverage all --weight_only --calib max

# 对照实验
python inference.py --model conformer --quant_format int4_block --coverage all --calib max
python inference.py --model conformer --quant_format int4_block --coverage gemm --calib max
python inference.py --model conformer --quant_format int4_block --coverage all --weight_only --calib max
python inference.py --model conformer --num_bits 4 --calib max --weight_only
python inference.py --model conformer --num_bits 4 --calib max --act_dynamic_token
python inference.py --model conformer --num_bits 4 --calib max --act_dynamic_token --coverage gemm
python inference.py --model conformer --num_bits 4 --calib max --act_dynamic_tensor --coverage all
python inference.py --model conformer --num_bits 4 --calib max --act_dynamic_tensor --coverage gemm

# 补充实验：块缩放格式的激活全局 scale 动态
python inference.py --model conformer --quant_format nvfp4 --coverage gemm --act_dynamic_tensor --calib max
python inference.py --model conformer --quant_format int4_block --coverage gemm --act_dynamic_tensor --calib max

# HiFloat4（arXiv:2602.11287）
python inference.py --model conformer --quant_format hif4 --coverage all --calib max
python inference.py --model conformer --quant_format hif4 --coverage gemm --calib max

# GEMM 层输出 MSE（method A，约 15 分钟）
python scripts/layer_mse_gemm.py

# 单元测试
python -m unittest tests.test_nvfp4 tests.test_nvfp4_model tests.test_hif4 tests.test_hif8 -v

# HiF8 Cast（无 PTS）
python inference.py --model conformer --quant_format hif8 --coverage all --calib max
python inference.py --model conformer --quant_format hif8 --coverage gemm --calib max
# 或: bash scripts/run_hif8_conformer.sh
```

从宿主机运行时，在命令前加上 `wsl -d Ubuntu-26.04 -- docker exec -w /workspace/esc mpsenet`。

日志位置：`logs/conformer_*.log`、`logs/conformer_b4/`、`logs/nvfp4/`、`logs/int4_block/`、`logs/hif4/`、`logs/hif8/`、`logs/controls/`、`logs/layer_mse_gemm.out`。

## 12. 代码改动一览

| 文件 | 改动 |
|---|---|
| `models/mpsenet/mpnet.py`、`models/mpsenet/utils.py` | 数据和 checkpoint 路径；兼容官方 checkpoint 的 `generator` 键 |
| `models/conformer/conformer.py` | LibriSpeech 路径改为 `/data/librispeech` |
| `inference.py` | 新增 `--fp32_only`；修复 Max 校准报错；新增 `--quant_format {int,nvfp4,int4_block,hif4,hif8}`、`--coverage {all,gemm}`、`--weight_only`、`--act_dynamic_token`、`--act_dynamic_tensor`；HiF4/HiF8 跳过静态校准 |
| `utils/utils.py` | 日志文件命名；自动创建日志目录 |
| `utils/quant_utils.py` | `set_block_quant`（含 `hif4`/`hif8`）、`set_gemm_only`、`set_dynamic_tensor_act`、`set_dynamic_token_act`、`set_weight_only`；`compute_amax` 跳过被关闭的量化器 |
| `pytorch_quantization/tensor_quant.py` | `fake_block_quant` / `fake_nvfp4_quant`；`fake_hif4_quant`；`fake_hif8_quant`（HiF8 Cast TA） |
| `pytorch_quantization/nn/modules/tensor_quantizer.py` | `use_block_quant`、`use_hif4`、`use_hif8`、`use_dynamic_amax` |
| `tests/test_nvfp4.py`、`tests/test_nvfp4_model.py`、`tests/test_hif4.py`、`tests/test_hif8.py` | NVFP4 / HiF4 / HiF8 单元测试 |
| `scripts/run_hif8_conformer.sh` | Conformer HiF8 Cast all+gemm 一键评测 |
