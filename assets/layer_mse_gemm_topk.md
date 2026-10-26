### Top-10 layers by mean MSE

**FP32 vs NVFP4 GEMM**

| Rank | Layer | mean MSE |
|---:|---|---:|
| 1 | `decoder.linear` | 7.592400e-01 |
| 2 | `encoder.layers.15.ff2.module.1` | 1.076087e-01 |
| 3 | `encoder.conv_subsample.module.2` | 1.042025e-01 |
| 4 | `encoder.layers.14.ff2.module.1` | 9.646997e-02 |
| 5 | `encoder.layers.15.ff1.module.1` | 8.927004e-02 |
| 6 | `encoder.layers.13.ff2.module.1` | 8.291407e-02 |
| 7 | `encoder.layers.14.ff1.module.1` | 7.749739e-02 |
| 8 | `encoder.layers.12.ff2.module.1` | 6.956723e-02 |
| 9 | `encoder.layers.13.ff1.module.1` | 6.816503e-02 |
| 10 | `encoder.layers.12.ff1.module.1` | 5.612756e-02 |

**FP32 vs INT4-block16 GEMM**

| Rank | Layer | mean MSE |
|---:|---|---:|
| 1 | `decoder.linear` | 7.464558e-01 |
| 2 | `encoder.layers.15.ff2.module.1` | 1.059752e-01 |
| 3 | `encoder.conv_subsample.module.2` | 9.861491e-02 |
| 4 | `encoder.layers.14.ff2.module.1` | 9.642544e-02 |
| 5 | `encoder.layers.15.ff1.module.1` | 8.942847e-02 |
| 6 | `encoder.layers.13.ff2.module.1` | 8.143901e-02 |
| 7 | `encoder.layers.14.ff1.module.1` | 7.785873e-02 |
| 8 | `encoder.layers.12.ff2.module.1` | 6.975891e-02 |
| 9 | `encoder.layers.13.ff1.module.1` | 6.920100e-02 |
| 10 | `encoder.layers.12.ff1.module.1` | 5.635539e-02 |
