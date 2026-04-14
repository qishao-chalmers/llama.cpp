# Measured vs layerwise: throughput by batch and KV quant

**Setup:** Qwen3-8B **Q8_0** weights, `prompt_len=2048`, `decode_len=4096` → `mid_ctx=4096`, **H100**.

**Sources:**

- **Real:** `research/results/qwen3-8b/profile/kv_timing_h100.json` (`measured_ms`)
- **Layerwise:** `research/results/qwen3-8b/profile/layerwise_kv_timing_h100.json` (`layerwise_ms_per_tok`, η from `layerwise_eta_h100.json`; calibrated column uses `layerwise_calibration_h100.json`)

**Simulator meta (this export):**

- `attn_impl`: `simple`  ·  `fa_bc`: `128`

**Definitions:**

- `tok/s = 1000 / ms_per_tok`
- `s/tok = ms_per_tok / 1000`
- `meas/lay` = measured ÷ layerwise raw (η-fitted sim); `meas/cal` = measured ÷ calibrated

---

## Real timing (`measured_ms`)

| KV   | B=1              | B=4              | B=8              | B=16             | B=32              |
|------|------------------|------------------|------------------|------------------|-------------------|
| f16  | 7.599 ms/tok · 131.6 tok/s | 2.783 ms/tok · 359.3 tok/s | 1.786 ms/tok · 559.8 tok/s | 1.143 ms/tok · 874.5 tok/s | 0.816 ms/tok · 1225.1 tok/s |
| q8_0 | 8.230 ms/tok · 121.5 tok/s | 2.986 ms/tok · 334.9 tok/s | 2.017 ms/tok · 495.8 tok/s | 1.340 ms/tok · 746.2 tok/s | 0.993 ms/tok · 1007.6 tok/s |
| q4_0 | 8.211 ms/tok · 121.8 tok/s | 2.944 ms/tok · 339.7 tok/s | 1.993 ms/tok · 501.8 tok/s | 1.338 ms/tok · 747.2 tok/s | 1.002 ms/tok · 997.8 tok/s |

### Real — seconds per token (`s/tok`)

| KV   | B=1      | B=4      | B=8      | B=16     | B=32     |
|------|----------|----------|----------|----------|----------|
| f16  | 0.007599 | 0.002783 | 0.001786 | 0.001143 | 0.000816 |
| q8_0 | 0.008230 | 0.002986 | 0.002017 | 0.001340 | 0.000993 |
| q4_0 | 0.008211 | 0.002944 | 0.001993 | 0.001338 | 0.001002 |

---

## Layerwise sim — raw η-fit (`layerwise_ms_per_tok`)

| KV   | B=1              | B=4              | B=8              | B=16             | B=32              |
|------|------------------|------------------|------------------|------------------|-------------------|
| f16  | 8.562 ms/tok · 116.8 tok/s | 2.843 ms/tok · 351.7 tok/s | 1.890 ms/tok · 529.0 tok/s | 1.414 ms/tok · 707.4 tok/s | 1.175 ms/tok · 850.8 tok/s |
| q8_0 | 8.121 ms/tok · 123.1 tok/s | 2.402 ms/tok · 416.3 tok/s | 1.449 ms/tok · 690.2 tok/s | 0.972 ms/tok · 1028.5 tok/s | 0.734 ms/tok · 1362.3 tok/s |
| q4_0 | 7.893 ms/tok · 126.7 tok/s | 2.174 ms/tok · 459.9 tok/s | 1.221 ms/tok · 818.9 tok/s | 0.745 ms/tok · 1343.1 tok/s | 0.506 ms/tok · 1975.3 tok/s |

### Layerwise — calibrated (`layerwise_ms_per_tok_calibrated`)

| KV   | B=1              | B=4              | B=8              | B=16             | B=32              |
|------|------------------|------------------|------------------|------------------|-------------------|
| f16  | 8.565 ms/tok · 116.8 tok/s | 2.858 ms/tok · 349.9 tok/s | 1.907 ms/tok · 524.4 tok/s | 1.432 ms/tok · 698.5 tok/s | 1.194 ms/tok · 837.7 tok/s |
| q8_0 | 8.114 ms/tok · 123.2 tok/s | 2.408 ms/tok · 415.3 tok/s | 1.457 ms/tok · 686.4 tok/s | 0.981 ms/tok · 1019.0 tok/s | 0.744 ms/tok · 1344.9 tok/s |
| q4_0 | 7.882 ms/tok · 126.9 tok/s | 2.176 ms/tok · 459.7 tok/s | 1.224 ms/tok · 816.7 tok/s | 0.749 ms/tok · 1335.3 tok/s | 0.511 ms/tok · 1956.4 tok/s |

---

## Measured vs layerwise (ratios)

| KV   | B   | meas ms/tok | lay raw | cal     | meas/lay | meas/cal |
|------|----:|------------:|--------:|--------:|---------:|---------:|
| f16  |   1 |       7.599 |   8.562 |   8.565 |     0.89 |     0.89 |
| f16  |   4 |       2.783 |   2.843 |   2.858 |     0.98 |     0.97 |
| f16  |   8 |       1.786 |   1.890 |   1.907 |     0.95 |     0.94 |
| f16  |  16 |       1.143 |   1.414 |   1.432 |     0.81 |     0.80 |
| f16  |  32 |       0.816 |   1.175 |   1.194 |     0.69 |     0.68 |
| q8_0 |   1 |       8.230 |   8.121 |   8.114 |     1.01 |     1.01 |
| q8_0 |   4 |       2.986 |   2.402 |   2.408 |     1.24 |     1.24 |
| q8_0 |   8 |       2.017 |   1.449 |   1.457 |     1.39 |     1.38 |
| q8_0 |  16 |       1.340 |   0.972 |   0.981 |     1.38 |     1.37 |
| q8_0 |  32 |       0.993 |   0.734 |   0.744 |     1.35 |     1.33 |
| q4_0 |   1 |       8.211 |   7.893 |   7.882 |     1.04 |     1.04 |
| q4_0 |   4 |       2.944 |   2.174 |   2.176 |     1.35 |     1.35 |
| q4_0 |   8 |       1.993 |   1.221 |   1.224 |     1.63 |     1.63 |
| q4_0 |  16 |       1.338 |   0.745 |   0.749 |     1.80 |     1.79 |
| q4_0 |  32 |       1.002 |   0.506 |   0.511 |     1.98 |     1.96 |

---

## Notes

- Regenerate layerwise JSON: `python3 research/scripts/export_layerwise_kv_json.py --template-json research/results/qwen3-8b/profile/kv_timing_h100.json --out research/results/qwen3-8b/profile/layerwise_kv_timing_h100.json --hw h100-sxm --eta-json research/results/qwen3-8b/profile/layerwise_eta_h100.json --calibration-json research/results/qwen3-8b/profile/layerwise_calibration_h100.json`
- Compare rows to cluster: `python3 research/scripts/compare_layerwise_to_traces.py research/results/qwen3-8b/profile/kv_timing_h100.json --eta-json research/results/qwen3-8b/profile/layerwise_eta_h100.json --calibration-json research/results/qwen3-8b/profile/layerwise_calibration_h100.json`

