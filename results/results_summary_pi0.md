# Evaluation Results Summary — pi0

**Policy:** `pi0`

**Baseline (no delay):** avg = 69.8
| libero_spatial | libero_object | libero_goal | libero_10 | Average |
|:---:|:---:|:---:|:---:|:---:|
| 70.0 | 78.0 | 80.0 | 51.0 | 69.8 |

### Table 1: Symmetric Delay ($\delta_{\mathrm{img}}=\delta_{\mathrm{state}}$)

Both image and state delayed by the same amount.

| Delay | Comp. | libero_spatial | libero_object | libero_goal | libero_10 | Average |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $\delta=0$ ★ | none | 70.0 | 78.0 | 80.0 | 51.0 | 69.8 |
| $\delta=1$ | none | 57.0 | 86.0 | 82.0 | 41.0 | 66.5 |
| $\delta=1$ | **belief** | 65.0 | 87.0 | 81.0 | 39.0 | 68.0 |
| $\delta=2$ | none | 48.0 | 69.0 | 71.0 | 35.0 | 55.8 |
| $\delta=2$ | **belief** | 54.0 | 74.0 | 83.0 | 46.0 | 64.2 |
| $\delta=4$ | none | 22.0 | 28.0 | 28.0 | 21.0 | 24.8 |
| $\delta=4$ | **belief** | 56.0 | 44.0 | 66.0 | 32.0 | 49.5 |
| $\delta=8$ | none | 0.0 | 4.0 | 15.0 | 3.0 | 5.5 |
| $\delta=8$ | **belief** | 36.0 | 39.0 | 58.0 | 28.0 | 40.2 |
| $\delta=16$ | none | 0.0 | 0.0 | 5.0 | 0.0 | 1.2 |
| $\delta=16$ | **belief** | 23.0 | 24.0 | 36.0 | 12.0 | 23.8 |
| $\delta=32$ | none | 0.0 | 0.0 | 3.0 | 0.0 | 0.8 |
| $\delta=32$ | **belief** | 14.0 | 10.0 | 16.0 | 1.0 | 10.2 |

### Table 2: Image-Only Delay ($\delta_{\mathrm{state}}=0$)

Only image observations are delayed, state is real-time.

| Delay | Comp. | libero_spatial | libero_object | libero_goal | libero_10 | Average |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $\delta_{\mathrm{img}}=0$ ★ | none | 70.0 | 78.0 | 80.0 | 51.0 | 69.8 |
| $\delta_{\mathrm{img}}=1$ | none | 60.0 | 79.0 | 82.0 | 41.0 | 65.5 |
| $\delta_{\mathrm{img}}=1$ | **belief** | 70.0 | 79.0 | 87.0 | 48.0 | 71.0 |
| $\delta_{\mathrm{img}}=2$ | none | 56.0 | 83.0 | 88.0 | 48.0 | 68.8 |
| $\delta_{\mathrm{img}}=2$ | **belief** | 62.0 | 88.0 | 87.0 | 42.0 | 69.8 |
| $\delta_{\mathrm{img}}=4$ | none | 65.0 | 81.0 | 84.0 | 45.0 | 68.8 |
| $\delta_{\mathrm{img}}=4$ | **belief** | 74.0 | 78.0 | 77.0 | 53.0 | 70.5 |
| $\delta_{\mathrm{img}}=8$ | none | 67.0 | 88.0 | 85.0 | 53.0 | 73.2 |
| $\delta_{\mathrm{img}}=8$ | **belief** | 69.0 | 86.0 | 89.0 | 50.0 | 73.5 |
| $\delta_{\mathrm{img}}=16$ | none | 61.0 | 83.0 | 80.0 | 45.0 | 67.2 |
| $\delta_{\mathrm{img}}=16$ | **belief** | 69.0 | 80.0 | 86.0 | 39.0 | 68.5 |
| $\delta_{\mathrm{img}}=32$ | none | 74.0 | 79.0 | 83.0 | 44.0 | 70.0 |
| $\delta_{\mathrm{img}}=32$ | **belief** | 66.0 | 79.0 | 85.0 | 41.0 | 67.8 |

### Table 3: State-Only Delay ($\delta_{\mathrm{img}}=0$)

Only state/proprioception is delayed, images are real-time.

| Delay | Comp. | libero_spatial | libero_object | libero_goal | libero_10 | Average |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $\delta_{\mathrm{state}}=0$ ★ | none | 70.0 | 78.0 | 80.0 | 51.0 | 69.8 |
| $\delta_{\mathrm{state}}=1$ | none | 65.0 | 84.0 | 81.0 | 40.0 | 67.5 |
| $\delta_{\mathrm{state}}=1$ | **belief** | 67.0 | 86.0 | 80.0 | 47.0 | 70.0 |
| $\delta_{\mathrm{state}}=2$ | none | 48.0 | 70.0 | 65.0 | 32.0 | 53.8 |
| $\delta_{\mathrm{state}}=2$ | **belief** | 59.0 | 78.0 | 83.0 | 41.0 | 65.2 |
| $\delta_{\mathrm{state}}=4$ | none | 18.0 | 35.0 | 37.0 | 21.0 | 27.8 |
| $\delta_{\mathrm{state}}=4$ | **belief** | 44.0 | 41.0 | 69.0 | 43.0 | 49.2 |
| $\delta_{\mathrm{state}}=8$ | none | 0.0 | 3.0 | 17.0 | 3.0 | 5.8 |
| $\delta_{\mathrm{state}}=8$ | **belief** | 37.0 | 45.0 | 65.0 | 28.0 | 43.8 |
| $\delta_{\mathrm{state}}=16$ | none | 0.0 | 0.0 | 3.0 | 0.0 | 0.8 |
| $\delta_{\mathrm{state}}=16$ | **belief** | 19.0 | 20.0 | 43.0 | 14.0 | 24.0 |
| $\delta_{\mathrm{state}}=32$ | none | 0.0 | 0.0 | 4.0 | 0.0 | 1.0 |
| $\delta_{\mathrm{state}}=32$ | **belief** | 9.0 | 9.0 | 21.0 | 2.0 | 10.2 |

---
★ = baseline (no delay)

*Policy: pi0 — Generated from 37 unique experiment configs.*
