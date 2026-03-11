# Evaluation Results Summary — pi05

**Policy:** `pi05`

**Baseline (no delay):** avg = 98.0
| libero_spatial | libero_object | libero_goal | libero_10 | Average |
|:---:|:---:|:---:|:---:|:---:|
| 97.0 | 100.0 | 98.0 | 97.0 | 98.0 |

### Table 1: Symmetric Delay ($\delta_{\mathrm{img}}=\delta_{\mathrm{state}}$)

Both image and state delayed by the same amount.

| Delay | Comp. | libero_spatial | libero_object | libero_goal | libero_10 | Average |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $\delta=0$ ★ | none | 97.0 | 100.0 | 98.0 | 97.0 | 98.0 |
| $\delta=1$ | none | 93.0 | 98.0 | 96.0 | 90.0 | 94.2 |
| $\delta=1$ | **belief** | 95.0 | 98.0 | 98.0 | 91.0 | 95.5 |
| $\delta=2$ | none | 84.0 | 96.0 | 94.0 | 81.0 | 88.8 |
| $\delta=2$ | **belief** | 85.0 | 98.0 | 91.0 | 86.0 | 90.0 |
| $\delta=4$ | none | 46.0 | 73.0 | 75.0 | 75.0 | 67.2 |
| $\delta=4$ | **belief** | 43.0 | 68.0 | 69.0 | 71.0 | 62.7 |
| $\delta=8$ | none | 10.0 | 10.0 | 48.0 | 20.0 | 22.0 |
| $\delta=8$ | **belief** | 10.0 | 8.0 | 47.0 | 17.0 | 20.5 |
| $\delta=16$ | none | 0.0 | 0.0 | 11.0 | 0.0 | 2.8 |
| $\delta=16$ | **belief** | 1.0 | 0.0 | 7.0 | 0.0 | 2.0 |
| $\delta=32$ | none | 0.0 | 0.0 | 5.0 | 0.0 | 1.2 |
| $\delta=32$ | **belief** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### Table 2: Image-Only Delay ($\delta_{\mathrm{state}}=0$)

Only image observations are delayed, state is real-time.

| Delay | Comp. | libero_spatial | libero_object | libero_goal | libero_10 | Average |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $\delta_{\mathrm{img}}=0$ ★ | none | 97.0 | 100.0 | 98.0 | 97.0 | 98.0 |
| $\delta_{\mathrm{img}}=1$ | none | 92.0 | 98.0 | 98.0 | 96.0 | 96.0 |
| $\delta_{\mathrm{img}}=1$ | **belief** | 95.0 | 98.0 | 95.0 | 91.0 | 94.8 |
| $\delta_{\mathrm{img}}=2$ | none | 83.0 | 98.0 | 90.0 | 83.0 | 88.5 |
| $\delta_{\mathrm{img}}=2$ | **belief** | 85.0 | 98.0 | 89.0 | 85.0 | 89.2 |
| $\delta_{\mathrm{img}}=4$ | none | 42.0 | 72.0 | 75.0 | 72.0 | 65.2 |
| $\delta_{\mathrm{img}}=4$ | **belief** | 46.0 | 64.0 | 76.0 | 70.0 | 64.0 |
| $\delta_{\mathrm{img}}=8$ | none | 9.0 | 10.0 | 42.0 | 15.0 | 19.0 |
| $\delta_{\mathrm{img}}=8$ | **belief** | 9.0 | 7.0 | 45.0 | 14.0 | 18.8 |
| $\delta_{\mathrm{img}}=16$ | none | 0.0 | 0.0 | 12.0 | 0.0 | 3.0 |
| $\delta_{\mathrm{img}}=16$ | **belief** | 0.0 | 0.0 | 7.0 | 0.0 | 1.8 |
| $\delta_{\mathrm{img}}=32$ | none | 0.0 | 0.0 | 3.0 | 0.0 | 0.8 |
| $\delta_{\mathrm{img}}=32$ | **belief** | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

### Table 3: State-Only Delay ($\delta_{\mathrm{img}}=0$)

Only state/proprioception is delayed, images are real-time.

| Delay | Comp. | libero_spatial | libero_object | libero_goal | libero_10 | Average |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| $\delta_{\mathrm{state}}=0$ ★ | none | 97.0 | 100.0 | 98.0 | 97.0 | 98.0 |
| $\delta_{\mathrm{state}}=1$ | none | 99.0 | 99.0 | 98.0 | 96.0 | 98.0 |
| $\delta_{\mathrm{state}}=1$ | **belief** | 96.0 | 100.0 | 97.0 | 96.0 | 97.2 |
| $\delta_{\mathrm{state}}=2$ | none | 97.0 | 98.0 | 98.0 | 95.0 | 97.0 |
| $\delta_{\mathrm{state}}=2$ | **belief** | 97.0 | 100.0 | 97.0 | 96.0 | 97.5 |
| $\delta_{\mathrm{state}}=4$ | none | 99.0 | 100.0 | 97.0 | 95.0 | 97.8 |
| $\delta_{\mathrm{state}}=4$ | **belief** | 98.0 | 100.0 | 98.0 | 97.0 | 98.2 |
| $\delta_{\mathrm{state}}=8$ | none | 95.0 | 99.0 | 97.0 | 92.0 | 95.8 |
| $\delta_{\mathrm{state}}=8$ | **belief** | 98.0 | 100.0 | 96.0 | 93.0 | 96.8 |
| $\delta_{\mathrm{state}}=16$ | none | 99.0 | 98.0 | 96.0 | 95.0 | 97.0 |
| $\delta_{\mathrm{state}}=16$ | **belief** | 96.0 | 99.0 | 98.0 | 92.0 | 96.2 |
| $\delta_{\mathrm{state}}=32$ | none | 97.0 | 98.0 | 99.0 | 91.0 | 96.2 |
| $\delta_{\mathrm{state}}=32$ | **belief** | 98.0 | 100.0 | 96.0 | 93.0 | 96.8 |

---
★ = baseline (no delay)

*Policy: pi05 — Generated from 37 unique experiment configs.*
