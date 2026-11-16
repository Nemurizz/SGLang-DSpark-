# Production-shaped arrival replay

| Phase | Rate req/s | Full E2E mean | Paired E2E mean | Change | Full p95 | Paired p95 | Paired hit / eviction / recovery |
|---|---:|---:|---:|---:|---:|---:|---:|
| night_low | 1 | 4405.81 | 5669.31 | -28.68% | 6352.76 | 7340.40 | 2038 / 0 / 0 |
| day_medium | 4 | 4886.21 | 4624.24 | 5.36% | 7463.93 | 7084.19 | 4151 / 0 / 0 |
| peak_burst | 20 | 10369.77 | 9831.11 | 5.19% | 18279.79 | 17437.52 | 12343 / 16 / 0 |
| cooldown | 2 | 2574.52 | 2642.44 | -2.64% | 3222.53 | 3327.33 | 3587 / 2 / 0 |
