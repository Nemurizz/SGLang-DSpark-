# Paired cache prime and eviction experiment

| Phase | Full E2E mean | Paired E2E mean | Change | Full p95 | Paired p95 | Paired hit / evictions |
|---|---:|---:|---:|---:|---:|---:|
| cold | 8019.07 | 8408.35 | -4.85% | 9704.31 | 10442.91 | 40 / None |
| prime | 2614.51 | 2942.82 | -12.56% | 2888.29 | 3197.43 | 68 / None |
| warm | 3989.42 | 5618.62 | -40.84% | 5371.09 | 6990.69 | 4156 / None |
| evict | 11438.03 | 11112.47 | 2.85% | 20317.72 | 19766.11 | 4473 / None |
| rehit | 3816.99 | 3744.70 | 1.89% | 5132.52 | 5037.32 | 6545 / None |
