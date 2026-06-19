> WARNING: runs span multiple FLOP budgets [0.0, 2000000000.0, 10000000000.0, 40000000000.0] -- rank by final bpb only within an equal budget; the plot shows the curves across budgets.

| rank | run | protocol | model | params | final FLOPs | final bpb | detail |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | transformer_b4e10 | prequential | transformer | 95,568 | 3.933e+10 | 4.2059 | stream=512B, pretrain=3.92e+10 |
| 2 | fast_weight_b4e10 | prequential | fast_weight | 95,568 | 3.936e+10 | 4.4017 | stream=512B, pretrain=3.92e+10 |
| 3 | context_mixing_reference | prequential | context_mixing | 0 | 4.283e+06 | 4.7779 | stream=512B, pretrain=0.00e+00 |
| 4 | fast_weight_b1e10 | prequential | fast_weight | 95,568 | 9.995e+09 | 5.9130 | stream=512B, pretrain=9.79e+09 |
| 5 | transformer_b1e10 | prequential | transformer | 95,568 | 9.962e+09 | 6.0125 | stream=512B, pretrain=9.79e+09 |
| 6 | fast_weight_b2e9 | prequential | fast_weight | 95,568 | 1.604e+09 | 7.3953 | stream=512B, pretrain=1.40e+09 |
| 7 | fast_weight_b0 | prequential | fast_weight | 95,568 | 2.052e+08 | 7.4095 | stream=512B, pretrain=0.00e+00 |
| 8 | transformer_b2e9 | prequential | transformer | 95,568 | 1.571e+09 | 7.6914 | stream=512B, pretrain=1.40e+09 |
| 9 | transformer_b0 | prequential | transformer | 95,568 | 1.727e+08 | 8.0003 | stream=512B, pretrain=0.00e+00 |
