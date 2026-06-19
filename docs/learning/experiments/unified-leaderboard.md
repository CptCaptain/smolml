> WARNING: runs span multiple FLOP budgets [0.0, 2000000000.0, 10000000000.0, 40000000000.0] -- rank by final bpb only within an equal budget; the plot shows the curves across budgets.

| rank | run | protocol | model | params | final FLOPs | final bpb | detail |
| ---: | --- | --- | --- | ---: | ---: | ---: | --- |
| 1 | transformer_b4e10 | prequential | transformer | 95,568 | 4.001e+10 | 4.1564 | stream=512B, pretrain=3.98e+10 |
| 2 | fast_weight_b4e10 | prequential | fast_weight | 95,568 | 4.005e+10 | 4.3585 | stream=512B, pretrain=3.98e+10 |
| 3 | transformer_b1e10 | prequential | transformer | 95,568 | 9.812e+09 | 4.6139 | stream=512B, pretrain=9.64e+09 |
| 4 | fast_weight_b1e10 | prequential | fast_weight | 95,568 | 9.844e+09 | 4.7534 | stream=512B, pretrain=9.64e+09 |
| 5 | context_mixing_reference | prequential | context_mixing | 0 | 4.283e+06 | 4.7779 | stream=512B, pretrain=0.00e+00 |
| 6 | fast_weight_b2e9 | prequential | fast_weight | 95,568 | 2.133e+09 | 6.8241 | stream=512B, pretrain=1.93e+09 |
| 7 | transformer_b2e9 | prequential | transformer | 95,568 | 2.101e+09 | 7.0550 | stream=512B, pretrain=1.93e+09 |
| 8 | fast_weight_b0 | prequential | fast_weight | 95,568 | 2.052e+08 | 7.4095 | stream=512B, pretrain=0.00e+00 |
| 9 | transformer_b0 | prequential | transformer | 95,568 | 1.727e+08 | 8.0003 | stream=512B, pretrain=0.00e+00 |
