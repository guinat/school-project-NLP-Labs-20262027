## Context length and number of heads, 2000 steps

| run | params | train loss | val loss | best val loss | time (s) |
|---|---|---|---|---|---|
| block_size=16 | 208,705 | 1.9122 | 2.0148 | 2.0148 (step 1999) | 40 |
| baseline | 209,729 | 1.8729 | 1.9679 | 1.9679 (step 1999) | 59 |
| block_size=64 | 211,777 | 1.8158 | 1.9359 | 1.9359 (step 1999) | 112 |
| n_head=1 | 209,729 | 1.8467 | 1.9623 | 1.9623 (step 1999) | 51 |

## Dropout, first 50,000 training characters, 3000 steps

| run | params | train loss | val loss | best val loss | time (s) |
|---|---|---|---|---|---|
| dropout=0.0 | 209,729 | 1.4836 | 2.3745 | 2.3124 (step 2000) | 87 |
| dropout=0.2 | 209,729 | 1.6812 | 2.3140 | 2.3116 (step 2500) | 106 |
