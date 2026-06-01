# Experiment Overview

## synthetic_auxiliary_ablation
This experiment tests whether a block proposer for dependent future tokens
produces longer verifier-accepted prefixes than a same-scale system that
predicts future tokens independently. It also checks that the comparison remains
at comparable model scale, so improvements cannot be explained by
parameter-count growth.

## synthetic_restricted_compute_n8
This experiment evaluates the same comparison after limiting the accepted
proposal horizon to 8 tokens. It checks whether the advantage remains when the
parallel proposer is constrained to a shorter prediction budget.
