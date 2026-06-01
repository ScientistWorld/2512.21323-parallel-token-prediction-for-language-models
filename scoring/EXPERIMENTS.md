# Experiment Overview

## codecontests_auxiliary_ablation
This experiment tests the central ablation behind the paper's multi-token
generation claim: a system that proposes dependent future tokens should produce
longer teacher-verified accepted prefixes than a system that predicts future
tokens independently. It also checks that the comparison remains at comparable
model scale, so improvements cannot be explained by parameter-count growth.
