# Synthetic comparisons

| Fixture | B2 component decisions | B2 removed | Ncut removed | Pass-through removed |
|---|---|---:|---:|---:|
| clean | ABSTAIN | 0 | 0 | 0 |
| weak | SPLIT | 2 | 2 | 0 |
| dense | ABSTAIN | 0 | 0 | 0 |
| repeat | RETAIN | 0 | 9 | 0 |
| unequal | SPLIT | 2 | 2 | 0 |
| sparse_true | ABSTAIN | 0 | 1 | 0 |
| noisy | SPLIT | 2 | 2 | 0 |
| disconnected | ABSTAIN, ABSTAIN | 0 | 0 | 0 |
| isolated | ABSTAIN, ABSTAIN, ABSTAIN, ABSTAIN, ABSTAIN, ABSTAIN, ABSTAIN, ABSTAIN, ABSTAIN, ABSTAIN, ABSTAIN, ABSTAIN, ABSTAIN, ABSTAIN, ABSTAIN, ABSTAIN | 0 | 0 | 0 |
| ambiguous_single | SPLIT | 2 | 2 | 0 |

Weak and ambiguous_single have identical observations and opposite intended decisions. The latter is a false split of one scene. Dense and repeat encode misleading edges between cliques. These fixtures measure behavior, not competition accuracy.
