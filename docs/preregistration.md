# Pre-registered hypotheses (written before the confirmatory tests)

- Written: 2026-10-26 17:01 IST, after the exploration split was explored (notebook sections 1-6) and before any
  test touched the confirmation split.
- Machine-readable version: `docs/preregistration.json`
- SHA-256 of that file at the time of writing: `24da1c62de4068c1e93eb4ae3cc1d2e21bec14cb5eed2e4b5119655fecd42321`
  (the notebook prints the hash again before running the tests; the two must match).

| Test | Problem | H0 | H1 | Side | Test | Smallest difference that matters |
|---|---|---|---|---|---|---|
| T1 | P1 | beta = 0 (within-stratum rating gap, large minus single hosts) | beta != 0 | two-sided | stratum fixed effects + host-clustered SE, t with G-1 df | 0.10 star |
| T2 | P1 | common odds ratio of a rating below 4.8 = 1 | != 1 | two-sided | Cochran-Mantel-Haenszel | OR outside 0.80-1.25 |
| T3 | P1 | host-level mean rating gap = 0 | != 0 | two-sided | Welch t-test | 0.10 star |
| T4 | P2 | within-stratum accuracy gap, hyped minus plain = 0 | != 0 | two-sided | stratified permutation (10,000), bootstrap CI | 0.10 star |

Family-wise alpha = 0.05 with Holm-Bonferroni across T1-T4. Seed 558 for every random step.
Population: established entire homes (10+ reviews) in the confirmation split.
Hyped = 2 or more hype words (largest count still covering >= 15% of exploration listings); plain = none.
