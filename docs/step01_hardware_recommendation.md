# Step 1 — Hardware recommendation and readiness

Audit date: **2026-11-10, Asia/Karachi**.

## Recommendation

Use this laptop for source inspection, CSV/manifest checks, light tests and report writing. Use an existing university Linux GPU or an available Kaggle GPU environment for later reproduction. No resource was provisioned or paid service used.

The local constraints are measured: 4 CPU cores / 8 logical threads, 7.884 GiB usable RAM with only 1.233 GiB available during audit, 2 GiB Quadro K1100M, CPU-only torch, and just 2.43 GiB free on C:. D: has 132.37 GiB free, but dataset size is unknown. Evidence and commands: [environment audit](step01_environment_audit.md), [JSON](evidence/environment.json), [CIM/import results](evidence/confirmed_diagnostics.txt). These observations support declining a full local MASt3R reproduction; GPU functionality with modern CUDA dependencies was not demonstrated.

## Planning targets — estimates, not minimum requirements

| Work | Suggested compute | Rationale / uncertainty |
|---|---|---|
| Audit, report, tiny metadata tests | Existing laptop | Already sufficient for Step 1; minimize memory pressure |
| Organizer baseline pilot | Modern CUDA GPU with 16 GiB VRAM; 16–32 GiB host RAM | Conservative engineering starting point, not a measured guarantee |
| First-place reproduction | Prefer Linux, 24 GiB GPU and >=32 GiB host RAM; 8+ CPU cores | Headroom for matching and reconstruction; exact peak usage unmeasured |
| Closest author environment | Two T4 GPUs if available | Author reports this configuration; availability/quota for this user unknown |
| Storage | Manifest-derived capacity plus caches, weights and headroom | Do not assume the 132.37 GiB free locally is sufficient |

The author's two-T4 configuration and optimizations are documented in the [first-place report](https://github.com/ns-rokuyon/kaggle-image-matching-challenge-2025/blob/main/REPORT.md); bundled Linux CPython 3.11 wheels and CUDA-related dependencies appear in [pyproject.toml](https://github.com/ns-rokuyon/kaggle-image-matching-challenge-2025/blob/main/pyproject.toml). These sources motivate Linux/CUDA alignment. A single 24 GiB GPU is an alternative planning choice, not a claim of identical runtime or numerical behavior. No pricing or quota claim is made.

Prefer university resources already available to the user; otherwise investigate Kaggle account access and accelerators after rules review. Consider paid cloud only if those fail and the user separately authorizes a concrete budget. First measure a small pilot before buying capacity or forecasting full-run time.

## Readiness gates

| Gate | Status | Required next action |
|---|---|---|
| Workspace audit and scaffold | Complete | Supervisor review |
| Hardware/software inventory | Complete with CUDA-runtime caveat | Select later execution environment |
| Competition identity/task/schema | Documented via official indexed sources | Recheck against downloaded sample |
| Current ongoing runtime/internet/weight rules | Unverified | Read current signed-in rules/code requirements |
| Kaggle authenticated data access | Blocked: HTTP 401 | User refreshes local authentication; no secrets in chat |
| Rule acceptance | Unknown | User checks/reviews in Kaggle |
| Dataset manifest and storage bound | Blocked by access | List files, retrieve small metadata, compute footprint |
| Exact official evaluator parity | Unresolved | Obtain organizer source/thresholds; resolve code/prose ordering |
| Top-10 reproduction candidate | First place recommended provisionally | Supervisor selects baseline and compatible source revision |
| Environment lock / model assets | Not created | Later isolated setup and hash manifest |
| Baseline or novel-improvement result | Not started | Outside Step 1 |

**Exact readiness:** Step 1 documentation is ready for supervisor review. Step 2 execution is **not ready**. Authentication, rules verification, dataset metadata, evaluator parity and a suitable compute environment remain gates. No baseline score, training result, novel improvement, full CVPR report, NVS-NAR-IQA experiment or Kaggle submission is claimed.

## Verification and risk record

- Passed: environment JSON generation after serialization fix; hardware identification via read-only CIM; core package imports outside sandbox; credential-presence-only inspection; public source retrieval and hashes; tiny NumPy ordering check.
- Failed/unavailable: initial sandbox CIM/network calls, initial bounded imports, `uv` executable launch, `nvidia-smi`/`nvcc` discovery, missing Kornia/pycolmap/faiss/COLMAP, Kaggle listing (401), guessed seventh-place requirements (404), GitHub tree API (403 rate limit), and direct readable ongoing rules/data pages.
- Security controls applied: no secret values, account identifiers, response headers or signed download URLs printed; no repeated failed authentication, submissions, rule acceptance, heavy installation or large download.
- Resource risks: very low system-drive headroom and available RAM; full GPU workload unsupported by the observed installed stack. Dataset and weight footprints remain unknown.
- Research risks: source drift, missing compiled wheels, reference evaluator ordering discrepancy, and lack of verified current runtime rules. These are explicit limitations rather than filled-in assumptions.

Stop here. The next action is supervisor review, not automatic progression to Step 2.

## Step 1.5 review addendum

Original findings above are retained as the Step 1 record. The later [supervisor review](step015_supervisor_review.md) qualifies CLI availability/authentication, evaluator validation and current rule uncertainty. The [execution plan](step015_execution_environment.md) supersedes the initial baseline order with starter-first integration, followed by a verified top-10 target. Current readiness: [BLOCKED](step015_step2_readiness.md).
