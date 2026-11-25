# Step 1 — Baseline research

Audit date: **2026-11-10, Asia/Karachi**. No baseline has run; no leaderboard result has been reproduced. Burden estimates below are qualitative engineering judgments, not measurements.

## Organizer baseline

The [organizer notebook](https://www.kaggle.com/code/eduardtrulls/imc25-submission) is public and lists DINOv2-base, ALIKED-n16, LightGlue-ALIKED, a pycolmap/LightGlue/Kornia package bundle and `imc25_utils` inputs. Indexed metadata identifies Apache 2.0 licensing for the notebook; dependency/weight licenses still require separate review. Its displayed nine-second run is not evidence of full hidden-test runtime.

The intended baseline is DINOv2 retrieval, ALIKED features, LightGlue matches and COLMAP reconstruction, as specified in the project brief and corroborated by the notebook inputs and [winner's description of the inherited COLMAP pipeline](https://github.com/ns-rokuyon/kaggle-image-matching-challenge-2025/blob/main/REPORT.md). Exact organizer notebook code and parameter defaults were not retrieved, so they remain unverified. [LightGlue's upstream implementation](https://github.com/cvg/LightGlue) provides the matcher/feature integration source.

**Dependency plan:** PyTorch, model-loading dependencies, ALIKED/LightGlue, pycolmap, NumPy and notebook utilities. Pin actual notebook package versions when accessible. **Estimated burden:** lowest of the three principal candidates; feature extraction and selected-pair matching need a suitable GPU, with CPU/RAM demand from reconstruction. **Risks:** inaccessible source snapshot, offline wheel/weight packaging, COLMAP API drift, and excessive pair count. This is a sanity baseline, not fulfillment of the top-10 reproduction requirement.

## First place — ns64 / Shuhei Nishimura

Public code: [repository](https://github.com/ns-rokuyon/kaggle-image-matching-challenge-2025). Author's primary explanation: [write-up](https://www.kaggle.com/c/image-matching-challenge-2025/discussion/583058), [repository report](https://github.com/ns-rokuyon/kaggle-image-matching-challenge-2025/blob/main/REPORT.md).

The reported final approach unions MASt3R-ASMK, MASt3R-SPoC, DINOv2 and ISC retrieval candidates; matches with MASt3R plus ALIKED/SuperPoint keypoints; then reconstructs with COLMAP. Crucially, the report says experimental pre-clustering was **not used in the final version**. It describes MASt3R at 512 pixels, ALIKED at 1280 and SuperPoint at 1600, CUDA RoPE, attention/AMP changes, and dataset-level parallelism on two T4s. These are author-reported methods, not our experiments. Sources: [report](https://github.com/ns-rokuyon/kaggle-image-matching-challenge-2025/blob/main/REPORT.md), [write-up](https://www.kaggle.com/c/image-matching-challenge-2025/discussion/583058).

Dependencies include torch 2.5.1, torchvision 0.20.1, transformers 4.47.0, NumPy <2, pycolmap >=3.11.1, faiss-cpu, CuPy CUDA12x, ASMK, curope and numerous bundled packages. Linux CPython 3.11 wheel paths are explicitly configured. Source: [pyproject.toml](https://github.com/ns-rokuyon/kaggle-image-matching-challenge-2025/blob/main/pyproject.toml). Upstream model code/build guidance: [MASt3R](https://github.com/naver/mast3r).

**Estimated burden:** highest of these candidates; multi-retriever features and pairwise MASt3R inference add compute and storage. Batch-one inference on a modern 16–24 GiB GPU is a planning target, not a proven minimum. **Risks:** absent prebuilt ASMK/curope wheels, numerous adapted dependencies, checkpoint provenance, Linux/CUDA extension builds, and mutable source compatibility. The scoring-order discrepancy must be resolved before declaring reproduction parity. The README's best-config path is `conf/pipeline/imc2025/mast3rhybrid/mast3rhybrid-022-c.yaml`; a future checkout must verify that artifact and its revision before execution. Source: [repository](https://github.com/ns-rokuyon/kaggle-image-matching-challenge-2025).

## Second place — Nazarko99

The official [second-place write-up](https://www.kaggle.com/competitions/image-matching-challenge-2025/writeups/nazarko99-2nd-place-solution) documents SuperPoint with a GIM checkpoint, Alike/LightGlue and SIFT/nearest-neighbor matching, followed by match-guided crops, additional local matching, concatenation, RANSAC and COLMAP. It reports 1280-pixel longest-side images and filters at 30 initial / 150 final matches. Preserve the author's “Alike” naming until code establishes the exact model; do not silently equate it with ALIKED.

**Public code:** no complete repository or executable notebook was identified in the accessed write-up. **Dependencies inferred from described components:** PyTorch/GIM, the specified local feature models, LightGlue, SIFT/RANSAC tooling and COLMAP; versions unknown. **Estimated burden:** multiple full-image/crop matching passes, likely greater than the organizer baseline; no defensible wall-clock or peak-memory figure. **Reproduction risks:** exact weights, crop-generation rules, matcher settings and software environment are missing. This is a documented top-10 comparison, but currently a weaker reproduction candidate than first place. All method/rank claims above come from the linked author write-up.

## Additional public-code option — seventh place

[Official seventh-place write-up](https://www.kaggle.com/competitions/image-matching-challenge-2025/writeups/kaqln-7th-place-solution) describes all-pair matching/rotation correction followed by inlier-graph clustering; small components and isolated nodes are treated as outliers. [Public repository](https://github.com/AymenLass/Image-Matching-Challenge-2025-7th-place-solution) contains reconstruction and evaluator code. The guessed root `requirements.txt` returned 404, so there is no verified dependency lock from this audit. Experimental filenames mentioning other models must not be treated as proof those models were in the final submission.

**Estimated burden:** all-pairs work scales quadratically in image count; rotation/crop passes can increase it. **Risk:** dependency/final-entrypoint ambiguity and the same reference scorer ordering issue. This is a fallback to investigate, not a reproduced system.

## Recommendation for supervisor review

Select first place provisionally for the required top-10 reproduction because code, method description and a candidate configuration are public. Use the organizer baseline later as a plumbing check. Before approving any full run, obtain valid Kaggle access, resolve evaluator provenance, pin an internally compatible commit, and demonstrate offline dependency/weight availability. Do not design or claim a novel improvement yet. Wall-clock estimates should follow a later authorized small-scene pilot with measured pair counts, preprocessing, matching, SfM, and peak resource use.

## Step 1.5 review addendum

Original findings above are retained as the Step 1 record. The later [supervisor review](step015_supervisor_review.md) qualifies CLI availability/authentication, evaluator validation and current rule uncertainty. The [execution plan](step015_execution_environment.md) supersedes the initial baseline order with starter-first integration, followed by a verified top-10 target. Current readiness: [BLOCKED](step015_step2_readiness.md).
