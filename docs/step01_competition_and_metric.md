# Step 1 — Competition and metric

Audit date: **2026-11-10, Asia/Karachi**. “Verified” below means source content was accessible, not that a submission was executed.

## Competition identity and access quality

The workshop explicitly identifies the ongoing 2025 competition as its 2026 challenge. Its June 4, 2026 workshop date is now past; the page states no hard challenge deadline. Source: [official workshop](https://image-matching-workshop.github.io/).

Direct Kaggle page opens returned empty content or access errors. Search-indexed official content recovered the ongoing overview/evaluation/submission text. It describes a combined public/private leaderboard and no prizes, medals, points, or deadlines. These are documented statements; live account-specific UI state was not verified. Source: [ongoing evaluation](https://www.kaggle.com/competitions/image-matching-challenge-2025-ongoing/overview/evaluation).

## Task and documented aggregation

Partition each dataset's images into scenes or `outliers`, then estimate scene camera poses. Ground-truth scenes may look alike without overlapping. For each true scene, the prose selects the predicted cluster with maximum mAA, breaking ties by clustering score; a cluster may serve multiple scenes. Predicted outliers are excluded. Dataset clustering precision is `sum_i |S_i intersect C_j(i)| / sum_i |C_j(i)|`. Pose accuracy is aggregated over scenes, combined with clustering by harmonic mean, then averaged across datasets. Source: [ongoing evaluation](https://www.kaggle.com/competitions/image-matching-challenge-2025-ongoing/overview/evaluation).

Interpretation: scene clustering and geometric registration both matter. Rejecting difficult images may improve cluster purity while losing pose coverage. A visually plausible reconstruction is insufficient evidence of official score quality.

## CSV contract

The documented columns, in order, are:

```csv
dataset,scene,image,rotation_matrix,translation_vector
example_set,cluster1,image_a.png,1;0;0;0;1;0;0;0;1,0;0;0
example_set,outliers,image_b.png,nan;nan;nan;nan;nan;nan;nan;nan;nan,nan;nan;nan
```

These are synthetic illustration rows, not downloaded competition data. Each test image needs one row. Rotation has nine semicolon-separated values in row-major order; translation has three. Scene names are arbitrary except reserved `outliers`, and may repeat across datasets. A clustered but unregistered image may retain its scene with NaN pose values. Intrinsics are not submitted. Source: [original official submission specification](https://www.kaggle.com/competitions/image-matching-challenge-2025). The same schema was recovered in the [ongoing page](https://www.kaggle.com/competitions/image-matching-challenge-2025-ongoing/overview/evaluation).

Proposed validation: check exact columns, unique `(dataset,image)` keys, complete expected image coverage, numeric field lengths, row-major serialization, valid rotations for estimated poses, and NaN conventions. Use the actual ongoing sample submission as the authority once accessible.

## Reference evaluator inspection

An organizer-associated submission notebook is public and lists `imc25_utils` as input, but its utility source was not obtained directly from Kaggle: [organizer notebook](https://www.kaggle.com/code/eduardtrulls/imc25-submission). Do not label a participant copy as an authenticated current official scorer.

The first-place repository's [evaluation driver](https://github.com/ns-rokuyon/kaggle-image-matching-challenge-2025/blob/main/evaluate_imc2025.py) calls `utils.imc25.metric.score` with training labels, threshold CSV, `inl_cf=0`, `strict_cf=-1`, and no split mask. Its [metric source](https://github.com/ns-rokuyon/kaggle-image-matching-challenge-2025/blob/main/utils/imc25/metric.py) was fetched as text only; SHA256 `c7ae94306ba0ae4ce9c00e9abc7e15bd99f3091fa063b5826b0dd567a6c3f261`. It is a useful public reference, with parity still unverified.

Code observations from that reference:

- Camera center is `-R.T @ t`, consistent with world-to-camera extrinsics.
- Horn-style registration estimates a similarity transform including scale, with triplet hypotheses and inlier refinement; a best model is retained per threshold.
- Threshold arrays are read per dataset/scene from CSV. Their actual values are unavailable here.
- Defaults skip the first two thresholds and subtract three registration cameras. For a scene with `n` ground-truth cameras, the code uses `sum_t max(count(error_t < threshold_t)-3,0) / (L*(n-3))`, returning zero when its numerator is zero.
- Dataset mAA pools corrected inlier counts over `L*(sum_i n_i - 3*number_of_scenes)` with zero handling, rather than blindly averaging scene mAAs. Harmonic fusion is zero when both components sum to zero. The final dataset mean is multiplied by 100.

These implementation details are **reference-code observations**, not invented ongoing guarantees. Source: [metric functions](https://github.com/ns-rokuyon/kaggle-image-matching-challenge-2025/blob/main/utils/imc25/metric.py). Do not substitute older pairwise angular-pose thresholds or a generic clustering metric.

## Material discrepancy to resolve

At line 760 of the downloaded first-place source, selection is `np.lexsort((-mAA_table[i], -cluster_table[i]))[0]`. NumPy uses the last key as primary; here `cluster_table` stores overlap counts. A tiny local check with mAA `[0.9,0.1]` and overlaps `[4,10]` selected index 1. Thus this code prioritizes overlap count, unlike the prose's mAA-first selection. The [seventh-place metric copy](https://github.com/AymenLass/Image-Matching-Challenge-2025-7th-place-solution/blob/main/imc25metric.py) contains the same ordering. Evidence: [local check](evidence/metric_order_check.txt), saved source snapshots under `evidence/public_sources/`.

This is not proof of an error in Kaggle's deployed scorer. It blocks a claim of exact local parity. Obtain and hash current organizer utilities; compare ordering, normalization, thresholds and small-scene/NaN behavior. The fetched first-place scorer returns two nested tuples containing full/public/private results; preserve that structure when integrating it. Do not silently “fix” reference code before selecting the reproduction version.

## Runtime, internet, weights and rules

| Question | Verified finding / unresolved status |
|---|---|
| Original 2025 execution | Notebook submission, CPU/GPU runtime <=9 hours, internet disabled, freely/publicly available external data and pretrained models permitted, output `submission.csv`. [Historical official source](https://www.kaggle.com/competitions/image-matching-challenge-2025) |
| Ongoing exact runtime | **Unknown**; no readable current runtime clause recovered. Do not automatically inherit nine hours. |
| Ongoing internet / model weights | **Unknown**; design for offline execution provisionally, then verify current rules. |
| Ongoing legal rules / acceptance | Rules page unreadable; account acceptance not verified. [Rules URL](https://www.kaggle.com/competitions/image-matching-challenge-2025-ongoing/rules) |
| Data page and sizes | Access failure; no reliable live manifest. [Data URL](https://www.kaggle.com/competitions/image-matching-challenge-2025-ongoing/data) |
| Daily limits / accelerator availability | **Unknown** for this account and competition. |

The general Kaggle documentation requires acceptance of competition rules before data download/submission; it is not a substitute for this competition's specific terms. [Kaggle documentation](https://www.kaggle.com/docs/competitions).

## Step 1.5 review addendum

Original findings above are retained as the Step 1 record. The later [supervisor review](step015_supervisor_review.md) qualifies CLI availability/authentication, evaluator validation and current rule uncertainty. The [execution plan](step015_execution_environment.md) supersedes the initial baseline order with starter-first integration, followed by a verified top-10 target. Current readiness: [BLOCKED](step015_step2_readiness.md).
