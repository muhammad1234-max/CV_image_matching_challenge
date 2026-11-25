# Phase 3B baseline results

**Registered organizer_baseline_v1_dev_full** on all 11 development datasets (1,563 images). Fold A scored 0.408975, Fold B scored 0.429336, and the equal-dataset mean was 0.418230 under both scoring variants. See [full report](../reports/phase3b_full/report.json) and [per-dataset metrics](../reports/phase3b_full/dataset_metrics.csv).

Batch 11 registered 225/225 images in one cluster across three reference scenes. Its mAA was 0.846065, clusterness 0.333333 and combined score 0.478247. Reconstruction took 7,886 seconds. The final session took 2 h 53 min and the prior timed-out session took 2 h 27 min. Across development, 1,435 cameras registered. Recorded session times including the timed-out attempt total at least 13 h 27 min, excluding export and any lost sessions. Peak worker RSS was 4.12 GiB and peak PyTorch allocation was 1.59 GiB. Use the archived four-hour batch 11 reconstruction allowance for paired comparisons. Supervisor review is next.

## Historical execution notes

| Required result | Current value |
|---|---|
| Per-dataset metrics and runtimes | Batches 01 through 10 validated |
| Fold A aggregate, six datasets | 0.4089749063699912 (primary and sensitivity) |
| Fold B aggregate, five datasets | Unavailable |
| Combined development aggregate, eleven datasets | Unavailable |
| Primary versus sensitivity differences | Zero for all ten completed datasets |
| Phase 3B GPU/CPU peaks and compute time | Per completed dataset in report.json; full run unavailable |
| Immutable full baseline record | Not created |
| Full report ZIP | Not created; requires all eleven verified results |

The primary score uses the pinned participant scorer, not a claim of hidden Kaggle scorer equivalence. Scores are fractions in [0,1]. Sensitivity uses the written mAA-first assignment interpretation on exactly the same predictions and metadata. Each dataset's combined score is produced by the pinned scorer. Fold and combined means are **derived equal-dataset-weighted means of those dataset scores**, not a harmonic mean of global component averages. No image-weighted score is used as the benchmark. Fold means are emitted only when every dataset in that fold is present; the combined mean requires all eleven.

`scripts/merge_baseline_batches.py` independently recomputes both score variants and diagnostics. It emits `report.json`, `dataset_metrics.csv`, and `scores_and_coverage.svg`. All exported input hashes, stages, resource measurements, raw logs, predictions, environments and diagnostic contingency tables remain in its evidence directory. Completed zero-registration inference is scored as actual all-outlier/NaN predictions and included in means. Interrupted inference is incomplete and blocks full aggregation/registration.

The earlier ETs smoke score (0.3589420655, both variants) is retained only in Phase 3A evidence. Batch 01 used a fresh execution. There are no new research improvements or leaderboard claims.

## Batch 01 validated result (2026-11-13)

ETs: 22/22 registered, one reconstructed cluster, zero predicted outliers. Primary mAA 0.28846153846153844; clusterness 0.4318181818181818; primary and sensitivity combined scores both 0.345873786407767. Both true scenes merged and all three true outliers joined the model. Selected/retained/verified pairs: 226/153/137. Session wall time 156.084 seconds; successful inference stages 115.913 seconds. GPU allocated/reserved peaks 1.58/2.69 GiB.

Source, configuration, weights, core environment, membership, assignments and independently recomputed scores passed validation. This fresh result differs from the smoke result; retain it without score-driven reruns. The source/settings match, but bitwise numerical reproducibility is not established.

Cache archive content integrity passed. F/colmap.db is absent from its expected checkpoint inventory, so that stage is not portable from this cache; the exporter omits files whose post-run bytes differ from their saved stage hashes. Recompute F if stage resumption is required. Completed dataset reuse does not require that database. Fold and combined aggregates remain unavailable.

## Batch 02 validated result (2026-11-14)

stairs: 51 required prediction rows; 40 registered (78.43%), 11 predicted outliers, two reconstructed models but one final predicted cluster. The original worker overwrites an image pose when multiple reconstructions contain that image; preserve this executed behavior. Both true scenes merge in the final cluster. All 11 predicted outliers are false outliers (no true outliers). Primary/sensitivity mAA 0.027777777777777776, clusterness 0.5, combined score 0.05263157894736842. Both score variants independently recomputed with zero difference.

Selected/retained/verified pairs: 883/391/350. Verified graph component sizes 50 and 1. Successful inference stage time 251.852 seconds; session wall time 312.091 seconds. Peak allocated/reserved GPU bytes 1396183552/1988100096; worker CPU RSS peak 1840939008 bytes. Across returned batches, session wall time sums to 468.175 seconds (excludes archive export).

Source/settings/weights/core environment and identity coverage match the frozen implementation. Cache archive hashes pass, but F/colmap.db is absent and cannot support direct F-stage reuse, as in batch 01. No model settings changed. All fold and full-development aggregates remain unavailable; baseline is NOT REGISTERED. Next planned batch: 03, imc2023_haiper, 54 images.

## Batch 03 validated (2026-11-14)

{
  "images": 54,
  "registered_images": 54,
  "predicted_clusters": 2,
  "scores": {
    "participant": {
      "combined_score": 0.6315135834411384,
      "maa": 0.6277777777777778,
      "clusterness": 0.6352941176470588
    },
    "prose_sensitivity": {
      "combined_score": 0.6315135834411384,
      "maa": 0.6277777777777778,
      "clusterness": 0.6352941176470588
    }
  },
  "successful_stage_seconds": 220.5679627520003,
  "diagnostics": {
    "false_outliers": 0,
    "missed_outliers": 0,
    "merged_clusters": {
      "cluster1": [
        "bike",
        "chairs"
      ]
    },
    "split_scenes": {},
    "contingency": [
      {
        "true_scene": "bike",
        "predicted_scene": "cluster1",
        "images": 15
      },
      {
        "true_scene": "chairs",
        "predicted_scene": "cluster1",
        "images": 16
      },
      {
        "true_scene": "fountain",
        "predicted_scene": "cluster0",
        "images": 23
      }
    ]
  }
}

Source/config/environment and independent score recomputation passed. Cache archive integrity passed; non-reusable checkpoint artifacts: ['datasets/imc2023_haiper/F/colmap.db']. Full baseline remains NOT REGISTERED.

## Batch 04 validated result

The theatre/church dataset registered 74 of 76 images. All 50 church images and 24 theatre images joined one cluster. Two theatre images were predicted as outliers. Both combined scores independently recomputed to 0.3832599118942731, with mAA 0.3107142857142857 and clusterness 0.5. Session wall time was 864.20 seconds. Cache integrity passed for archived files. The stage F geometry database is absent, so resuming from that checkpoint requires recomputation.

## Batch 05 validated result

Dioscuri/Baalshamin registered 131 of 138 images across seven clusters. Both scorers returned 0.5628763937749686. All 24 true outliers entered clusters and seven scene images were classified as outliers. The session took 1415.14 seconds. Archived cache files passed their integrity checks. The stage F database is missing from the portable cache.

## Batch 06 validated result

Both scorers returned 0.24329896907216497 for fbk_vineyard. All 163 images registered in one cluster containing all three true scenes. Session wall time was 3029.67 seconds. Archive integrity checks passed for both files. The portable cache omits the stage F geometry database.

## Batch 07 validated result

amy_gardens has 200 prediction rows, with 171 registered images in three clusters of sizes 126, 41 and 4. Its single true scene is split across all three clusters, with 29 false outliers. Both combined scores independently recomputed to 0.29252437703141926, mAA to 0.1713197969543147 and clusterness to 1.0. The main Kaggle session took 4026.04 seconds. A second 8.73-second session reused the completed result. Cache file integrity passed. The stage F geometry database is omitted from the portable cache.

## Batch 08 validated result

All 200 images registered. The two scenes of 100 images each merged into one cluster. Both combined scores independently recomputed to 0.6146971201588878. Local verification reused deterministic Horn alignment results for identical input arrays and parameters across scoring calls. The completed Kaggle session took 7110.01 seconds, including 3469.97 seconds of evaluation. An earlier incomplete session lacks a final wall time. Cache file hashes passed. The stage F geometry database is absent from the portable cache.

## Batch 09 validated result

Heritage registered 132 of 209 images in eight clusters. Both combined scores independently recomputed to 0.5310435931307793. Session wall time was 2398.03 seconds. Both archive inventories passed. The portable cache omits the stage F geometry database. The cumulative review rechecked archive and prediction hashes for batches 01 through 08 and reused their previously validated scores. For batch 09, identical Horn alignment inputs reused locally computed alignments across score components.

## Batch 10 validated result

The baseline registered 223 of 225 images in one cluster across three true scenes. Two Brandenburg Gate images were false outliers. The primary and sensitivity combined scores independently recomputed to 0.4645614035087719. Session wall time was 9619.17 seconds. Both archives passed integrity checks. The portable cache omits the stage F geometry database. Earlier validated scores were reused after archive and prediction hash checks. Identical alignment inputs reused local alignment computations during batch 10 scoring.
