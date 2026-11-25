Use notebooks/step03c_full_ns64_kaggle_export.ipynb on trusted CPU Kaggle. Set BATCH to a list of the authorized dataset names. The exporter writes separate opaque ZIPs and prints package hashes. Use notebooks/step03c_full_ns64_reproduction.ipynb in a fresh clean T4 runtime. Set the same BATCH and SESSION_SECONDS, then upload the corresponding opaque archives. The worker setup uses the pinned clean-runtime ZIP. Save every completed or partial report immediately. The selected order should follow image count. Four-hour per-dataset caps remain conservative defaults. Each new attempt needs 140 minutes of declared remaining session time and reserves at least 20 minutes for exports.

After returning archives, independently evaluate them and extend the trusted archive index with reviewed hashes. Subsequent clean sessions select only remaining datasets. A local existing attempt directory prevents rerun. Across runtime resets, the reviewed administrative index is the authority. No numerical mid-stage checkpoint resume has been demonstrated. The immutable receipts provide evidence, while live local intermediate files should be preserved on timeout. Exported lightweight archives omit large intermediates; they cannot restart COLMAP.

Authorized order and counts follow.
- imc2024_dioscuri_baalshamin, 138 images
- fbk_vineyard, 163 images
- amy_gardens, 200 images
- pt_stpeters_stpauls, 200 images
- imc2023_heritage, 209 images
- pt_brandenburg_british_buckingham, 225 images
- pt_sacrecoeur_trevi_tajmahal, 225 images
