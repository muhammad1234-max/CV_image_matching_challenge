# Selected solution

**BLOCKED at dependency and checkpoint verification.** Selected Shuhei Nishimura (ns64). The author's README explicitly names mast3rhybrid-022-c.yaml as the best submission and evaluate_imc2025.py as a dataset-specific evaluation driver. kernel.py is the submission entry point. The frozen scorer must run separately after predictions.

Revision 07b5695b99a8ae4725d5d952f0065c2157fadd7d. Configuration SHA256 e7829be60865192d3ddcc8555b94bc8678415e265a566d8ce8386ef17b0cc763. The shortlist combines MASt3R ASMK (10 seeds, k=25), retrieval SPoC, DINOv2 and ISC. The latter three use top 10. MASt3R at size 512 matches dense features and sparse points from ALIKED (1280) and SuperPoint (1600), each capped at 4096. The published YAML disables explicit pre-clustering. Incremental mapping allows up to 25 models with minimum size 3. The author's experimental pre-clustering and VGGT branches do not belong in this run.
