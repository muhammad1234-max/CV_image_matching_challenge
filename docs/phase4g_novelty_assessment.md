# Prior work and novelty

The normalized-cut objective and spectral proposal are established. Reusing it to nominate an image-graph boundary is an engineering choice. [Shi and Malik](https://doi.org/10.1109/34.868688)

COLMAP's incremental multiple-model reconstruction, bundle adjustment, triangulation and filtering supply the underlying models and diagnostics. Viewing-graph optimization has already used geometric constraints to improve SfM. [Schonberger and Frahm](https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html) [Sweeney et al.](https://openaccess.thecvf.com/content_iccv_2015/html/Sweeney_Optimizing_the_Viewing_ICCV_2015_paper.html)

Disambiguation using geodesic context already addresses repeated structures. Graph partitioning also has a separate role in distributing bundle adjustment for consensus, where optimizing a shared reconstruction is the objective. [Yan et al.](https://openaccess.thecvf.com/content_cvpr_2017/html/Yan_Distinguishing_the_Indistinguishable_CVPR_2017_paper.html) [Zhang et al.](https://openaccess.thecvf.com/content_iccv_2017/html/Zhang_Distributed_Very_Large_ICCV_2017_paper.html)

The hypothesis under test is that a single budgeted alternative reconstruction, judged on paired surviving pixel observations with per-camera loss accounting and an untouched pass-through fallback, can screen proposed scene boundaries. This focused comparison establishes no publication-priority claim. Two previously studied development datasets cannot establish generalization.
