# Closest mechanisms and research scope

SfM systems organize feature observations into tracks and triangulate them using geometry. Incremental SfM can extend, merge and filter tracks with geometric checks. Here a conflict-free observation component is only a hypothesis. [Schonberger and Frahm, Structure-from-Motion Revisited](https://openaccess.thecvf.com/content_cvpr_2016/html/Schonberger_Structure-From-Motion_Revisited_CVPR_2016_paper.html)

Zhou et al. optimize affinity and cycle consistency jointly using a shared factorization. Maset et al. jointly update noisy partial matches for consistency. Both address agreement across views more directly than counting closed triangles. [Multi-Image Matching via Fast Alternating Minimization](https://openaccess.thecvf.com/content_iccv_2015/html/Zhou_Multi-Image_Matching_via_ICCV_2015_paper.html) [Practical and Efficient Multi-View Matching](https://openaccess.thecvf.com/content_iccv_2017/html/Maset_Practical_and_Efficient_ICCV_2017_paper.html)

Tron et al. formulate multi-image matching as feature clustering with consistent clique structure. Their approach seeks a consistent association; our conservative conflict flag discards ambiguous components from clean support without resolving them. [Fast Multi-Image Matching via Density-Based Clustering](https://openaccess.thecvf.com/content_ICCV_2017/papers/Tron_Fast_Multi-Image_Matching_ICCV_2017_paper.pdf)

Yan et al. use geodesic context in the image collection to detect visually ambiguous connections. Cai et al. learn a visual pair classifier and filter scene-graph edges before SfM. Both target incorrect connections surviving local matching. [Geodesic context](https://openaccess.thecvf.com/content_cvpr_2017/html/Yan_Distinguishing_the_Indistinguishable_CVPR_2017_paper.html) [Doppelgangers](https://openaccess.thecvf.com/content/ICCV2023/html/Cai_Doppelgangers_Learning_to_Disambiguate_Images_of_Similar_Structures_ICCV_2023_paper.html)

The narrower research question is whether observation-level corroboration adds useful boundary discrimination after controlling for match support and image topology. Cycle counting, conflict detection and grouping track evidence across a proposed boundary do not establish novelty. These results supply neither calibrated abstention nor a new conflict-resolution algorithm.

