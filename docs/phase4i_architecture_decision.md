# Architecture decision

**Select Option A, one reconstruction trial judged on corresponding feature observations.** The design question is whether a boundary can improve the fit of the original measurements while retaining the cameras and measurements that made the original reconstruction useful. It needs no new training labels. The alternative would be a boundary reliability estimator built from verified match counts, graph neighborhoods and feature tracks.

| Evidence available before model freeze | Consequence for Option A | Consequence for Option B |
|---|---|---|
| 4B–4C could validate calibrated rotation checks in simulation. None of the 454 Haiper pairs passed the real focal-prior requirement. | Use native optimized cameras in each branch to measure pixel error. | Relative-pose confidence would frequently abstain. |
| 4D produced the same partition evidence for a true boundary and a single-scene viewpoint bottleneck. In 4E, 56 of 68 false edges had exact track witnesses. | A graph proposes one test. Reconstruction provides a second measurement. | A classifier needs varied, source-disjoint negative **and** positive boundary examples. One numerical parent supplies neither a defensible training population nor a calibration set. |
| 4F demonstrated a transactionally audited cut and paired mapper replay. 4H rejected it on point count before examining matched observations. | A comparison indexed to the input database is implementable and falsifiable. The known Haiper score supplied no parameter fit. | There is no evidence that a pre-mapper predictor recovers the missed decision. |
| Stairs and Theater/Church failed before capture on free infrastructure. | Cost of up to one further map is significant. Use one immutable parent and a hard stop. | One map would be cheaper, but the costly first matching/capture would still be required. |

The course-report contribution is an auditable selection experiment with a negative control and a reproducible failure condition. The candidate may abstain on every new group. A more flexible split can lower in-sample reprojection error even for the wrong semantic partition. A publishable new SfM algorithm and independent accuracy gain both remain unestablished.

## Closest work

- Shi and Malik introduced normalized cut. The proposal stage here inherits that objective. [Normalized Cut and Image Segmentation](https://www2.eecs.berkeley.edu/Pubs/TechRpts/1997/5487.html)
- Multi-stage SfM already partitions camera collections into independently reconstructed components and later merges them. Gong et al. already use local clusters and pose consistency to disambiguate matching. Both precede the use of reconstruction as feedback on partitioning. [Multi-stage SfM](https://arxiv.org/abs/1512.06235) · [Gong et al.](https://doi.org/10.1016/j.isprsjprs.2024.02.016)
- Zach et al. use loop constraints to disambiguate image relations. Wilson and Snavely use local visibility-network structure to score repeated features. TC-SfM uses spatial track communities to guide reconstruction. Exact cycles, neighborhood overlap and feature communities all have clear precedent. [Loop constraints](https://doi.org/10.1109/CVPR.2010.5539801) · [Visibility networks](https://www.cs.cornell.edu/projects/disambig/) · [TC-SfM](https://arxiv.org/abs/2206.05866)
- Damblon et al. learn an outlier-edge filter for relative translations on a line graph. Their intervention addresses global pose estimation. Training a small edge classifier here would require a fair comparison and source-disjoint calibration. [Damblon et al., CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.html)

Our narrower hypothesis concerns the **whole-boundary decision** made after one trial, using per-image paired pixel costs, explicit missing-observation penalties and difficult-feature retention. This targeted review establishes close precedents. Priority for that particular combination remains unverified.
