# ETs scores on the frozen development split

| Condition | mAA | Clusterness | Combined | Sensitivity combined | Delta vs organizer | Registered | Clusters | Outliers | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Registered organizer | 0.288462 | 0.431818 | 0.345874 | 0.345874 | 0 | 22 | 1 | 0 | 156.084 |
| Historical ns64 | 0.461538 | 1 | 0.631579 | 0.631579 | +0.285705 | 19 | 2 | 3 | 435.882 |
| Neutral 20261111 | 0.423077 | 1 | 0.594595 | 0.594595 | +0.248721 | 19 | 2 | 3 | 364.174 |
| Neutral 20261112 | 0.461538 | 1 | 0.631579 | 0.631579 | +0.285705 | 19 | 2 | 3 | 355.410 |

All four runs completed on the same 22 development images. Both neutral runs reconstructed two models, registered all 19 inlier cameras and correctly excluded all three true outliers. Relative to the organizer, mAA increased by 0.134615 and 0.173077 respectively and clusterness increased by 0.568182 for both. Both score variants were independently recomputed after identity restoration. Historical rescoring reproduced the complete prior evaluation JSON.
