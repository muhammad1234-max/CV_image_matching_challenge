# Literature review and search record

Search cutoff 19 November 2026. Searched CVF proceedings, arXiv, publisher records and author repositories using the four requested papers, both competition implementations and combinations of visual disambiguation, view-graph pruning, pose consistency, subgraph reconstruction, repeated structures and scene partitioning. Backward tracing from Damblon and Xu leads to Zach’s loop constraints. Forward development from Doppelgangers through Doppelgangers++ leads to XDG and MVDG. Gong, Xu and HiSfM materially narrow the contribution. This is a targeted review, with no assertion of exhaustive indexing or first publication. Venue labels distinguish the archival record from manuscripts reviewed via arXiv. Hardware timings refer to the original setups unless explicitly measured locally. Release availability is distinct from having a verified local artifact.

### Normalized Cuts and Image Segmentation

Jianbo Shi, Jitendra Malik. IEEE Transactions on Pattern Analysis and Machine Intelligence, 2000.

Minimize boundary affinity normalized by both volumes via generalized eigenvectors. Unsupervised partitioning of natural-image pixel affinity graphs. The sparse eigensolver is the main cost. Classical algorithm, no weights; a release for this exact version was not checked. Supplies A’s proposal and a mandatory topology-only control. A low-cost cut alone cannot distinguish two views of one scene with little overlap. [Primary record](https://doi.org/10.1109/34.868688)

### Disambiguating Visual Relations Using Loop Constraints

Christopher Zach, Manfred Klopschitz, Marc Pollefeys. CVPR, 2010.

Compose transformations around loops and infer conflicting edges from inconsistent loops using a Bayesian model. No learned image encoder or fitting labels. Experiments include ambiguous image-based structure and motion. Cost depends on sampled loops; a reusable original release was not verified. A’s internal closure check inherits this idea; the unit of intervention here is a whole cut. [Author PDF](https://people.inf.ethz.ch/pomarc/pubs/ZachCVPR10.pdf) [Primary record](https://doi.org/10.1109/CVPR.2010.5539801)

### Network Principles for SfM: Disambiguating Repeated Structures with Local Context

Kyle Wilson, Noah Snavely. ICCV, 2013.

Score features from local visibility-network clustering to detect links caused by repeated structures. Label-free, evaluated on repeated architectural structures. Graph neighborhoods avoid exhaustive reconstruction hypotheses. Public code, no learned checkpoint. A plain local-support filter is a necessary control; high support is possible for consistently repeated patterns. [Code](https://github.com/wilsonkl/sfm-disambig) [Primary record](https://doi.org/10.1109/ICCV.2013.69)

### A cluster-based disambiguation method using pose consistency verification for structure from motion

Ye Gong, Pengwei Zhou, Changfeng Liu, Yan Yu, Jian Yao, Wei Yuan, Li Li. ISPRS Journal of Photogrammetry and Remote Sensing, 2024.

Obtain local clusters, verify a maximum spanning tree between them, and use global poses to test uncertain relations via cluster-level rotation and translation deviations. Geometric method; authors report 14 ambiguous sequences as well as regular and ground-truth sequences. Reconstruction and consistency verification carry the cost. Public MA-SfM code, no trained checkpoint needed for the core rule. Direct priority against claiming cluster-level pose verification. [Code](https://github.com/gongyeted/MA-SfM) [Primary record](https://doi.org/10.1016/j.isprsjprs.2024.02.016)

### Robust Global Structure-from-Motion via View Graph Pruning

Jiamin Xu, Lixing Yao, Weichen Dai, Renshu Gu, Zunjie Zhu, Weiwei Xu, Gang Xu. arXiv manuscript, 2026.

Run global SfM inside locally consistent subgraphs, use RANSAC across subgraphs to prune edges and solve global SfM again. No scene-label training for the geometric rule. Tests ambiguous sets, sequential captures including Tanks and Temples, and unordered internet imagery. Several local/global solves. Public implementation not established here; no learned head required. Closest contemporary counterexample to broad claims for A and C. Compare a consensus-based per-edge pruning control at the same input and budget. [Primary record](https://arxiv.org/abs/2608.22054)

### HiSfM: Disambiguating Structure-from-Motion via Scaffold-Anchored Hierarchical Reconstruction

Ziding Zhao, Hainan Cui, Peilin Tao, Shuhan Shen. IROS, 2026.

Build strong communities, pack edge-disjoint spanning trees and verify skeleton edges with a two-view disambiguator before scaffold reconstruction and registration. Tests ambiguous and ordinary reconstruction collections. Cost includes learned verification on selected edges and hierarchical SfM. Public code; check detector checkpoint requirements before comparison. Edge-disjoint evidence and selective geometry already have precedent; endpoint-disjoint votes across a single proposed boundary are a narrower test. [Code](https://github.com/3dv-casia/HiSfM) · [Authors’ venue record](https://3dv.ac.cn/en/publication/iros-f/) [Primary record](https://arxiv.org/abs/2609.04718)

### Learning to Filter Outlier Edges in Global SfM

Nicole Damblon, Marc Pollefeys, Daniel Barath. CVPR, 2025.

A line-graph transformer classifies relative translations after rotation averaging; subgraph processing controls line-graph memory growth. BCE labels derive from translation angular error. Training uses Piazza San Marco, validation Taj Mahal; evaluation includes 1DSfM, PhotoTourism and ScanNet. Their setup trains for 300 epochs. Public code; exact pretrained binary not hash-verified. Direct precedent for B; a bad translation and a cross-scene edge are different targets. [Paper](https://openaccess.thecvf.com/content/CVPR2025/html/Damblon_Learning_to_Filter_Outlier_Edges_in_Global_SfM_CVPR_2025_paper.html) · [Code](https://github.com/DmblnNicole/LFOE-GlobalSfM) [Primary record](https://doi.org/10.1109/CVPR52734.2025.01079)

### Doppelgangers: Learning to Disambiguate Images of Similar Structures

Ruojin Cai, Joseph Tung, Qianqian Wang, Hadar Averbuch-Elor, Bharath Hariharan, Noah Snavely. ICCV, 2023.

CNN predicts real versus illusory matches from image pairs plus spatial keypoint and correspondence cues. Trained and tested with labeled similar-structure pairs and downstream ambiguous SfM. Cost per evaluated pair. Author page releases data, code and model resources. Already targets geometrically plausible false matches; same-surface classification is narrower than membership in a whole physical scene. [Project](https://doppelgangers-3d.github.io/) [Primary record](https://arxiv.org/abs/2309.02420)

### Doppelgangers++: Improved Visual Disambiguation with Geometric 3D Features

Yuanbo Xiangli, Ruojin Cai, Hanyu Chen, Jeffrey Byrne, Noah Snavely. CVPR, 2025.

Pair classifier on 3D foundation-model features, supervised with Doppelgangers and VisymScenes. Includes reconstruction checks via registered geolocated imagery; pairwise backbone inference costs grow with edges. Public code and model links. A strong learned comparison if weights and compute are available. A tests the agreement of distinct camera pairs without new image features. [Code](https://github.com/doppelgangers25/doppelgangers-plusplus) [Primary record](https://arxiv.org/abs/2412.05826)

### MVDG: Efficient Multi-view 3D Disambiguation on Unconstrained Real-World Images

Hanyuan Xiao, Gonglin Chen, Haolin Xiong, Wenbin Teng, Haiwei Chen, Yajie Zhao. arXiv manuscript, 2026.

VGGT-based multi-view disambiguation, fine-tuned with pseudo-pairwise examples from AerialMegaDepth. Evaluates pair classification and SfM on held-out scenes and studies proxy validation. Group processing amortizes pairwise backbone work. Public checkpoint/code for the added model not verified. Multi-view learned disambiguation is itself established prior art; count added processing in comparisons. [Primary record](https://arxiv.org/abs/2610.01098)

### XDG: Accelerated Visual Disambiguation

Gonglin Chen, Ben Southall, Hanyuan Xiao, Wenbin Teng, Haolin Xiong, Tianwen Fu, Junyi Ouyang, Kshitij Singh Minhas, Supun Samarasekera, Rakesh Kumar, Yajie Zhao. arXiv manuscript, 2026.

LoRA-adapt Depth Anything 3 and classify camera tokens with a compact MLP for same-surface prediction. Learned pair labels, pairwise and downstream reconstruction tests including LaMAR; exact training mixture not verified here. Authors report more than 3× inference speedup against their disambiguation baseline. Public code; checkpoint bytes unverified. Rules out claiming novelty for a compact reliability head alone. [Code](https://github.com/xtcpete/xdg) [Primary record](https://arxiv.org/abs/2608.29733)

### MAGSAC++, a Fast, Reliable and Accurate Robust Estimator

Daniel Barath, Jana Noskova, Maksym Ivashechkin, Jiri Matas. CVPR, 2020.

Marginalize noise scale in robust model scoring and iterative reweighting; use Progressive NAPSAC sampling. No training; six real datasets for homography and fundamental-matrix fitting. Repeated minimal solves per pair. Open implementation, no weights. A valid epipolar model constrains a pair but cannot certify which physical scene generated it. Keep baseline verification fixed. [Code](https://github.com/danini/magsac) [Primary record](https://arxiv.org/abs/1912.05909)

### CoMatcher: Multi-View Collaborative Feature Matching

Jintao Zhang, Zimin Xia, Mingyue Dong, Shuhan Shen, Linwei Yue, Xianwei Zheng. CVPR, 2025.

Group co-visible views, aggregate their learned context and enforce cross-view projection consistency to form tracks. Synthetic homography pretraining followed by pose-supervised MegaDepth tuning; HPatches and multi-view MegaDepth-1500 evaluation. Authors report six days on two RTX 4090s. Public training/evaluation code; verify weight availability before scheduling. Better tracks may help, but swapping the matcher would confound this experiment. [Paper](https://openaccess.thecvf.com/content/CVPR2025/papers/Zhang_CoMatcher_Multi-View_Collaborative_Feature_Matching_CVPR_2025_paper.pdf) · [Code](https://github.com/EATMustard/CoMatcher) [Primary record](https://doi.org/10.1109/CVPR52734.2025.02046)

### Self-Supervised Partial Cycle-Consistency for Multi-View Matching

Fedor Taggenbrock, Gertjan Burghouts, Ronald Poppe. VISAPP, 2025.

Pseudo-mask the cycle loss for feature assignments under partial overlap; training uses self-supervised cross-camera sampling. DIVOTrack evaluates object association. Matrix products grow with detections and views. Public implementation; checkpoint not verified. Supports masking unknown evidence but operates on object identities, not camera graphs. [Code](https://github.com/FedorTaggenbrock/Self-Supervised-Partial-Cycle-Consistency) [Primary record](https://arxiv.org/abs/2501.06000)

### Deep Graph Infomax

Petar Veličković, William Fedus, William L. Hamilton, Pietro Liò, Yoshua Bengio, R Devon Hjelm. ICLR, 2019.

Contrast positive local graph embeddings against a summary and corrupted features using a graph convolutional encoder. Label-free pretraining, downstream node classification including citation networks and PPI. Requires encoder passes on original and corrupted graphs. Public code; no transferable geometric-reliability checkpoint established. A self-supervised alternative for B, but graph corruption can produce much easier negatives than real aliases. [Code](https://github.com/PetarV-/DGI) [Primary record](https://arxiv.org/abs/1809.10341)

### On Calibration of Modern Neural Networks

Chuan Guo, Geoff Pleiss, Yu Sun, Kilian Q. Weinberger. ICML, 2017.

Fit one temperature to held-out logits by minimizing negative log likelihood. Study supervised image/text classifiers including CIFAR and ImageNet; cheap one-parameter calibration. Public implementation, no SfM predictor weights. Reliability diagrams and Brier scores can audit B; calibrating edges need not calibrate a proposed cut. [Proceedings](https://proceedings.mlr.press/v70/guo17a) [Primary record](https://arxiv.org/abs/1706.04599)

### GraphMatch: Efficient Large-Scale Graph Construction for Structure from Motion

Qiaodong Cui, Victor Fragoso, Chris Sweeney, Pradeep Sen. 3DV, 2017.

Graph-guided sampling and propagation find likely image pairs, avoiding all-pairs matching. No learned graph predictor; tests landmark internet photo collections including Taj Mahal, with thousands of images. Matching/reconstruction time is explicitly evaluated. Theia-based implementation in paper, separate release not verified. Relevant to active matching in C. Keep original selected pairs for A. [Primary record](https://arxiv.org/abs/1710.01602)

### MATCHA: Towards Matching Anything

Fei Xue, Sven Elflein, Laura Leal-Taixé, Qunjie Zhou. CVPR, 2025.

Attention fuses diffusion features across scales and incorporates DINOv2 cues for semantic, geometric and temporal correspondence. Multi-task learning includes SPair-71K and MegaDepth; evaluation includes HPatches, Aachen and TAP-Vid. Dense image features require backbone inference. Public code and pretrained models. Matching similar objects semantically can be correct for MATCHA and wrong for SfM. [Code/data](https://github.com/nv-dvl/matcha) [Primary record](https://arxiv.org/abs/2501.14945)

### DUSt3R: Geometric 3D Vision Made Easy

Shuzhe Wang, Vincent Leroy, Yohann Cabon, Boris Chidlovskii, Jerome Revaud. CVPR, 2024.

Paired transformer regresses dense pointmaps with confidence-weighted 3D losses; global alignment combines pairs. Multi-dataset 3D training includes MegaDepth, Habitat and BlendedMVS; evaluated on geometry including relative pose and depth. Dense backbone per pair and optional alignment. Public models/code. A shares the original feature lineage and adds no pointmap predictor. [Code](https://github.com/naver/dust3r) [Primary record](https://arxiv.org/abs/2312.14132)

### Grounding Image Matching in 3D with MASt3R

Vincent Leroy, Yohann Cabon, Jérôme Revaud. ECCV, 2024.

Extend pointmap regression with a dense descriptor and contrastive correspondence objective, use fast reciprocal matching. Mixed synthetic/real 3D supervision and metric data; includes Map-free relocalization evaluation. Dense pair inference is already expensive in our measured path. Public weights and code; pinned artifact in local receipts. Freeze these so improvements can be attributed to graph decisions. [Code](https://github.com/naver/mast3r) [Primary record](https://arxiv.org/abs/2406.09756)

### VGGT: Visual Geometry Grounded Transformer

Jianyuan Wang, Minghao Chen, Nikita Karaev, Andrea Vedaldi, Christian Rupprecht, David Novotny. CVPR, 2025.

Alternating frame/global attention predicts cameras, pointmaps, depth and tracks from multiple images under 3D supervision. Mixed real/synthetic data include CO3D and Virtual KITTI; evaluated on pose, depth and tracking. Joint attention trades repeated pair processing for memory as frames increase. Public code and weights. MVDG already adapts this class of model to disambiguation. [Code](https://github.com/facebookresearch/vggt) [Primary record](https://arxiv.org/abs/2503.11651)

### Structure-from-Motion Revisited

Johannes L. Schönberger, Jan-Michael Frahm. CVPR, 2016.

Incremental initialization, registration, triangulation and bundle adjustment optimize reprojection error from verified correspondences. Label-free; evaluated on internet photo collections. Nonlinear mapping and BA can dominate runtime. Public COLMAP code, no learned checkpoint. Provides camera poses and track feedback for C; optimizing reprojection on an incorrect merged model can reinforce aliases. [Code](https://github.com/colmap/colmap) [Primary record](https://doi.org/10.1109/CVPR.2016.445)

## Translation consistency and a direct triplet-filter comparator

### Robust Global Translations with 1DSfM

Kyle Wilson and Noah Snavely. ECCV, 2014. Project measured pair directions onto lines, reject inconsistent order constraints, then average translations in 3D. No training or weights; public numerical data and code for internet landmark photo collections. Repeated one-dimensional subproblems cost CPU time. For a triangle of unit bearing directions in a common orientation frame, closure needs positive unknown baseline lengths: a*u12 + b*u23 + c*u31 = 0. Setting all lengths to one creates spurious errors. A uses rotations only for this reason. Internally consistent rotations can still leave a translation-only alias undetected. [Primary record](https://doi.org/10.1007/978-3-319-10578-9_5) · [Authors’ data and code](https://www.cs.cornell.edu/projects/1dsfm/)

### Unifying Viewgraph Sparsification and Disambiguation of Repeated Structures in Structure-from-Motion

Lalit Manam and Venu Madhav Govindu. International Journal of Computer Vision, 2026. The authors’ camera-triplet line of work scores edges from triplet geometry and redundancy, and sparsifies the view graph using an edge-selection objective. This unsupervised geometric approach is evaluated on generic and ambiguous SfM sets; reduced mapping work is part of the claim. The authors provide separately versioned code for the CVPR 2024 and IJCV 2026 methods; no trained weights are needed for this geometric comparator. This is a stronger control than merely testing whether an edge has a triangle. Evaluate with shared verified inputs if supported; distinguish the author's method from our deliberately simple rotation-closure ablation. [Journal record](https://doi.org/10.1007/s11263-025-02681-3) · [Authors’ implementation and datasets](https://ee.iisc.ac.in/cvlab/research/camtripsfm/)

## Original competition implementations

**Image Matching Challenge 2025 — first-place solution.** ns-rokuyon, 2025. Published hybrid retrieval, local and MASt3R matching, verification and mapping. Repository identifies the 022-c preset; pin our audited source and weights rather than repository HEAD. Four measured development pilots establish the comparison scope. [First-place write-up](https://www.kaggle.com/competitions/image-matching-challenge-2025/discussion/583058) [Repository](https://github.com/ns-rokuyon/kaggle-image-matching-challenge-2025)

**Image Matching Challenge 2025 — seventh-place solution.** AymenLass and team, 2025. Author repository links a seventh-place write-up and provides MASt3R, VGGT, vggsfm and COLMAP scripts. Multiple scripts do not establish which ablation or ensemble produced a specific score. IMC25 benchmark, exact checkpoint set and comparable T4 cost not independently verified here. [Seventh-place write-up](https://www.kaggle.com/competitions/image-matching-challenge-2025/writeups/kaqln-7th-place-solution) [Repository](https://github.com/AymenLass/Image-Matching-Challenge-2025-7th-place-solution)
