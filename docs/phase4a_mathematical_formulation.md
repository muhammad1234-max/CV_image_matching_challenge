# When do verified image pairs justify splitting a model?

**Question.** If two image groups belong to one rigid reconstruction, their local camera orientations should admit one relative frame rotation. Different baselines, unknown translation scale and different choices of local origin do not invalidate this necessary condition. Violating it can also indicate poor calibration or noisy internal poses. Passing it cannot prove common scene membership.

### Propose a boundary

Let \(G=(V,E)\) contain the original post-verification, post-pruning pairs and their inlier counts \(n_{ij}>0\). Set \(w_{ij}=\log(1+n_{ij})\), symmetric with zero diagonal. Work within each connected component. For disjoint sides \(A,B\) covering that component minimize the *proposal* cost
\[
J(A,B)=w(A,B)(1/\mathrm{vol}(A)+1/\mathrm{vol}(B)),\qquad
\mathrm{vol}(A)=\sum_{i\in A,j\in V}w_{ij}.
\]
Use the minimum feasible sweep cut of the second generalized eigenvector of \((D-W)x=\lambda Dx\), not a claim of exact combinatorial minimization. Each side must induce a connected subgraph and contain at least six cameras. Break ties by canonical opaque order. One proposal per eligible component, no recursion. The sweep sees counts and topology only. An arbitrary cut through a well-connected scene can also minimize this objective.

### Can each side supply a trustworthy reference frame?

Use only edges with valid calibrated relative rotations \(R_{ji}\) taking camera \(i\) coordinates into camera \(j\) coordinates. Reverse edges by transposition. Construct a maximum-weight internal spanning tree of valid edges on each side; disconnected means abstain. Set the canonical root orientation \(U_r=I\), propagate \(U_j=R_{ji}U_i\). Each \(U_i\) maps its side's local frame to camera coordinates. For internal **non-tree** edges compute
\[
e_{ij}=d(U_j,U_i;R_{ji})=\theta(U_j^\top R_{ji}U_i),\quad
\theta(R)=\arccos(\mathrm{clip}((\mathrm{tr}(R)-1)/2,-1,1)).
\]
Use degrees for all thresholds. Require at least four non-tree edges per side and nearest-rank 80th percentile \(s_A,s_B\le10^\circ\) separately. Tree edges would have zero residual by construction. Endpoints far from checked internal loops can still drift. Log depth and non-tree coverage for diagnosis.

### Test the crossing edges together

For each valid crossing edge oriented from \(i\in A\) to \(j\in B\), compute \(Q_{ij}=U_j^\top R_{ji}U_i\). In a rigid noise-free scene all these rotations are the same. Changing either local gauge sends all \(Q\) to \(S_B^\top Q S_A\), which preserves pairwise geodesic distances. No estimated crossing rotation enters the internal trees. The candidate boundary still depends on full graph topology, so the test is descriptive rather than a calibrated significance test.

Take a deterministic greedy bipartite matching \(M\) of crossing edges, descending inlier weight then canonical endpoint IDs. Skip used endpoints and cap at 32 pairs. Greedy can abstain even when a larger assignment exists. Require \(|M|\ge6\). Define
\[
\tau=\max(15^\circ,2(s_A+s_B)),\quad
h=\max_{a\in M}\frac{1}{|M|}\sum_{b\in M}
{\bf1}[\theta(Q_a^\top Q_b)\le\tau].
\]
Accept the split if **strictly fewer than half** the endpoint-disjoint witnesses agree with even the best observed center, i.e. \(2|M|h<|M|\) using integer counts. Otherwise retain the baseline graph. This observed-center search is discrete. A compact cluster whose center lies between samples may be missed. Witnesses share tree estimates, so distinct endpoints limit repeated votes without ensuring statistical independence. Dense coherent aliases can all support a single false rotation.

A split suppresses *all* crossing pairs on the mapper input copy, even a locally consistent minority. This tests a scene-boundary hypothesis, a stronger action than outlier-edge pruning. Missing or degenerate geometry always yields unknown evidence. Never interpret low support as disagreement. No training objective, fitted coefficients, external labels or threshold search for A. Initial numeric choices are engineering hypotheses frozen for the first protocol, not calibrated probabilities.

### Task-level endpoint

For each dataset record the frozen evaluator's mAA \(a\), clusterness \(c\) and combined \(H=2ac/(a+c)\) with the evaluator's zero convention. Scene assignment affects which poses contribute to mAA. A large cut score or improved edge precision need not improve \(H\). Run the official local evaluator on the full prediction set. Use its existing dataset/fold aggregation, never substitute the harmonic mean of averaged components.
