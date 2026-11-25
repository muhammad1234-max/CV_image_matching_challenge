# Development graph

There were 1,803 image triangles and 38,031 unique indexed correspondence cycles. Of 734,692 linked observations, 57,673 participate in at least one cycle. Of 607,136 links, 77,321 have an exact third-view witness and 25,733 have witnesses in two or more distinct images. The observation graph contains 170,442 linked hypotheses, including 14,114 conflicted and 89,927 clean multi-view hypotheses.

| Component | Region | Edges | Median cycle fraction | Median conflict fraction |
|---|---|---:|---:|---:|
|23|internal|115|.158|.506|
|23|boundary|92|.109|.540|
|31|internal|179|.066|.301|
|31|boundary|68|.044|.235|

Both boundaries have weaker median cycle support. Only the 23-image boundary has elevated conflict. B2 retained this component and abstained on the other. The observation data adds a measurable distinction within both components, but the direction of the cycle difference alone cannot justify splitting one and preserving the other. All edge denominators, grid occupancies and flags are in features.json.
