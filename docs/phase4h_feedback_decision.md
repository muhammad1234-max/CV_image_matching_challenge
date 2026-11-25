# Automatic decision

REJECT. The registered 31-image model had 73,841 points. Its 15- and 16-image successors had 37,757 and 26,072 respectively. The point ratio 63,829 / 73,841 = 0.8644113703768909 falls below 0.95 (minimum integer count 70,149). The unaffected 23-image model had 61,723 points in both branches. Registration and the partition checks passed before the affected-point veto fired. The per-image 95% retention, 30-observation minimum and 0.05 px rules were never reached.

The returned reason is "affected point loss". selected.csv is byte-identical to the validated pass-through prediction. Its previously evaluated combined score is 0.685551871151113 and the selected prediction has exactly zero paired difference from pass-through. The separately evaluated fixed split scored 0.8498402555910544. The missed gain is a development diagnostic. No new scoring run or new independent validation was performed.
