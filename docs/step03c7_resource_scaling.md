# Measured ns64 scaling at fixed order seed 20261111

Dataset | Images | Possible / selected pairs | Matching s | Mapping s | Wall s | GPU allocated / reserved GiB | Worker RSS GiB | Final workspace MB | Registered | Clusterness | Combined
---|---:|---|---:|---:|---:|---|---:|---:|---|---:|---:
ETs | 22 | 231 / 231 | 106.99 | 173.53 | 364.17 | 5.58 / 8.77 | 6.05 | unmeasured | 19/22 | 1.000 | 0.594595
stairs | 51 | 1275 / 855 | 510.89 | 366.46 | 1016.56 | 5.09 / 5.51 | 6.05 | 76.46 | 51/51 | 0.500 | 0.166667
imc2023_haiper | 54 | 1431 / 733 | 479.45 | 427.52 | 1086.62 | 5.00 / 5.52 | 6.49 | 202.77 | 54/54 | 0.635294 | 0.685552

Matching and mapping together consumed 83.5% of the medium run. Final workspace includes staged images and excludes shared weights. Peak disk use, peak separate cache size, and actual remaining provider session lifetime are unknown. At preflight the T4 reported 15,524,954,112 bytes free VRAM and the VM reported 11,203,912 KiB available RAM.

## Planning scenarios

Applying the medium wall time proportionally to images or to all possible pairs gives the following single-T4 compute hours. These scenarios omit setup, retries and evaluation. Actual shortlisting, scene difficulty, resolution and SfM can move runtime outside both scenarios.

Dataset | Images | Image-proportional hours | Pair-proportional hours
---|---:|---:|---:
imc2023_theather_imc2024_church | 76 | 0.42 | 0.60
imc2024_dioscuri_baalshamin | 138 | 0.77 | 1.99
fbk_vineyard | 163 | 0.91 | 2.78
amy_gardens | 200 | 1.12 | 4.20
pt_stpeters_stpauls | 200 | 1.12 | 4.20
imc2023_heritage | 209 | 1.17 | 4.58
pt_brandenburg_british_buckingham | 225 | 1.26 | 5.32
pt_sacrecoeur_trevi_tajmahal | 225 | 1.26 | 5.32

The eight remaining datasets total 8.03 or 28.99 GPU-hours under the respective assumptions. These are planning scenarios from three small pilots.
