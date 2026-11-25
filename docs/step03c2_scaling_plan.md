Measured ETs total was 435.882 s for 231 pairs, with 5.58 GiB peak allocated GPU and 6.05 GiB peak worker RAM. Organizer development took 13 h 27 min including a timed-out attempt. Planning scenarios below use the ETs total scaled between image count and exhaustive pair count. These are workload scenarios, not calibrated bounds. Shortlisting can cut matching time and nonlinear SfM can exceed the upper scenario.

Dataset | Images | Hours (image to exhaustive-pair scenarios)
---|---:|---:
ETs | 22 | 0.12 to 0.12
amy_gardens | 200 | 1.10 to 10.43
fbk_vineyard | 163 | 0.90 to 6.92
imc2023_haiper | 54 | 0.30 to 0.75
imc2023_heritage | 209 | 1.15 to 11.39
imc2023_theather_imc2024_church | 76 | 0.42 to 1.49
imc2024_dioscuri_baalshamin | 138 | 0.76 to 4.95
pt_brandenburg_british_buckingham | 225 | 1.24 to 13.21
pt_sacrecoeur_trevi_tajmahal | 225 | 1.24 to 13.21
pt_stpeters_stpauls | 200 | 1.10 to 10.43
stairs | 51 | 0.28 to 0.67

Aggregate scenarios 8.6 to 73.6 hours. After the gate, propose stairs (51 images), the smallest remaining frozen development dataset. Use a single two-hour attempt and record a timeout as a result. Budget roughly 3.8 GB for active weights plus runtime, source and intermediates. Measure scratch disk, descriptor/cache growth and peak RAM on stairs before committing larger sessions. The current adapter requires a fresh workspace and has no verified mid-dataset resume. Export after each dataset. Colab session availability and nonlinear reconstruction remain material uncertainties.
