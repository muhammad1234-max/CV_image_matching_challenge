# Mask integrity

Parent SHA-256 is 9d088fd433baee7e4101ab25a8b7f5fedf405c8342c1bc779f88ef911c32051a. An independently created transactional SQLite backup reproduced its complete logical fingerprint, schema and row counts. The original bytes were verified again after copying. Each pair table contains 1,431 rows, including empty verified records. The graph has 454 nonempty verified pairs. The active hypothesis removes 68 verified edges, leaving 386. The database denylist spans all 240 pair IDs between the two sides, including raw-only or empty geometries, to prevent resurrection of the boundary. Camera, image, keypoint and retained pair records must compare exactly. After parity passed, independent audit of the masked numerical copy reproduced the complete returned before/after receipt.

