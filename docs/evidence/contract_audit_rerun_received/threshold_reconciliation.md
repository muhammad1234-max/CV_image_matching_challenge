# Threshold inspection

{
  "sha256": "5b92db83d510ab19e66bd1ef4c23865245d75b4a981986e6d8eb1aee06fc9fe2",
  "expected_sha256": "5b92db83d510ab19e66bd1ef4c23865245d75b4a981986e6d8eb1aee06fc9fe2",
  "hash_matches_previous": true,
  "columns": [
    "dataset",
    "scene",
    "thresholds"
  ],
  "rows": 30,
  "interpretation": "UNRESOLVED: preserve raw tokens; participant parser uses float(token) on semicolon-separated strings. A representable float is not proof of official intended semantics.",
  "configuration_created": false,
  "complete": true
}

Haiper raw diagnostics (no substituted values):
[
  [
    8,
    "imc2023_haiper",
    "bike",
    "0;0;0.001;0.025;0.05;0.1",
    "[{\"raw\": \"0\", \"type\": \"zero\", \"finite_value\": 0.0}, {\"raw\": \"0\", \"type\": \"zero\", \"finite_value\": 0.0}, {\"raw\": \"0.001\", \"type\": \"positive\", \"finite_value\": 0.001}, {\"raw\": \"0.025\", \"type\": \"positive\", \"finite_value\": 0.025}, {\"raw\": \"0.05\", \"type\": \"positive\", \"finite_value\": 0.05}, {\"raw\": \"0.1\", \"type\": \"positive\", \"finite_value\": 0.1}]",
    "zero",
    "{\"dataset\": \"imc2023_haiper\", \"scene\": \"bike\", \"thresholds\": \"0;0;0.001;0.025;0.05;0.1\"}"
  ],
  [
    9,
    "imc2023_haiper",
    "chairs",
    "0;0;0.001;0.025;0.05;0.1",
    "[{\"raw\": \"0\", \"type\": \"zero\", \"finite_value\": 0.0}, {\"raw\": \"0\", \"type\": \"zero\", \"finite_value\": 0.0}, {\"raw\": \"0.001\", \"type\": \"positive\", \"finite_value\": 0.001}, {\"raw\": \"0.025\", \"type\": \"positive\", \"finite_value\": 0.025}, {\"raw\": \"0.05\", \"type\": \"positive\", \"finite_value\": 0.05}, {\"raw\": \"0.1\", \"type\": \"positive\", \"finite_value\": 0.1}]",
    "zero",
    "{\"dataset\": \"imc2023_haiper\", \"scene\": \"chairs\", \"thresholds\": \"0;0;0.001;0.025;0.05;0.1\"}"
  ],
  [
    10,
    "imc2023_haiper",
    "fountain",
    "0;0;0.001;0.025;0.05;0.1",
    "[{\"raw\": \"0\", \"type\": \"zero\", \"finite_value\": 0.0}, {\"raw\": \"0\", \"type\": \"zero\", \"finite_value\": 0.0}, {\"raw\": \"0.001\", \"type\": \"positive\", \"finite_value\": 0.001}, {\"raw\": \"0.025\", \"type\": \"positive\", \"finite_value\": 0.025}, {\"raw\": \"0.05\", \"type\": \"positive\", \"finite_value\": 0.05}, {\"raw\": \"0.1\", \"type\": \"positive\", \"finite_value\": 0.1}]",
    "zero",
    "{\"dataset\": \"imc2023_haiper\", \"scene\": \"fountain\", \"thresholds\": \"0;0;0.001;0.025;0.05;0.1\"}"
  ]
]

A failed positivity check does not prove invalid metadata. Review raw tokens and actual evaluator behavior before creating an executable threshold configuration. The source was not modified.
