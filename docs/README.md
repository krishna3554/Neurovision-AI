# Docs — report figures 3.1–3.6

Keep architecture (3.1), use-case (3.2), class (3.3), state (3.4),
deployment (3.5), and CFD/DFD (3.6) diagrams here, consistent with code.

- `backend/app/services/pipeline.py` implements Fig 3.4 states.
- CFD decision diamonds: `Valid?` → `validator.validate_study`,
  `Quality OK?` → `triview.quality_ok` (max slice score >= tau).
- Fig numbering note: the report has two figures numbered 4.2 and two
  tables numbered 2.1; Fig 4.1 (Dice 0.886) vs Table 6.1 (Dice 0.8062)
  cover different subsets — document which is which in your write-up.
- Fig 6.1 numbers (94.8% confidence, 12.4 cm³, Level 3, #NV-8842-A) are a
  design mock-up unless replaced by a real screenshot.
