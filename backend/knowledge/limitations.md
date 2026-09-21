# Ischemic lesions vs mimics

Not every DWI-bright focus is acute ischemia. Mimics include artifacts,
T2 shine-through (bright DWI without ADC correlate), chronic lesions, and
other pathologies. Always verify ADC correlation and anatomical plausibility.

Small-vessel disease, demyelination, and postictal changes can confound
automated segmentation. AI masks need clinician review.

# Limitations of AI segmentation

Automated segmentation can over- or under-segment, especially for small or
diffuse lesions. Model confidence reflects predicted probability, not a Dice
score (Dice requires ground truth). Severity rules based on volume bands are
unvalidated heuristics for triage display only, not clinical guidance.
