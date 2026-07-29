# Clinical Disclaimer

**This software is not a medical device and is not registered, cleared or
approved as one in any jurisdiction.**

IVF Digital Twin is a research and decision-support tool. It produces
probabilistic estimates of IVF outcomes from historical cohort data and
published coefficients. It does not diagnose, treat, cure or prevent any
condition.

## Limits of the predictions

- Outputs are **population-level probabilities**, not statements about what
  will happen to an individual patient.
- Model performance depends on how closely a clinic's population and
  laboratory practice resemble the training cohorts. Applying it to a
  different population without recalibration will degrade accuracy in ways
  the reported metrics do not capture. See `calibrate_for_clinic.py` and
  `validate_clinic_data.py`.
- Confidence and prediction intervals reflect model uncertainty only. They
  do not account for data-entry error, protocol deviation, or unmodelled
  clinical factors.
- The LLM-based narrative components (`llm_consultant.py`,
  `patient_brief.py`, `protocol_guidance.py`) generate explanatory text.
  Generated text can be wrong or incomplete and must be reviewed by a
  clinician before it reaches a patient.

## Required conditions of use

1. Predictions must never replace professional medical judgment, clinical
   examination, or consultation with a qualified reproductive medicine
   specialist.
2. A qualified clinician must review every output before it informs a
   treatment decision or is shown to a patient.
3. Before any clinical deployment, the user is responsible for local
   validation on their own historical data, and for obtaining whatever
   regulatory approval their jurisdiction requires for clinical decision
   support software.
4. Coefficient sources are cited in the accompanying documentation. Users
   applying the tool outside the cited populations should verify that those
   sources remain applicable.

## No warranty

Consistent with the No Liability section of the `LICENSE`, the software is
provided as is, without any warranty or condition of fitness for any clinical
purpose. Use for clinical decision-making without appropriate validation and
regulatory approval is at the user's sole risk.

## Data protection

The software processes patient data. Users are solely responsible for
complying with applicable data protection law (GDPR, HIPAA, national health
data regulations) in their jurisdiction, including lawful basis for
processing, data minimization, and any obligations arising from sending data
to third-party LLM providers if the optional narrative features are enabled.
