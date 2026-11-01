# Final synthesis reorganization

Requested order: 01 audit, 02 distribution/quality, 03 temporal patterns,
04 weather, 05 users, 06 hypothesis testing, 07 clustering, 08 feature selection,
09 synthesis. Retain the existing input data and manual report.

- Rename scripts, tests and generated outputs with simultaneous prefix mapping;
  update source references, plans, reports and README links.
- Test that synthesis rejects missing upstream evidence and input hash mismatch,
  and that its report includes every analysis, actual statistics and valid links.
- Recompute base EDA evidence from current input; audit current raw files using
  step 01 helpers. Read extension tables only after matching their run metadata
  to the current processed input hash. Record all consumed evidence hashes.
- Generate 09 tables/figures and reports/synthesis_summary.md with separate
  observational, inferential, clustering and predictive interpretations.
- Run the full test suite, rerun extensions 06–08, then final synthesis (which
  revalidates raw data and recomputes the core evidence of steps 02–05);
  inspect the final report and check local links and filename consistency.

The summary does not retune models or select features from final test results.
Missing or incompatible inputs fail before the summary is published. Script
files stay together because synthesis imports step 01/05 helper functions.
