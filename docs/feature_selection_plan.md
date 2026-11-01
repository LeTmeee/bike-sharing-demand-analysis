# Feature selection implementation plan

Extend the existing numbered scripts with `08_feature_selection.py`; preserve the
proposal's original EDA scope and label this as an optional predictive extension.
Use the supplied specification: exploratory Spearman, cross-validated Lasso and
RandomForest permutation importance. Existing raw and processed data stay intact.

1. Test explicit input allowlisting and rederived calendar fields, invalid input,
   whole-date chronological partitions, and training-only preprocessing.
2. Keep continuous weather variables plus year numeric; one-hot encode hour,
   weekday, month, season, holiday, workingday and weather. Exclude all other
   columns, including target components, shares, log counts and quality flags.
3. Reserve the last 20% of observed dates as final test. Use five expanding
   chronological splits of observed training dates (unequal elapsed spans are
   reported). GridSearchCV wraps the complete preprocessing/Lasso pipeline,
   selecting alpha by validation MSE. Never fit preprocessing before CV.
4. Use a fixed RandomForest baseline. Measure primary permutation importance on
   the last inner validation block, training that forest only on earlier dates.
   This allows training-side feature interpretation without test-driven selection.
   Fit final models on all training dates and evaluate the untouched final test
   once; do not tune or claim reduced-model gains from these results.
5. Export `08_*` CSVs, PNGs, input hash, software/settings metadata and a generated
   Chinese findings report. Record raw coefficients and categorical reference
   levels, aggregate absolute coefficients only as a model-specific summary.
   Correlations are continuous-weather associations, never an automatic filter.
6. Verify focused tests, full existing suite, real-data execution, deterministic
   rerun and plot inspection. Document correlated substitutes, noncausal meaning,
   temporal dependence, limited month coverage, observed-weather availability,
   and permutation repeat SD versus sampling confidence intervals.

No Git repository is present. Use the existing workspace and installed system
Python (the old .venv lacks sklearn). No additional dependency is required.
