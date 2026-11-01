# Daily-demand clustering implementation plan

Scope: extend the existing numbered-script EDA with `07_clustering.py`, focused
unittest checks, generated `07_*` outputs and `reports/clustering.md`. Preserve
steps 01–07, original data, proposal and the manually maintained final report.

1. Test daily matrix construction: exactly 24 ordered hours, absent days/hours
   stay missing, duplicate/off-hour/invalid demand is rejected, daily calendar
   flags are constant, tied weather modes are explicitly ambiguous.
2. Build profiles from the existing processed dataset, rederive time fields,
   and audit every date in the input's bounded calendar span. Retain complete
   positive-total days for both representations, with no imputation. Require
   at least 30 such days and at least half of observed dates; if insufficient,
   save the audit and stop with an actionable message rather than fabricate
   complete curves. This is a conservative analysis guard, not a statistical
   sufficiency guarantee. Here 414/456 observed dates are complete.
3. Compare unscaled raw counts (magnitude + shape) and per-day shares (shape).
   All hours already have common units: do not standardize individual hours,
   which would upweight low-variance overnight hours. This deliberately
   refines the earlier conversation's suggested StandardScaler step.
4. Evaluate K=2–6 on each representation, plus K=1 inertia for elbow context.
   Use fixed seed 18, n_init=20, full-sample Euclidean silhouette. Select the
   highest silhouette within each representation (smaller K on exact ties),
   display inertia as supporting evidence; do not compare inertia across units.
   Fit final models; daily shares are primary by research intent, raw is a
   sensitivity comparison. Save centroids, labels and all candidate metrics.
5. Generate mean count/share profiles, cluster sizes, metadata distributions,
   calendar retention and input checksums/software settings. Metadata never
   enters the feature matrix. Keep target-derived user columns excluded.
6. Run tests, actual workflow, repeatability check and inspect plots. Document
   empirical results and selection bias, temporal dependence, post-hoc use,
   Euclidean/KMeans assumptions and lack of causal/predictive validation.

Review focus: zero-total days, missing complete dates, weather ties, repeated
profiles/degenerate K, malicious or stale prederived calendar/target columns.
No Git checkout is present, so work is performed additively in this directory.
