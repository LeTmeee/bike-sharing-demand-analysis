# Hypothesis Testing extension

The requested extension is step 06, following scripts 01–05; final synthesis is step 09.

- Validate timestamps, numeric fields and within-date calendar flags.
- Save one observation per date; primary analysis uses 24 observed hours, sensitivity analysis uses all dates.
- Compare working-day distributions with two-sided Mann–Whitney U; compare unique modal daily weather groups with Kruskal–Wallis and conditional pairwise Mann–Whitney with Holm adjustment.
- Estimate daily temperature–demand Spearman correlation. Save daily bootstrap intervals for signed Cliff's delta and rho, and rank epsilon squared for the weather omnibus test.
- Save tables, a figure, reproducibility metadata and interpretation notes. Verify aggregation, group exclusions, multiplicity adjustment and reproducibility with the actual data.

Implementation choice: unique modal weather summarizes daily predominant conditions; ties are excluded only from weather comparisons. Minimum five dates per weather group is a pragmatic guard for asymptotic tests, not a guarantee of power or validity. Results remain exploratory because dates may be temporally dependent.
