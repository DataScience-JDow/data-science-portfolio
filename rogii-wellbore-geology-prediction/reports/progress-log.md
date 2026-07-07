# Progress Log

## baseline_last_known_tvt

- Set up project scaffold for the ROGII wellbore geology prediction competition.
- Downloaded and inspected the Kaggle data locally.
- Extracted the task framing from the competition PowerPoint deck.
- Confirmed the target is post-start horizontal-well `TVT`, evaluated by RMSE.
- Built a last-known-TVT carry-forward baseline.
- Validated on all training-well suffix rows where `TVT_input` is null and true `TVT` is available.
- Submitted via Kaggle kernel `jdow76/rogii-last-known-tvt-baseline`, version 2.

### Result

| Metric | Value |
| --- | ---: |
| Training wells scored locally | `773` |
| Local validation rows | `3,783,989` |
| Local RMSE | `15.90985` |
| Kaggle public RMSE | `15.883` |
| Kaggle submission ref | `54374628` |

### Baseline Rationale

The first baseline should be hard to fool and easy to explain. Since `TVT_input` is known up to the prediction-start point, carrying forward the final known value gives a continuity baseline for each well. It avoids fitting a slope that may extrapolate poorly after the well path changes direction or enters flatter/dipping geology.

### Next Ideas

- Add typewell GR-to-TVT alignment features.
- Use pre-start horizontal GR patterns to align post-start gamma ray signatures.
- Use neighboring training wells and azimuth/coordinate features for dip-aware corrections.
- Compare per-well errors to identify when carry-forward fails.

## residual_correction_hgb

- Built a residual model on top of the carry-forward baseline.
- Targeted the model at `true TVT - last known TVT_input` instead of predicting absolute `TVT` from scratch.
- Used only test-available fields: post-start distance, MD/XYZ deltas, last known TVT, prefix TVT slope, prefix Z slope, GR values, GR missingness, and prefix GR statistics.
- Validated with 5-fold grouped cross-validation by well, so rows from the same well never appear in both train and validation for the same fold.
- Sampled at most `1,400` post-start rows per training well for model fitting, while scoring all `3,783,989` validation target rows.
- Blended the residual correction at `0.60` and clipped raw residual corrections to `+/-60` feet to avoid letting the model oversteer the strong baseline.
- Submitted Kaggle kernel `jdow76/rogii-residual-correction`, version 1, as submission ref `54410984`.

### Result

| Metric | Baseline | Residual correction | Impact |
| --- | ---: | ---: | ---: |
| Local RMSE | `15.90987` | `15.33328` | `-0.57659` |
| Relative local improvement | - | - | `3.62%` |
| Kaggle public RMSE | `15.883` | `14.304` | `-1.579` |

### Why This Helped

The baseline assumed `TVT` stays flat after the prediction-start point. That was strong, but it missed wells where geology continued drifting up or down. The residual model keeps the baseline as an anchor, then learns a controlled correction from geometry and prefix behavior. Blending mattered: the unblended model helped some folds but hurt others, while the `0.60` blend gave the best overall validation RMSE.

### Validation Notes

The alpha sweep showed the best local blend around `0.60`:

| Blend alpha | Local RMSE |
| ---: | ---: |
| `0.0` | `15.90987` |
| `0.3` | `15.47854` |
| `0.5` | `15.34909` |
| `0.6` | `15.33328` |
| `0.7` | `15.35030` |
| `1.0` | `15.59615` |

### Public Leaderboard Impact

The public leaderboard improved from `15.883` to `14.304`, a `9.94%` public RMSE reduction from the baseline submission.

## typewell_gr_residual_hgb

- Added simple typewell GR matching features to the residual model.
- For each horizontal row with `GR`, searched the paired typewell within `+/-180` TVT feet of the last known horizontal `TVT_input`.
- Added five GR-alignment features: typewell GR at the baseline TVT, horizontal-minus-typewell GR at the baseline, nearest-GR matched TVT delta, nearest-GR match error, and match availability.
- Kept the same modeling guardrails as the residual model: grouped validation by well, at most `1,400` sampled training rows per well, residual clipping at `+/-60`, and `0.60` residual blending.
- Submitted Kaggle kernel `jdow76/rogii-typewell-gr-residual`, version 1, as submission ref `54412802`.

### Result

| Metric | Residual correction | Typewell GR residual | Impact |
| --- | ---: | ---: | ---: |
| Local RMSE | `15.33328` | `15.28302` | `-0.05026` |
| Relative local improvement | - | - | `0.33%` |
| Kaggle public RMSE | `14.304` | `14.395` | `+0.091` |

### Interpretation

The typewell GR features helped locally, but the public score got worse. This suggests the simple "nearest GR value inside a TVT window" feature contains some signal, but it is not stable enough on the public test wells. A stronger version should compare short GR sequences rather than one row at a time.

## typewell_gr_window_experiment

- Tested a short-window GR pattern match instead of a single-row GR match.
- For each horizontal row, summarized a centered `41`-row GR window using count, mean, standard deviation, and end-to-end GR delta.
- Built rolling typewell GR windows with the same summary shape.
- Used nearest-neighbor matching to find the most similar typewell GR window within `+/-220` TVT feet of the last known horizontal `TVT_input`.
- Added match features: matched TVT delta, match distance, GR-mean delta, and match availability.
- Validated on a grouped-by-well sampled frame of `926,518` target rows because the full multi-million-row HGB fit was too slow with the extra window features.

### Result

| Metric | Base sampled residual | Window-match sampled residual | Impact |
| --- | ---: | ---: | ---: |
| Sampled local RMSE | `15.21663` | `15.20325` | `-0.01339` |
| Relative sampled improvement | - | - | `0.09%` |
| Kaggle public RMSE | `14.304` best current | not submitted | not applicable |

### Decision

The window feature moved validation in the right direction, but the gain was too small to justify a Kaggle submission. The prior typewell-GR model already showed that small local GR gains may not transfer to public score, so the safer move is to stop here and rethink the GR alignment rather than submit a weak candidate.
