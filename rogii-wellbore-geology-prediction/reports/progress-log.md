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

## typewell_gr_sequence_residual_hgb

- Tested a more direct sequence-shape match instead of summary-stat window matching.
- For each horizontal target row, extracted a centered `31`-row GR sequence.
- Linearly filled missing values inside the local window when at least `14` values were present, then normalized the mini-sequence to zero mean and unit variance.
- Built matching normalized `31`-row GR sequences from the paired typewell within `+/-240` TVT feet of the last known horizontal `TVT_input`.
- Used nearest-neighbor matching on the normalized sequence shape, then added matched TVT delta, match distance, mean delta, standard-deviation delta, match availability, and local valid-fraction features.
- Validated on `618,007` grouped-by-well sampled target rows. The model trained on up to `700` sampled rows per well and used the same `0.60` residual blend and `+/-60` residual clip.
- Pushed Kaggle kernel `jdow76/rogii-sequence-gr-residual`, version 1, and submitted output as Kaggle submission `54443515`.

### Result

| Metric | Base sampled residual | Sequence-match sampled residual | Impact |
| --- | ---: | ---: | ---: |
| Sampled local RMSE | `15.15077` | `15.11805` | `-0.03272` |
| Relative sampled improvement | - | - | `0.22%` |
| Sequence match availability | - | `80.53%` | - |
| Kaggle public RMSE | `14.304` best current | `14.596` | worse by `0.292` |

### Interpretation

This is the first GR approach that compares the actual shape of the GR curve rather than one value or a few window summaries. The local improvement was larger than the previous window-summary experiment and had much better match coverage, but the public score got worse. That means the sequence-shape signal is probably real on the sampled validation rows, but it does not transfer cleanly to the hidden public wells in this form. The current best remains the simpler residual correction model.

## bagged_shrink_residual_hgb

- Tested a robustness upgrade to the winning residual model instead of adding more GR matching features.
- Trained five HGB residual models with different random per-well training samples.
- Averaged their clipped residual predictions, keeping the same `0.60` blend that worked for the original residual model.
- Also tested dynamic shrinkage, where the correction gets smaller when the five models disagree. The alpha sweep selected the plain fixed `0.60` blend instead, so the final submitted model used bagging without dynamic shrinkage.
- Validated on `618,007` grouped-by-well sampled rows, with `900` sampled training rows per well per model.
- Pushed Kaggle kernel `jdow76/rogii-bagged-residual`, version 1, and submitted output as Kaggle submission `54477669`.

### Result

| Metric | Single sampled residual | Bagged residual | Impact |
| --- | ---: | ---: | ---: |
| Sampled local RMSE | `15.11501` | `15.04806` | `-0.06694` |
| Relative sampled improvement | - | - | `0.44%` |
| Best blend rule | fixed `0.60` | fixed `0.60` | dynamic shrinkage did not help |
| Kaggle public RMSE | `14.304` previous best | `14.161` | `-0.143` |

### Interpretation

The useful part was not making the model more complicated at prediction time. It was asking five slightly different versions of the same strong residual model, then averaging their answers. That smoothed out some noisy overcorrections and transferred to the public leaderboard. Dynamic shrinkage sounded sensible, but the validation data did not reward it; the simple fixed blend remained best. The current best is now the bagged residual model.

## lightweight_particle_filter_hgb

- Built a lightweight particle filter to track a plausible TVT path through the post-PS lateral.
- Each particle tracks a `TVT + Z` state, moves forward using the prefix TVT/Z trend, and is reweighted by how closely the horizontal GR matches the calibrated typewell GR at that candidate TVT.
- Tested the raw PF path directly and as features inside the residual HGB model.
- Raw PF alone was noisy: sampled RMSE was `17.13177`, worse than the `15.60039` carry-forward baseline. A partial baseline/PF blend helped, with best alpha `0.40` scoring `14.46795`.
- PF features were much stronger. Adding `pf_delta`, `pf_std`, and PF-vs-prefix-slope features to the residual model improved sampled grouped validation from `15.11501` to `14.16951`.
- Pushed Kaggle kernel `jdow76/rogii-lightweight-particle-filter`, version 1, and submitted output as Kaggle submission `54480888`. Kaggle accepted the submission, but the public score is still pending.

### Result

| Metric | Base sampled residual | PF-feature residual | Impact |
| --- | ---: | ---: | ---: |
| Sampled local RMSE | `15.11501` | `14.16951` | `-0.94550` |
| Relative sampled improvement | - | - | `6.26%` |
| Raw PF sampled RMSE | - | `17.13177` | worse than baseline alone |
| Best baseline/PF path blend | - | `14.46795` | alpha `0.40` |
| Kaggle public RMSE | `14.161` best current | pending | pending |

### Interpretation

The particle filter is not good enough to be trusted by itself yet, but it gives the residual model a much better sense of the likely path. That is exactly the pattern from stronger public approaches: use path trackers as signals, then let a model correct them. This is the first local result that looks like a step into the right solution family rather than another small row-level tweak.
