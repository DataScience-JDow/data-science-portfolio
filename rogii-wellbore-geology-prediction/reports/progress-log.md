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
