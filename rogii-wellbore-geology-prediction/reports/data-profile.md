# Data Profile

## Dataset And Grain

The unit of modeling is a row in a horizontal well after the prediction-start point, where `TVT_input` is missing and true `TVT` must be predicted. Training files include the true suffix `TVT`, so they provide a natural local validation target. Test files include the same visible fields but omit suffix `TVT`.

## File Inventory

| Split | Horizontal CSVs | Typewell CSVs | PNGs |
| --- | ---: | ---: | ---: |
| Train | `773` | `773` | `773` |
| Test | `3` | `3` | `0` |

## Horizontal Well Data

| Measure | Value |
| --- | ---: |
| Total training rows | `5,092,255` |
| Known pre-start `TVT_input` rows | `1,308,266` |
| Post-start target rows | `3,783,989` |
| Global `TVT_input` null rate | `74.31%` |
| Global `GR` null rate | `29.61%` |
| True `TVT` nulls in training | `0` |

Training row counts per well:

| Statistic | Rows | Known rows | Target rows |
| --- | ---: | ---: | ---: |
| Mean | `6,587.7` | `1,692.5` | `4,895.2` |
| Median | `6,576` | `1,703` | `4,840` |
| 95th percentile | `8,614.2` | `2,053` | `6,918.4` |
| Max | `12,141` | `2,392` | `10,052` |

Horizontal columns:

- `MD`, `X`, `Y`, `Z`: complete in training.
- `GR`: sparse, with `1,507,972` training nulls.
- `TVT_input`: intentionally null after prediction start.
- `TVT`: target, complete in training.
- `ANCC`, `ASTNU`, `ASTNL`, `EGFDU`, `EGFDL`, `BUDA`: formation-top columns available in training horizontal files. Most are complete, but `ANCC` has `45,634` nulls and `EGFDL` has `6,067` nulls.

## Typewell Data

| Measure | Value |
| --- | ---: |
| Total training typewell rows | `1,567,045` |
| Typewell `GR` nulls | `0` |
| Typewell `Geology` nulls | `523,474` |
| Median typewell rows per well | `1,874` |
| Max typewell rows per well | `10,043` |

Most common non-null geology labels:

| Label | Rows |
| --- | ---: |
| `ANCC` | `294,268` |
| `EGFDL` | `205,397` |
| `ASTNL` | `172,223` |
| `BUDA` | `140,640` |
| `ASTNU` | `118,025` |
| `EGFDU` | `70,013` |

## Test And Submission Shape

The sample submission has `14,151` target rows across three wells:

| Well | Submission Rows | Known Rows Before PS | GR Nulls |
| --- | ---: | ---: | ---: |
| `000d7d20` | `3,836` | `1,442` | `2,258` |
| `00bbac68` | `6,014` | `1,545` | `942` |
| `00e12e8b` | `4,301` | `2,083` | `584` |

## Data Quality Notes

- The target `TVT` is complete in training, which makes validation straightforward.
- `TVT_input` missingness is intentional and defines the post-start prediction region.
- `GR` is materially sparse in horizontal wells, so the first baseline does not depend on it.
- Typewell `GR` is complete, making it a promising next feature source.
- Typewell `Geology` labels are partially missing, so any geology-label feature should handle missing labels explicitly.

## Baseline Implication

The shape of the data supports a continuity baseline first: every test well has a known pre-start `TVT_input` history, and every requested target row is after that known segment. Carrying forward the last known value is crude, but it is stable, transparent, and locally validated before adding GR alignment or spatial dip modeling.
