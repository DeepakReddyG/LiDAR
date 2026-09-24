# M5 bounded development-data quality report

Generated 2026-09-24T00:09:27.194957+00:00. Source: `revision_work/runs/phase1_fresh/input.las`.

This is descriptive evidence on fixed development data. No reference classes, model predictions, A/B inputs or full-survey inputs were read. It was not used to select an experiment.

Sensor, acquisition date, colorization, radiometric encoding and color-registration accuracy: **UNKNOWN - needs provider/PI**. CRS: UNKNOWN - needs provider/PI. Horizontal unit status: assumed; CRS absent; vertical unit status: assumed; CRS absent. Vertical datum: UNKNOWN - needs provider/PI. The configured distance unit is US survey foot (1200/3937 metre).

## Bounded context and nested pilot

| Measure | Tile-C context | Pilot |
|---|---:|---:|
| Points | 2,505,912 | 158,618 |
| Fixed XY rectangle area (ft²) | 22,500.0 | 1,200.0 |
| Points per fixed rectangle ft² | 111.374 | 132.182 |
| Observed XYZ extent (ft) | 149.999, 149.999, 56.670 | 39.999, 29.999, 55.644 |
| Occupied 1 ft XY cells / total | 22,499 / 22,500 | 1,200 / 1,200 |
| Minimum points per 1 ft cell (including empty cells) | 0.00 | 24.00 |
| Median points per 1 ft cell (including empty cells) | 94.00 | 118.00 |
| 90th percentile points per 1 ft cell (including empty cells) | 176.00 | 243.00 |
| 99th percentile points per 1 ft cell (including empty cells) | 347.00 | 376.06 |
| Maximum points per 1 ft cell (including empty cells) | 4120.00 | 425.00 |
| Any RGB channel zero | 87,118 (3.4765%) | 13,905 (8.7663%) |
| All RGB channels zero | 1,400 (0.0559%) | 17 (0.0107%) |
| Any RGB channel at 65535 | 513 (0.0205%) | 8 (0.0050%) |
| Negative HAG points | 99,211 (3.9591%) | 1,636 (1.0314%) |
| HAG minimum / maximum (ft) | -2.63124 / 54.24360 | -0.43856 / 54.01186 |

Density includes all heights and possible multiple surfaces/overlap in each XY cell. These counts are not independent observations or a terrain-surface sampling rate. Fixed rectangle area and actual observed XYZ bounds are separate fields in the JSON.

## Stored RGB values

| Region | Channel | Min | Median | p90 | p99 | Max | Values equal to 255 | Values equal to 65535 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| context_c | red | 0.0 | 24977.0 | 37979.0 | 57609.0 | 65535.0 | 58 | 422 |
| context_c | green | 0.0 | 25595.0 | 37762.0 | 57184.0 | 65535.0 | 61 | 354 |
| context_c | blue | 0.0 | 16750.0 | 36920.0 | 56380.0 | 65535.0 | 14 | 286 |
| pilot_c | red | 0.0 | 24042.5 | 36010.0 | 43988.8 | 65535.0 | 6 | 5 |
| pilot_c | green | 0.0 | 25911.0 | 35455.0 | 42662.0 | 65535.0 | 2 | 4 |
| pilot_c | blue | 0.0 | 15672.0 | 24461.0 | 32639.0 | 65535.0 | 0 | 1 |

The endpoint 65535 is the unsigned 16-bit storage limit; **255 is not called saturation**. Storage endpoint counts do not establish optical saturation or original radiometric bit depth. The JSON includes exact zero/endpoint rates, distinct values, and divisibility patterns as observations only. No provider encoding or color-registration quality is inferred.

## Generated terrain support

The final 300 × 300 DTM at 0.5 ft resolution has 90,000 finite cells (100.0000%); 11 finite cells contain no input point in their XY cell. All-height input occupancy is not observed-ground support.

**NOT SAVED by Phase 1 pipeline; cannot distinguish observed-ground/interpolated/fallback-filled terrain cells from final DTM alone.** No independent terrain checkpoints or ground reference were used. Negative HAG is a measured residual against this estimated DTM, not proof of an acquisition error or a vertical-accuracy metric.

## Provenance and unresolved work

Input crop SHA-256: `4052f08a0785b61e28db98e5ce2b2614729047efe9d35d705bf5a0394edc97dc`. Script SHA-256: `d57985ddbbc638f01264572afba709a47f4114b5e180771d56f377e895748eff`. The JSON records all input/artifact hashes and exact statistics.

- One fixed development context and one nested pilot, not independent sites.
- High XY point counts include multiple heights, vertical surfaces and potential acquisition overlap; points are not independent observations.
- Observed RGB storage endpoints and code patterns do not establish optical saturation, bit-depth provenance, colorization method or registration accuracy.
- Negative HAG and DTM coverage are descriptive residual/support checks; no absolute or relative terrain accuracy is claimed.
- Acquisition and color-registration provenance require provider/PI confirmation.
