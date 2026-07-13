# Forecast Generation Process

How each `hp30` forecast is produced end-to-end, including how missing or late
upstream data is handled. This is the canonical reference for the runtime
behaviour; see [architecture.md](architecture.md) for the overall system design.

## At a glance

```
trigger (cron, 3 attempts/anchor)
   → fetch NOAA + GFZ           ── fetch fails ─┐
   → aggregate + align (impute) ── unfillable ──┤
   → model + MCD uncertainty                    │
   → write forecast                             │
   → classify: ok / imputed / failed ◄──────────┘ (failed)
   → publish (don't-downgrade) + deploy page
```

## 1. Trigger

- A GitHub Actions schedule runs the pipeline every 10 minutes:
  `cron: '8,18,28,38,48,58 * * * *'`.
- Each 30-minute anchor gets **three attempts**:
  - anchor `:00` → runs at `:08`, `:18`, `:28`
  - anchor `:30` → runs at `:38`, `:48`, `:58`
- A run can also be started manually (Actions → **Forecast** → **Run workflow**).
- **Caveat:** GitHub schedules are best-effort and may be delayed or dropped
  under load, so not all three attempts always fire.

## 2. Data collection

The inference CLI (`run_realtime.py`) computes the **anchor**
`t_end = floor(now − 2 min, 30 min)` (all times **UTC**) and fetches the live
feeds, each with up to 3 retries and a 30-second timeout:

- **NOAA SWPC** — solar wind plasma + interplanetary magnetic field.
- **GFZ Potsdam** — Hp30 / hp30 geomagnetic nowcast.

If a feed cannot be retrieved after its retries, the run exits with code **2**
("data gap") and goes to the failure path in step 6.

## 3. Preprocessing and missing-data handling

1. Resample the 1-minute feed to 30-minute bins and align onto the 24-step
   (12-hour) input window ending at the anchor.
2. **Impute missing values ("always emit" policy):**
   - **Forward-fill** — carry the last available value forward (up to 48 steps),
     which covers gaps at the recent/tail end of the window.
   - **Linear interpolation** (both directions) for interior gaps.
   - Proceed unless a single variable is **almost entirely** missing
     (`max_gap_fraction = 0.9`); the most-recent steps are **not** required to be
     real (`require_recent_steps_present = 0`).
   - As a last resort, the anchor may roll back by 30 minutes (up to 2 times) if
     the current window cannot be filled at all.
3. If, after imputation, a usable window still cannot be built, the run exits
   with code **2** (failure path).

The fraction of cells that had to be imputed is recorded as
`missing_data_filled_fraction` and drives the status in step 5.

## 4. Inference

Normalize the window with the training statistics, run the model
(GNN + PatchTST), compute a Monte Carlo Dropout (MCD) uncertainty interval
(±2σ), denormalize, and write the 24-step forecast (JSON + CSV). The run exits
with code **0**.

## 5. Status classification

Each run is classified and the result is recorded with the forecast.

| Condition | Status |
|---|---|
| exit 0 and `missing_data_filled_fraction` ≤ 0.05 | **ok** (normal) |
| exit 0 and `missing_data_filled_fraction` > 0.05 | **imputed** |
| non-zero exit (fetch failure / unfillable) | **failed** |

## 6. Publishing and the retry rule

`update_site_data.py` records the run:

- **Success (ok / imputed):** write `site/data/latest.json` (with the observed
  history embedded), set `status.json = "ok"`, and append the forecast to the
  per-anchor archives `forecast_history.json` / `.csv` (each carrying a `status`
  field/column).
- **Failure:** **keep the previous `latest.json`** (the page continues to show
  the last good forecast), set `status.json = "warn"` (data gap) or `"error"`,
  and record a **`failed`** marker for the current anchor in the archives.

**Don't-downgrade rule.** Because each anchor is attempted up to three times, a
later attempt only overwrites the stored record for that anchor when its status
is the **same or better** (`ok` > `imputed` > `failed`). So a later transient
failure never clobbers an earlier good forecast, and an `imputed` result is
upgraded to `ok` if a later attempt gets clean data.

## 7. Page display

The job itself always succeeds (failures are handled in software), so the static
site is re-deployed every run. The page shows a status **banner** — green
(current), yellow (data gap / stale / heavily imputed), or red (error) — and the
**plot**: observed hp30 history, the current 12-hour forecast with its MCD
uncertainty band, and a vertical "now" divider.

## 8. Banner messages

The status banner shows one of three colours — green (ok), yellow (warn), red
(error) — with the messages below, evaluated top-down (the first match wins).

| Banner | Message | When | Meaning |
|---|---|---|---|
| error | `Status file unavailable …` | `status.json` cannot be fetched | Status file missing (pipeline not run yet / Pages issue) |
| error | `Forecast data unavailable …` | `latest.json` cannot be fetched | No forecast output exists yet |
| error | `Pipeline error: Inference exited with code N. Showing last successful forecast.` | `status = "error"` (unexpected non-0/2 exit) | Inference crashed unexpectedly; last good forecast shown |
| warn | `InsufficientDataError — upstream data gap, waiting for next cycle. Showing last successful forecast.` | `status = "warn"` (exit 2) | Data unavailable / unfillable; last good forecast kept (archive `failed`) |
| warn | `Data is stale: last successful run was X.X hours ago.` | ok but last run > 2 h old | Forecast not updated recently (runs dropped) |
| warn | `X.X% of input data was filled from upstream gaps.` | ok, fresh, but imputed > 5% | Forecast produced on imputed inputs (archive `imputed`) |
| ok | `Forecast is current.` | ok, fresh (< 2 h), imputed ≤ 5% | Fresh, clean forecast (archive `ok`) |

The `InsufficientDataError …` text comes from
`status.json.last_error.message`, while `Showing last successful forecast.` is
appended by the page. The banner status corresponds to the per-anchor archive
status: ok ↔ `ok`, "X% filled" ↔ `imputed`, "InsufficientDataError" /
"Pipeline error" ↔ `failed`.

## When upstream data is missing — summary

- **Partial gap** (some data fetched): missing values are imputed (forward-fill
  + interpolation) and a forecast is still produced → status **imputed**.
- **Total failure** (feed unreachable, or window unfillable): no forecast is
  produced → exit 2 → the last good forecast is kept on the page (yellow banner)
  and the anchor is recorded as **failed**.
- **Retries**: the same anchor is attempted three times, 10 minutes apart; if
  data returns on a later attempt the slot is filled with `ok` / `imputed` and
  the `failed` marker is overwritten (don't-downgrade).
- **Limit**: if an upstream feed stays unavailable across all attempts, that
  anchor remains `failed` and the previous forecast keeps showing until data
  returns.

## Notes

- All times — anchor and data timestamps — are **UTC**.
- This repository runs its own pipeline and accumulates its own data
  independently; a second deployment running the same code will generally be at
  a different anchor at any given moment (independent, best-effort scheduling),
  so the two can show different values even though the code and model are
  identical.
