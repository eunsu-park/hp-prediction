# hp-prediction

Public dashboard for 12-hour hp30 geomagnetic index forecasts.

> **Status: awaiting first hp30 checkpoint.** The site shell, inference engine,
> and pipeline are wired for hp30, but the matched checkpoint
> (`model_best.pth` + `table_stats.pkl` from `hp_in12h_out12h_gnn_patchtst`) is
> not committed yet — the hp30 model is still training. Until it is dropped into
> `vendor/realtime-regression-sw/checkpoint/` (see
> [that folder's PLACEHOLDER.md](vendor/realtime-regression-sw/checkpoint/PLACEHOLDER.md))
> and GitHub Pages is enabled, the page reports "forecast unavailable".

- Deployed site: https://www.eunsu.me/hp-prediction/
  (also at https://eunsu-park.github.io/hp-prediction/)
- Inference engine: bundled in-tree under `vendor/realtime-regression-sw/`
  (engine developed in [eunsu-park/geoindex-realtime](https://github.com/eunsu-park/geoindex-realtime),
  target selected via `profile.target: hp30`)
- Update cadence: a new anchor every 30 min; the cron fires every 10 min with
  three attempts per anchor (cron `8,18,28,38,48,58 * * * *`, a backup against
  transient upstream outages)
- Architecture details: [docs/architecture.md](docs/architecture.md)

This repo is **self-contained**: the engine is inlined and (once training is
done) the checkpoint is committed in-tree, so a run needs only a checkout — no
submodule, no GitHub Release download.

## How It Works

1. `.github/workflows/forecast.yml` runs on a 10-min cron (three attempts per
   30-min anchor; a later attempt only overwrites an earlier one at equal-or-
   better status, so a transient failure never clobbers a good forecast).
2. It checks out this repo — the inference engine and (once trained) the model
   checkpoint are committed in-tree — and runs `scripts/run_realtime.py` with
   `--config ../../configs/realtime.ci.yaml` (which sets `profile.target: hp30`).
   If an upstream feed is unreachable the run exits with a "data gap" warning
   (exit 2) instead of failing hard.
3. `scripts/update_site_data.py` copies the newest forecast JSON into
   `site/data/latest.json`, refreshes `site/data/status.json`, and appends to
   the past-forecast archives (`forecast_history.json` / `.csv`).
4. The `site/` directory is published as a GitHub Pages artifact.
5. `site/index.html` fetches `data/latest.json` (+ `forecast_history.json`) on
   load and renders a Chart.js plot of the 24-step (12-hour) hp30 forecast, the
   observed history, and the past-forecast line.

## Repository Layout

```
hp-prediction/
├── .github/workflows/forecast.yml   cron-triggered pipeline
├── vendor/realtime-regression-sw/   inlined inference engine (+ committed checkpoint, TBD)
├── configs/realtime.ci.yaml         CI path overrides + profile.target: hp30
├── scripts/update_site_data.py      post-process inference output
├── site/
│   ├── index.html                   page shell
│   ├── main.js                      Chart.js render + metadata
│   └── data/
│       └── status.json              pipeline status for the banner
└── README.md
```

## One-Time Setup

Enable GitHub Pages: Settings → Pages → Build and deployment → Source:
**GitHub Actions**. Then commit the matched hp30 checkpoint pair (below).

## Adding / Updating the Checkpoint

The engine and weights are vendored in-tree, so an upgrade is a payload refresh:

1. Train `hp_in12h_out12h_gnn_patchtst` in
   [eunsu-park/geoindex-model](https://github.com/eunsu-park/geoindex-model)
   (`server_hp` profile) and pick the best epoch.
2. Copy the matched pair into the checkpoint folder:

   ```
   cp <run>/checkpoint/model_best.pth  vendor/realtime-regression-sw/checkpoint/
   cp <run>/checkpoint/table_stats.pkl vendor/realtime-regression-sw/checkpoint/
   git rm vendor/realtime-regression-sw/checkpoint/PLACEHOLDER.md
   git add vendor/realtime-regression-sw/checkpoint/model_best.pth \
           vendor/realtime-regression-sw/checkpoint/table_stats.pkl
   ```

3. Fill `model_provenance` (val loss/MAE/RMSE, epoch) in
   `configs/realtime.ci.yaml`, then commit.
4. Matched-pair invariant: `model_best.pth` and `table_stats.pkl` must come from
   the same training run (no runtime validation).

## Trigger a Run Manually

Actions tab → "Forecast" workflow → "Run workflow".
Optionally provide an ISO8601 `now` to replay a specific anchor.

## Failure Handling

`run_realtime.py` exit codes mapped by `scripts/update_site_data.py`:

- `0` → `status.json.status = "ok"`, `latest.json` updated
- `2` → `status.json.status = "warn"` (InsufficientDataError — upstream data
  gap), `latest.json` preserved
- other → `status.json.status = "error"`, `latest.json` preserved

The workflow itself always succeeds (the Actions badge stays green); the page
banner is the true health indicator.

## License

MIT License. See [LICENSE](LICENSE).
