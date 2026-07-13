# Architecture

This document explains how `hp-prediction` works end-to-end: which pieces
exist, how data flows from the live upstream feeds to the browser, and how
the dashboard page connects to the main personal site at `www.eunsu.me`.

For the runtime behaviour of a single forecast (data collection, imputation,
status classification, the retry rule), see
[docs/forecast-process.md](forecast-process.md).

---

## 1. Overview

`hp-prediction` publishes a live 12-hour hp30 geomagnetic-index forecast
chart at `https://www.eunsu.me/hp-prediction/`. A GitHub Actions cron
re-runs the inference pipeline every 10 minutes (three attempts per 30-min
anchor), writes a fresh `latest.json`, and deploys the updated static site
to GitHub Pages.

**Design tenets**

- **Self-contained in-tree.** The inference engine is inlined under
  `vendor/realtime-regression-sw/`, and the model weights
  (`model_best.pth`) and normalizer stats (`table_stats.pkl`) are committed
  in-tree under `vendor/realtime-regression-sw/checkpoint/`. A forecast run
  needs only a plain checkout — no git submodule, no GitHub Release
  download, no cache step.
- **Matched-pair invariant.** `model_best.pth` and `table_stats.pkl` must
  come from the same training run. Mismatched files silently produce
  miscalibrated forecasts; there is no runtime check, so the pairing is
  enforced by process (they are swapped together, see §5).
- **Static site.** Everything the browser consumes is JSON on disk
  (`site/data/latest.json` + `status.json`). No backend API, no database,
  no server-side rendering.

---

## 2. Component map

This repository is self-contained: the workflow, the inlined engine, the
committed checkpoint, and the site all live together. Two other
repositories are involved only at the edges — one upstream (where the engine
is developed) and one downstream (the homepage that links to the dashboard).

```
┌─────────────────────────────────────────────────────────────────────────┐
│  github.com/eunsu-park/geoindex-realtime               (engine dev)      │
│    Train + validate the model here. On an upgrade, the engine source     │
│    and the checkpoint pair are re-inlined into hp-prediction (§5).       │
└──────────────────────┬──────────────────────────────────────────────────┘
                       │ payload refresh (re-inline engine + swap checkpoint)
                       ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  github.com/eunsu-park/hp-prediction      (this repo, self-contained)    │
│    ├── .github/workflows/forecast.yml   ← cron + build + deploy          │
│    ├── vendor/realtime-regression-sw/   ← inlined engine (in-tree dir)   │
│    │   ├── src/, scripts/run_realtime.py   inference engine + CLI        │
│    │   └── checkpoint/                                                   │
│    │       ├── model_best.pth           ← committed weights             │
│    │       └── table_stats.pkl          ← committed normalizer stats     │
│    ├── configs/realtime.ci.yaml         ← CI path overrides             │
│    ├── scripts/update_site_data.py      ← JSON post-process             │
│    ├── site/index.html                  ← page shell                    │
│    ├── site/main.js                     ← Chart.js renderer             │
│    └── site/data/                                                       │
│        ├── latest.json                  ← most recent forecast          │
│        ├── status.json                  ← pipeline health               │
│        └── forecast_history.json/.csv   ← per-anchor archives           │
└──────────────────────┬──────────────────────────────────────────────────┘
                       │ actions/deploy-pages@v4 (artifact)
                       ▼
          www.eunsu.me/hp-prediction/       (served page)
          eunsu-park.github.io/hp-prediction/ (alias, auto-redirect)

┌─────────────────────────────────────────────────────────────────────────┐
│  github.com/eunsu-park/eunsu-park.github.io                             │
│    ├── _config.yml   (url: https://www.eunsu.me)                        │
│    ├── CNAME         (www.eunsu.me)                                     │
│    └── _includes/navigation.html   ← sidebar link to /hp-prediction     │
└──────────────────────┬──────────────────────────────────────────────────┘
                       ▼
          www.eunsu.me/                     (main CV site)
```

**Why keep them separate**

- `geoindex-realtime` is the engine's development home. Training and
  validation happen there; `hp-prediction` only ever receives a vetted,
  re-inlined payload, so day-to-day model-code churn never destabilizes the
  live dashboard.
- `eunsu-park.github.io` is a separate Jekyll CV site. It stays clean — the
  forecast auto-commits land in `hp-prediction`, not here, so a 10-min cron
  never triggers its Jekyll rebuild.

---

## 3. Data flow

Every anchor (new every 30 min, up to three attempts), one full cycle from
upstream feed to browser happens:

```
┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────────┐
│ NOAA SWPC plasma     │   │ NOAA SWPC magnetic   │   │ GFZ Hp30/ap30        │
│ (1-min cadence)      │   │ (1-min cadence)      │   │ (30-min cadence)     │
└──────────┬───────────┘   └──────────┬───────────┘   └──────────┬───────────┘
           └──────────────┬──────────────┘                       │
                          ▼                                      ▼
                ┌─────────────────────────────────────────────────────┐
                │ vendor/realtime-regression-sw — run_realtime.py     │
                │                                                     │
                │  1. Fetch the three HTTP feeds (requests + retry)   │
                │  2. Aggregate 1-min → 30-min bins                   │
                │  3. Compute anchor t_end = floor(now - 2min, 30min) │
                │  4. Build the 24-row × 22-col event window (impute) │
                │  5. Normalize with table_stats.pkl                  │
                │  6. Run model_best.pth (GNN + PatchTST, CPU)        │
                │  7. Denormalize; emit 24-step hp30 forecast + MCD   │
                │  8. Write JSON + CSV to results/predictions/…       │
                └──────────────────────┬──────────────────────────────┘
                                       │
                                       ▼
                ┌─────────────────────────────────────────────────────┐
                │ hp-prediction — update_site_data.py                 │
                │                                                     │
                │  1. Locate newest JSON under vendor/.../results/    │
                │  2. Read it                                         │
                │  3. Locate the paired event CSV                     │
                │     (dataset/events/{anchor_stem}.csv)              │
                │  4. Embed the last 96 rows of observed (datetime,   │
                │     hp30) as the "history" array (48 h of display)  │
                │  5. Write site/data/latest.json (don't-downgrade)   │
                │  6. Refresh status.json + forecast_history.json/csv │
                └──────────────────────┬──────────────────────────────┘
                                       │
                                       ▼ (git commit + push site/data to main)
                                       │
                                       ▼ (actions/deploy-pages artifact)
                                       │
                                       ▼
                ┌─────────────────────────────────────────────────────┐
                │ Browser — site/main.js                              │
                │                                                     │
                │  1. fetch("./data/latest.json", {cache:"no-store"}) │
                │  2. fetch("./data/status.json", {cache:"no-store"}) │
                │  3. Populate metadata block (UTC + KST)             │
                │  4. Paint status banner based on status.json        │
                │  5. Render Chart.js: gray history + blue forecast   │
                │     + MCD band + "now" divider at the anchor        │
                │  6. x-axis tick labels formatted in UTC             │
                └─────────────────────────────────────────────────────┘
```

### 3.1 Input

- **NOAA SWPC RTSW real-time solar wind** — plasma (density, speed, temp) and
  IMF magnetic field (Bx/By/Bz/Bt) from the SOLAR1 primary (ACE/IMAP fill gaps
  only where SOLAR1 is missing). ~24-hour rolling JSON.
- **GFZ Potsdam Hp30/ap30 nowcast** — 30-min geomagnetic index observed
  values. Text file, published within minutes of each 30-min boundary.

### 3.2 Anchor computation

The "anchor time" `t_end` is the most recent completed 30-min boundary,
minus a 2-minute safety offset to let the publishers finish posting:

```
t_end = floor(now - 2min, to 30-min boundary)
```

Example: at 14:13 UTC → `t_end = 14:00 UTC`. At 14:45 UTC
→ `t_end = 14:30 UTC`.

If the input window cannot be filled even after imputation, `t_end` may
roll back one 30-min step (up to 2 attempts). Beyond that, the CLI exits
with code 2 (data gap). See [forecast-process.md](forecast-process.md) §3
for the full imputation policy.

### 3.3 Model I/O shape

Active profile: **`in12h_out12h_gnn_patchtst`** — an 8-node GNN with a
PatchTST temporal backend.

| Tensor | Shape | Description |
|--------|-------|-------------|
| Input  | `(1, 24, 22)` | 1 batch × 24 timesteps (12 hours × 30-min) × 22 vars |
| Output | `(1, 24, 1)`  | 1 batch × 24 timesteps (12 hours × 30-min) × 1 var (hp30) |

22 input variables: 21 solar-wind parameters (v/np/t ×avg/min/max,
Bx/By/Bz/Bt ×avg/min/max) + hp30.

The input ordering and normalization schema are **safety-critical
invariants** — the input window and the `table_stats.pkl` used to normalize
it must match the trained model.

> Note: the 24-row figure above is the **model input window** (12 h). The
> `history` array embedded into `latest.json` for the chart is a separate,
> longer 96-row (48 h) slice of observed hp30 used only for display.

---

## 4. The GitHub Actions workflow

File: [.github/workflows/forecast.yml](../.github/workflows/forecast.yml)

### 4.1 Triggers

```yaml
on:
  schedule:
    - cron: '8,18,28,38,48,58 * * * *'   # every 10 min; 3 attempts per anchor
  workflow_dispatch:                      # manual trigger from the UI
    inputs:
      now: {description: 'ISO8601 anchor override', required: false}
```

- **Cron** — fires every 10 minutes. Each 30-min anchor gets three attempts:
  the `:00` anchor at `:08`/`:18`/`:28`, the `:30` anchor at
  `:38`/`:48`/`:58`. The `:08` offset gives publishers time to post. A
  later attempt only overwrites an earlier one when its status is the same
  or better (**don't-downgrade**, see §4.5), so a transient failure never
  clobbers a good forecast. GitHub schedules are best-effort and may be
  delayed or dropped, so not all three attempts always fire.
- **workflow_dispatch** — manual trigger with optional `now` parameter for
  replaying a specific anchor (debugging / backfill).

### 4.2 Concurrency

```yaml
concurrency:
  group: forecast
  cancel-in-progress: false
```

If the previous run is still going, queue the next one rather than
cancel it. Prevents the pipeline from eating its own tail under heavy
scheduler drift.

### 4.3 Permissions

```yaml
permissions:
  contents: write       # auto-commit site/data/*.json
  pages: write          # for actions/deploy-pages
  id-token: write       # OIDC token required by deploy-pages
```

### 4.4 Steps

Because the engine and checkpoint are committed in-tree, the workflow is a
plain checkout followed by install → infer → post-process → deploy. There
is **no** `submodules: true`, **no** `actions/cache` for weights, and **no**
`gh release download` step.

| # | Step | Purpose |
|---|------|---------|
| 1 | `actions/checkout@v4` (no submodules) | Pull this self-contained repo, including the inlined engine + committed checkpoint |
| 2 | `actions/setup-python@v5` (3.12, pip cache keyed on `vendor/realtime-regression-sw/requirements.txt`) | Python runtime + speed up subsequent installs |
| 3 | `pip install torch --index-url .../cpu` | **CPU-only** PyTorch wheel (~200 MB instead of ~1.5 GB for CUDA) |
| 4 | `pip install -r vendor/realtime-regression-sw/requirements.txt` | numpy, pandas, pyarrow, omegaconf, pyyaml, requests, tqdm, matplotlib |
| 5 | `python scripts/run_realtime.py --config ../../configs/realtime.ci.yaml --device cpu --verbose` (in `vendor/realtime-regression-sw`) | **Inference**. Optional `--now`. Real exit code captured via `set +e` → `$GITHUB_OUTPUT`; the step always `exit 0` |
| 6 | `python scripts/update_site_data.py --exit-code X` | Post-process: copy JSON, embed history, update `status.json` + archives |
| 7 | `git add site/data` + commit + push | Persist `site/data/*` changes to `main` (skipped if nothing changed) |
| 8 | Job summary | Append anchor + first-horizon hp30 to the Actions run summary |
| 9 | `actions/configure-pages@v5` | Signal to Pages: "we're deploying now" |
| 10 | `actions/upload-pages-artifact@v3 path:site` | Upload the `site/` tree as a Pages artifact |
| 11 | `actions/deploy-pages@v4` | Publish the artifact to the live site |

### 4.5 Failure handling

The inference step always exits 0 (its real code is captured separately), so
the workflow **never fails** on inference errors. The failure state is
recorded in `status.json` and rendered as a banner on the page:

| Inference exit code | `status.json.status` | Page banner |
|---------------------|----------------------|-------------|
| `0` (success)       | `"ok"`               | Green: "Forecast is current." |
| `2` (data gap)      | `"warn"`             | Yellow: upstream data gap |
| other non-zero      | `"error"`            | Red: inference error |

When the run fails, `latest.json` is **not overwritten** — the page keeps
showing the last successful forecast with the warning banner on top. Every
run is also recorded into the per-anchor archives
(`forecast_history.json`/`.csv`) with an `ok`/`imputed`/`failed` status, and
the **don't-downgrade** rule (`ok` > `imputed` > `failed`) governs whether a
later attempt overwrites an earlier record for the same anchor. See
[forecast-process.md](forecast-process.md) §5–§8 for the full classification
and banner logic.

---

## 5. Model asset delivery

### 5.1 Why commit the weights in-tree

- `model_best.pth` (~4–5 MB) is small enough that committing it — rather
  than fetching it from a Release at runtime — makes every run reproducible
  from a single checkout, with no external dependency to break.
- The weights and `table_stats.pkl` live side by side under
  `checkpoint/` and are **always swapped together**, making the matched-pair
  coupling a git-level fact rather than a runtime hope.

### 5.2 Updating the checkpoint (payload refresh)

Because the engine and weights are vendored in-tree, an upgrade is a
**payload refresh**, not a submodule bump or a Release upload:

1. **Develop and validate** the new engine/model in
   `eunsu-park/geoindex-realtime`.
2. **Re-inline the engine** into `vendor/realtime-regression-sw/` (source
   under `src/` + `scripts/`) and **swap the checkpoint pair** under
   `vendor/realtime-regression-sw/checkpoint/` — replace both
   `model_best.pth` and `table_stats.pkl` from the **same training run**.
3. **Update `configs/realtime.ci.yaml` if the architecture/shape changed**
   (`profile.*`, `experiment.name`, `window.lookback_steps`,
   `window.forecast_steps`, and `model_provenance.*`). For a same-shape
   retrain, only `model_provenance.*` (the displayed val metrics) needs to
   change. Commit the config change **together with** the checkpoint swap so
   the repo is never in a mismatched state.
4. **Commit and push.** The matched-pair invariant holds by construction —
   both files come from the same run, so there is nothing to reconcile at
   runtime.

> The `sync_to_njit.py` script performs exactly this payload refresh when
> promoting a validated version from this dev/staging repo to the
> production `njit-research/hp-prediction` repo: it copies the engine trees,
> the checkpoint pair, and the web/config payload, applying dev→prod
> reference rewrites, and never touches `.github/` or the bot-maintained
> `site/data/`.

#### Rolling back

Reverting is a `git revert` of the payload-refresh commit (which restores
the previous engine + checkpoint pair together), followed by a push and an
optional manual run. Because the pair is versioned atomically in git, the
rollback can never desync the weights from the stats.

---

## 6. GitHub Pages deployment

### 6.1 "Actions" source vs branch source

We use **Source: GitHub Actions** (not "Deploy from a branch"). This
means:

- No `gh-pages` branch exists. Publishing is done by uploading a Pages
  artifact (`actions/upload-pages-artifact@v3`) and then calling
  `actions/deploy-pages@v4`.
- Each run re-deploys the full `site/` directory. This keeps the build
  deterministic and means `main` branch history is not mixed with a
  parallel `gh-pages` history.

### 6.2 URL resolution

The repo name `hp-prediction` becomes the URL path:

`github.com/eunsu-park/hp-prediction` repo name (`hp-prediction`)
→ project Pages URL path (`/hp-prediction/`).

Because the user account has a user-page repo (`eunsu-park.github.io`)
with a custom domain (`CNAME = www.eunsu.me`), the custom domain is
**automatically inherited** by all project Pages. Therefore both of the
following URLs serve the same content:

- Primary: `https://www.eunsu.me/hp-prediction/`
- Alias: `https://eunsu-park.github.io/hp-prediction/` (301 redirects
  to the primary)

### 6.3 Cache behavior

- JSON files (`latest.json`, `status.json`) are fetched with
  `cache: "no-store"` in `main.js`, so browsers always request a fresh
  copy.
- HTML and JS files (`index.html`, `main.js`) use GitHub Pages' default
  cache headers. The browser may cache them aggressively — if the page
  visibly lags behind, a hard refresh (`Cmd+Shift+R` / `Ctrl+F5`)
  forces a fresh pull.

---

## 7. Homepage integration

The main site (`www.eunsu.me`) is a Jekyll blog in
`github.com/eunsu-park/eunsu-park.github.io`. Integration is **one
line** in `_includes/navigation.html`:

```html
<li><a href="{{ site.baseurl }}/hp-prediction">
  <i class="fas fa-chart-line"></i> HP Forecast
</a></li>
```

**How the link actually works**

1. Jekyll renders `{{ site.baseurl }}/hp-prediction` → `/hp-prediction`
   (since `baseurl` is empty in `_config.yml`).
2. Browser clicks on `<a href="/hp-prediction">` → navigates to
   `https://www.eunsu.me/hp-prediction`.
3. GitHub Pages receives the request for `/hp-prediction/` and serves
   the content from the `hp-prediction` project Pages artifact (i.e.
   the `site/` directory this repo publishes).

Nothing else is shared between the two sites — no CSS, no JavaScript,
no layout. They just happen to live under the same domain.

---

## 8. Files & responsibilities

### 8.1 In `hp-prediction`

| Path | Purpose |
|------|---------|
| [`.github/workflows/forecast.yml`](../.github/workflows/forecast.yml) | Cron-triggered build+deploy pipeline (plain checkout, no submodules/Release) |
| [`configs/realtime.ci.yaml`](../configs/realtime.ci.yaml) | CI path overrides for the inlined engine (checkpoint, stats, event_dir, results_dir all relative to `vendor/realtime-regression-sw/`) + active profile + model provenance |
| [`scripts/update_site_data.py`](../scripts/update_site_data.py) | Post-process: read latest forecast JSON, embed 96-step observed history from the event CSV, write `site/data/latest.json` + `status.json` + `forecast_history.json`/`.csv` (don't-downgrade) |
| [`scripts/sync_to_njit.py`](../scripts/sync_to_njit.py) | Promote a validated payload (engine + checkpoint + web/config) from this dev/staging repo to production `njit-research/hp-prediction` |
| [`site/index.html`](../site/index.html) | Static page shell. Inline CSS. Loads Chart.js v4 + date-fns adapter from jsDelivr CDN |
| [`site/main.js`](../site/main.js) | Fetches `latest.json` + `status.json`, fills metadata, paints banner, renders history + forecast + MCD band with a "now" divider at the anchor, UTC-formatted x-axis ticks, tooltips showing both UTC and KST |
| [`site/data/latest.json`](../site/data/latest.json) | Most recent forecast payload (auto-committed by the workflow) |
| [`site/data/status.json`](../site/data/status.json) | Pipeline health (auto-committed by the workflow) |
| [`vendor/realtime-regression-sw/`](../vendor/realtime-regression-sw) | **Vendored in-tree directory** — the inlined inference engine (`src/`, `scripts/`) + committed `checkpoint/model_best.pth` and `table_stats.pkl`. Not a git submodule. |

### 8.2 `latest.json` schema

```json
{
  "run_timestamp_utc":    "2026-04-25T00:00:07Z",
  "anchor_timestamp_utc": "2026-04-24T14:30:00Z",
  "model": {
    "profile":          "in12h_out12h_gnn_patchtst",
    "checkpoint_path":  "./checkpoint/model_best.pth",
    "checkpoint_sha256":"d5d87bcbf905...",
    "val_loss_at_train": 0.245454,
    "val_mae_at_train":  0.3781,
    "val_rmse_at_train": 0.4956
  },
  "input": {
    "event_csv": "/.../dataset/events/20260424143000.csv",
    "sources": {
      "noaa_plasma_url": "...",
      "noaa_mag_url":    "...",
      "gfz_hpo_url":     "..."
    },
    "missing_data_filled_fraction": 0.017
  },
  "forecast": [                                // 24 entries = 12 hours
    {"horizon_steps":1, "horizon_minutes":30, "target_timestamp_utc":"...", "hp30":7.2},
    ...
  ],
  "history": [                                 // 96 entries = 48 hours (added by update_site_data.py)
    {"timestamp_utc":"...", "hp30":9.0},
    ...
  ]
}
```

### 8.3 `status.json` schema

```json
{
  "status":            "ok" | "warn" | "error",
  "last_success_utc":  "2026-04-25T00:00:07Z",
  "last_attempt_utc":  "2026-04-25T00:00:07Z",
  "last_error": null | {
    "code":    <int>,
    "message": "..."
  }
}
```

---

## 9. Cost and quota

- GitHub Actions Linux runner minutes are **unlimited and free for
  public repos**. The 10-min cron uses roughly 2,000–4,000 runner minutes
  per month depending on how many attempts fire; cost is $0.
- GitHub Pages bandwidth: 100 GB/month soft limit per user. Our static
  site is a few hundred KB; nowhere near the limit.
- NOAA and GFZ feeds are unauthenticated public JSON/text; no API key
  or quota to worry about.

---

## 10. Known limitations

1. **Scheduler drift** — GitHub Actions cron is best-effort. A run
   scheduled for :18 UTC may start late, and some attempts may be dropped
   entirely under load. The three-attempts-per-anchor design absorbs most
   of this, and the anchor computation always aligns to the most recent
   30-min boundary, but the "last updated" timestamp on the page reflects
   the actual run time, not the slot time.
2. **Matched-pair invariant is process-enforced** — `model_best.pth` and
   `table_stats.pkl` must come from the same training run; there is no
   runtime check that they match. Committing them together in-tree and
   always swapping them as a pair (§5) makes a mismatch a git-review-visible
   mistake rather than a silent runtime one, but the guarantee is still
   procedural, not automatic.
3. **No historical archive on the page** — the page renders `latest.json`
   plus the recent observed history. Longer-term forecasts are accumulated
   in `forecast_history.json`/`.csv` but are not yet surfaced in the UI
   (kept for future re-exposure; also present in git history of
   `site/data/`).

---

## 11. Extending the dashboard

Candidate next steps, in rough order of effort:

1. **Historical accuracy view** — `forecast_history.json`/`.csv` already
   archive each anchor's first-horizon forecast with its status. Surface a
   secondary chart: "forecast-vs-realized MAE over the last N days" by
   joining archived forecasts against the later observed hp30.
2. **hp30 as a second target** — currently only hp30 is on the page. The
   engine also supports hp30 variants. Add a second line to the chart with a
   toggle.
3. **Attention heatmap** — `plot_attention` exists in the engine but emits
   PNG. For interactive use, serialize attention weights to JSON and render
   with a canvas heatmap library.

Each of these would be additive — none require restructuring the
current pipeline.
