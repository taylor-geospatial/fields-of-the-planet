# Planet Data API issues — FTW global pipeline

Observed across a single end-to-end run pairing every Fields of the World v2 patch (~70.5k patches × 2 windows = ~141k extracts, ~6.1k unique PSScene IDs) with PlanetScope `ortho_analytic_4b_sr` + `ortho_udm2`. Run started 2026-05-09 ~20:35; activation phase still in progress at time of writing.

The pipeline is straightforward: **search → activate → COG range-read** via the official `planet` Python SDK v2 + `rasterio` `/vsicurl/`. Issues below are surfaced as we hit them; numbers in parentheses are observed magnitudes from this run.

## Issues

### 1. Cold-storage thaw is extreme and unpredictable

PSScene `ortho_analytic_4b_sr` activation ranges from **2 s** (warm) to **>45 min** (deeply cold) for the same account. Mean ~200 s, p95 ~580 s, p99 measured at 2,763 s. Long-tail blocks concurrency slots — even at 96-concurrent activations our effective throughput tops out at ~10/min, ~40% of theoretical peak.

**Ask:** bulk pre-warm by item_id list. We can give Planet the 6.1k IDs ahead of time and sidestep this entirely.

### 2. UDM2 activation is consistently 2–3× slower than SR

For the same item_id activated in the same call: SR median 113 s vs UDM2 median 315 s; UDM2 mean 447 s vs SR mean 154 s. Both sit on the same archive, so this is internal to Planet's pipeline. We worked around it by running the two activations in parallel, but it still doubles the wall-time on long-tail scenes.

### 3. ~23% of activations return broken URLs

This is the biggest surprise. Planet's `:activate` returns `status: active` and a signed Google Cloud Storage URL, but the URL itself is malformed in ways that cause:

```
HTTP response code: 400 - Failure writing output to destination,
passed 107 returned 0
```

…on the first byte-range read via GDAL `/vsicurl/`. **31,890 of 140,325 attempted patch extracts** (23%) hit this on activated scenes. Per-scene: ≈1,386 of 6,113 unique scenes return non-functional URLs. Workaround is to re-call `:activate` for those item_ids — Planet often returns a different (working) URL on the second try, suggesting a stale-cache or signing race in their activation worker.

**Ask:** what causes this? Is there a way to detect it server-side and not return the bad URL? Or a retry mechanism that doesn't require us to track and re-activate by item_id list?

### 4. Aggressive default rate limits, no SDK auto-retry

Search calls at 8 shards × 64-way concurrency (512 inflight) hit `TooManyRequests: max rate reached: retry-in 200ms`. The SDK doesn't auto-retry on 429; we had to wrap every Data API call in our own exponential-backoff retry loop. We've throttled down to 8 × 16 = 128 inflight which is below the limit but feels low for an account on a paid plan.

**Ask:** documented per-account rate limits + an SDK-side retry default would save every consumer from re-implementing this.

### 5. Transient 5xx + connection errors during long runs

Across a ~2-hour activation phase we observed:

- One PSScene endpoint 503 (`Service error -27`), killed a search shard mid-run
- Two `httpx.ConnectTimeout` to `api.planet.com` mid-call — killed another shard
    We absorb these into the same retry wrapper now, but the SDK does not.

### 6. Orders API is the wrong tool for our access pattern

We initially used the Orders API with clip + reproject tools (the recommended path per Planet docs). For 1.5 km × 1.5 km clips out of ~24 km × 8 km strips, **per-clip wall time is 1–5 minutes** dominated by the queue + worker spin-up + full-strip load. This is ~100× slower than direct COG range reads.

Switching to `:activate` + `/vsicurl/` HTTP range reads brought per-patch network time from minutes down to **~0.5 s** — the COG headers + the few tiles overlapping the patch transfer are ~1–2 MB per patch out of a ~500 MB strip.

**Ask:** flag this in docs. The Orders API is great for fully-realized derived products (mosaics, NDVIs) but for "give me a window of pixels" the COG range-read path is the only viable option at scale.

### 7. Documentation gap on the COG range-read path

The fact that PSScene `ortho_analytic_4b_sr` is a Cloud-Optimized GeoTIFF and that the activated `location` URL supports HTTP range reads is not documented anywhere we could find. We had to discover it by activating an asset, copying the URL, and trying `gdalinfo`. The whole pipeline above relies on this; without it, Orders API scaling is the only option Planet documents and it doesn't work for our scale.

## Pipeline scale (for context)

|                                                                        |                                        |
| ---------------------------------------------------------------------- | -------------------------------------: |
| FTW patches                                                            |                                 70,484 |
| (patch, window) extract targets                                        |                                140,968 |
| Unique PSScene candidates after search                                 |                                  6,113 |
| Mean patches per scene                                                 |                                   23.0 |
| Total bytes transferred over network                                   | ~150 GB (range reads, not full strips) |
| Cluster                                                                |     1× cpu_amd node (64 cores, 128 GB) |
| Concurrency: search 16 / activate 96 / extract 16 per shard, 64 shards |                                        |

## Issue summary table

| #   | Issue                                           | Impact                                  |
| --- | ----------------------------------------------- | --------------------------------------- |
| 1   | Cold-storage activation latency, long tail      | ~10× wall-time blowup on extract phase  |
| 2   | UDM2 2–3× slower than SR                        | Forces parallel activation in client    |
| 3   | **~23% of activations yield HTTP 400 URLs**     | Largest single failure source           |
| 4   | Aggressive 429 rate limits, no SDK retry        | Forces every consumer to roll their own |
| 5   | Transient 5xx/network errors not retried by SDK | Same as above                           |
| 6   | Orders API ~100× slower than COG range reads    | Major architectural footgun             |
| 7   | COG range-read access pattern undocumented      | Required reverse-engineering            |

## Numbers to bring to Planet

- 6,113 unique PSScene IDs hit, mostly 2016–2021 acquisitions
- 23% activation failure rate (returns active + bad URL)
- p99 activation latency ≥ 45 min observed
- Account: `<your account/org name>`
- Time window of run: 2026-05-09 20:35 CDT onward
- Item IDs of broken activations available in `data/planet/_global/extract/shard_*.jsonl` (status=open_failed)
