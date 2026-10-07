# Race-day mode: capture, learn the track, show results

## Context

At a race, a laptop (Apple M5, 24 GB) with an HDMI capture card receives the race director's output: **4 pilot feeds in a 2×2 grid, 1080p, no borders, no overlays** (so each quadrant is 960×540, the same as track1). Most of the time the feeds are gray, because no drone is connected.

The goal is a system that runs a race day with as little input as possible: you set up the track once, and from then on it records every pilot's run, learns the track from the first runs, and shows each run's gates, laps and statistics in a browser that anyone on the Wi-Fi can open. The same system replays recorded 2×2 videos, for testing and for analysing events afterwards.

This is a living document: add ideas, answer the open questions and change anything here as the work goes on.

## What I think

It's doable with what exists:
- `extract_race.py` finds passes;
- `gate_decoder.py` labels gates and laps;
- `learn_track.py` learns a track from unreviewed videos: 98.2% gate ID on track1, and 97% on a track it had never seen (USGQ2025).

The new parts are capture, the race-day app, orchestration and the dashboard. The gate count is now typed in at setup, which removes the biggest unknown. The remaining risks:

1. **Pass detection on unseen cameras and gates.** On USGQ2025 (HDZero, fisheye), pass recall was 89%, with most misses on two gate types. A race mixes analog, HDZero and DJI feeds. The decoder tolerates missed gates. Analog static and breakup are new, and must not be taken for a live feed.
2. **Throughput — measured, about 2× headroom.** YOLO is the only heavy step. Four live feeds at 30 fps need 120 detections/s. On the race laptop (M5 Pro, 24 GB), running the analysis in parallel with a Core ML copy of the detector gives up to 276/s (see *Parallel processing*). What remains is checking that the full pipeline gives the same results with Core ML.
3. **The first runs must be good enough to learn from.** The track is learned from the first 24 runs of Live mode. Crashes or very short runs are filtered by the 40 s minimum, and the learning tolerates missed passes.

## Two modes (chosen when the app starts)

`race_day.py` is a desktop app (PyQt6, like learn_ui), run by the operator. It owns all settings and state. The browser dashboard only displays.

**1. Live track** — at the race
1. Setup:
   - track name (a new or existing folder in `FPVdatasets/`);
   - **number of gates**;
   - channel layout of the 2×2 (see below);
   - capture device, with a preview.
2. Records every run, analyses it, and **learns the track from the first 24 runs** (the learn state).
3. Then switches to the result state: every run, including the first 24, is labelled and shown in the dashboard. More runs keep improving the learned track: the app re-learns every K runs and relabels.

**2. Replays** — testing and after the event
1. Choose an existing track folder.
2. Add video files: 2×2 recordings in the same format as the live feed.
3. Each file is split into the 4 channels, cut into runs (≥ 40 s), analysed **with the track's existing learned data** and shown in the dashboard.
4. **No learning, and no recording** in this mode. The track must already be learned, from Live mode or with `learn_track.py`.

Whichever track is open is the only one the dashboard shows. Opening another track switches the dashboard to that track's data.

## Architecture (new `race_day/` package, reusing the pipeline)

```
race_day.py (desktop app: mode, setup, live view, progress)
   │
   ├─ Live:    HDMI capture ─┐
   └─ Replays: 2×2 video file ┴─► splitter + run detector ──► <track>/videos/<run>.mp4 + <run>.meta.json
                                        │
                                        ▼
                                  job queue ──► extract_race.run_race_extraction (passes, embeddings, motion)
                                        │
                Live, learn state:      │ first 24 runs → learn_track (gate count from setup) → gate memory + timing
                                        ▼
                result state / Replays: gate_decoder.label_race on every run → stats
                                        ▼
                         dashboard server (read-only web page, open track only; phones on the same Wi-Fi)
```

- **Track folder** (`dataset_paths.py`), `FPVdatasets/<track>/`:
  - `videos/`, `runs/`, `gate_memory.json`, `gate_timing.learned.json`;
  - `race_day.json` with the setup (gate count, channel layout) and the state (learn/result, runs, heats, pilots).
  - Every file made from a run starts with the run's name, e.g. `h012_R3.mp4`, `h012_R3.meta.json`, `h012_R3.race_data.json`.
- **Channel layout:** defaults to
  ```
  R1  R3
  R6  R8
  ```
  It can be changed in setup: each quadrant gets a channel from a dropdown, and it's saved with the track. The run's name and the heat info use the channel. *(To confirm: whether "from top right" means R1 is the top-right quadrant; the setup screen makes it easy to set either way.)*
- **Splitter and run detector** (`race_day/capture.py`), the same code for live capture and replay files:
  - splits each frame into 4 quadrants (no border detection needed);
  - per quadrant, a feed is **live** while its image has texture and changes between frames. Flat gray isn't live, and neither is static noise (noise everywhere, no coherent motion);
  - **pre-roll:** a buffer of a few seconds, so take-off isn't lost;
  - **dropouts:** a feed must stay gray for **about 10 s** before the run counts as ended, so analog breakups or a reboot don't split one run in two. **While other pilots of the heat are still flying, it waits up to 45 s**, because a crash with a video blackout mid-heat must not cut the run. Q25 Maman's video goes blank for 15 s during a crash;
  - **saving:** a run of **≥ 40 s** is kept — **the video file and its data** (`<run>.mp4` + `<run>.meta.json` with channel, start/end time, heat, pilot). Shorter runs are discarded.
- **Heats:** runs that are live at the same time form one heat. The heat number and pilot names come from the website's heat info.
- **Restarted heats (attempts).** A heat (stage + round + race) can be flown more than once. Tested on a session with Qualifying Race 3 run twice:
  - From fril's `phase`, each time a heat becomes `running` an attempt starts. It ends **completed** at `finished`, or **abandoned** if another heat replaces it first or it's reset.
  - Per heat, **the latest completed attempt is official**, and earlier attempts are **void** — the same rule as fril, which keeps only the last result.
  - Runs are matched to attempts by time on our own clock. **Void runs keep their video and data on disk, are still used for learning, and are hidden completely from the dashboard** (lists, pages, comparisons and video).
  - A video that covers both attempts, because the drone stayed powered, is kept whole. Only its official part counts in the stats.
  - Labels read "Stage · Round · Race", because each stage restarts at Round 1 · Race 1.
- **Pilot and heat info** (`race_day/pilots.py`):
  - `FrilLiveProvider` polls `https://fril.co.il/api/live/state`, the public API the fril live page reads, every 2 s. It reads `currentHeat.round`, `currentHeat.race`, and each pilot's `pilotName` and `channel` {band, number}, which maps onto the 2×2 layout. The layout's channel names must match the channels set in the timing system.
  - Only that API is used. It needs a named User-Agent; Cloudflare refuses Python's default.
  - The provider is used in Live mode, and can be switched off in setup. **Without internet or data**, names are a sequence: heat number plus channel, e.g. `H012-R1`.
- **Controller** (`race_day/controller.py`):
  - **Parallel workers:** one analysis process per channel (see *Parallel processing*). A run is analysed, then labelled and its stats computed. A heat's 4 runs are analysed at the same time.
  - **Live:** the state is `learn` until 24 runs are analysed. Then it learns the track with the gate count from setup, installs the result, relabels all runs and switches to `result`. Re-learning happens every K new runs.
  - **Replays:** label with the existing learned track; no learning.
  - **Resumable:** everything restarts from `race_day.json` after a crash or restart, so nothing is lost.
- **Stats** (`race_day/stats.py`), per run and per pilot:
  - laps and lap times, best and average lap, **consistency** (standard deviation / coefficient of variation of lap times);
  - **section times** (each gate-to-gate leg), best section per pilot, **theoretical best lap** (sum of best sections);
  - **speed vs the field** (per-section time against the field median, plus an overall speed index), rank per section;
  - **holeshot** (time to the first gate, G1), missed or skipped gates, crash or run-ended-early flags;
  - progression across a pilot's runs.

## Parallel processing (run time)

The 4 runs of a heat are analysed **at the same time**, not one after another.

**Measured on the race laptop (Apple M5 Pro, 24 GB), YOLO detections per second:**

| Setup | Detections/s | 4 live feeds at 30 fps need 120/s |
|---|---|---|
| 1 process, GPU (today's pipeline) | 51 | no, about 0.4× real time |
| 4 processes, all GPU | 105 | almost |
| 1 process, Core ML (Neural Engine) | 104 | — |
| 4 processes, all Core ML | 210 | yes, about 1.75× |
| **2 Core ML + 2 GPU processes** | **276** | **yes, about 2.3×** |

The Core ML model is the same detector converted once for Apple's Neural Engine. It found the same boxes as the `.pt` model on 60 of 60 test frames. Mixing Neural Engine and GPU processes is fastest, because two different chips work at once. Everything else per frame (decoding at about 1,770 frames/s, optical flow, tracker, pass logic, CLIP only at pass candidates) is light and spreads over the CPU cores. Labelling gates and laps, and learning the track, take seconds.

**Design:**
- **Core ML: checked and turned off.** The full pipeline was run on 2 reviewed videos (Q25 Verso, Q3 Aviv) and scored with `score.py`:

  | Detector | Errors (missed + false + off-time) | Gate ID | Time, Q25 Verso (88 s) |
  |---|---|---|---|
  | `.pt` on CPU (Ultralytics' default until now) | 7 | 91/92 | 221 s |
  | **`.pt` on GPU (mps)** | **7 — identical to CPU** | **91/92** | **54 s** |
  | Core ML fp16 + NMS, 640×640 | 24 | 74/85 | 38 s |
  | Core ML fp32, 640×640 | 24 | 74/85 | 212 s |
  | Core ML fp16, 384×640 | 22 | 82/87 | 27 s |

  Core ML is faster but less accurate; the reason (likely fp16 confidences near thresholds) is still open. **Race day uses the GPU**, with Core ML available as a switch (`DayConfig.use_coreml`, off by default).
- **One worker process per channel,** 4 in total, all on the GPU, each with its own copy of the models. About 3–4 GB of memory in total. Measured: a heat's runs (64–143 s each) are analysed in 90–200 s, so results are ready about 2–3 minutes after the heat ends.
- **Live, version 1:** when a heat's runs are saved, the 4 are analysed in parallel. Results are ready about 1 minute after the heat ends, before the next heat. Learning (after 24 runs) and re-learning run between heats.
- **Live, version 2 (later, if wanted):** each feed is analysed *while it's flown*. Frames go to the channel's worker as they arrive, so results appear a few seconds after landing. This needs `extract_race` turned into a frame-by-frame analyser. It's only worth it once version 1 works.
- **Replays:** each 2×2 file is decoded once and split, and its 4 channels are analysed in parallel. That's about 2× faster than real time.
- **The desktop app** shows each worker's progress, and warns if analysis falls behind the capture.

## Desktop app — `race_day_app.py` (operator)

(Named `race_day_app.py`, because a `race_day.py` next to the `race_day/` package confuses Python imports.)

- **Start screen:** two big buttons, **Live track** and **Replays**.
- **Live setup:**
  - track name (new or existing folder) and number of gates;
  - channel layout grid (default R1 R3 / R6 R8);
  - capture device with a live preview;
  - advanced options, usually untouched: minimum run length 40 s, end-of-run gray time 10 s, runs to learn from 24.
  - **Start** begins recording.
- **Live screen:**
  - the live 2×2 feed, each quadrant labelled with its channel, pilot and status: `waiting – no signal`, `LIVE 0:37`, `saved ✓`, `too short ✗`;
  - **runs captured** (total and per channel);
  - the learn-state progress `learning the track: 9 / 24 runs`, then `results`;
  - the processing queue (waiting / analysing / done) and the current heat from the website;
  - the dashboard address and a **QR code** for phones;
  - **Stop.**
- **Replays screen:**
  - choose a track folder (it must already be learned) and add video files, by dragging them in or with a file picker;
  - per file: splitting into runs, then analysing, then done (runs found);
  - the dashboard address and QR code.

## Browser dashboard — what everyone sees (read-only)

A local web server started by `race_day.py`. Anyone on the same Wi-Fi opens it on a phone through the QR code, or on a laptop or TV. It **only shows data**: nobody can change settings, the mode or the state from it. It shows **only the track currently open** in the desktop app, and refreshes itself every few seconds. The look: dark theme, large readable numbers, one colour per pilot, and green / yellow / red for faster / equal / slower than the field.

1. **Header** (every page):
   - track name and mode (Live / Replays);
   - in Live, the state (`learning the track 9/24` or `results`), the current heat and the 4 pilots with live/waiting status.
2. **Learn-state page** (Live, before results):
   - progress and the runs collected so far;
   - after learning: the gate count, a **gate gallery** with a few learned images per gate, and a **lap bar** — one lap drawn as sections proportional to the learned section times. The lap bar is the "map" reused on every result page.
3. **Results home:**
   - a **latest-runs feed:** one card per run, with pilot, heat, laps, best lap, consistency badge and flags;
   - a **leaderboard:** best lap, best 3 consecutive laps, theoretical best, consistency, speed index and runs flown, sortable and filterable by round or heat;
   - **fastest-section badges** per section.
4. **Run page** — the main analysis screen. **Any phone can open and play any run's replay**; the video streams from the laptop.
   - **Video:** with a timeline marking every gate pass and lap; tapping a mark jumps there.
   - **Lap table:** time, difference from the pilot's best and from the field median, best lap highlighted.
   - **Section grid:** laps × sections, each cell coloured by how it compares with the field.
   - **Sections against the field:** the pilot's best and average against the field median and the fastest pilot. Shows where time is gained or lost.
   - **Consistency:** lap-time chart and spread.
   - **Run facts:** holeshot, total time, missed gates (linked to the moment in the video), crash or run-ended-early.
5. **Heat page:**
   - the 4 recordings of a heat side by side, played in sync;
   - finish order and the gap between pilots at every gate;
   - a gap chart showing where each pilot gained or lost.
6. **Pilot page:**
   - all of the pilot's runs and a best-lap chart over the day;
   - strengths and weaknesses per section compared with the field;
   - consistency trend.
7. **Track page:** the learned gates with images and typical section times, and **the hardest sections** (the largest spread between pilots).

## Milestones

| # | Milestone | Done when |
|---|---|---|
| 0 | This plan in `docs/RACE_DAY_PLAN.md`, committed | in the repo, open for edits |
| 1 | ✅ `race_day_app.py`: start screen, Live / Replays setup, track folder and `race_day.json`, channel layout | both modes can be opened and set up, and settings are saved per track |
| 1b | Detector on the GPU + one worker process per channel | ✅ GPU gives results identical to the CPU, 3–4× faster; Core ML failed the score check and is off by default |
| 2 | ✅ Splitter and run detector, plus a 2×2 test-video maker (from existing videos, with gray gaps and static) | from a test 2×2 file, exactly the runs ≥ 40 s are saved (video + data) under the right channels, with no gray or static |
| 3 | ✅ Controller: queue, Live learn → result after 24 runs, relabel, re-learn; Replays with the existing learned track | a simulated Live day runs unattended and survives a restart; Replays analyses added files |
| 4 | ✅ Stats | numbers match the reviewed videos (lap times = reviewed G1 times) |
| 5 | ✅ Browser dashboard (read-only, open track only, phone replay) | a simulated event shows live on a laptop and a phone, and runs play on the phone |
| 6 | ✅ Heat and pilot info from the website: `GET https://fril.co.il/api/live/state` only, polled every 2 s — round, race, and each pilot's name and channel (Raceband 1 → `R1`). Asked when a heat starts and again when each run ends, since the timing PC may load the heat late. Falls back to `H012-R3` names without internet or data. | names and heats per run, with and without internet |
| 7 | Dry run on the M5 with the real capture card | a full simulated heat analysed within about 1 minute of ending, race-day checklist written |

## How to run (implemented)

```bash
python race_day_app.py                       # the desktop app: Live track or Replays
python -m race_day.simulate out.mp4 --heat "a.mp4,b.mp4,c.mp4,d.mp4" --heat "…"   # a 2×2 test video
```

- **Live:**
  - type a track name and the number of gates, check the channel layout, pick the capture card, then **Start**;
  - for a test without the card, pick *Test: play a 2×2 video file as if live*.
- **Replays:**
  - pick a learned track, add 2×2 files, then **Analyse**.
- **Dashboard:** the address and QR code are on the running screen. Open it on any phone on the same Wi-Fi.
- **Code:** `race_day/` holds:
  - `capture.py` — splitter and run detector;
  - `analysis.py` — GPU workers;
  - `labeling.py` — gates, laps, learning;
  - `stats.py`;
  - `controller.py`;
  - `server.py` and `static/` — the dashboard;
  - `pilots.py` — heat and pilot names;
  - `models.py`;
  - `simulate.py`.

## Decided
- **Capture:** 1080p, no borders, no overlays.
- **Laptop:** M5, 24 GB.
- **Gate count:** entered in the desktop app's setup when opening a track for Live (or chosen with the track in Replays).
- **Start gate:** the first gate passed is always G1.
- **Internet:** available; without it, pilots are named by heat and channel.
- **Viewing:** phones on the same Wi-Fi, through the dashboard. The dashboard is read-only and shows only the open track.
- **Run end:** about 10 s of gray, not 3 s.
- **Saving:** both video and data.
- **Learning:** only in Live mode.
- **Laptop speed:** the M5 Pro analyses 4 feeds at about 2× real time with Core ML + GPU workers (measured).

## Open questions
- **Channel layout:** confirm the default reading order (is R1 top-left or top-right?).
- **Re-learning:** how often during Live (every K runs; K = 12?).
- **Dashboard language:** Hebrew (right to left), English, or both?
- **TV mode:** a TV or projector at the venue for a kiosk mode that cycles the leaderboard and the latest run?
- **Replays on an unlearned track:** if a track has never been learned, should Replays offer to learn from the added files once?
