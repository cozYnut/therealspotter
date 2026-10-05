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
2. **Throughput.** Analysis takes about 1× real time per video on the current Mac. Four runs per heat means about 4× the heat's length of compute, so a queue can build up. The M5 should be faster; the dry run will measure it. If needed, the analysis can run at lower resolution or skip frames.
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
  - **dropouts:** a feed must stay gray for **about 10 s** before the run counts as ended, so analog breakups or a reboot don't split one run in two;
  - **saving:** a run of **≥ 40 s** is kept — **the video file and its data** (`<run>.mp4` + `<run>.meta.json` with channel, start/end time, heat, pilot). Shorter runs are discarded.
- **Heats:** runs that are live at the same time form one heat. The heat number and pilot names come from the website's heat info.
- **Pilot and heat info** (`race_day/pilots.py`): an interface `heat_info(time) → {heat, round, {channel: pilot}}`. The fril provider is implemented later (there will be internet). **Without internet or data**, names are a sequence: heat number plus channel, e.g. `H12-R1`, `H12-R3`.
- **Controller** (`race_day/controller.py`):
  - **One queue worker:** analyse a run, then label it and compute its stats, in order.
  - **Live:** the state is `learn` until 24 runs are analysed. Then it learns the track with the gate count from setup, installs the result, relabels all runs and switches to `result`. Re-learning happens every K new runs.
  - **Replays:** label with the existing learned track; no learning.
  - **Resumable:** everything restarts from `race_day.json` after a crash or restart, so nothing is lost.
- **Stats** (`race_day/stats.py`), per run and per pilot:
  - laps and lap times, best and average lap, **consistency** (standard deviation / coefficient of variation of lap times);
  - **section times** (each gate-to-gate leg), best section per pilot, **theoretical best lap** (sum of best sections);
  - **speed vs the field** (per-section time against the field median, plus an overall speed index), rank per section;
  - **holeshot** (time to the first gate, G1), missed or skipped gates, crash or run-ended-early flags;
  - progression across a pilot's runs.

## Desktop app — `race_day.py` (operator)

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
| 1 | `race_day.py` skeleton: start screen, Live / Replays setup, track folder and `race_day.json`, channel layout | both modes can be opened and set up, and settings are saved per track |
| 2 | Splitter and run detector, plus a 2×2 test-video maker (from existing videos, with gray gaps and static) | from a test 2×2 file, exactly the runs ≥ 40 s are saved (video + data) under the right channels, with no gray or static |
| 3 | Controller: queue, Live learn → result after 24 runs, relabel, re-learn; Replays with the existing learned track | a simulated Live day runs unattended and survives a restart; Replays analyses added files |
| 4 | Stats | numbers match the reviewed videos (lap times = reviewed G1 times) |
| 5 | Browser dashboard (read-only, open track only, phone replay) | a simulated event shows live on a laptop and a phone, and runs play on the phone |
| 6 | Heat and pilot info from the website, with sequential naming as the fallback | names and heats per run, with and without internet |
| 7 | Dry run on the M5 with the real capture card | real-time throughput measured, race-day checklist written |

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

## Open questions
- **Channel layout:** confirm the default reading order (is R1 top-left or top-right?).
- **Re-learning:** how often during Live (every K runs; K = 12?).
- **Dashboard language:** Hebrew (right to left), English, or both?
- **TV mode:** a TV or projector at the venue for a kiosk mode that cycles the leaderboard and the latest run?
- **Replays on an unlearned track:** if a track has never been learned, should Replays offer to learn from the added files once?
