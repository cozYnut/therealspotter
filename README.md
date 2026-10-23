# therealspotter — Lazy Spotter v2 🏁

**The first version of the realspotter app.** One desktop app, `lazy_spotter_v2.py`,
for FPV drone races: it watches the pilots' video feeds, learns the track's gates,
finds every gate pass and lap, checks FPVTrackside's lap times against the video,
and shows each run's results in a browser that anyone on the Wi-Fi can open.

Everything runs locally on a laptop (no cloud), in Python. Demo of an earlier
version: https://www.youtube.com/watch?v=6A2gmpzc6iU

---

## Install

```bash
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision          # first: the right build for your machine (Apple MPS is included on macOS)
pip install -r requirements.txt
brew install ffmpeg                    # recommended: makes recordings stream in phone browsers
```

The first run downloads the CLIP model (once, ~350 MB) into the Hugging Face
cache; after that the app works without internet.

## Run

```bash
source venv/bin/activate
python lazy_spotter_v2.py
```

**The first time**, the app asks for the **data folder**: the folder that holds
your tracks and FPVTrackside events (for example `FPVdatasets`). It's remembered
(`~/.lazy_spotter_v2.json`) and can be changed on the start screen.

---

## The start screen

- **Data folder: …  [Change…]** — the folder every event is picked from.
- **Browser address** — the dashboard (see below), already running.
- Three tabs, one per way of working:

| Tab | For | You pick |
|---|---|---|
| ⏱ **FPVTrackside event** | checking a race that FPVTrackside timed | one of the FPVTrackside events in the data folder |
| ● **Live capture** | recording heats from the capture card yourself | a track of the data folder, or a new one |
| ▶ **Replays** | looking at races afterwards | any track or FPVTrackside event of the data folder |

Events are always chosen from a list of what's in the data folder (newest first,
with a filter), never browsed for.

### ⏱ FPVTrackside event

For races timed by FPVTrackside (LapRF). The app reads the event folder in place
— it never copies or moves the videos — and checks every pilot's laps against
the video.

- **Heat list (left):** every finished heat, newest first, with its pilots and
  their LapRF lap counts, and a status: `•` new · `⏳` analysing · `◦` analysed ·
  `✓` checked · `↻` laps updated · `⚠ n` lap findings to look at.
  Tick ☑ a heat (or single pilots in the heat view) to learn the track from it.
- **Learn track / Relearn track (top):** learns the gates from the ticked heats
  (asks the number of gates). LapRF's start/finish times help, so one full lap
  per run is enough.
- **Auto (top):** new heats are analysed and checked by themselves, and confident
  lap corrections are written to FPVTrackside; unsure ones wait for you.
- **The heat (right):** the 2×2 video with each pilot's channel, name and lap;
  a timeline per pilot (white = LapRF laps, green = laps seen in the video,
  orange = things to check, blue = your marks); and per pilot a table of laps
  with what the video says about each.
- **Buttons:** **Analyse** · **Review** (see below) · **✓ Accept** / **✗ Reject** a
  finding · **+ Lap / HS here** (add a lap or the holeshot at the video's time) ·
  **− Remove lap** · **Update laps in FPVTrackside** (writes the changes like a
  marshal's correction) · **Restore original laps**.
- While FPVTrackside runs a race, the app pauses completely so FPVTrackside has
  the computer; it continues when the race ends.

### ● Live capture

For recording heats from the race director's HDMI output yourself.

1. **Setup:** **Choose track…** (a track of the data folder, or **New track…**),
   or one of the **Recent** three; **Feed**: 2×2 (four pilots) or single pilot;
   the **channel** of each quadrant; **Number of gates**; the **Capture device**
   (**Refresh**, **Preview device** to check it — the card should give 1920×1080);
   **Longest recording** (default 3:00). **Start**.
   *Test without a card:* choose "Test: play a video file as if live".
2. **The screen:** heat list on the left, the **live feed** top right, the chosen
   heat below it.
3. **● Record** saves the whole feed as `<track>/videos/heat_NNN.mp4` until
   **■ Stop** or the time limit. While it records, analysis and learning wait,
   so the recording is never choppy.
4. Select heats (⌘/Shift-click for several) → **Analyse selected heats**, or
   **Analyse this heat**. Nothing is analysed by itself.
5. Tick ☑ a few heats with full laps → **Learn track**. Every track is learned in
   the app at least once; after that analysed heats get gate numbers and laps.
6. Videos already in `videos/` that the app didn't record are asked about once
   (all 2×2 / all single / one by one, with optional pilot names).

Pilots are known by their channel unless a name was given.

### ▶ Replays

The Live capture screen without the setup, the recording and the live feed:
pick a track, and its videos (`<track>/videos/`) are the heat list. **Analyse**,
**Learn track** and **Review** work the same way. Picking an FPVTrackside event
opens it in the FPVTrackside screen, without Auto.

### Review — checking one pilot's run

In any tab: select a pilot in the heat view and press **Review**. The video shows
only that pilot's quadrant, enlarged; the timeline shows the passes the system
found; a list shows each with its time, lap and gate, next to the system's crop
and the learned picture of that gate. Go through them:

| Key | Action |
|---|---|
| `N` / `Shift+N` | next / previous pass to check |
| `0`–`9` | gate number (`1` then `2` quickly → G12) |
| `Enter` | accept, at the video's time |
| `←` / `→` | one frame (fix the exact pass frame first); `Shift` = one second; `Space` = play |
| `X` | false pass (no real pass here) |
| `A` | add a pass the system missed |
| `M` | a gate the pilot skipped |
| `T` | tag: crash/clipped → bad video → reverse → unsure → none |
| `Backspace` | undo |

Every change is saved (`lazySpotter/gt/<run>.gt.json`); **Done reviewing** goes
back. Reviews make gate numbers better on that track and are what the system is
measured and retrained on. A clipped gate counts as a pass if the drone went
through it; a crash in the gate only if the drone flies on.

### The browser dashboard

Starts with the app; the address is on the start screen and in every tab's top
bar. Open it on any phone or laptop on the same Wi-Fi. It shows **every event of
the data folder** (the one open in the app live): results and leaderboard, each
run (its video with gate and lap marks, laps, sections against the field), heats
(the pilots' videos side by side, gap chart), pilots, and the track's gates.
Read-only: nothing can be changed from the browser.

---

## Where the data is

```
<data folder>/
  <track>/                       a track (Live capture, Replays)
    videos/                      its heat videos (recordings go here)
    lazySpotter/                 everything the app keeps:
      race_day.json                setup and heat states
      videos.json                  what each video is (single / 2×2, channels, pilots)
      gate_memory.json             the learned gates (+ gate_timing.learned.json)
      runs/                        each pilot run's analysis
      gt/                          reviews
  events/<FPVTrackside event>/   an FPVTrackside event folder (heats read in place)
    lazySpotter/                 the same, plus backups/ of every Race.json the app changed
  training/                      models and datasets (FPV_YOLO_tool)
```

Learning data is only ever taken from `lazySpotter/`.

---

## How it works

1. **Gate detector** — YOLO (`cyn_current-20261021_best.pt`) finds the gates in every frame.
2. **Tracking** (`tracking.py`) — each gate is followed from frame to frame; CLIP
   gives each gate crop a picture fingerprint.
3. **Pass detection in two stages** (`pass_scorer.py`, `pass_scorer.json`) — every
   moment a big gate disappears is a candidate; a trained scorer keeps almost all
   of them (low threshold), and the gate-sequence decoder drops the ones that don't
   fit the track.
4. **Gate numbers and laps** (`gate_decoder.py`) — the most likely sequence of
   gates from their order, the time between gates, and how each gate looks;
   each pass's time is corrected by its gate's usual offset.
5. **Learning a track** (`learn_track.py`) — gates and gate-to-gate timing are
   learned from a few runs, given only the number of gates.

On 52 reviewed runs of three tracks: 96.2% of passes found on time, 55 false
passes, gate number right 98.7% (against 93.2%, 162 and 97.1% before the
two-stage pass detection).

## Project structure

```
therealspotter/
├── lazy_spotter_v2.py       # the app
├── race_day/                # its screens, controller, capture, review, analysis workers, dashboard (static/)
├── extract_race.py          # the analysis of one video (run by the app's workers)
├── tracking.py              # gate tracker and CLIP embedder
├── pass_detector.py         # pass candidates
├── pass_scorer.py           # two-stage pass detection   (+ pass_scorer.json, the trained scorer)
├── gate_decoder.py          # gate numbers and laps
├── learn_track.py           # learning a track
├── extract_motion.py        # camera motion (for the pass scorer)
├── gate_db.py               # gate memory
├── pipeline_cfg.py          # pipeline settings (debug_ui_defaults.json)
├── score.py                 # measures the system against the reviews
├── dataset_paths.py         # where the data is
└── cyn_current-20261021_best.pt   # the gate detector
```

**Tools without the app**

- `python score.py` — score the system against every reviewed run.
- `python pass_scorer.py --save` — retrain the pass scorer on the reviewed runs.
- `python learn_track.py <track> --gates N` — learn a track from the command line.
- Labelling gates and training detectors: the **FPV_YOLO_tool** repo
  (`gate_annotator.py` and the dataset tools). Older detectors are in
  `<data folder>/training/models/`.

More detail and history: `docs/RACE_DAY_PLAN.md` (the app) and `docs/PLAN.md`
(the pass detection and gate numbering work).
