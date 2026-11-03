# M1 speed test — instructions for the agent on the Mac Mini M1

Eyal wants to know whether ideas from gilboash/therealspotter's Live mode make **our heat analysis faster on the
Mac Mini M1 (8 GB)**, and how much memory each one costs. On the M5 Pro they were tested already (results in
`results/` from that machine). This test repeats them on the M1, so the two can be compared.

**What not to do:**
- Don't change any of the app's code. The test swaps the ideas in from `speed_test.py` only.
- Don't write anything into the event folder. The test only reads it; its outputs go to `/tmp/m1_speed_out`.
- Commit only this test's results folder (step 7). Never commit `.vscode/` or `keypoint_models/`.

## 1. Get the branch
```bash
cd <the therealspotter repo on this Mac>
git status                 # local changes? tell Eyal before going on — don't discard them
git fetch origin
git checkout m1-speed-test
git pull
```

## 2. Check the Python setup
```bash
venv/bin/python -c "import ultralytics, coremltools, open_clip, cv2; print('ok')"
```
If something's missing, tell Eyal; don't install into the venv without asking.

## 3. Find the event
The event is **"Race 1 Season 2027"**, folder name `00ac5796-375f-47d7-afc0-da2d5c359798`. Eyal copied his
events to this Mac. Find where they are:
```bash
mdfind -name 00ac5796-375f-47d7-afc0-da2d5c359798
# or: find ~ -maxdepth 5 -type d -name '00ac5796-375f-47d7-afc0-da2d5c359798' 2>/dev/null
```
The test heat is the folder starting with **`b9661a9b`**. It must hold `2.mp4` (~780 MB), `2.recordinfo.xml` and
`Race.json`. That heat has 4 pilots, a 93 s race, about 42 gate passes each and 3 LapRF laps per pilot.

## 4. Get the machine ready
- **Quit** the Lazy Spotter app, FPVTrackside and any other heavy app: browser tabs with video, OBS, etc.
- Keep the Mac awake and plugged in. Run the test with `caffeinate`, as in step 5.
- Make the detector's Core ML copy first, if it isn't there yet. This takes about 30 s to a few minutes, once:
  ```bash
  venv/bin/python -c "from race_day.models import *; print(ensure_coreml(DEFAULT_DET_MODEL))"
  ```

## 5. Run the test (about 25 minutes on an M1)
```bash
caffeinate -i bench/m1_speed/run_all.sh "<the event folder from step 3>"
```
- It analyses the one heat 3 ways, one after another, with a line of results after each: `base4`, `split2_4`,
  `split2_6`.
- Run it in the background if your tool times out on long commands, and wait for `done: …`. Don't start anything
  heavy on the Mac while it runs: that would distort the timings.
- **Memory guard:** a run that fills the memory is stopped, and the test stops with it. You'll see
  `STOPPED: … filled the memory`.
  - "Fills the memory" means macOS reports critical memory pressure, or swap grows by more than 1 GB during the run.
  - That's a result, not a failure: don't restart it, just report it.

## 6. Check the results
In `bench/m1_speed/results/<this Mac>-<date>/`:
- `results.jsonl`: one line per run.
  - `exit 3` with `stopped`: the run filled the memory, and was stopped `after_s` seconds in.
  - `wall_s`: the analysis time.
  - `x_realtime`: the analysis time ÷ the video analysed. Under 1 means faster than the race itself.
  - `det_busy_s`: how long YOLO was busy.
  - `peak_rss_mb`: peak memory of the analysis.
  - `peak_swap_mb` and `swap_before_mb`: swap used.
  - `peak_pressure`: the worst memory pressure seen (1 normal, 2 warning, 4 critical).
  - `passes`: per pilot, the number of gate passes and a fingerprint.
- **The passes must match between runs:** pilots 1–4 should have the same `n` and `hash` in every run (in
  `split2_6`, pilots 5 and 6 repeat pilots 1 and 2). Note it if they don't.
- The `<run>.log` files hold each run's errors, if any.
- `system.txt`: the chip, memory, macOS version, and what else was running.

## 7. Send the results
Commit only the results folder and push the branch:
```bash
git add bench/m1_speed/results/
git commit -m "M1 speed test results"
git push origin m1-speed-test
```
Then give Eyal a short summary in the chat:
- each run's `wall_s`, `x_realtime`, `peak_rss_mb`, `peak_swap_mb` and `peak_pressure`, or that it was stopped;
- whether the passes matched;
- anything unusual: swapping, errors, a run that was much slower.

Eyal reviews the results with Claude on the other Mac.

## What the variants are (for the summary)
| Run | What it changes |
|---|---|
| `base4` | The app as it is: one process, 4 pilot threads, YOLO on the Neural Engine. |
| `split2_4` | The pilots split over 2 processes (2 each), each with its own YOLO and CLIP. |
| `split2_6` | 6 pilots (the heat's 4 pictures plus 2 of them again), split over 2 processes (3 each). |

On the M5 Pro (24 GB), splitting the pilots over 2 processes was the only idea that was clearly faster. It used
about twice the memory (~3.2 GB instead of ~1.6 GB). The M5 Pro's results for this same heat are in `results/`.
The question for the M1 is whether that still holds with 8 GB, or whether it fills the memory.
