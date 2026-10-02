# Gate recognition by sequence and position — plan

> **This branch:** Milestone 1, Branch 2: automatic pass labels from Velocidrone's race websocket, saved in the same test-dataset format.

Living version: https://claude.ai/code/artifact/3f94089a-4c4b-4f06-9882-2c39bdc995e0 (snapshot of Oct 2, 2026)

## Goal and constraints

Work out which gate each detected pass belongs to, and flag any gate a pilot skipped, by combining how a gate looks with where it sits in the lap. No 3D map. This builds on [therealspotter](https://github.com/cozYnut/therealspotter) and later feeds the missed-gate flag in FPVTrackside.

- **Post-race review first.** Decoding runs offline over the whole video, so it can use context before and after each pass. Live use comes later.
- **Video only.** No blackbox or gyro data; motion has to come from the image.
- **Several lenses, a range of fields of view.** Cues must not depend on knowing the focal length, or must fit one scale per video.
- **Gate sizes and shapes are known**, so two gates of the same type can be compared by apparent size.

## Where it stands today

Pass detection works well. Deciding *which* gate each pass was is the weak spot, because it is made one pass at a time, from appearance alone.

- **Pass detection** (`pass_detector.py`): a per-track idle → aligned → passed state machine with frame-edge checks. Out of scope for this plan.
- **Gate identity** (`gate_db.py`, `race_match`): CLIP embedding of the padded crop 4 frames before the pass. It is compared only against the next few expected gates, and accepted at similarity ≥ 0.88 with a margin ≥ 0.03 over the runner-up.
- **Greedy and final.** Each decision moves the expected index forward, so one wrong match can shift every later pass in the lap. A pass under threshold becomes NOMATCH and is never revisited.
- **Appearance only.** Same-type gates with similar backgrounds, or a change in light between videos, look alike to CLIP. Nothing about timing or layout is used.
- **Useful for us:** `extract_race.py` already saves every frame's tracks (bbox, type, score) and every pass's CLIP embedding to `race_data.json`. The decoder can run on that file in seconds, without re-running YOLO.
- **Gap:** ground-truth marks placed in `debug_ui` are not saved to a file, and they carry no gate identity.

## Milestone 1: test dataset and scoring

Before changing any matching logic, build scored test data and record how today's system scores. Every later change is judged against that number. Both branches below exist only to build the test dataset for the pass detection system. Neither changes how passes are detected. Branch 1 is real videos you review (the final judge), and Branch 2 is Velocidrone runs labelled automatically (for fast development). Each track has its own gate memory, and gates don't move within a track.

### Branch 1: real videos, fast review

You have 2 tracks with about 30 videos each from different pilots. The goal is to make marking a review job, not a from-scratch job.

1. **Pre-fill.** The current system runs on the video first and proposes every pass with a gate ID.
2. **Review in `learn_ui`.** The tool jumps from pass to pass: Enter accepts, a number key fixes the gate ID, M marks a missed gate, T adds a tag. You add passes the system missed while watching between marks.
3. **Save** the reviewed marks per video as `<video>.gt.json` (time, gate ID, lap, tag), and allow loading them back to fix mistakes.
4. **Marking-only mode**, so test videos never add embeddings to the gate memory.
5. **Start small.** About 5 test videos per track are enough for a first test set. The rest are marked later, as needed.

**Optional tag per mark**, only when something unusual happens: *crash / clipped gate*, *bad video* (static or breakup) or *unsure*. `score.py` reports accuracy per tag, unsure marks are left out of the score, and tagged frames are candidates for future YOLO training data.

**Pass rule (decided):** a clipped gate counts as a pass if the drone went through. A crash in the gate counts only if the drone flies on.

### Branch 2: Velocidrone, labelled automatically

Velocidrone sends live race data over a websocket (`ws://<pc>:60003/velocidrone`). For each pilot it reports lap, gate number and race time as they fly. FPVTrackside's Velocidrone connector already reads this feed (`Timing/Velocidrone/VelocidroneProtocol.cs` in [FPVTracksideCore](https://github.com/uewepuep/FPVTracksideCore)). This has been read from code, not yet tested live.

1. **Record** the screen and log the websocket at the same time on your Mac, both stamped with the same clock.
2. **Convert** the log into the same `<video>.gt.json` format as Branch 1, so `score.py` treats both alike. A skip in Velocidrone's gate number is a missed gate.
3. **Build a gate memory** per Velocidrone track from a few recorded laps, the same way as for a real track.
4. **Check** first that the YOLO model detects Velocidrone gates well enough. If it doesn't, Branch 2 can still test the gate-order logic by using Velocidrone's own pass times.

**Limit:** sim footage doesn't look like real FPV video. Branch 2 scores show whether the logic works, but a milestone is done only when the Branch 1 score improves.

### Shared pieces

**Dataset location (decided).** All training and test data lives outside the git repos, in one folder on your Mac: `/Users/eyalcozac/Codes and apps/FPVdatasets`. Nothing in it is committed.

```
FPVdatasets/
  real/<track>/                 Branch 1, one folder per real track
    gate_memory.json            built from the memory videos only
    memory_videos/              1-2 videos used to learn the gates
    test_videos/                videos used only for scoring
    gt/<video>.gt.json          reviewed marks
    runs/<video>.race_data.json output of extract_race.py
  velocidrone/<track>/          Branch 2, same layout
    gt/<video>.gt.json          converted websocket log
    logs/<video>.ws.jsonl       raw websocket log
  training/                     YOLO images and labels for future retraining
```

**How the code finds it.** One data-root setting, read in this order: the `FPV_DATA_ROOT` environment variable, then `data_root` in a local `local_config.json` next to the code, then the default path above. Files inside the dataset refer to each other by paths relative to the root, so the folder can move. The repo's `.gitignore` excludes `local_config.json`, videos, files named \*.race\_data.json or \*.gt.json, and any local data folder (settings like debug\_ui\_defaults.json stay tracked), and the README names the dataset folder and its layout. The path contains spaces, so it must be quoted in shell commands.

**Score script** `score.py`: compares `race_data.json` with `gt.json` and prints the metrics in "How we measure": pass detection (a detected pass within ±0.15 s of a mark), gate-ID accuracy, missed-gate recall, false flags and lap count, split by tag. It runs on all tracks at once and prints real and Velocidrone results separately.

**Done when** the first 5 test videos per real track are reviewed, at least one Velocidrone track is recorded and labelled, `score.py` runs on both, and the current system's scores are written down as the baseline.

## Approach

Treat a lap as a sequence and label every pass at once with the most likely gate order, allowing skips. Position enters as relational cues (timing, bearings, size ratios, turns) that need no map and no focal length.

**Sequence decoder.** Hidden states are the track's gates G1…GN in order. Each detected pass is one observation. Transitions allow the next gate (normal), jumping ahead by k (k−1 missed gates, with a penalty per skip), or a "false pass" state for detector noise. Viterbi finds the best labelling for the whole video; forward-backward gives a confidence for each label. A missed gate is a skip on the best path.

**Scoring a pass against gate Gi** (log-likelihoods added, weights tuned on marked videos):

```latex
\text{score}(p, G_i) = w_c\,\text{CLIP}(p, G_i) + w_t\,\log P(\text{type}_p \mid G_i) + w_\tau\,\log P(\Delta t_p \mid G_{i-1} \to G_i) + w_g\,\log P(\text{geom}_p \mid G_i)
```

**Cues**

| Cue | What it measures | Lens-independent | Needs a new video pass | Milestone |
| --- | --- | --- | --- | --- |
| CLIP similarity | Look of the gate and its surroundings | Yes | No | 2 |
| Gate type | square / arch / circle / flagpole from YOLO | Yes | No | 2 |
| Gate-to-gate time | Time since previous gate as a fraction of lap time | Yes | No | 2 |
| Next-gate bearing | Where the next visible gate sits in the frame at the pass (normalised x, y) | Mostly (fisheye bends edges) | No, from saved tracks | 3 |
| Same-type size ratio | Apparent size of the next gate vs the current one; equals their distance ratio | Yes | No, from saved tracks | 3 |
| Turn between gates | Yaw and pitch change from background optical flow between passes | One scale factor per video | Yes, low resolution | 4 |

**Learning the cues.** `learn_ui` already has you mark every gate over several laps. Each gate slot then also stores: the gate-to-gate time distribution, the next-gate bearing and size ratio, and (later) the turn angle. No extra marking is needed beyond what you do today.

**Lens scale.** Angles from flow scale with field of view. The decoder fits one scale factor per video (the track shape has to repeat lap after lap), or takes it from a lens dropdown.

## Milestones

Start with the decoder on signals you already have, because it needs no new model or video pass and shows quickly whether sequence decoding pays off.

1. **Milestone 1: test dataset and scoring.** Branch 1 reviews pre-filled passes on real videos, and Branch 2 labels Velocidrone runs from its race websocket. `score.py` scores the current system on both, and the real-video score is the baseline.
2. **Milestone 2: sequence decoder on existing signals.** Viterbi over G1 to GN with skip and false-pass states, scored by CLIP, gate type and gate-to-gate time. Done when it clearly beats greedy matching on held-out videos.
3. **Milestone 3: geometry from saved tracks.** Next-gate bearing and same-type size ratio for each gate slot. Each cue stays only if it improves held-out accuracy.
4. **Milestone 4: turn angle from optical flow.** Yaw and pitch change between passes, with one scale fitted per video. It stays only if it helps on multi-lens videos.
5. **Milestone 5: the working pass detector as an FPVTrackside feature.** Pass detection and missed-gate flags shown to the user inside FPVTrackside. Afterwards, if possible, a 3D map of the track inside FPVTrackside.

Each milestone moves on only if it beats the one before on held-out real videos.

Milestones 1 and 2 need only your marked videos and existing `race_data.json` files. Milestone 4 is the only one that reads the video again.

## How we measure

Every change is scored against your marked videos, and a cue stays only if it improves gate-ID accuracy on videos it was not tuned on.

**Ground truth.** For each pass: the time and the gate ID (G1…GN), plus a flag for deliberately skipped gates. Include at least one video with a real missed gate and one with a different lens.

**Split.** Tune weights on some videos, report on the rest (leave one video out when there are few).

| Metric | Definition | Proposed target |
| --- | --- | --- |
| Gate-ID accuracy | Correct gate on passes the detector found | Beats greedy `race_match` clearly, then ≥ 97% |
| Missed-gate recall | Real skips flagged | ≥ 95% |
| False missed-gate flags | Flags on laps where every gate was flown | ≤ 1 per 20 laps |
| Lap count | Laps found vs marked | Exact |

Pass-detection errors (missed or extra passes) are reported separately, so they aren't blamed on the decoder. A false flag costs the pilot trust, so it is weighted above a missed flag.

## Risks and open questions

- **Pilot-dependent timing.** Gate-to-gate times differ between pilots and classes. Normalising by lap time helps; if not enough, learn times per pilot after a first decode.
- **Pass-detector misses.** A pass the detector never fired looks like a skipped gate. The decoder should report "gate not seen" separately from "gate skipped" where the next-gate cues show the gate was flown past.
- **Next gate not in view** at the pass (tight turns, gates behind). The bearing cue is then missing, not zero; the decoder must skip it.
- **Fisheye distortion** bends bearings near the frame edge. Use normalised positions near the centre, or a rough per-lens undistortion.
- **Analog breakup** can split one pass into two or hide a gate. The false-pass state absorbs extras.
- **Decided: one memory per track.** Gates don't move within a track; a changed layout means a new memory.
- **FPVTrackside comes last.** Integration starts only once the pass detection system works (Milestone 5).
