# Bubble Highway – the whole simulation as one pipeline

All five parts of the team guide, joined into one program. The CV person's part is final. **Every other part is a working stand-in**, clearly marked at the top of its file, so that each teammate can replace their file with their own version (golden rule). Nothing else changes as long as the inputs and outputs stay the same.

## The parts

| File | Owner (guide) | Receives | Passes on |
|---|---|---|---|
| `shared_settings.py` | EE2 – rule 2 | – | every number, tagged GUIDE / ENG3 / TEAM / LIT / ASSUME |
| `bubble_physics.py` | shared by EE1 + EE2 + EE3 | air, depth, time | Bulson plume speed, push shape, depth profile, tide, compressor power |
| `bloom.py` | EE1 Job A | settings, time | new jellyfish (agreed fields) |
| `water.py` | EE1 Job B | firing pipes + air | every jellyfish's position, 20 × per second |
| `jelly_scorecard.py` | EE1 Job C | jellyfish records | out / back / sideways shares, transit, roughness, time above pipes |
| `key_test.py` | EE1 key test | `water.py` alone | colour map + three sanity checks |
| `camera.py` … `camera_system.py` | CV person | jellyfish positions | count, crowding, drift (final version) |
| `controller.py` | EE2 Job B | camera reading + current meter | on/off, air, wave speed, firing pipes |
| `run_simulation.py` | EE2 Job A | all parts | one run: records, scorecard, energy |
| `experiments.py` | EE2 Job C | `run_simulation.py` | tuning + six tests × 3 options × 3 seeds |
| `energy.py` | EE3 Job A | controller record | kWh, saving vs curtain, kWh per 1,000 moved |
| `charts.py` | EE3 Job B | saved results | Chart 1, Chart 2, key-test map, adaptability charts, animation |
| `make_deliverables.py` | everyone | – | runs everything and saves the hand-in items |

## Run it

```bash
pip install numpy scipy opencv-python-headless matplotlib pillow ultralytics
python run_simulation.py --quick              # one 20-min bloom, prints the scorecard
python key_test.py                            # EE1 key test (~15 min)
python experiments.py --seeds 3               # all tests, quick mode (~30-45 min)
python make_deliverables.py                   # main runs + charts + animation
```

`--full` runs the experiments with the real camera and 2-hour blooms (hours on a laptop). The camera uses the trained YOLO model in `runs/jelly/weights/best.pt` if present, otherwise the blob finder. `camera_source="truth"` uses the guide's stand-in camera for fast sweeps.

## Key numbers and where they come from

* **Air: 374 L/min per metre nominal (285–534 controller range).** This gives 8.99 kW per pipe and 459 kW for all 51 pipes, matching the guide's 9 kW / 450 kW check. The range is the tested 8–15 L/min/m ratio (Circiumaru et al. 2024) scaled up. The eng3 document's 10.5 L/min/m comes from a 10 cm deep flume and was judged too low for 5 m depth (Haberlin et al. 2021: low-flow curtains at 5 m had no significant effect).
* **Water speed: theory, not measurement.** Vmax = 1.46 (g q)^(1/3) (Bulson 1961; 1.2 tested as sensitivity) → 0.58 m/s at 374 L/min/m. The surface current is H/4 = 1.25 m deep, falling linearly to zero (Bulson 1961), and fastest about d/2 from the pipe.
* **Pressure:** water only, 0.503 bar gauge (no hose loss, team decision). Compressor efficiency 60 %.
* **Current direction for wall mode:** a separate current meter (team decision).

## Assumptions to state openly in the slides

Push shape between the literature points, background current −0.05 m/s toward the intake, the swarm appearing across the first 10 m of the entrance, jellyfish depths (exponential, mean 1 m), response time growing with size, the random swimming, the gentleness limit 1.0 /s (no published limit), controller levels and gain, and the wave gap of 10 m. Every one is tagged `[ASSUME]` in `shared_settings.py`.
