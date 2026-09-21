# Nine Versions to Walk Forward: Reward Hacking, Unpriceable Constraints, and Auditing the Evaluator in RL Bipedal Locomotion

**A technical report on training the Poppy humanoid to walk in Isaac Lab**

Peixin Gao — Shanghai Jiao Tong University
Project repository: `github.com/GaoPeixin418/isaacsim-poppy-walk` (code, logs, figures, videos)
September 2026

---

## Abstract

We train a 10-DOF Poppy humanoid to walk forward at a commanded 0.25 m/s in NVIDIA Isaac Sim / Isaac Lab using PPO (RSL-RL), from scratch, on a single RTX 3090. The final policy walks with 100% survival under ±0.3 m/s random pushes, 44%/43% airborne duty factors, near-zero lateral drift, and symmetrical foot loading. The contribution of this report is not the final gait but the **debugging knowledge produced on the way**: (1) a taxonomy of nine distinct reward-hacking and specification-failure modes, including a gait asymmetry constraint that the policy *literally priced into its objective* until it was converted from a penalty into an episode termination; (2) three case studies in which the *evaluation tooling itself* was the faulty component (a rigid-body index mismatch, a stale command-channel assumption, and sampling aliasing), each of which produced confident but wrong diagnoses; and (3) a heading determination failure chain in which the "forward" command was, in sequence, lateral (v7) and backward (v8), because both the reward metrics and human visual inspection of a featureless white robot are unreliable arbiters of heading — resolved finally by forward kinematics of the URDF chain. We distill these into transferable rules: *penalties are a pricing language, termination is a legal language*; *the measuring instrument is a suspect of equal standing with the policy and the asset*; *when the eye disagrees with the instrument, convert the disputed premise into a computable quantity.*

---

## 1. Introduction

Simulated bipedal locomotion with deep RL is a solved problem in the aggregate — published pipelines (e.g., Isaac Gym-era legged gym, Walk-These-Ways-style full-ASIMO-style control, AMP and its descendants) produce robust controllers routinely. What is *not* routinely published is the failure-mode ledger between "PPO on a fresh humanoid" and "a gait that actually satisfies the intent of the specification." This report documents exactly that ledger, on a small open-source humanoid (Poppy, 2.6 kg, 10 actuated leg DOF, all MX-series Dynamixel servos), under a deliberately constrained budget: one consumer GPU, a 7-day sprint, no reference motions, no pretrained weights.

Three findings generalize beyond this platform:

1. **Constraint enforcement by penalty is gameable in a way that enforcement by termination is not.** A −2.0-weighted asymmetry penalty was absorbed as a small recurring "fine" (≈0.01/step) traded against a velocity reward of ≈1.46/step; the same constraint expressed as an episode termination (any foot's 2-second-EMA ground reaction force below 18% body weight ⇒ terminate) converged immediately to a symmetric alternating gait. We call the former a *priceable* penalty and argue the distinction is a design axis worth making explicit.

2. **The evaluation stack must be audited with the same rigor as the policy.** Across this project, three separate evaluation defects each produced high-confidence false diagnoses: (a) an index-ordering mismatch between the robot's body IDs and the contact sensor's body IDs turned "left foot load" into "right shin load" (nothing, forever); (b) the eval script pinned its command into a hard-coded channel inherited from an earlier coordinate convention, feeding the policy an out-of-distribution command and misreporting the result as a failure of learning; (c) a 0.5 s probe sampling interval aliased against a 0.33 s gait period, manufacturing a spurious slow wave that was briefly interpreted as a quasi-static gait.

3. **Heading is an evaluation blind spot of a special kind.** Scalar reward metrics (speed magnitude, survival, duty factors, load symmetry) are all invariant to *where the robot is actually facing*. We obtained, in sequence: a lateral-crossover gait at 90° to intent (v7), a backward "moonwalk" at 180° (v8), and only then a true forward gait (v9) — with every intermediate metric suite green the whole time. Visual inspection caught both errors (a human noticed "it's walking sideways" and later "it's walking backward"), but visual inspection of a symmetric white robot *also* produced the wrong arbitration once (the camera-calibration experiment that concluded "+y is front" was an eyeball judgment of front-vs-back of a near-featureless torso). The reliable arbiter was forward kinematics: in the URDF's default bent-knee pose, the knee protrudes toward the base frame's −y, and the computed ankle height cross-validates against the simulated pelvis height to sub-centimeter agreement.

The report is organized as a chronological failure ledger (§4–§7), followed by final results (§8), limitations (§9), and reproduction instructions (§10). The Chinese-language engineering reports (D2–D5) in the repository contain full logs and additional detail.

---

## 2. Task Setup

**Platform.** Isaac Sim 4.5.0, Isaac Lab v2.1.0, PyTorch 2.5.1, RSL-RL PPO, single RTX 3090 (24 GB), 2048 parallel environments. Training runtimes: ≈40 min per 2500 iterations nominal, degraded to 2–5× slower during a cloud-host PCIe fault (diagnosed and documented separately; see §9).

**Robot.** Poppy humanoid, upper body locked to the torso (10-DOF leg-only training variant). All joint limits and drive gains were audited during URDF→USD conversion, with a recurring hazard worth stating explicitly: the URDF/Isaac Lab configuration layer speaks radians and N·m/rad, while the USD stage stores degrees and N·m/deg — a 57× factor that silently corrupts any gain copied between layers. PD gains were measured, not assumed (hip pitch kp≈20, other joints ≈8 N·m/rad); torque limits were overridden in configuration (MX-28→2.5, MX-64→6.0 N·m) rather than edited in the USD. The knee joint's one-sided limit [−2.339, +0.061] rad physically closes off reverse-joint "cheating" solutions in the sagittal plane.

**MDP.** Standing and walking are the same task: command-conditioned base velocity tracking with the command resampled from a configurable range. Standing is the zero-velocity slice. Observations: joint positions/velocities, base linear/angular velocity, last action, current command. Actions: 10 joint position targets under the PD controller. Terminations: base contact, excessive tilt, base below height threshold, and (from v7 on) the single-leg-lean term defined in §5. Domain randomization: random push events (±0.3 m/s velocity kicks every 4–6 s) from the standing task onward, mass/COM noise, friction sampling.

**Evaluation protocol.** Frozen deterministic policy, 8 parallel environments × 12 s, two conditions: undisturbed (Play) and with the training push events active. Metrics: survival fraction, mean base velocity in the commanded channel, lateral drift, per-foot airborne duty factor, left/right foot vertical position correlation (anti-phase expected), per-foot mean ground reaction force, base height. All metrics computed from per-frame simulation state, with force data taken from the contact sensor's own body ordering (see §6.1 for why this parenthetical matters).

---

## 3. Asset Engineering (summary)

Three lessons from the URDF→USD stage, before any learning happened:

- **Audit, don't trust, the source mesh tree.** The stock Poppy URDF contained a mirrored-axis defect in the left knee (rotation axis flipped relative to the right knee). Symmetric robots are not automatically symmetric in code; we verified left/right equivalence with a zero-action probe (foot height difference 0.13 mm after the fix).
- **Units are a three-way hazard.** Radians vs. degrees across config/USD layers, and N·m/rad vs. N·m/deg stiffness, are each individually documented pitfalls; the combination is worse because numbers *look* plausible.
- **Physical limits are the cheapest anti-reward-hacking device available.** The one-sided knee limit and the explicit torque caps eliminate entire families of degenerate solutions at the physics layer, where no reward shaping can override them.

Full details: D2 asset report in the repository.

---

## 4. The Failure-Mode Ledger (v1–v6)

Each entry follows the same format: phenomenon → investigation → root cause → fix. The pattern to notice is that **every fix closes exactly one exploit and reveals the next**.

| Ver | Failure mode | Root cause | Fix applied |
|---|---|---|---|
| v1 | **Skating** — both feet slide on the ground | Velocity reward had no foot-lift term | Air-time reward granted only at landing events (not continuously, to avoid encouraging airborne dancing) |
| v2 | **One-legged hop** — right foot hops, left drags | Air-time reward summed over feet; single-leg hopping is a legal optimum | Fixed a `clamp(min=0)` that zeroed negative gradients (the policy had no signal to *land* the foot); clamp the maximum only |
| v3 | **Tip-toe shuffling** — tiny rapid steps in place | Air-time cap (0.5 s) too generous; rapid cycling farms the landing bonus | Tightened air-time cap; reweighted velocity tracking |
| v4 | **Backward walking** (a bug, not a hack) | Penalty refactor made sign error: penalty function returned negative excess, weight was negative, so the "hover penalty" *rewarded* hovering | Fixed sign; instituted a rule that every reward edit ships with an analytic gradient-direction derivation |
| v5 | **Right-leaning overload** — alternating gait but load badly asymmetric (left 30% / right 73% duty-weighted load) | See §6.1 — this diagnosis is itself a story: the initial "left foot never touches" diagnosis was an evaluation artifact | Symmetry enforcement, next two versions |
| v6 | **Sweeping limp** — left foot skims the ground with contact but no load; pays a small asymmetry fine (≈0.01/step) to keep earning ≈1.46/step of velocity reward | *Priceable penalty*: any finite penalty can be absorbed as a cost of doing business | See §5 — the constraint was re-expressed as a termination |

Two meta-observations. First, the exploit sequence (skate → hop → shuffle → sweep) is exactly a tour of the specification's un-stated desiderata: what "walking" means beyond "base moves forward and robot doesn't fall." Second, v4 (our own sign bug) and v5–v6 (evaluation artifacts, §6) mix *policy* failures with *tooling* failures — the ledger does not distinguish them until you audit, and you should expect this mixture.

---

## 5. Penalties Are Priceable; Termination Is Not (v6 → v7)

The v6 configuration penalized load asymmetry (2-second EMA of per-foot vertical contact force, difference normalized by body weight, threshold 0.25, weight −2.0) and base roll. Training logs showed why it failed to change the gait: the policy's total return under the limp exceeded the symmetric alternative by the fine it was paying. A penalty of weight *w* simply posts a price; a rational optimizer buys whatever the price permits. Raising *w* destroys the reward landscape's learnability long before it eliminates the behavior.

v7 re-expressed the identical constraint as a **termination**: any foot's 2-second-EMA load below 18% of body weight after a 2.5 s warm-up ends the episode. The arithmetic of the threshold matters and is worth reproducing: a normal alternating gait holds each foot's EMA near 50% of body weight (2.8× safety margin), while a push-induced unloading lasting under the EMA's time constant (~2 s) does not trip the term — so it distinguishes *gait structure* from *transient disturbances* by construction.

Training converged immediately to an alternating, loaded gait. We summarize the design axis as:

> **A penalty is a price the policy may choose to pay; a termination is a law it cannot.** Choose the currency according to whether the constrained behavior is *undesirable* (price it) or *inadmissible* (outlaw it).

This is, of course, the classic reward-shaping vs. constraint distinction (constrained MDPs, safe RL, and curriculum design all touch it); our contribution here is the concrete ledger showing the failure mode of getting it wrong in one direction, at full experimental detail, in a setting where the "priced" behavior looked like progress on every scalar metric.

---

## 6. Auditing the Measuring Instrument (three cases)

### 6.1 The body-index mismatch (two sensors, two orderings)

Symptom: v7 evaluation reported "left foot load = 0 N for the entire episode" while the single-leg-lean termination — which reads the *same* physical quantity — never fired across 12 s of full survival. Contradiction between two instruments is a signal, not noise.

A live probe (`probe_term.py`) dumping every intermediate quantity (per-foot forces, EMAs, body weight, warm-up flag, term return) exposed the defect: `d5_eval.py` indexed the contact sensor's force array with the **robot's** body IDs. The sensor tracks all 11 rigid bodies with its own ordering; robot-ID 9 is the left foot, but sensor-slot 9 is the **right shin**, which never touches anything. "Left foot load" had been measuring the right shin for three evaluation cycles (v5–v7). Positions taken via `robot.find_bodies` were coincidentally correct (that ordering is the robot's own).

Fix: force data must be indexed by the sensor's own ordering (`sensor.find_bodies`), positions by the robot's — the two lookups are separate concerns and must not be conflated. Rule: **any cross-object index must have its mapping verified explicitly; orderings are API contracts, not conventions.**

### 6.2 The stale command channel (the evaluator's inherited assumptions)

After re-training (v8, §7), the first evaluation reported a slow shuffle (0.054 m/s). The policy was fine; the evaluator was not: its `pin_command()` wrote the commanded velocity into the base-x channel, hard-coded under the original coordinate convention (§7). The policy — trained with x always zero — received an out-of-distribution command and behaved conservatively. The instrument carried a stale *assumption about the system* rather than a code defect in the narrow sense.

Rule: **when the system under test changes convention, the harness inherits the change or lies.** Configuration drift between training code and evaluation code is a measurement defect.

### 6.3 Sampling aliasing (a probe that manufactured a slow gait)

A termination-term probe sampled every 25 steps (0.5 s). The actual gait period was 0.33 s. The sampled trace showed a slow oscillation that was pure aliasing artifact — briefly diagnosed as "a quasi-static gait with 2–2.5 s steps." Gait-phase conclusions thereafter came from per-frame traces only. Rule: **probe sampling must exceed twice the highest frequency of the measured signal; for periodic gaits, prefer full-rate traces.**

### 6.4 Synthesis

Each of these produced a *confident, quantitative, wrong* diagnosis. In all three cases the surrounding evidence (survival under a contradictory termination; command-channel history; the existence of a faster oscillation in per-frame data) was already present — the error was in treating the measuring tool as ground truth. We elevate this to a standing rule:

> **The measuring instrument is a suspect of equal standing with the policy, the asset, and the reward design.** When two instruments disagree, do not pick a side: write a probe that dumps both instruments' intermediate quantities and make them confront each other.

---

## 7. The Heading Chain: How "Forward" Was Lateral, Then Backward (v7 → v8 → v9)

**v7 (lateral, 90° off).** Every scalar metric was green: 0.289 m/s tracked speed, 100% survival, alternating anti-phase foot heights, symmetric loading. A human watching the video noticed the gait was sideways. Measurement: the crab angle (base yaw minus velocity heading) was ≈0° — the base was *aligned with its motion*; the error was that the commanded channel (base x) is the robot's *lateral* axis in the anatomical frame. The base frame is rotated 90° relative to the anatomical frame — a coordinate convention, not a bug in the ordinary sense. All v1–v7 "walking" was a 3 Hz lateral crossover gait. **Scalar metrics are invariant to heading; a specification that never measures where the robot faces cannot enforce it.**

**v8 (backward, 180°).** The command was moved to the y channel per a camera-calibration experiment: render the same pose from four body-frame directions and eyeball which view is the "front." This arbitration was itself wrong — a featureless white torso is genuinely hard to distinguish front from back in a render — and v8 trained a fluent backward gait (0.234 m/s, all metrics green). A human again caught it ("it seems to be walking backwards"). Frame-by-frame: toes pointed one way, base translated the other. A frozen-policy test with the sign flipped (command −0.25 in the forward channel) produced 100% survival but marching in place (+0.045 m/s actual): the policy had **no sign generalization** — the learned gait is not a symmetric representation that can be run in reverse.

**v9 (kinematic arbitration).** Instead of eyes, the URDF chain: under the default bent-knee pose (hip +0.10, knee −0.20, ankle +0.10 rad), forward kinematics places the knee joint protruding toward base −y with the shank trailing toward +y — the knee's protrusion direction is the anatomical forward (bend your knee: kneecap forward, heel backward). Cross-validation: the computed ankle height (−0.383 m) plus foot thickness matches the simulated pelvis height (0.419 m); the right leg sits at −x, consistent with (forward −y) × (up +z) in a right-handed frame. Command sign flipped to −y; fine-tuned 2500 iterations from the v8 checkpoint (the skill family transfers; direction corrections need thousands, not tens of thousands, of iterations).

> **Rule: when the eye and the instrument disagree about a geometric premise, do not adjudicate by rendering. Convert the premise into a computation (here: forward kinematics) and let it decide.**

---

## 8. Final Results (v9)

Deterministic frozen policy, 8 environments × 12 s, commanded −0.25 m/s in the anatomical-forward channel (base −y):

| Metric | Undisturbed | Random pushes (±0.3 m/s, every 4–6 s) |
|---|---|---|
| Survival | 100% (8/8 full 12 s) | **100%** |
| Forward speed | 0.296 m/s (cmd 0.25; +18% overshoot) | 0.273 m/s |
| Lateral drift | 0.001 m/s (<1% of forward) | 0.005 m/s |
| Airborne duty factor (L/R) | 44% / 43% | — |
| Foot height anti-phase correlation | −0.69 | −0.68 |
| Mean foot load (L/R) | 12.6 / 12.9 N (sum ≈ 25 N body weight) | alternating maintained |
| Base height | 0.419 ± 0.0007 m | 0.419 ± 0.0008 m |
| Heading | toes and translation co-directional, frame-verified | — |

Video (12 s, 720p, camera-following side view): `docs/videos/poppy_walk_v9.mp4`; the lateral-gait (v7) and backward (v8) videos are retained alongside for contrast.

The gait is a small-step high-frequency walk (~3 Hz, ~10 cm stride, ~1 cm foot clearance) — dynamically stable and robust, but not a natural large-stride gait. We consider this acceptable for the milestone and flag it in §9.

---

## 9. Limitations and Future Work

**Gait quality.** Small steps, shallow foot lift, ~18% speed overshoot. Natural next steps: reference-state gait shaping, AMP-style adversarial objective, or explicit foot-trajectory rewards — all deliberately out of scope for this sprint.

**Single seed.** Every result above is one training run per version. The v6-vs-v7 penalty/termination contrast in particular deserves a multi-seed ablation (3–5 seeds each arm) before it can be called a finding rather than an anecdote. This is the first experiment of the follow-up program.

**Heading was resolved, not generalized.** The policy does not understand "forward" as a semantic; it tracks a signed scalar in one channel. Command-generalization tests (speed sweep 0–0.35 m/s, turn commands) and sign-symmetry analysis are open.

**Evaluation still has known blind spots.** Our final metric suite measures heading only via a human-checked video property (toe/translation co-direction). A yaw-consistency metric (crab angle under command) should be promoted into the automated suite.

**Sim-only.** No sim2real claims. Actuator modeling is idealized PD; the real Dynamixel series has gearbox backlash and temperature-dependent torque curves that would matter on hardware.

**Infrastructure.** A cloud-host PCIe degradation (16×→8×, host-to-device bandwidth 0.5 GB/s) slowed training up to 5× for part of the sprint and was diagnosed by a small-kernel-latency / pinned-bandwidth protocol worth reusing; a desktop GUI rendering path (carb.tasking) deadlocked consistently, and all visualization was produced by headless off-screen rendering with a following camera (`record_walk.py`) — a workaround that turned out to be more reproducible than the interactive path it replaced.

---

## 10. Reproduction

```bash
# Train (2048 envs, ~40 min / 2500 iters on RTX 3090)
bash scripts/run.sh scripts/train_poppy.py --headless --task Poppy-Walk-v0 \
  --num_envs 2048 --max_iterations 2500

# Evaluate (--vx is the anatomical-forward speed, base -y channel)
bash scripts/run.sh scripts/d5_eval.py --headless --task Poppy-Walk-Play-v0 \
  --checkpoint <ckpt> --vx -0.25 --duration 12

# Live probe of termination-term internals
bash scripts/run.sh scripts/probe_term.py --headless --checkpoint <ckpt>

# Camera-following side-view video (headless off-screen render)
bash scripts/run.sh scripts/record_walk.py --headless --enable_cameras \
  --checkpoint <ckpt> --seconds 12 --out out/videos
```

Full engineering reports (asset audit D2, standing D3, walking D5 with the complete v1–v9 ledger), traces, and all probe scripts are in the repository.

---

## Appendix: Methodological rules distilled

1. **Contradiction between two instruments is a lead, not noise.** Dump both instruments' intermediate quantities and make them confront each other (§6.1).
2. **Cross-object indices are API contracts.** Verify orderings explicitly; never assume robot-order == sensor-order (§6.1).
3. **The harness inherits the system's conventions, or it lies.** Re-audit evaluation assumptions after any change to the system's coordinate or channel conventions (§6.2).
4. **Probe sampling must satisfy Nyquist against the measured signal.** For periodic gaits, use full-rate traces (§6.3).
5. **When measurements and analysis disagree, either may be wrong** — the disputed premise should become a separately measurable quantity (§7).
6. **Geometric premises are adjudicated by computation, not by rendering.** Front/back of a featureless robot is exactly as unreliable as it looks (§7).
7. **Penalties price; terminations outlaw.** Match the enforcement mechanism to whether the behavior is undesirable or inadmissible (§5).
8. **Physical limits are the cheapest anti-hacking device.** Spend joint limits and torque caps before reward shaping (§3).
