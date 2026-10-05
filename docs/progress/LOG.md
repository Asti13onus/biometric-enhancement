# Progress Log

Newest entries at the top. One entry per working session: what was done, what was decided,
what is blocked, what is next. This is the raw material for supervisor updates and for the
thesis narrative — write it as if a reader in month nine needs it.

---

## 2026-10-05 — Session 13: U12 complete — the benchmark, the crossover, and closure

**U12 — 45 registry cells: held-out sets, the severity axis, position strata**
- Held-out real sets (EER, none / SNFEN / WAFEN-aligned): CrossMatch 0.0066 / 0.0209 /
  0.0344; MINEX 0.0911 / 0.0690 / 0.0882; FVS 0.0410 / 0.0711 / 0.0671. **Enhancement
  hurts clean prints — SOTA included — and helps only the degraded set.**
- **The severity crossover (AE3), the thesis's second figure**
  (`results/figures/severity_crossover.png`): CrossMatch degraded by the wear model at
  0.3/0.5/0.7, same 408 prints throughout. Unenhanced EER climbs 0.0066 → 0.2387;
  SNFEN holds 0.0209 → 0.0750; **WAFEN holds 0.0344 → 0.1206 — halving the unenhanced
  error at 0.5 and 0.7 with disjoint CIs**, at one network against four. Crossover
  near severity 0.3. Caveat carried: the degradation is our wear model, WAFEN's
  training family (SNFEN never saw it either; inputs are real prints).
- MINEX position strata: little fingers are the hard stratum (none 0.18–0.19);
  per-position intervals are wide (≈50 fingers each) — reported as context, not claims.
- Two harness defects found and fixed by the run itself: MINEX's validation imagery
  ships four calibration patterns that parsed as fingers (roster now 797; identities
  from manifest columns after the subject/finger fix); and bozorth3's -M mode
  malloc-fails on a 634k-line mates list — run_baseline now exposes the seeded
  impostor cap (30k) and oversized templates are quality-trimmed non-destructively.
- MINEX cells re-logged under the clean roster and capped pairs: values stable
  (0.0911 / 0.0690 / 0.0882) — the original numbers were robust.

**U11 — paired verdicts, completed on the mindtct chain**
- **minex, SNFEN vs none: −0.0221 [−0.0461, −0.0060] — resolved.** "Enhancement helps
  genuinely degraded real prints" is now significance-tested, the deployment claim.
- Standing trio: arm2 > arm1 (resolved, −0.0247); SNFEN > none on MINEX (resolved);
  SNFEN > none on FVC2004 (not resolved even paired).
- **LEADER chain, FVC2004, SNFEN vs none: −0.0141 [−0.0284, −0.0008] — resolved.**
  The enhancement effect's direction survives a change of extractor (R17), and notably
  *resolves* under LEADER where mindtct left it open. Absolute EERs are far higher under
  LEADER + bozorth3 (0.19 vs 0.10) — the extraction chain dominates absolute numbers,
  one more reason the thesis reports within-chain comparisons only.
  **With this, every planned experiment of the semester is complete or ADR-deferred.**

**Results-chapter tooling**
- `make_tables.py` renders the benchmark (absent cells reported as absent), the
  precision–coverage table and the paired verdicts from the registry;
  `make_severity_figure.py` draws the crossover. The results chapter regenerates from a
  clean checkout plus the registry — U12's verification criterion met.

**The semester's findings, as the discussion chapter will argue them**
1. Post-hoc domain alignment beats corrected synthesis (resolved, disjoint CIs) —
   the first controlled head-to-head of the literature's responses.
2. Enhancement must be quality-gated: harmful on clean prints (every method),
   breakeven at light wear, dramatically beneficial at heavy wear (crossover ≈ 0.3).
3. Image quality metrics do not predict matching (NFIQ 2 up, EER flat or worse —
   shown on three independent axes).
4. Abstention falsified on both domains (U9's sweep; FVC2004's gates).
5. The degradation model is corrected synthesis's ceiling (four improving proxies,
   one immobile matcher — session 10's diagnosis, unrefuted since).

---

## 2026-10-01 — Session 12: U9 and U11 — the thesis figure, and abstention answered

**U9 — minutiae precision and the precision–coverage curve (synthetic pseudo-GT)**
- Held-out test split (1,290 impressions, 430 fingers), deterministic wear at severity
  0.5; pseudo-GT = mindtct on each clean master; Cappelli correspondence (14 px, π/9).
- **Enhancement works on its home domain**: precision 0.559 → **0.815**, recall 0.397 →
  **0.634** over no enhancement. WAFEN vs SNFEN: precision 0.815 vs 0.832, recall 0.634
  vs 0.575 — **higher F1 (0.713 vs 0.680) at ~3× less compute**.
- **AE2 answered: no.** The sweep (t = 0.10…0.40, block 16, pre-registered) is nearly
  flat in precision (+0.007 at 90% coverage) while EER worsens 0.0001 → 0.0155.
  Discarding uncertain minutiae does not beat keeping them — on either domain, since
  FVC2004 showed the same. A designed, falsifiable claim, tested and falsified.
- `results/figures/precision_coverage.png`: our curve, every baseline a point at 1.0.
  All eight conditions in the registry; figures labelled synthetic throughout.

**U11 — paired verdicts (mindtct chain)**
- **SNFEN vs none: −0.0095 [−0.0208, +0.0017] — not resolved.** The 2026 SOTA's gain
  over no enhancement on FVC2004-B does not survive the powered paired test at this
  benchmark's size.
- **Arm 2 vs arm 1: −0.0247 [−0.0359, −0.0140] — resolved, alignment better.** The
  thesis's central comparison excludes zero by its full interval width; it is
  statistically stronger than the SOTA's own improvement claim on the same benchmark.
- Pending: the same comparison under the LEADER chain (extractor independence), queued
  behind the U12 benchmark for machine time.

**U11 — paired bootstrap and the second chain (built; comparisons running)**
- `paired_bootstrap_eer_difference`: pair indices drawn once per resample, applied to
  both conditions; tests assert the paired interval is strictly narrower than the
  marginal comparison and that shared pair noise cancels.
- LEADER wrapped as a second extraction chain emitting `.xyt` (conventions converted
  and tested against mindtct); `run_baseline` gains an `extract` hook;
  `experiments/paired_compare.py` reports the EER difference with a verdict row.

---

## 2026-09-27 — Session 11: arm two moves the matcher — the comparison has its finding

**Arm two — unsupervised domain alignment (Joshi line), on the same harness**
- Pseudo-annotation: the *synthetic-only* checkpoint labels the 1,160 real non-test
  prints with its own five outputs (`scripts/build_pseudo_annotations.py`); its
  confidence becomes the evidence target. No pyfing, no ground truth.
- Photometric mode: real prints enter undamaged; paired views differ by contrast/gamma/
  blur/noise jitter (CDC-GAN's recipe), never by our wear model. Consistency in both
  domains (Anguli wear-pairs + real photometric-pairs). 8 × 400 from `wafen/best.pt`.
- Training: cross-view consistency 0.0458 → 0.0372; ridge flat; orientation on jittered
  real prints stayed fragile (~0.3–0.4) — self-labels add stability, not information.
- **FVC2004: EER 0.1105 [0.0985, 0.1232]** — vs arm one's 0.1353 [0.1238, 0.1489]:
  **disjoint CIs, an 18% relative improvement**, after sixteen statistically flat
  evaluations of arm-one variants. Still short of the unenhanced 0.0991, but the CI now
  touches it. 88.1 minutiae/img, NFIQ 2 59.1, **336 ms/img** (fastest yet).
- **The comparison finding the thesis was built for:** same network, benchmark and
  budget — post-hoc domain alignment substantially outperforms corrected synthesis, and
  neither yet beats the unenhanced control on worn-print data outside the alignment
  literature's home benchmark. Also note the arms compose: the alignment started from
  the corrected-synthesis checkpoint.
- Pre-committed follow-up (allowed by yesterday's criterion "0.0991–0.1310 → one longer
  run is justified"): a single 20-epoch alignment run, same recipe, no other changes.

**Pre-registered 20-epoch run — plateau confirmed, arm two closed**
- Training curves matched the 8-epoch run (best real-val ridge 0.2917 vs 0.2929;
  consistency floor ~0.0365 vs 0.0372); the annealing phase added nothing.
- FVC2004: EER 0.1139 [0.1026, 0.1253] — statistically identical to the 8-epoch run's
  0.1105 [0.0985, 0.1232]. Per the criterion fixed in advance, **0.1105 stands as arm
  two's final number** and no further arm-two variants will be run.
- Arm-two per-DB: DB1 0.1520, DB2 0.1715, DB3 0.0470, **DB4 0.0461** — the best cell of
  any method on DB4 (vs none 0.0607, SNFEN 0.0661), with the caveat that DB4 is
  SFinGe-generated, the domain nearest the training distribution.
- Final standings on FVC2004, one harness: SNFEN 0.0895 · none 0.0991 · **arm two
  0.1105** · arm one (best) 0.1310 · arm one (plain) 0.1353. The comparison's finding:
  post-hoc alignment > corrected synthesis (disjoint CIs); neither beats the unenhanced
  control on worn-print data outside the alignment literature's home benchmark.
- Remaining: arm three (unpaired translation, Karabulut) — needs an ADR on GAN scope
  within the parameter budget before any build; then U9–U12 and the write-up.

**Machine note:** the leaked-`explorer.exe` commit exhaustion recurred twice (~1.4 GB/h);
killing it restores ~13 GB. Root-cause with ShellExView after the thesis runs.

---

## 2026-09-26 — Session 10: first trained model, first real-data result (negative)

**Training — three causes stood between us and a single completed epoch**
- **fp16 is broken on this GPU.** Under autocast a `Conv2d` with inputs near 5 returns NaN
  on the GTX 1650; fp32 on the same batch is exact. Every loss, train and val, was NaN —
  one NaN forward also poisons BatchNorm running stats. The canary never caught it
  because it runs in fp32. AMP is now opt-in; a non-finite loss stops the run at batch 1.
  The card has no tensor cores, so fp32 costs almost nothing.
- **DataLoader workers exhaust the pagefile.** Each Windows worker re-imports torch and
  its CUDA DLLs (~5.7 GB commit per torch process); two hit WinError 1455 and the main
  process hung for good on the dead worker. Workers now default to 0.
- **The machine itself.** A leaked, elevated `explorer.exe` held 13.5 GB of commit (0.03 GB
  free system-wide) — the cause of the recurring "bun crashed" / terminal / IDE failures.
  After it was ended, RAM (8 GB, one stick) became the limit: the run crawled at 2% GPU
  while VS Code, Dropbox and Edge held ~5 GB; closing them took epochs 8 → 3.5 min.
  **16 GB of RAM would remove most of this project's friction.**
- `scripts/train_simple.bat`: one process, short epochs, atomic checkpoints, auto-resume
  and retry. Must be started from its own window — started from inside the IDE, it
  died with the IDE.

**Run:** 10 × 400 steps, batch 4 × accumulate 4, fp32, 51 min. Best val ridge **0.260**
(epoch 7, from 0.446), plateaued over the last three epochs. Orientation, segmentation and
confidence all learned; **the period head barely did** (6.2 px MAE on ~8–10 px periods).

**First real-data result — FVC2004 B, all four DBs, coverage 1.0**

  | method | EER [95% CI] | NFIQ 2 | minutiae/img | CPU ms |
  |---|---|---|---|---|
  | none | 0.0991 [0.088, 0.112] | 46.0 | 62.1 | — |
  | GBFEN | 0.0955 [0.084, 0.110] | 57.5 | 69.9 | 1054 |
  | SNFEN | 0.0895 [0.077, 0.101] | 59.2 | 66.1 | 1061 |
  | **WAFEN** | **0.1353 [0.124, 0.149]** | 57.3 | **92.8** | **576** |

- **Worse than no enhancement, intervals disjoint.** NFIQ 2 rises as much as SNFEN's while
  minutiae per image rise 50%: the network writes convincing ridges where evidence is
  gone, which looks better to a quality metric and matches worse. This is exactly the
  "invention becomes a minutia" failure the thesis argues about — and it is a
  synthetic-only model scored without its abstention mechanism.
- CPU inference 576 ms/image at 480×640 — just over budget; SNFEN's chain is 1,061.

**U9 abstention sweep — same checkpoint, confidence-gated, 8 px guard around declined pixels**

  | coverage | threshold | EER [95% CI] | minutiae/img | dropped/img | FTA |
  |---|---|---|---|---|---|
  | 100% | — | 0.1353 [0.124, 0.149] | 92.8 | — | 0% |
  | 85.2% | 0.15 | 0.1983 [0.183, 0.218] | 49.8 | 37.8 | 0% |
  | 68.4% | 0.20 | 0.4138 [0.393, 0.435] | 30.6 | 46.7 | 1.3% |
  | 47.8% | 0.30 | 0.3405 [0.307, 0.374] | 26.2 | 56.5 | **39%** |

- **Abstention makes it worse, monotonically.** The 0.30 row looks better than 0.20 only
  because 125 of 320 images failed to enrol, so its EER is on a survivor subset. It is
  not comparable with the others.
- **On real prints the confidence head is uncalibrated.** Median foreground confidence is
  0.29 (p10 0.14, p90 0.75). Declining 15% of the finger drops 41% of minutiae, so the
  declined pixels are speckled across the finger and the guard dilation removes real
  minutiae with the invented ones. If confidence marked the invented ridges, EER would
  fall as coverage falls. It rises.
- Reading: a confidence head supervised only by the *synthetic* wear model's evidence map
  does not transfer to real damage. That is the synthetic→real gap this thesis measures.
  It is a negative result for corrected synthesis *alone*, not a harness bug.
- Found and fixed a latent bug on the way: mindtct's default `.xyt` is **bottom-origin**
  (100% of minutiae on the finger read that way, 90.6% top-origin). `filter_minutiae`
  assumed top-origin. It was never used in a reported number; `filter_xyt_file` now
  flips it, with a test.

**Block-level abstention (16 px, t = 0.15 fixed in advance):** EER 0.1319 [0.118, 0.147],
coverage 89.9%, 71.5 minutiae/img. It undoes the pixel-level damage but does not beat
coverage 1.0 (0.1353) by more than noise. Abstention cannot rescue the ridge head. Closed.

**Hybrid — WAFEN mask + orientation → SNFFE → SNFEN (replaces SUFS and SNFOE):**
EER **0.3550** [0.339, 0.371], 103.5 minutiae/img, 902 ms/img. No convention bug: both
networks emit angles in (−π/2, π/2], and the result does not change mod π. On real prints
WAFEN's orientation is 11–15° (median) from SNFOE's and **fails near the core**, and its
mask edge is ragged. Both generate minutiae exactly where identity is decided.

**Where this leaves the model (honest):** no WAFEN variant beats *no enhancement* on real
prints. 51 minutes of training on synthetic data alone gives estimates clearly worse
than Cappelli's released networks. The fix is data, not post-processing: the model must see
real prints.

**Distillation, done — fine-tune on Anguli + 1,048 real teacher-labelled prints**
- Labelled FVC2000, FVC2002, U.are.U (1,160 prints, 0 skipped) with SUFS/SNFOE/SNFFE and
  SNFEN as ridge target. pyfing's SNFEN crashes at 569 dpi (image and period map
  rescaled to different sizes), so the teachers run at 500 dpi throughout.
- Fine-tune: 8 × 400 steps at 1e-4 from `best.pt`, real prints ×4. Real-val ridge
  0.399 → **0.342**, orientation 0.106 → 0.059.
- **FVC2004: EER 0.1347 [0.122, 0.149]**, against 0.1353 before. **No change.**
  Minutiae/img 93 → 80, NFIQ 2 59.5 (= SNFEN's 59.2), CPU 444 ms/img. Still worse than
  no enhancement (0.0991).
- Reading: lower loss against SNFEN's output did not buy matching accuracy. The prints
  look as good as SNFEN's to NFIQ 2 and carry 20% more minutiae than SNFEN's (80 vs 66).
  The loss rewards pixel overlap, and matching is decided by the ridge *endings* and
  *junctions* that pixel overlap barely weighs.

**Per-database breakdown, hybrid re-run, confidence blend — the closing set**

  | EER | none | SNFEN | WAFEN_ft |
  |---|---|---|---|
  | DB1 (optical) | 0.1142 | 0.1051 | 0.1509 |
  | DB2 (optical) | 0.1444 | 0.1449 | 0.2225 |
  | DB3 (thermal) | 0.0481 | **0.0278** | 0.0444 |
  | DB4 (SFinGe) | **0.0607** | 0.0661 | 0.1053 |

- **Even SNFEN beats "none" on only 2 of 4 databases** and loses on DB4. Enhancement's
  benefit is sensor-dependent at the state of the art — a genuine finding of the harness,
  and central to the "when does enhancement help?" story.
- WAFEN_ft reaches parity with none on DB3 (CIs overlap); worse everywhere else.
- Border-minutiae audit (finger-region distance, both polarities handled): WAFEN_ft and
  SNFEN are equally contaminated at the mask border (24 vs 21 per image); the gap is
  **interior** — ~56 vs ~45 interior minutiae/img, small ridge breaks/joins the loss
  barely weighs.
- Hybrid re-run with the fine-tuned checkpoint: 0.3550 → **0.2223**. Fine-tuning did fix
  the orientation (as diagnosed), but the line stays closed.
- **Confidence blend** (soft abstention, pre-committed, no threshold:
  conf·reconstruction + (1−conf)·original inside the mask): **0.1310** [0.118, 0.144]
  vs 0.1347 plain. Direction right, magnitude noise-level. 416 ms/img.

**Assessment after 13 FVC2004 evaluations, all in the registry**
- No WAFEN variant beats the unenhanced control on aggregate. The search stops here —
  further variants would be test-set fishing, and every run is already logged.
- What stands: (1) SOTA reproduced under one harness, per-sensor; (2) enhancement gains
  are sensor-dependent even for SNFEN; (3) NFIQ 2 improves while EER worsens — three
  separate demonstrations that quality metrics do not predict matcher accuracy;
  (4) WAFEN matches SNFEN's NFIQ 2 (59.5 vs 59.2) at 2.4× lower CPU cost in one pass;
  (5) the synthetic→real gap measured end to end — the thesis's comparison framing
  (STRATEGY.md §1) was chosen to be robust to exactly this outcome.
- The one remaining *research* lever, not tried: a minutia-aware loss (weight ridge
  endings/bifurcations of the teacher's map heavily). Diagnosed cause fits; cost ~2 h;
  outcome uncertain. Decision deferred to the supervisor conversation.

**Minutia-aware loss (the last lever) — trained as designed, did not transfer**
- New loss term: L1 on the ridge map inside 7 px disks over the target's skeleton
  endpoints and bifurcations (crossing number), border artifacts excluded, weight 2.0 —
  the heaviest in the objective. Fine-tuned 8 × 400 from wafen_ft.
- It learned what it was told to: val minutiae-site error 0.276 → 0.226 (−18%), val
  ridge 0.355 → 0.291 — both better than any previous run, no trade-off.
- **FVC2004: EER 0.1376 [0.126, 0.148].** Within noise of plain wafen_ft (0.1347) and
  the blend (0.1310). Minutiae/img 83.4. The pre-committed criterion (beat 0.1310 or the
  lever is spent) says: spent.
- The pattern across all 14 evaluations is now consistent and worth stating as a
  finding: **every proxy we optimised improved — NFIQ 2, pixel overlap, minutiae-site
  L1 — and the matcher moved for none of them.** What bozorth3 rewards is not per-image
  fidelity to the teacher but *cross-impression consistency* of the reconstruction, which
  none of our losses (or any per-image loss) sees. That is a diagnosis with a citation
  trail, not an excuse: it names the requirement for any future attempt (pair-consistent
  training) and closes the corrected-synthesis arm with a mechanism, not a shrug.

**Pair-consistent training — the diagnosis's own experiment, and the arm's last word**
- Each item became two independently-damaged, pixel-aligned views of one print, with a
  masked L1 between the two reconstructions (weight 1.0, fixed in advance). Fine-tuned
  8 × 400 from wafen_mn.
- It worked as training: cross-view disagreement 0.1253 → 0.1068 (−15%), and the best
  real-val ridge (0.2680) and minutiae-site error (0.2073) of any run — the consistency
  term helped the supervised objectives rather than trading against them.
- **FVC2004: EER 0.1359 [0.121, 0.149].** Statistically identical to 0.1347 / 0.1376 /
  0.1310. Pre-committed criterion: the corrected-synthesis arm is closed.
- **The complete negative result, now fully characterised.** Four distinct objectives were
  each improved substantially — pixel overlap, NFIQ 2, minutiae-site error, and
  cross-view consistency under our wear model — and the matcher moved for none of them.
  The last one sharpens the diagnosis once more: consistency was trained over *our wear
  model's* variation, and the EER says real cross-impression variation (pose, pressure,
  skin state, sensor contact) is not spanned by it. The degradation model itself is the
  ceiling — which is the thesis's central question answered, not dodged: corrected
  synthesis fails on real prints *because the correction is itself synthetic*.
- Future-work section, now with two named requirements instead of hand-waving:
  (1) consistency across *real* impression pairs, which needs registration
  (alignment) machinery we do not have; (2) degradation models validated against real
  cross-impression variation, not only against single-image realism (U3 measured the
  latter and the wear model won it — this result shows why that was not enough).

**Next — teacher distillation on real, non-test prints**
- Label real prints from the *non-test* datasets (FVC2000/2002, Neurotechnology, MINEX,
  L3-SF; FVC2004 stays held out) with the pyfing teachers (SUFS, SNFOE, SNFFE, and SNFEN
  output as the ridge target) using the existing supervision pipeline. Fine-tune WAFEN on
  those plus Anguli.
- Target: match SNFEN's EER at one forward pass (~0.5 s against 1.06 s). That is the
  thesis's four-networks-into-one claim, and it is testable.
- Leakage check before running: no FVC2004 image or derivative in training.
  If precision is the problem, EER should fall as coverage drops. Cheap — no retraining.
- Check where the extra minutiae fall (low-confidence regions or not) before tuning.
- Only then consider a longer run or a higher period weight.

---

## 2026-09-15 — Session 9: Phase B complete, GPU unblocked, canary passes

**Where the plan stands**
- **U1–U8 done.** Phase A (week 2) and Phase B (week 3) of the implementation plan are
  complete. 183 tests. Branch `feat/wear-degradation-model`, `main` untouched.
- Remaining: U9 (minutiae precision + precision–coverage curve), U10 (ablations),
  U11 (paired bootstrap + LEADER), U12 (full benchmark).

**The milestone: the wiring is proven**
- The over-fit canary passes on real cached data — **ridge loss 0.1457** against a 0.35
  threshold, in under a minute on the GPU. The whole chain (cached targets → augmentation →
  wear degradation → five-headed network → five losses → optimiser) is correct end to end.
  The same check took over an hour on CPU.

**The GPU works — three separate blockers, three different causes**
- torch 2.14 will not load: the x64 VC++ 2015-2022 runtime here is **14.32.31332** and
  recent torch needs ≥14.40 (the *x86* redist is already 14.44; only x64 is stale).
- The cu121 builds need driver ≥525; this machine has **517.48** (2022), which supports
  CUDA 11.x only.
- **torch 2.5.1+cu118 satisfies both** and reports CUDA available on the GTX 1650. No driver
  update needed. Updating the x64 redist remains a deferred nice-to-have.

**Two regressions that appeared because the project got *faster***
- Installing CUDA torch broke pyfing everywhere — the supervision build *and* the
  GBFEN/SNFEN baselines — without a line of our code changing. Keras places pyfing's models
  on whatever device it finds and pyfing calls `.numpy()` straight on the output, which
  fails on a CUDA tensor.
- The first fix failed because **Keras resolves its torch device at *import* time** from
  `KERAS_TORCH_DEVICE`, so no context manager entered after `import pyfing` can help.
  `fpe.models.pyfing_runtime.load_pyfing()` now owns the import and sets the variable
  first; every call site routes through it and no direct `import pyfing` remains. Pinning
  pyfing to CPU is right regardless — teachers and baselines are cached one-offs at
  ~0.7 s/image, and 4 GB of VRAM is wanted for training. Verified our own network still
  runs on the GPU with pyfing pinned.
- The parameter budget was checked *after* construction, so an absurd config exhausted
  memory while building and surfaced as an allocation failure rather than the scope
  decision it is. Now estimated from the config and refused before allocating.

**U2 — supervision build**
- Resumable by atomic rename: a cache file exists only once complete. Proven in practice —
  the build was killed and relaunched mid-run and resumed at 442 without losing anything,
  and it survived a session restart.
- **The atomicity test paid for itself immediately**: `np.savez_compressed` *appends* `.npz`
  to any path lacking it, so writing `x.npz.tmp` produced `x.npz.tmp.npz` and the rename
  found nothing. Without that test the "resumable" build would have written zero files.
- Splits by blake2b hash of the finger id, not the built-in `hash()` whose seed is
  randomised per process. 4,418 / 560 / 606 fingers; no finger crosses a split.
- **Two plan assumptions were wrong.** The identity parser did *not* handle the Anguli
  layout (files are `5.png` with the impression in the *directory*), so every manifest row
  parsed to `None` and would have been silently dropped. And the test split yields
  **1,649,835 impostor pairs** — the better part of an hour per evaluation, and U9 sweeps
  many. Added seeded impostor subsampling drawing uniformly over *pairs*, not finger pairs.

**U3 — realism, measured**

  | arm | KS vs real | NFIQ 2 | classifier accuracy |
  |---|---|---|---|
  | latent | 0.1593 | 45.1 | 0.823 [0.81, 0.83] |
  | **wear** | **0.1313** | 53.9 | **0.708 [0.70, 0.72]** |
  | real | — | 50.3 | — |

- Wear is closer on both, intervals disjoint. Both remain well above chance, so neither
  model is indistinguishable from real damage — reported as the limitation it is (AE4).
- **The first design was worthless and was thrown away.** Degrading Anguli and classifying
  against real degraded prints scores 0.95 — but the control, clean Anguli vs clean FVC with
  *no degradation*, scores **1.000**. It measured the corpora, not the models. The rebuilt
  version never crosses corpora: real prints split by measured NFIQ 2 quality *within each
  sensor*, clean half degraded, tested against the genuinely degraded half.
- Two guards came out of it. The cross-validated accuracy is biased *below* chance when
  samples are few relative to features (30/class scored 0.317 on identical distributions),
  and below-chance is the *flattering* direction — it now refuses rather than returning it.
  And **NFIQ 2 silently fails on Neurotechnology's palette TIFFs**, which had dropped all
  928 images from the pool without a word.

**U5–U8 — the network**
- **5,462,710 parameters** against the 10M budget. CPU inference **309 ms** at 256×256 and
  **478 ms** at full Anguli resolution, against a 500 ms budget and the ~1,050 ms that
  Cappelli's four-network chain costs here.
- Orientation predicted as (cos 2t, sin 2t); ridge loss is Tversky α = 0.7. A test asserted
  that adding 16 false pixels and erasing 16 true ones cost the same at α = 0.5 — they do
  not, because erasing also reduces true positives. Replacing it exposed the sharper
  property: **α = 0.7 must *reverse* which error is worse**, and it does.
- Abstention writes background where confidence is low and drops minutiae landing there;
  coverage is measured over the foreground so a tight crop cannot claim coverage for free.

**Throughput**
- Dataset was 239 ms/item, which would have starved the GPU. The wear model's oriented-noise
  bank was a **91×91 kernel applied 8 times**; it now runs at 0.4 scale with correlation
  lengths scaled to match. Wear 163 → 67 ms, dataset 239 → **101 ms**, epoch 11 → **5 min**.
  Realism revalidated afterwards: unchanged.
- The frequency teacher emits **negative ridge periods** where it could not estimate. Marked
  invalid and excluded from the loss rather than clamped — clamping would invent a number
  and train the network to reproduce it.

**Running / next**
- Supervision build detached at **~11,250/16,310**, ~70 min left. If it dies, just re-run
  `.venv/Scripts/python.exe scripts/build_supervision.py` — it skips what is built.
- **The training run has not been started.** ~5 min/epoch × 25 epochs ≈ 2 h on the GPU:
  `.venv/Scripts/python.exe experiments/train_wafen.py --epochs 25`
- Then U9: minutiae precision on Cappelli's protocol, and the precision–coverage curve.
- Still unstarted, still longest lead time: CASIA registration, IAB licence paperwork.

---

## 2026-09-11 — Session 8: baselines measured, plan written, wear model built

**Baselines — the 2026 SOTA, run by us**
- `pyfing` installed and running on the **PyTorch** backend. All five pretrained models
  (SUFS, SNFOE, SNFFE, SNFEN, LEADER) ship in the wheel, so baselines needed no downloads
  and no training.
- Full four-database FVC2004, 1,120 genuine pairs, sensor-matched impostors:

  | Method | EER | 95% CI | Minutiae/img | NFIQ 2 |
  |---|---|---|---|---|
  | No enhancement | 0.0991 | [0.0876, 0.1117] | 62.1 | 46.0 |
  | GBFEN | 0.0955 | [0.0837, 0.1102] | 69.9 | 57.5 |
  | SNFEN | **0.0895** | [0.0770, 0.1010] | 66.1 | 59.2 |

- **The ordering reproduces Cappelli's published ranking on a benchmark he never used**,
  since SD27 is unobtainable. That is an independent replication and it is ours to cite.
- **Nothing is resolved**: every interval overlaps the control. Diagnosed as *my* error, not
  the data's — all conditions score the **same** pairs, so comparing independent bootstrap
  CIs is under-powered. A paired bootstrap on the EER difference is now planned (U11).
- The cross-column pattern is the thesis argument, measured on our own benchmark: NFIQ 2
  climbs hard (46.0 → 57.5 → 59.2) and minutiae counts rise 6–13%, while EER moves inside
  noise. Quality gain is not matching gain; extra minutiae buying no accuracy is what
  spurious minutiae look like. SNFEN adds *fewer* minutiae than GBFEN and scores better.
- Deployment number: **~1.05 s/image on CPU** for the four-network chain, against a 500 ms
  budget. Direct evidence for collapsing four networks into one.

**Two environment findings**
- **torch 2.14 will not load here.** The x64 VC++ 2015-2022 runtime is 14.32.31332; recent
  torch needs ≥14.40. Oddly the *x86* redist is already 14.44 — only the 64-bit one is
  stale. torch 2.5.1 works against what is installed. Updating needs admin; deferred.
- pyfing calls `.numpy()` on model outputs, which raises under torch because tensors carry
  `requires_grad`. Every call wrapped in `torch.no_grad()`.

**Correctness fix in the pairing protocol**
- `build_pairs` allowed impostor pairs **across** FVC databases — DB1's optical sensor
  against DB3's thermal sweep. Trivially separable for reasons unrelated to identity, which
  depresses EER without any method improving: **38,400 of 49,920** pairs on a full FVC2004
  run. Impostors are now sensor-matched by default. Single-subset rows already written are
  unaffected.

**Implementation plan (weeks 2-5)**
- `docs/plans/2026-09-11-001-feat-wear-aware-abstaining-enhancement-plan.md`. Twelve units,
  four phases, traced to the origin requirements. Written *after* week 1, so it rests on
  measured facts rather than assumptions.
- **R4's premise was false.** Anguli exports only pattern type and singular points — no
  orientation or frequency fields. Resolved by distilling from Cappelli's pretrained
  estimators run on the **clean** impression. Honest cost, to be stated in the thesis: two
  heads are then supervised by another model's output, not human annotation. The ridge-map
  and confidence heads, which carry the claims, are unaffected.
- **The Anguli master is binary** at 275×400 — already the near-binary target form Cappelli
  builds by hand — and impressions are greyscale and **pixel-aligned** with it. So: degrade
  the greyscale impression as input, reconstruct the binary master. No registration.
- Decided with the user: R17 satisfied by varying the **extractor** (LEADER vs mindtct)
  rather than the matcher, since spurious minutiae originate in extraction; a flat
  precision–coverage curve is reported as falsification, not pivoted around.
- Self-review caught two gaps, both fixed before commit: no justification for training from
  scratch rather than warm-starting from SNFEN's released weights (rejected — incompatible
  stem, single-head decoder, and it would contaminate the training-data ablation), and
  nothing produced an **Anguli manifest**, without which U7 could not evaluate a trained
  model and U9 had no pairs to sweep.

**U1 done — wear degradation model** (branch `feat/wear-degradation-model`)
- `src/fpe/degradation/wear.py`: ridge amplitude attenuation, flexion creases, dryness
  fragmentation, pressure-dependent partial contact. Parallel in shape to
  `latent.py` so the two arms of the central comparison are interchangeable.
- **Two properties hold by construction, not by luck.** Severity 0 is exactly the identity,
  and damage is monotone in severity for a fixed seed — because every random quantity is
  sampled at unit scale *before* severity is applied. Crease positions are pre-sampled and
  severity selects a prefix; fragmentation thresholds a fixed noise field so break area only
  grows.
- Fragmentation is **anisotropic** via a steerable filter bank aligned to structure-tensor
  orientation. Dry skin breaks a ridge into dashes *along* its length; round blobs are the
  common shortcut and are wrong. The test discriminates — an isotropic config scores aspect
  0.97 against the required 1.3.
- **Foreground mask support came from reading the figure, not the plan.** Without a mask the
  model drew creases and breaks across blank background and reported destroyed evidence
  where no ridge ever existed — supervision that would have taught the confidence head to
  distrust empty paper.
- 33 tests. `results/figures/wear_degradation_severity.png` shows evidence falling
  1.00 → 0.23 across the sweep.

**Next**
- U2: supervision build (pyfing teachers over 5,584 fingers), splits by finger id, and the
  Anguli manifest. Long single-threaded pass — make it resumable before starting it.
- U3: realism validation and the three-way degradation figure.
- Branch `feat/wear-degradation-model` is unmerged; `main` is untouched.
- Still unstarted and still the longest lead time: CASIA registration, IAB licence paperwork.

---

## 2026-09-11 — Session 7: two-month reframe, dataset dossier, NBIS built, harness live

**The deadline changed the thesis**
- Two months to **final submission, thesis document included** — not eight. That is roughly
  five build weeks and three writing weeks. The four-contribution plan does not fit.
- Re-scoped to **one coupled contribution**, written up in
  `docs/brainstorms/2026-09-11-wear-aware-abstaining-enhancement-requirements.md`:
  a single network estimating segmentation, orientation, frequency, ridge map and per-pixel
  confidence, supervised throughout by our own wear degradation model.
- The coupling *is* the argument: because we synthesise the degradation, we know per pixel
  how much ridge evidence was destroyed, which is a direct supervision target for the
  confidence head. No method that inherits its degradation from elsewhere can train it.
  *You can only abstain honestly if you know what was destroyed.*
- Architecture is deliberately unremarkable, per the literature's own conclusion. Novelty
  sits in what the network estimates and what supervises it. Collapsing Cappelli's four
  networks into one is also a *simplification* that serves the 500 ms CPU budget.
- **Cut:** multi-impression fusion, the domain-alignment and unpaired-translation comparison
  arms, demographic stratification. Approach A (four-stage pipeline, only the enhancer
  replaced) is a **contracted fallback at the end of week 4**, not an aspiration.
- Headline metric is matcher EER on real prints with bootstrap intervals; minutiae precision
  against synthetic pseudo-ground-truth is supporting and labelled as such.

**SOCOFing dropped**
- Supervisor guidance: no Kaggle datasets for training. SOCOFing was the only Kaggle-sourced
  set in the corpus. Our own measurement agrees independently — 96×103 px declared as
  500 dpi describes a 4.9 mm finger, so true sampling is nearer 200 dpi.
- Cost: the damage-severity axis and the only demographic label. Severity is recovered, and
  improved, by applying our own wear model at controlled severities to real clean prints —
  known parameters instead of a vendor's undocumented easy/medium/hard. Demographic
  stratification has no substitute; **CASIA registration is now the highest-priority
  outstanding request.**

**Dataset dossier for the supervisor**
- `docs/datasets/DATASET_DOSSIER.md` + `.pdf` (6pp). Requested as a gate before project work
  continues: what was acquired, what was not, links and descriptions for both.
- Figures are script-produced from the committed manifests (`scripts/dataset_dossier_stats.py`),
  which caught two wrong published specs — SOCOFing's dpi above, and L3-SF's advertised
  1200 dpi applying to only 740 of its 8,140 images.
- `scripts/render_pdf.py` renders any project markdown to PDF via markdown-it-py plus
  headless Chrome. No LaTeX, no new dependencies.

**NBIS built from source — the critical path is clear**
- This machine had **no C compiler at all**. Provisioned portable winlibs MinGW-w64 GCC 16.2,
  cmake 3.31.6 from pip, and a `make` shim, all under `E:	oolchains` — nothing on C:,
  nothing installed system-wide.
- NBIS 5.0.0 needed **four non-obvious patches**, all recorded in `docs/nbis-build.md`:
  CMake must be 3.31 (4.x dropped `cmake_minimum_required(2.6)`); `setup.sh` must use
  `pwd -W` because native make cannot resolve Git Bash's `/e/...`; `-fpermissive -fcommon`
  because GCC 14 made implicit declarations errors and GCC 10 made `-fno-common` the
  default; and `make-depend`'s sed strips to the first colon, which on `E:/...` is the drive
  letter, producing `.d` files make rejects.
- **mindtct reads none of the formats our datasets ship in** — no TIFF, BMP or PNG — and
  cannot size a headerless raw file. WSQ was rejected as lossy; putting lossy compression
  under every number in the thesis is not acceptable when `cjpegl` exists. Lossless JPEG
  round-trip verified **byte-identical**, all 307,200 bytes of an FVC2004 image.

**Harness live, Phase 0 gate closed**
- `fpe.eval.registry` (git SHA + config hash + manifest hash per row), `fpe.eval.protocol`
  (identity parsed from filenames — the manifests do not carry it for FVC, Neurotech or
  FVS), `fpe.metrics.matching` (EER + bootstrap CI), `fpe.metrics.nbis`, `fpe.metrics.nfiq2`,
  `fpe.eval.baseline`, and `experiments/baseline_nbis.py`.
- EER implementation verified against the analytic Φ(−d/2) to within 0.0006.
- First two registry rows, and the evidence the harness measures what it claims:

  | Dataset | EER | 95% CI | NFIQ 2 mean | FTA |
  |---|---|---|---|---|
  | FVC2004 DB1_B (dry / distorted) | 0.1142 | [0.089, 0.141] | 56.0 | 0% |
  | Neurotech CrossMatch (clean) | 0.0066 | [0.004, 0.011] | 73.4 | 0% |

  A 17× gap in the expected direction with quality tracking it. 83,028 pairs matched.

**Findings worth keeping**
- **Bootstrap intervals are not optional at this corpus size.** An FVC "B" subset yields 280
  genuine pairs; the interval on a 0.16 EER spans ±2.2 points. A point estimate would be
  misleading.
- Impostor pairing uses all cross-finger impression pairs, not the official FVC
  first-impression rule, which would give 45 pairs per database. Documented deviation.
- `pyfing` is **Keras 3**, so it runs on the PyTorch backend — the TF-2.10 Windows-GPU dead
  end in `STRATEGY.md` §5 is avoidable and WSL2 is not needed. It also ships **LEADER**, an
  end-to-end minutiae extractor.

**Next**
- Second independent matcher for R17 — SourceAFIS or MCC — still unchosen and still a
  resolve-before-planning question.
- Install `pyfing`; run GBFEN and SNFEN as baselines through this harness.
- Baseline rows for the remaining real sets (FVC2000/2002, U.are.U, FVS, MINEX).
- Wear degradation model and its label export — week 2 of the plan.
- CASIA registration; IAB licence paperwork still unstarted.

---

## 2026-09-10 - Session 6: machine ran out of memory; corpus reconciled

**What broke**
- Pylance kept dying and the terminal kept crashing. Root cause measured, not guessed.
  Two compounding causes:
  1. **7.8 GB of RAM against ~211,000 workspace files** (`data/` alone is 196,000 --
     SOCOFing extracts to 110k, Anguli's filter bank to 40k). VS Code's file watcher and
     Pylance were indexing all of them.
  2. **The pagefile lives on C:, which has 16.6 GB free while the pagefile is 24 GB
     allocated and 21 GB in use.** Windows-managed, so it cannot grow. E: has 469 GB free
     and is where every dataset already lives. The user spotted this.
  The same exhaustion had already failed git's credential helper mid-push ("Not enough
  memory resources") and kept killing the IDE MCP server.
- Dropbox ruled out: it syncs only the user's Dropbox folder on C:, never the project on E:.

**Fixed**
- `.vscode/settings.json` -- excludes `data/`, `.venv/`, `vendor/`, `tools/bin/` from
  `files.watcherExclude`, `python.analysis.exclude` and `search.exclude`; git
  autorefresh/autofetch off (status walks are slow with 200k ignored files).
  **These excludes are load-bearing; the file says so.**
- Same file fixes a second Pylance complaint that was real rather than memory-related:
  `python.analysis.extraPaths: ["src"]`, without which every `from fpe...` import reads as
  unresolved, since the package lives in `src/` and scripts add it to `sys.path` at runtime.
  Interpreter pinned to `.venv`.
- Constraint recorded in `PROJECT_RULES.md` (new non-negotiable 6) and `STRATEGY.md` 2.3, with the
  risk-register row marked **materialised**: cap generation at 4 threads while the editor is
  open, dataloader workers 0-2, stream rather than bulk-load.
- `docs/environment-tuning.md` -- the pagefile move to E: (needs admin + reboot) and
  redirecting temp off C:.

**Corpus: interrupted, reconciled, renamed**
- The 20,000-finger run died with the session at **5,584 complete fingers**. One finger
  (`fp_6/5745`) was torn -- master and Impression_1 written, Impression_2/3 and metadata not.
  Training on that would silently pair a clean image against a missing counterpart.
- `scripts/reconcile_anguli.py` intersects the finger ids present across Fingerprints /
  every Impression_N / Meta Info, reports and optionally prunes torn records, and writes the
  `generation.json` provenance the killed run never wrote (`partial_run: true`).
  Interrupted runs will happen again on this machine.
- Renamed `anguli_20k` -> **`anguli_dev_5584`**, because a directory called 20k holding
  5,584 fingers is a trap for anyone reading it in month eight. Usable: **5,584 fingers,
  22,336 images, 3 impressions each.**

**Judgement**
- Not resuming to 20,000 for now. SNFEN was trained on 360 images and beats FingerGAN's
  130,000, so 5,584 paired fingers is ample for every Phase 1 baseline. Scaling up is one
  4-thread overnight run *if* a measurement shows it is needed.

**Next**
- **Reload the VS Code window** so the excludes take effect (read at window start).
- Move the pagefile to E: per `docs/environment-tuning.md` -- needs admin and a reboot.
- Install NFIQ 2 + NBIS to close the Phase 0 gate.
- Still the critical path, still not started: IAB licence (Registrar's signature) and
  NIST SD302. Drafts ready in `docs/correspondence/`.
- CASIA registration outstanding; walkthrough in `docs/datasets/casia-access.md`.

---

## 2026-09-10 — Session 5: ChaLearn dead, baseline arm built here instead

**Blocked, permanently**
- **ChaLearn is unrecoverable.** The user exhausted login, password reset and repeated
  sign-up across several addresses; all fail. **ADR 0003 triggered the same day.** The email
  to Sergio Escalera is still worth sending and may yet recover the original data, but
  Phase 1 no longer waits on it.

**Correction to my own earlier work**
- The `chalearn-like` Anguli preset I wrote in session 4 **overstated what it produced.**
  ChaLearn applies **nine** artefact types — blur, brightness, contrast, elastic transform,
  occlusion, scratches, resolution reduction, rotation, and **compositing onto background
  textures**. Anguli's flags cover roughly four and cannot composite backgrounds at all.
  Preset renamed `anguli-noise`; ADR 0003 amended.

**Done**
- `fpe.degradation.latent` — the real nine-artefact recipe, in our own code. Seeded, and it
  returns a per-image record of which artefacts fired with which parameters, so any sample
  can be explained afterwards. Verified on FVC2004 DB1_B: all nine types fire, same seed
  reproduces exactly, different seed differs.
- `fpe.degradation.backgrounds` — DTD texture bank (5,640 images, 47 categories; the source
  SFP used). Downloaded and extracted, 625 MB.
- Division of labour now: **Anguli supplies clean masters plus acquisition variation across
  impressions; all degradation happens in our code.** Better for a thesis whose central
  claim is about degradation models — "what was applied" must be inspectable, not split
  across a closed binary and a script.
- `results/figures/latent_degradation_examples.png` — the visual audit, and the thesis
  figure for the control condition. The wear-model figure will reuse the layout so the two
  read side by side. Output looks correctly latent-like: prints on visible surfaces, partial
  contact, scratches cutting through ridges.
- Started the 20,000-finger x 3-impression corpus (`--preset clean`, seed 20260910) as a
  background run into `data/processed/anguli_20k`.

**Findings worth keeping**
- **Cappelli's leakage criticism is answered structurally, not procedurally.** DTD textures
  were never fingerprints, so no fingerprint test image can leak through them; and DTD is
  partitioned **by category** via a filename hash, so a training image cannot receive a
  texture from the same family as a test one. `split="test"` cannot return a training
  texture — it is not a flag someone can forget to pass. Verified disjoint at both category
  and file level: 35/8/4 categories = 4,200/960/480 textures, all 5,640 assigned.
- **Corpus size deliberately capped at 20,000 fingers, not 84,000.** Cappelli's SNFEN, trained
  on 360 images, beats FingerGAN trained on 130,000. Spending 22 h of wall clock to match a
  corpus size whose numbers we cannot compare against anyway (no ChaLearn access) is poor
  value. 20,000 pairs plus fusion triplets covers every Phase 1 need; scaling up later is
  one overnight run if a baseline demonstrably requires it.
- ChaLearn published artefact *types* but not their parameter distributions, so the severity
  ranges in `LatentConfig` are ours. Another reason exact comparability is not claimed.

**Next**
- Send the Escalera email (drafted); CASIA registration still outstanding.
- Group C emails still to draft: PrintsGAN, Tsinghua SD27 annotations, FVC segmentation GT.
- IAB licence (Registrar's signature) and NIST SD302 — **still the critical path, still not
  started.** Every other blocker so far has had a workaround; these two do not.
- Install NFIQ 2 + NBIS to close the Phase 0 gate.

---

## 2026-09-10 — Session 4: ChaLearn blocked, contingency armed and verified

**Blocked**
- **ChaLearn registration is broken on their side.** Sign-up returns HTTP 500; retrying the
  same address then reports the email is not unique. Several addresses tried, all identical.
  That is the classic Django failure where the user row commits before a failing
  confirmation-email step raises — so the accounts probably exist but are unusable.
  File lists are login-gated, so the archive URLs cannot be discovered anonymously.

**Done**
- Diagnosed the failure and identified two self-service recovery routes for the user:
  log in with the **username** (the form field is `username`, not email) from the attempt
  that errored, or use the working reset page at `/password_reset/`.
- Drafted the access request to Sergio Escalera: `docs/correspondence/chalearn-data-access.md`.
- **Armed and verified the regeneration contingency** (ADR 0003). Anguli produces
  **275×400 8-bit greyscale — exactly ChaLearn's geometry**, confirming ChaLearn was built
  with it at defaults. Wrote `scripts/generate_anguli.py` with a `chalearn-like` preset
  mapping the published artefact list onto Anguli's noise/scratch/rotation/translation flags.
- Measured throughput: **1.41 fingers/sec, 4.2 images/sec on 8 threads.** A ChaLearn-equivalent
  84,000-pair corpus is a **~17 h overnight run**; a 20,000-pair dev corpus ~4 h. Generation
  is not a bottleneck. (The apparent slowness of a 4-finger test was startup overhead.)

**Findings worth keeping**
- Regeneration is in three ways *better* than the download: seeded and provenance-recorded
  (`generation.json` beside the output), **multiple impressions per finger** (`-ni`) which
  Contribution 4 needs and ChaLearn does not provide, and pattern-class labels (`-meta` gives
  class plus singular-point coordinates) — a free stratification axis.
  The cost is real though: exact numerical comparability with published ChaLearn numbers is
  lost, and the evaluation chapter must say so.
- Three Anguli traps, none documented upstream: it resolves `Filterbank/` and `Densitymaps/`
  relative to the **cwd**, not the executable (`Error: Not able to load Filter Bank`
  otherwise); its **default image type is jpg**, whose compression would corrupt ridge detail;
  and it will not create its own output directory. All three handled in the wrapper.

**Next**
- User: try the two ChaLearn recovery routes; send the Escalera email; CASIA registration.
- Deadline: if ChaLearn is not resolved by end of Phase 1 (week 8), ADR 0003 triggers.
- Group C emails still to draft: PrintsGAN, Tsinghua SD27 annotations, FVC segmentation GT.
- IAB licence (Registrar's signature) and NIST SD302 — still the critical path, still not started.

---

## 2026-09-10 — Session 3: SOCOFing in, Group B access mapped

**Done**
- **SOCOFing downloaded and manifested: 55,270 images, 811 MB.** A Kaggle token was already
  present on this machine and still valid, so this needed nothing from the user. Counts match
  the published spec exactly (6,000 real + 17,931 easy + 17,067 medium + 14,272 hard).
- `scripts/download_group_b.py` — automates SOCOFing; prints exact next steps for the two
  datasets that genuinely cannot be scripted.
- Manifest now parses SOCOFing's filename labels into columns.

**Findings worth keeping**
- **SOCOFing carries the entire stratification grid we need, for free.** All 55,270 filenames
  parse: 600 subjects, gender (M 44,203 / F 11,067), hand, 5 finger types, 3 damage types
  (obliteration / central rotation / z-cut) x 3 severities. This is Gap D's demographic axis
  and the severity axis in one dataset. **Caveat for the fairness chapter: gender is 80/20
  male-skewed**, so female-subgroup estimates will have wide confidence intervals — report
  intervals, not point estimates.
- **The Kaggle upload ships the whole dataset twice** — a byte-identical nested
  `SOCOFing/SOCOFing/` copy (110,540 files extracted, 55,270 real). Verified identical by
  per-file hash. Excluded in the manifest rather than deleted, so `data/raw` stays faithful
  to the archive and the manifest is reproducible on any machine.
- **ChaLearn is login-gated at the file-list level.** Confirmed: `/dataset/32/data/55/files/`
  (Training), `/56/` (Validation), `/57/` (Test) all redirect to `/login/`. The archive URLs
  cannot be discovered anonymously, so a free account is genuinely required.
- **idealtest.org (CASIA) is a JavaScript portal with an expired TLS certificate.** Alive but
  unscriptable; browser download only.

**Next**
- User: ChaLearn account + CASIA registration (SOCOFing is done).
- Group C emails: PrintsGAN agreement, Tsinghua SD27 annotations, FVC segmentation GT.
- IAB licence (Registrar's signature) and NIST SD302 — still the critical path, still not started.
- Install NFIQ 2 + NBIS; verify `pyfing`'s DL backend.

---

## 2026-09-10 — Session 2: Group A datasets downloaded

**Done**
- Downloaded and manifested all Group A datasets: **10,997 images, 952 MB** across 8 sets
  (FVC2000/2002/2004 "B" x4 each, Neurotechnology CrossMatch + U.are.U, FVS, MINEX III
  validation, L3-SF). Every published dimension matched the source spec exactly.
- Anguli synthetic generator fetched: Windows binary (`Anguli.exe`) **and** source (Qt
  `.pro`), so the WDM can be built on top of it rather than beside it.
- `scripts/download_group_a.py` — checksummed, resumable-by-cache, refuses to clobber.
- `scripts/make_manifest.py` + `src/fpe/data/minex.py` — per-image sha256, dimensions and
  declared dpi into committed CSVs; manifest hash identifies dataset content as a whole.
- Created `.venv` with numpy/scipy/opencv/scikit-image/pandas/matplotlib/pillow/gdown.

**Findings worth keeping**
- **MINEX ships per-image labels.** Its raw `.gray` files are headerless, but the validation
  driver's C++ header carries width, height, quality band and **finger position** for all
  801 images (54 subjects, 8 positions — no thumbs; 132 images labelled poor/fair). That is
  a ready-made finger-type stratification axis for the benchmark, on permissively licensed
  data. Parser in `src/fpe/data/minex.py`.
- **L3-SF's 1200 dpi applies to only 740 of its 8,140 images.** The pore-annotated subset is
  740 x 512x512 with TSV pore coordinates (x, y); the main set is 7,400 x 320x240 and cannot
  be 1200 dpi (that would be a 0.27 x 0.20 inch finger). Main-set dpi recorded as unknown
  rather than guessed — to be measured from ridge period.
- FVS and L3-SF are stored as RGB despite being greyscale content; loaders must convert.
- Two hosts have expired TLS certificates (`fvs.sourceforge.net`, `dsl.cds.iisc.ac.in`).
  Python `urllib` refuses them; `curl` accepts. Noted in the checksum record.

**Two bugs found and fixed in my own tooling**
- `.gitignore`'s bare `datasets/` rule matched `docs/datasets/` at any depth, silently
  excluding the dataset registry from the first commit. Anchored to `/data/`, `/datasets/`.
- The FVC "B" archives share internal filenames (`101_1.tif`, ...), so extracting four into
  one directory left only 80 of 320 images. Each archive now gets its own subdir, and
  extraction refuses to overwrite an existing file rather than silently losing data.

**Next**
- Group B accounts (Kaggle token, ChaLearn registration, idealtest.org) — needs the user.
- Group C emails: PrintsGAN agreement, Tsinghua SD27 annotations, FVC segmentation GT.
- IAB licence paperwork (Registrar's signature) and NIST SD302 request — still not started,
  and still the critical path.
- Install NFIQ 2 + NBIS; verify `pyfing`'s DL backend (decides native Windows vs WSL2).

---

## 2026-09-10 — Session 1: repository and strategy

**Done**
- Read the thesis plan, the literature review and the bibliography.
- Initialised the repository; wrote `STRATEGY.md`, `PROJECT_RULES.md`, `README.md`,
  `docs/datasets/REGISTRY.md`, ADR 0001 (record decisions) and ADR 0002 (harness first).
- Surveyed the environment: Python 3.11, no conda, GTX 1650 (4 GB), `gh` CLI not installed.

**Decided**
- Thesis framed as a **controlled comparison** of the three literature responses to the
  synthetic/real degradation mismatch — corrected synthesis (ours), post-hoc domain
  alignment (Joshi et al.), unpaired translation (Karabulut et al.) — rather than as an
  attempt to beat SNFEN on EER. Robust to a negative model result. (`STRATEGY.md` §1)
- **Evaluation harness is built first**, not in Phase 4. (ADR 0002)
- No phase may block on a licence; the Rural Indian DB is a confirmatory test set, with free
  substitutes named for every dependency. (`STRATEGY.md` §2.2)
- 4 GB GPU turned into a design goal: <= 10M parameters, <= 500 ms CPU inference.

**Blocked / open**
- GitHub remote not yet created — `gh` CLI absent.
- No datasets downloaded yet; no licence requests sent.

**Next**
- Start the IAB licence paperwork (needs Registrar's signature — longest lead time).
- Submit the NIST SD302 request.
- Download ChaLearn, SOCOFing, FVC "B" subsets.
- Install NFIQ 2, NBIS, pyfing; verify end-to-end on one FVC image.
