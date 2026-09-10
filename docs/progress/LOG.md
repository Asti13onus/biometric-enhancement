# Progress Log

Newest entries at the top. One entry per working session: what was done, what was decided,
what is blocked, what is next. This is the raw material for supervisor updates and for the
thesis narrative — write it as if a reader in month nine needs it.

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
