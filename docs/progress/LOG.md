# Progress Log

Newest entries at the top. One entry per working session: what was done, what was decided,
what is blocked, what is next. This is the raw material for supervisor updates and for the
thesis narrative — write it as if a reader in month nine needs it.

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
