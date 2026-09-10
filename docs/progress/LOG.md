# Progress Log

Newest entries at the top. One entry per working session: what was done, what was decided,
what is blocked, what is next. This is the raw material for supervisor updates and for the
thesis narrative — write it as if a reader in month nine needs it.

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
