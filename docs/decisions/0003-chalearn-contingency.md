# ADR 0003 — Contingency for inaccessible ChaLearn data: regenerate with Anguli

- **Date:** 2026-09-10
- **Status:** accepted (contingency armed, not yet triggered)

## Context

The ChaLearn LAP Track 3 set (84,000 paired clean/degraded fingerprints) is the field's
de-facto paired training set and the intended baseline arm of this thesis — the
"latent-style degradation" condition that Contribution 1 argues against.

Its file lists are login-gated: `/dataset/32/data/{55,56,57}/files/` all redirect to
`/login/`, so the archive URLs cannot be discovered anonymously. Registration is currently
**broken**: sign-up returns HTTP 500, and retrying the same address then reports the email is
not unique. That pattern is the well-known Django failure in which the user row is committed
before a failing confirmation-email step raises, so accounts are created but unusable.
Several addresses were tried, all failing the same way.

The dataset therefore may not be obtainable on any predictable timetable. Per `STRATEGY.md`
§2.2, no phase may block on access we do not control.

## Decision

Three paths, pursued in parallel rather than in sequence:

1. **Recover the existing account.** Log in with the *username* (the form field is
   `username`, not email) chosen on the attempt that 500'd; failing that, use the working
   reset page at `/password_reset/`. Cost: minutes.
2. **Email the maintainer.** Sergio Escalera is the listed contact. Draft in
   `docs/correspondence/`. Cost: minutes to send, unknown to resolve.
3. **Regenerate an equivalent corpus with Anguli** — verified working, see below.

If path 1 or 2 succeeds we use the original data and report it. If neither has resolved by
the **end of Phase 1 (week 8)**, path 3 becomes the baseline arm and the thesis states so
explicitly.

## Why regeneration is a genuine substitute

ChaLearn's training set was itself generated with Anguli, and measurement confirms the match:
Anguli's default output is **275×400 8-bit greyscale — exactly ChaLearn's geometry**. The
published ChaLearn artefact list (blur, brightness, contrast, elastic transformation,
occlusion, scratch, resolution, rotation, plus backgrounds) maps onto Anguli's `-noise`,
`-scratch`, `-rot`, `-trans` controls, captured as the `chalearn-like` preset in
`scripts/generate_anguli.py`.

Measured throughput on this machine (8 threads, GTX 1650 box, CPU-only generation):
**1.41 fingers/sec — 4.2 images/sec.** So a ChaLearn-equivalent 84,000-pair corpus is a
**~17 hour overnight run**, and a 20,000-pair development corpus is ~4 hours. Generation is
not a bottleneck.

Regeneration also gives us three things the download does not:

- **Reproducibility.** Seeded (`-seed`), with the exact config and command recorded to
  `generation.json` next to the output. The published ChaLearn set has no such provenance.
- **Multiple impressions per finger** (`-ni`), which Contribution 4 (multi-impression
  fusion) requires and the ChaLearn release does not provide.
- **Pattern-class control and labels** (`-cdist`, `-meta` → class plus singular-point
  coordinates), giving a pattern-class stratification axis for free.

## Consequences

- **Exact numerical comparability with published ChaLearn results is lost** if path 3 is
  used. This must be stated plainly in the evaluation chapter. It is a real cost, mitigated
  by the fact that the field's principal benchmark (SD27) is already unobtainable and the
  thesis is explicitly building a protocol on obtainable data — the same argument, applied
  consistently.
- The baseline arm becomes "latent-style synthetic degradation, regenerated to the published
  recipe" rather than "the ChaLearn release". The scientific role of the arm is unchanged:
  it is the degradation model the thesis argues is wrong for worn fingerprints.
- Anguli is needed regardless, as the source of clean masters the wear-degradation model is
  applied to. This decision only changes whether it also supplies the baseline arm.

## Operational notes discovered while verifying

Recorded here because both cost time and neither is documented upstream:

- Anguli resolves `Filterbank/` and `Densitymaps/` relative to the **current working
  directory**, not the executable. Run from anywhere else and it dies with
  `Error: Not able to load Filter Bank`. The wrapper sets `cwd` to the install directory.
- **Anguli's default image type is `jpg`.** Lossy compression on ridge structure would
  contaminate every downstream measurement. The wrapper forces `png` and refuses `jpg`.
- Anguli does not create its output directory, warning `Cannon create directory` instead.
  The wrapper creates it first.

## Alternatives rejected

- **Wait for ChaLearn access before starting Phase 1.** Rejected: violates §2.2, and the
  failure is on their side with no visible timetable.
- **Use SOCOFing-Altered as the paired baseline instead.** Rejected: its alterations
  (obliteration, central rotation, z-cut) are a *damage* model, not the latent-noise model
  the baseline arm must represent, and it is ~200 dpi against ChaLearn's 500. It remains the
  severity-stratification set, which is a different role.
- **Train the baseline arm unpaired** (CycleGAN-style). Rejected here: that is Karabulut's
  arm, which the thesis compares *against*, so it cannot double as the paired baseline.
