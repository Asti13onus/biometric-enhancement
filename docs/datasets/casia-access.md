# CASIA (idealtest.org) — access walkthrough

Reverse-engineered from the portal's own API on 2026-09-10, because the site is a
JavaScript single-page app that cannot be scripted and its documentation is behind login.

**Portal:** https://www.idealtest.org/
**Operator:** Institute of Automation, Chinese Academy of Sciences (CASIA)
**Note:** the site's TLS certificate is **expired**. Your browser will warn. That is the
site's own misconfiguration, not an interception — click through. The old host
`biometrics.idealtest.org` is dead (connection refused); only `www.idealtest.org` works.

---

## What we want from it

| id | Dataset | Images | Archive | Why |
|---|---|---|---|---|
| **7** | CASIA Fingerprint Image Database **V5.0** | 63,407 | 5 subset zips, ~1.6 GB total | The free stand-in for a worn population: the subject pool explicitly includes **workers and waiters**, and subjects were asked to give difficult impressions. Protects Phase 2 if the IAB licence is slow. |
| **15** | **CASIA Fingerprint Subject Ageing** v1.0 | 3,664 | `CASIA-Fingerprint-Subject-Ageing.zip` | ⭐ See below — likely the most on-topic public dataset we have found since the Rural Indian DB. |

V5.0 subset archives (id 7), sizes from the API:

| Subset | Bytes |
|---|---|
| `CASIA-FingerprintV5 (000-099).zip` | 320,642,925 |
| `CASIA-FingerprintV5 (100-199).zip` | 315,175,880 |
| `CASIA-FingerprintV5 (200-299).zip` | 314,211,620 |
| `CASIA-FingerprintV5 (300-399).zip` | 323,588,427 |
| `CASIA-FingerprintV5 (400-499).zip` | 327,982,423 |

Note the count discrepancy to record in the thesis: the portal reports **63,407** files
while the published description says 20,000 images (500 subjects × 8 fingers × 5
impressions). The archives likely carry extra per-subject files. Resolve by counting the
manifest after download rather than quoting either figure.

### Why dataset 15 matters

`CASIA Fingerprint Subject Ageing` appears in **neither the thesis plan nor the literature
review's dataset table**. Its sibling, `CASIA Iris Subject Ageing` (id 14), is a multi-year
time-lapse collection, and the naming convention implies the same design here.

That would make it a rare thing. A 2019 survey of longitudinal biometrics notes that
studies of fingerprint ageing effects rely "almost all" on **non-public forensic datasets,
because there is hardly any publicly available fingerprint database allowing such
analysis**. Ageing is one of the three degradation causes in this thesis's scope statement
(occupational wear, **ageing**, skin damage), and §2.5 of the thesis plan makes ageing and
demographic bias the motivation chapter.

**Unconfirmed until we can read `CASIA-Fingerprint-Subject-Ageing.pdf`**, which is behind
login. Do not build a contribution on it until its structure is verified. But request it in
the same registration — it costs nothing extra.

---

## The access flow

From the portal's API surface (`/js/app.*.js`): `getCaptcha` → `sendValidCode` →
`register` → **`admin/userReview`** → `downloadInspection` → `download`. So there is a
**manual approval step**; this is not instant.

1. **Register** at https://www.idealtest.org/register
   - Use your **institutional email address**, not a personal one. Approval is done by a
     human, and an academic address is the main signal they have.
   - The form issues a captcha and emails a validation code (`sendValidCode`). If the code
     does not arrive, check spam before retrying — repeated attempts may look like abuse.
2. **Wait for approval.** An administrator reviews the account (`admin/userReview`). Days,
   not minutes. There is no published SLA.
3. **Once approved**, open each dataset page and accept its terms:
   - `https://www.idealtest.org/datasetDetail/7` — Fingerprint V5
   - `https://www.idealtest.org/datasetDetail/15` — Fingerprint Subject Ageing
   - Read and keep `Note_CASIA-FingerprintV5.pdf` and
     `CASIA-Fingerprint-Subject-Ageing.pdf` — they carry the citation and licence terms we
     must honour. **Save them into `docs/datasets/licences/`.**
4. **Download** all five V5 subset zips plus the ageing zip into
   `data/raw/_archives/`, then hand them over:

   ```
   python scripts/download_group_b.py casia --casia-archive "CASIA-FingerprintV5 (000-099).zip" \
       --casia-archive "CASIA-FingerprintV5 (100-199).zip" ...
   python scripts/make_manifest.py casia_v5
   ```

---

## Licence constraints — these bind the thesis

CASIA data is released for research and education only, and **prohibits publishing,
copying, redistribution and dissemination.** Concretely:

- Never commit the images. `.gitignore` blocks the formats by default; do not override it.
- CASIA images **cannot appear in the public benchmark release.** Only split files that
  *reference* them by relative path and checksum, so another licensed holder can reproduce
  the protocol. Contribution 3 must be designed this way from the start.
- Figures in the thesis showing CASIA images need the licence checked first. Some CASIA
  notes permit illustrative figures in publications; verify against the PDF rather than
  assuming.
- Cite exactly as the note PDF specifies.

## Fallbacks if approval is refused or stalls

CASIA is itself a *fallback* for the Rural Indian DB, so a failure here is not fatal — but
it does thin the "real worn population" evidence. In descending preference: NIST SD300
(uncooperative captures, request form), FVC2004 DB1_B (dry/distorted, already in hand),
SOCOFing-Altered (damage labels, already in hand). Record the outcome in `REGISTRY.md`
either way.
