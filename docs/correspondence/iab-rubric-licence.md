# Draft — IAB Rubric licence request (Rural Indian DB, MOLF, MUST)

**This is the critical path.** It is the only request needing a signature you cannot
provide, and the Rural Indian Fingerprint Database has no adequate substitute.

**To:** `databases@iab-rubric.org`
**Cc:** your supervisor
**Send from:** institutional address
**Signature required:** **Registrar / Head of Institution** — not you, not your supervisor
**Status:** not sent
**Realistic wait:** 2–4 weeks *after* your institution produces the signed form

Datasets requested, all under the same licence route:

| Dataset | Why |
|---|---|
| **Rural Indian Fingerprint Database** | The only public dataset matching the motivating scenario — ridges degraded by manual labour. The benchmark of the Joshi et al. line at IIT-Delhi. |
| **IIIT-D MOLF** | 19,200 images, 100 Indian subjects, 5 capture methods. Cross-sensor variation on the target population. |
| **MUST** | ~21,000 impressions with **manually marked minutiae, PPI and semantic masks** — the annotations are what let us compute minutiae precision/recall/F1 against ground truth. |

Ask explicitly for the **public** Rural Indian database. Some papers distinguish it from a
separate *private* rural set held by Prof. Phalguni Gupta at IIT Kanpur; if your supervisor
has a contact there, that one is worth a separate request.

---

## Step 1 — start the internal paperwork today

This is the part that actually takes the time. Before emailing IAB, find out from your
department office:

- who signs institutional data-licence agreements (Registrar, Dean, or Head of Institution)
- what they need from you — usually a covering note, the licence PDF, and your supervisor's
  endorsement
- how long their turnaround is

Use the covering note below. Send the IAB email in parallel so you have the licence form in
hand when the signature slot comes up.

---

## Step 2 — the email to IAB

> **Subject:** Licence request — Rural Indian Fingerprint Database, IIIT-D MOLF and MUST
>
> Dear IAB Rubric team,
>
> I am a master's student at [institution], supervised by [supervisor, title], working on
> the restoration of fingerprints degraded by occupational wear and ageing, for use in
> large-scale civil identity systems rather than forensics.
>
> I would like to request licensed access to three of your fingerprint databases:
>
> 1. the **Rural Indian Fingerprint Database** (Puri, Narang, Tiwari, Vatsa and Singh, 2010)
>    — specifically the publicly licensable version;
> 2. **IIIT-D MOLF** (Sankaran, Vatsa and Singh, IEEE Access, 2015);
> 3. **MUST** (Malhotra, Vatsa, Singh, Morris and Noore, IEEE TIFS, 2023).
>
> The Rural Indian database is central to the work. My thesis argues that the synthetic
> degradation used to train fingerprint restoration models — blur, scratches, occlusion and
> background compositing — models crime-scene latent prints rather than the ridge-amplitude
> attenuation caused by manual labour, and proposes a physiologically grounded degradation
> model instead. Validating that against real prints from the affected population is the
> point at which the argument either holds or fails, and yours is the only public dataset I
> have found that supports it. MOLF would provide cross-sensor variation on the same
> population, and MUST's manually marked minutiae would let me report minutiae
> precision and recall against expert ground truth rather than pixel-similarity metrics.
>
> Could you please send the licence agreement forms? I understand they require signature by
> the Head of Institution; I have begun that process at my end so I can return them
> promptly. I am happy to provide any further detail about the intended use.
>
> On handling: the images would be stored only on institutional machines, used solely for
> this research, and never redistributed. Any public release from this project consists of
> evaluation scripts and dataset split files that reference images by relative path and
> checksum, so that only other licensed holders can reproduce the protocol.
>
> Thank you for maintaining and sharing these databases.
>
> With best regards,
> [name]
> [programme], [department]
> [institution]
> [institutional email]

---

## Step 3 — covering note for your Registrar

Attach the IAB licence PDF once it arrives. Keep this to one page; signatories read the
first paragraph and the risk line.

> **Subject:** Request for institutional signature — research data licence agreement (IAB Rubric)
>
> Dear [Registrar / title],
>
> I am requesting your signature on the attached data licence agreement, required for my
> master's thesis research in [department] under the supervision of [supervisor].
>
> **What it is.** A licence from the Image Analysis and Biometrics Lab (originally
> IIIT-Delhi, now IIT Jodhpur) granting academic access to three fingerprint research
> databases. The provider requires the agreement to be signed by the Head of Institution
> rather than by the student or supervisor, which is why it needs your office.
>
> **Why the research needs it.** Automated fingerprint authentication fails
> disproportionately for people whose ridges are worn by manual labour, which in
> large-scale civil identity systems means being unable to authenticate for services. My
> work develops a method to restore such fingerprints. These databases are the only
> publicly licensable data collected from the affected population.
>
> **Obligations it places on the institution.** Use restricted to non-commercial academic
> research; no redistribution or publication of the images; secure storage; citation of the
> source publications. The data contains no personally identifying information beyond the
> fingerprint images themselves, which are supplied already anonymised by the provider.
>
> **Handling arrangements.** Storage on institutional machines only, access limited to me
> and my supervisor, and deletion on completion of the degree if the licence requires it.
> No images will appear in any public repository; published outputs will contain only
> evaluation code and file listings.
>
> I would be grateful if this could be signed at your earliest convenience, as the provider's
> review typically takes a further two to four weeks and the research timeline depends on it.
>
> I am glad to answer any questions or provide the underlying publications.
>
> With thanks and best regards,
> [name]
> [programme], [department] — [student ID]
> [institutional email]
> Supervisor: [supervisor, title, email]

---

## Follow-up

- **Log the send date in `REGISTRY.md` immediately.**
- **No reply by week 4** → ask your supervisor to email **Prof. Mayank Vatsa** or
  **Prof. Richa Singh** directly. The thesis plan notes that a supervisor-to-supervisor
  email moves faster than the form, and this is the request where that matters most.
- On arrival: save the signed licence to `docs/datasets/licences/`, download, run
  `scripts/make_manifest.py`, and update `REGISTRY.md`.
- If refused: fall back per `STRATEGY.md` §2.2 — CASIA-V5 (workers/waiters), NIST SD300
  (uncooperative captures), FVC2004 DB1_B, SOCOFing-Altered. Record the refusal; "the only
  public dataset for this population is not obtainable" is itself a finding worth stating.
