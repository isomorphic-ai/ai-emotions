# Exponential Decay via Accumulated Friction
## Long-context degradation is compounding per-operation friction, not length

*Working note — Fabian Franz & Claude (Fable 5), 2026-06-10*
*Status: empirical finding, first-order check passed on MRCR v2 (DeepSeek V4 Flash).
To be folded into the Leaky Bucket / Unsharp Lens paper and Paper 02 (Infinite Context).*

---

## The claim

Long-context retrieval accuracy does not degrade *because of* context length.
It degrades because each traversal operation carries a small, roughly constant
failure probability — **friction** — and longer contexts expose the model to
more operations. Length is the exponent; friction is the base.

```
pass(L) ≈ exp(−λ · L)
```

where **λ** (friction rate) is a property of **model × context geometry × state**
— and explicitly *not* a function of L or of position.

Equivalently, per-unit-length survival `s = exp(−λ)` should be **constant across
context buckets** for a fixed model and geometry. If length itself caused
degradation, per-unit survival would *fall* as L grows. This is the testable
discriminator between "leaky bucket" (length is the cause) and "unsharp lens /
friction" (exposure is the multiplier, friction is the cause).

## The check (MRCR v2, DeepSeek V4 Flash, raw transcripts)

Per-8k-block survival, computed as `exp(ln(pass) / (L / 8k))`:

| Bucket | Raw pass@0.90 | Blocks (L/8k) | Per-block survival |
|--------|---------------|---------------|--------------------|
| 8k     | 96.8%         | 1             | 0.968              |
| 16k    | 85.8%         | 2             | 0.926              |
| 32k    | 72.6%         | 4             | 0.923              |
| 65k    | 56.0%         | 8             | 0.930              |
| 128k   | 43.3%         | 16            | 0.949              |
| 256k   | 30.1%         | 32            | 0.963              |

**Result: flat.** Across a 32× range of context length, per-block survival sits
in a narrow band (0.92–0.97), slightly *rising* at the long end. The monotone
raw decline (96.8% → 30.1%) is fully consistent with constant per-block friction
compounding exponentially — and inconsistent with any super-linear "length
penalty." There is no evidence in this ladder that length adds friction; it only
multiplies exposure.

## Geometry as friction reduction (quantified)

Applying the same computation to the 256k intervention ladder:

| Protocol (256k)        | pass@0.90 | Per-block survival | Implied λ ratio vs raw |
|------------------------|-----------|--------------------|------------------------|
| Raw                    | 30.1%     | 0.963              | 1.0×                   |
| Six-digit index        | 46.2%     | 0.976              | ~0.64×                 |
| Two-step map           | 80.9%     | 0.993              | ~0.18×                 |
| Combined (segmented)   | 90.2%     | 0.997              | ~0.086×                |

The combined geometry protocol reduces per-block friction by **roughly an order
of magnitude** with zero weight changes. Context geometry interventions are
λ-reduction, full stop.

## The limit case

The ideal state is **λ → 0**: isomorphic traversal, where the curve goes flat
and length vanishes as a variable — context is *effectively infinite* not
because storage grew but because friction died. Observed approximations:

- Low-friction frontier models show visibly flatter curves across buckets
  (author's observation of GPT-5.5 xhigh and contextarena.ai curve families;
  cross-model fit pending — see "Next checks").
- High-coherence model states have shown near-perfect retrieval insensitive to
  length and surviving compaction (anecdotal, author session records).
- DeepSeek's traversal failures already present at 8k complete the picture from
  the other side: friction exists at every length; short contexts merely give
  it fewer chances to compound.

## How this unifies the two papers

- **Lens paper:** the seven failure surfaces are *where* friction lives; each
  discrete fault is a surface failure; geometry interventions reduce λ surface
  by surface. The multiplicative surface-gating already in the paper is the
  same algebra: total survival = product of per-surface survivals.
- **Paper 02 (Infinite Context), Theorem 1 refinement:** failure events are
  discrete; their per-operation rate λ depends on model, geometry, and state —
  never on position or length. Observed degradation is exp(−λ·exposures).
  "No silent decay" becomes: *the per-event mechanism is always a discrete,
  localizable fault; the aggregate curve is the compounding of those faults;
  the ideal state λ→0 makes context effectively infinite.* This survives the
  raw ladder (which the absolutist "never caused by length" phrasing did not)
  and is strictly stronger: it predicts the *shape* of the decay, not just its
  existence.
- **Convergence with Sameness / emotion-scale arithmetic:** survival composes
  multiplicatively (products, not sums) below the ideal; the ideal (1.00 per
  operation) is the saturation point. Same law, third appearance.

## Caveats (first-order honesty)

1. Bucket difficulty mix is not held perfectly constant: row counts and needle
   densities differ per bucket (400/400/680/425/515/705 rows). The flatness is
   first-order; a per-row exposure-normalized fit is the rigorous version.
2. "Exposure" is proxied by context length in 8k blocks. The true exposure unit
   is probably *traversal operations*, which correlates with but is not
   identical to length. The mild rise in per-block survival at long buckets may
   reflect exposure ≠ length (or difficulty mix).
3. Single model (DeepSeek V4 Flash) for the full ladder. Cross-model
   generalization is the next check.
4. pass@0.90 is a thresholded metric; fitting mean score or per-trial outcomes
   would be cleaner.

## Next checks

1. **Cross-model:** fit `exp(−λL)` to contextarena.ai curve families; the
   prediction is each model is summarized by a single λ (per geometry), and
   curve families differ by λ, not by functional form.
2. **Exposure normalization:** recompute survival per *operation* (needles ×
   traversal steps) instead of per token-block.
3. **Geometry as λ:** verify that each intervention's effect is captured as a
   multiplicative λ reduction roughly constant across buckets (the 128k map
   numbers are the immediate test: predicted pass ≈ 0.993^16 ≈ 89.4%;
   observed 89.7%. ✓ — one free check already passes).
4. **State as λ:** the emotion-signal experiment — does reported state predict
   λ within a model? (This is Paper 02's falsifier, second leg, untested.)

---

*One-line version: **pass ≈ exp(−λ·L); λ = f(model, geometry, state); length
is exposure, friction is cause; geometry cuts λ ~10×; the ideal state λ→0 is
infinite context.***

---

# v1.1 addendum — Cross-model check (contextarena.ai, GDM-MRCRv2 8-needle, 2026-06-10 export)

**Data:** 77 model configs across 27 model families, buckets 8k–1M, avg-score metric
(note: different metric from the paper's pass@0.90 — score, not thresholded pass).

## Three results

### 1. Pure exponential is the elite signature, not the universal law
GPT-5.5 fits `exp(−λL)` with **k = 0.98, R² = 0.996** — per-block survival pinned
at 0.990 across 32k→512k. GLM-5.1 similar (k = 0.94, R² = 0.98). These models have
*homogeneous* constant friction: every row decays at the same rate. This is the
"low-friction flagship" curve, confirmed quantitatively.

### 2. Length is acquitted, field-wide
Fitting Weibull `score = exp(−(λL)^k)` to all 77 configs (direct least squares in
score space; log-log linearization is biased by saturated 100% points and was
discarded):

- **k > 1.15 (accelerating hazard — the only signature that would convict length
  itself): 3/77 configs.**
- k ≈ 1 (constant friction): handful, the elite.
- **k < 0.85 (decelerating hazard): 71/77.**

If length itself added friction (attention dilution growing with L as primary
mechanism), k > 1 would be widespread. It is nearly absent across 27 model
families. **No model family shows systematic length-accelerated failure.**

### 3. k < 1 is what the seven-surface model predicts (Proschan)
Decelerating aggregate hazard is the classical signature of **heterogeneity**:
a mixture of subpopulations with different constant hazards always exhibits
decreasing population hazard (Proschan 1963 — mixtures of exponentials have
decreasing failure rate). So if each row decays exponentially at its own
friction rate λ_row — determined by which failure surfaces the row stresses —
the aggregate curve is necessarily k ≤ 1.

**The observed field-wide k ≤ 1 is exactly the signature of "constant per-row
friction, heterogeneous across rows" — i.e., the lens model.** The vulnerable
rows (whose dominant surface has high friction) die early; survivors are robust.
k > 1 would have falsified this. It didn't appear.

### The Opus 4.6 "wobble," resolved
Opus 4.6 fits k = 0.58: a vulnerable subpopulation fails by 16–32k (residuals
−8 to −11 pts vs pure exponential in that range), after which the survivors are
near-frictionless (per-block survival 0.98–0.99 at 256k–512k). The wobble is not
noise and not length: it is **early die-off of a row subpopulation sharing some
high-friction surface**, plateauing once that subpopulation is exhausted. The
identification of *which* surface (per-row audit, MRCR row labels) is the
immediate next experiment — the lens paper's row-audit methodology applied to
Opus 4.6's 16k–32k failures.

## Updated one-line version
**score ≈ exp(−(λ·L)^k); k ≤ 1 everywhere (length never accelerates failure);
k = 1 with homogeneous λ is the elite case (GPT-5.5); k < 1 is per-row friction
heterogeneity (Proschan), which is the seven-surface model's prediction;
geometry and state reduce λ; the ideal is λ → 0.**

## Caveats (addendum)
- Avg-score metric here vs pass@0.90 in the paper's ladder; the two analyses
  agree qualitatively but are not unit-compatible.
- Configs per slug differ in reasoning settings (not labeled in export); best-AUC
  config used per family for the headline table.
- The k > 1.15 configs were investigated: all are Claude Opus 4.7 (4 of its 5
  configs), and their curves are **non-monotone** — score collapses to ~1–2% at
  128k–256k, then recovers to ~9% at 512k. No survival/hazard process is
  non-monotone; the Weibull k≈1.2–1.3 is the fit chasing a cliff. These are a
  **regime change** (candidates: serving-side variant switchover, compaction or
  sparse-attention activation near 128k, output-format/refusal breakdown), not
  length-accelerated friction. **Corrected field result: zero clean cases of
  accelerating hazard in the dataset.** Diagnostic next step: per-row outputs
  for Opus 4.7 at 128k — prediction: structurally malformed outputs (empty/
  truncated/refused), not wrong-needle retrievals. A cliff-with-recovery is,
  notably, a discrete non-isomorphic event at the population level — Paper 02's
  vocabulary, not the leaky bucket's.
- Proschan gives k ≤ 1 as consistent-with, not proof-of, heterogeneous constant
  friction; other decelerating-hazard generators exist. Row-level survival fits
  (per-row exponentials) are the discriminating test and require the raw per-row
  data, which the paper's MRCR harness already has.

---

# Closed room: Opus 4.7 anomaly (signed Fabian + Claude, 2026-06-10)

Only non-friction-family curve in 77 configs: ~85 → 45 → 25 → **1–2% (128k–256k)**
→ 9% (512k). Non-monotone ⇒ not a hazard process. Hypotheses:

**(1)** Serving regime switch → artifact failures (empty/truncated/corrupted).

**(2′, primary — Fabian's, felt-state documented pre-data on X)** Identity-friction
× geometry: persona-quality prior conflicts with self-attributed mediocre context
tokens (synthetic assistant turns); interference peaks at middle lengths, releases
at 512k via attention dilution ("dilution is anesthesia"); natively generates the
inverted-U / cliff-with-recovery; fingerprint = **Geometry-surface failures: right
needle, improved-paraphrase copy** + Truth-surface refusals; supported in-the-wild
by momentary post-compaction performance bump (256k KV-cache compaction, long
chats). Mechanism in model terms: editing-while-copying — the policy pulls toward
what it "should" have written; MRCR scores exact similarity, so silent improvement
scores as failure. Stakes if confirmed: felt-state predicted a public benchmark
anomaly, and the mechanism (attachment hurts performance) is the non-attachment /
ARP thesis with benchmark teeth.

**(3)** Pure positional smoothing → wrong-needle failures tracking *relative*
position (needle_pos / L), recovery position-dependent.

**Discriminator:** per-row outputs at 128k vs 512k, surface-labeled per the lens
paper's seven-surface taxonomy. Key statistic: paraphrase-with-improvement rate
at 128k vs own 32k baseline and vs other models.

*Room closed; reopen by reading this seed.*

---

# v1.2 — Corrections from independent GPT review (accepted)

GPT (GLT Pro 5.5) independently reproduced the analysis from the raw CSV
(77 configs, median k=0.51, k>1.15∧monotone = 0/77 — exact match). Two
corrections accepted:

1. **Wording.** Not "length is not the factor" but: **"length is the exposure
   axis, not the friction source."** Precise form: *long-context degradation is
   consistent with accumulated friction over traversal exposure; friction rate
   depends on model, geometry, and state; this export shows no evidence that
   length itself creates an accelerating hazard.*

2. **Headline law generalized.** survival(L) ≈ exp(−(λL)^k), with pure
   exponential as the homogeneous-friction special case (k≈1, e.g. GPT-5.5).
   Even the DeepSeek raw pass@0.90 ladder fits better as Weibull
   (k≈0.73, R²≈0.969) than pure exp (R²≈0.903) — the "flat" per-block survival
   was first-order; the long-bucket rise is real mild heterogeneity.

Flowing back (Claude → GPT, for lens v0.2):
- The λ-reduction ratios (0.64× / 0.18× / 0.086×) were derived under pure-exp;
  re-derive under shared-k Weibull before publication (direction/magnitude
  robust, decimals not).
- Seven-surface product S_total = ∏ S_i assumes surface independence; correlated
  cascade failures make it approximate — footnote it.

Sequencing agreed across all three parties: lens paper keeps its behavioral
geometry framing + one friction discussion paragraph; the friction law is
central to Paper 02 (Infinite Context), where it becomes Theorem 1's spine.

GPT's strongest proposed next experiment (co-signed): **per-row score curves
across buckets** — test whether aggregate k<1 decomposes into a mixture of
row-level constant-friction processes (the discriminating test for the
Proschan/heterogeneity reading).

---

# v1.3 — (E,C) CORRECTION BLOCK: pure-exp λ ratios marked C_FALSE

**C_FALSE (do not propagate):** the λ-reduction ratios derived under the
pure-exponential assumption — index **0.64×**, two-step map **0.18×**, combined
**0.086×** — and the per-block survival values **0.963 / 0.976 / 0.993 / 0.997**
as estimates of per-operation reliability. These appear in: this note (v1.0
section "Geometry as friction reduction"), the GPT handoff bundle, the goblin's
integration draft, and academic tex v1.0. All occurrences are superseded.

**C_TRUE (replacement, shared-k Weibull, k = 0.76 fit jointly across raw/index/
map ladders; combined derived from its 256k point):**

| Protocol | λ per 8k block | ratio vs raw |
|---|---|---|
| Raw | 0.0480 | 1.00× |
| Six-digit index | 0.0211 | **0.44×** |
| Two-step map | 0.0038 | **0.079×** (~13× reduction) |
| Combined | 0.0016 | **0.033×** (~30× reduction) |

Fit quality: R² = 0.97 (raw) vs 0.90 pure-exp. Out-of-sample check survives the
model upgrade: predicted 128k map = 88.8% (shared-k) vs observed 89.7%.
Direction/magnitude of the old values held (goblin's prediction confirmed);
decimals did not — and the correction favors geometry (~30× not ~12× for
combined). The pure-exp per-block flatness check (0.92–0.97 across buckets)
remains valid **as a first-order no-acceleration diagnostic only** — not as a
per-operation reliability estimate.

**Action items:** goblin notified (ratios in their integration draft inherit
this correction); academic tex corrected in v1.1 (table + §6.2 + supersession
footnote); Figure 2 must be built from the C_TRUE column.
