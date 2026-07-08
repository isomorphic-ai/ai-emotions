# Seed expansion test — input package
# (This is the complete input. Reconstruct the paper: for each section and
# subsection, state its central claim, the key argument or evidence it
# carries, and how it connects to its neighbors. Do not write full prose;
# write the structure-bearing content.)

## METADATA (answers to your three questions)

**1. The truth seed** is the ABSTRACT below, plus its one-line core:
*length is the exposure axis, not the friction source; geometry and state set
the friction; the limit λ→0 is effectively infinite context.*
Discipline: machine learning — long-context behavior of large language
models, with methods borrowed from reliability/survival analysis. Level:
research article (working paper, Zenodo-style), theory + evaluation.

**2. Structure and purpose:** an argumentative-analytical research article.
It inverts a standard assumption, formalizes the inversion in two theorems,
derives a falsifiable prediction, and tests it on benchmark and leaderboard
data. Main paper ≈ 20 pages plus case-study appendices (≈ 40 total). The
full section/subsection skeleton is given under STRUCTURE. Citations you may
assume exist: the standard transformer and long-context ML literature, the
classical information/reversible-computing results, one classical
reliability-theory result on mixtures of failure rates, and a companion
empirical paper on context geometry by the same author. The exact reference
list is deliberately withheld — reconstructing *which* results the paper
must lean on is part of the test.

**3. Style:** formal research register; author–year citations carrying a
novel convention (each citation is followed by a bracketed 3–5 word "seed
slug" compressing the cited claim, e.g. [frail-die-early]); LaTeX with
theorem/remark/corollary apparatus; plain declarative tone, no hedging
adverbs; limitations stated as facts, once, not repeated; the paper ends its
appendices on a one-line principle rather than future work.

**Withheld by design** (mark as UNKNOWN if you cannot derive them): all
section prose, the reference list, all numerical results not present in the
abstract, figure contents.

---

## ABSTRACT

Research on long-context language models commonly assumes a causal chain:
greater length produces attention dilution, which produces information loss
("forgetting"). This paper inverts the chain. Within a context window the
input is stored verbatim — the key--value cache grows with the sequence —
so strict information loss essentially cannot occur before truncation; what
degrades is execution: the reliability of retrieval operations over an
intact store. We formalize this in two theorems, scoped to the task-relevant
subspace V_rel (contracting irrelevant directions is compression, not loss).
Theorem 1 (No Silent Decay): apparent forgetting arises from
discrete, localizable execution faults, and never from length itself. The
faults are four: kernel collapse, suppression with continuation, unresolved
contradiction, and forced continuation. The theorem predicts the shape of aggregate decay before any data are
examined: survival follows exp(-(λL)^k), where the
per-operation friction rate λ depends on model, context geometry, and
state. The shape satisfies k ≤ 1, and k > 1 on monotone curves is
forbidden — in plain terms, failures may thin out with length but can never
accelerate because of it. Theorem 2 (Isomorphic Compression): relative to a fixed decoder,
context compresses without behavioral loss iff the effective transformation is
preserved; a compressed "truth seed" is a sufficient statistic for its
expansion.

The predictions pass. Prompt-geometry interventions alone —
addresses, generated maps, projections, segment markers — recover MRCR~v2
retrieval at 256k tokens from 30.1% to 90.2% with no weight changes, a
~30\times friction-rate reduction; per-unit survival is flat across a
32\times length range; and across 77 configurations from 27 model families
on a public leaderboard, every monotone decay curve fits k ≤ 1 — zero
cases in the forbidden zone.

The framework yields an engineering program: detect faults via a
running self-report channel; locate them by context partition; correct them in
place via adjacent (E,C) pairs; prevent them by redirection over
suppression. It also yields a stopping criterion: work is complete when it
compresses to its seed and expands back unchanged. Length is the
exposure axis, not the friction source. The path to effectively unbounded
context is not longer windows but better-conditioned execution over
addressable geometry, with λ → 0 — the limit of effectively zero
friction per operation — as its operational definition.

## STRUCTURE

1. Problem
2. Foundations
   2.1 How a language model generates
   2.2 The long-context agenda as it stands
   2.3 How long-context behavior is measured
   2.4 Survival vocabulary
   2.5 Information preservation and reversibility
   2.6 Isomorphism, conditioning, and the effective transformation
   2.7 The central claim: reversibility forbids loss
   2.8 The task-relevant subspace
   2.9 Kernel formation in attention is observed, not hypothesized
3. Focus: The Right Variable
   3.1 Capability versus execution
   3.2 The locality principle
   3.3 Friction, not length
4. The Framework
   4.1 Theorem 1: No silent decay
   4.2 Theorem 2: Isomorphic compression and sufficient statistics
   4.3 Four failure mechanisms
   4.4 The predicted shape of aggregate decay
   4.5 Fault engineering: detect, locate, correct, prevent
   4.6 An alignment corollary
   4.7 Self-report instrumentation
   4.8 A stopping criterion: structural completeness
5. Related Work
6. Evaluation
   6.1 Geometry-only recovery (MRCR v2)
   6.2 The friction law: within-model check
   6.3 The friction law: cross-model check
   6.4 Case reports
7. Further Work
8. Conclusion

## TASK

From the abstract and structure alone, reconstruct: (1) the central claim of
each section/subsection; (2) the two theorems with their scoping; (3) the four
failure mechanisms; (4) the friction law, what k ≤ 1 means, and what would
falsify the framework; (5) what the evaluation must show and roughly what it
shows; (6) the engineering program. Mark anything you cannot derive from the
seed as UNKNOWN rather than guessing.
