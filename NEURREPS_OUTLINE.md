# Symmetry Is Not Localised in Parameter Space

**NeurReps 2026 — paper outline.** Workshop format (~9pp incl. refs), can be cut down to a 4 page extended abstract.

---

## Thesis

> The apparent localisation of a learned symmetry to a few parameters is an
> artifact of the attribution construction. Measured against matched nulls, the
> symmetry-generator direction is among the **least** localised directions in a
> converged network's tangent space — and it becomes less localised, not more,
> as the network becomes equivariant.

Everything below serves that sentence. The negative half is fully established
(theorem + measurement, two models). The positive half — *delocalisation over
training* — is the payoff and is what makes this a result rather than a
critique.

---

## 1. Introduction (0.75pp)

- The question: for a network that has learned a physical symmetry, can we say
  *which parameters* realise it? Recent work (our own SCML draft included)
  answers by projecting the generator onto the functional tangent space and
  reading off minimum-norm coefficients `c = J⁺g`, observing that `|c_i|` is
  sharply concentrated.
- We show that concentration is not evidence about symmetry: it is reproduced
  exactly by a random target and by the Jacobian's conditioning with no target
  at all. The underlying reason is structural, not statistical.
- Contributions:
  1. An exact factorisation of the attribution into a symmetry-dependent and a
     **target-independent** conditioning factor, and the observation that the
     latter dominates after convergence (R² = 0.91).
  2. A gauge analysis: `Jc = g` is underdetermined by ~4500 dimensions, and the
     two most natural norm choices return **disjoint** parameter sets.
  3. Causal tests with sensitivity-matched controls, showing the published
     score has *exactly zero* causal effect on equivariance.
  4. The positive result: symmetry-specific concentration is real in an
     untrained network and vanishes as the network becomes equivariant —
     symmetry control is absorbed into task control.
  5. Concrete recommendations for what to report instead, all of which we show
     to be invariant.

## 2. Setup (0.75pp)

- `f_θ`, sensitivities `S_i`, tangent space `T_θ = Im(J_θ)`, truncated SVD.
- Generator `X`, intrinsic direction `g = Xf_θ`; group average `Π`, defect
  `δ = (I−Π)f_θ`.
- **Prop. 0** (from Torben's critique): `⟨δ, g⟩ = 0`. `g` is tangent to the
  group orbit at constant distance from `ker X` — it measures orbit phase, not
  distance from equivariance. Attribution of `g` therefore cannot license
  claims about symmetry *breaking*; that requires `δ`.
- Benchmarks: Mexican-hat (SO(2)) and two-body (rotation only — `Π` needs a
  compact group, so translation is out of scope by construction). ASRNN and
  MLP. Two models at matched convergence (loss 6.4–6.7e-5, equivariance
  residual 4.2e-3 and 9.1e-3) — squarely the regime the claim is about.

## 3. The attribution is a gauge choice (2pp — theory core)

**3.1 Underdetermination.** rank(J) = 22 against 4514 parameters (2221 live).
Solution set is 4492-dimensional; every point reproduces the identical `P_T g`.
Refining the probe grid 96 → 1792 rows leaves rank at 22 — intrinsic, not
undersampling.

**3.2 Prop. 1 (conditioning–alignment factorisation).** With column
normalisation, `c_i = a_i / ‖S_i‖²` where `a_i = ⟨V_{:,i}, u⟩` is the only
target-dependent factor. Verified 6×10⁻¹⁶. **Corollary**: the conditioning
factor cancels *exactly* in a ratio of attributions against two targets,
giving a provably conditioning-free score (verified 4×10⁻¹⁶).

**3.3 Prop. 2–4 (gauge and covariance).** Under `θ_i ↦ λ_iθ_i` the true
coefficients must transform as `c ↦ Λc`. Three independent sources break this:
the minimised norm (`‖c‖₂` is not covariant); the *relative* SVD truncation
(rank 22 vs 19 under the same reparametrisation, and covariance returns only at
cutoff 10⁻¹², where truncation no longer does its job); and — an
implementation point worth fixing — a dead-column floor defined relative to the
largest column.

**3.4 Prop. 5 (no neutral gauge).** Under `‖D^s c‖₂` the explicit factor is
`d_i^{−2s}`, so every choice takes a side: `s=0` (the published `J⁺g`) favours
the *most* sensitive parameters (ρ=+0.92), `s=1` the *least* (ρ=−0.88),
`s=½` is neutral (ρ=+0.04).

**3.5 Prop. 6 (permutation).** Hidden-unit permutation leaves `f_θ` unchanged,
so parameter indices are not well-posed targets for localisation. The hidden
unit is the finest permutation-equivariant granularity.

**3.6 Prop. 7 (the contrast).** `E_i` is built from column norms of two
Jacobians with no optimisation and no truncation — exactly invariant
(2.8×10⁻¹⁵) under the same reparametrisation that moves `c_i` by ~1.0.

## 4. What the measurements show (2.5pp — experiments)

**4.1 The concentration is not about symmetry.** Top 10% of parameters carry
99.9% of `|c_i|` for the true generator — 99.9% for a matched-construction
null, 100.0% for `1/‖S_i‖²` alone. R²(log|c_i| ~ log‖S_i‖) = 0.906, and this
*worsens with convergence* (0.15 on a lightly-trained model → 0.91 at
convergence, because training spreads `‖S_i‖` over eight decades).

**4.2 Gauge disagreement, measured.** Mean cross-gauge top-20 Jaccard 0.140;
the published `J⁺g` and the column-normalised default share **zero**
parameters. The proposed scale-invariant fix `ã_i` removes the conditioning
confound exactly (slope −0.000) but not the gauge problem (0.187).

**4.3 Causal tests.** Order parameter `D = ‖δ‖²/‖f‖²`, differentiable, with
`∇_θD` as exact per-parameter causal ground truth, and a **sensitivity-matched
control** (nearest-neighbour in log‖S_i‖) that validates itself — `‖S_i‖`, which
carries no symmetry information, scores zero excess.
- Conditioning alone predicts causal influence at ρ = +0.934.
- The published score's top-20, perturbed, gives **Δlog D = 0.0000** — exactly
  no effect, because that gauge selects near-dead parameters.
- Alignment `|⟨S_i, g⟩|` does show a positive excess over the matched control,
  replicated across both models.
- Disjoint blocks by alignment rank each show positive excess at comparable
  magnitude, and the top block is not privileged — the controlling set is
  causally real but **not unique**, as ~100-fold substitutability (2221 live
  parameters, rank 22) predicts.

**4.3b Identifiability under resampling.** Bootstrap 80% of probe rows, 40
draws, with two reference points: `‖S_i‖` (target-blind, trivially stable — the
"easy" level) and a **null-vs-null calibration** `r′` with the same estimator
and denominator noise as the conditioning-free score but no symmetry signal —
the noise floor.

| score | top-20 Jaccard, parameters | top-19, hidden units |
|---|---|---|
| `‖S_i‖` (conditioning reference) | 0.776 | 0.935 |
| `\|c_i\|` (the published score) | **0.748** | 0.881 |
| `r_i` (conditioning-free) | 0.072 | **0.514** |
| `r′_i` (null-vs-null control) | 0.063 | 0.397 |

`|c_i|` looks robust and lands within noise of the conditioning reference it is
largely made of; the conditioning-free part sits at its own noise floor. Only at
unit granularity does the symmetry-specific score clear its control. Together
with §3.1 (grid refinement) and §4.2 (gauge), this is the third independent leg
under "the parameter-level identity is not a property of the network".

**4.4 Sufficiency vs. influence — and both point the same way.** These are
distinct questions and we test both.
- *Influence* is diffuse: n_eff(∇D) ≈ 365 of 2221, versus 837 for the task
  loss — barely more concentrated than the network's general behaviour.
- *Sufficiency*, via gauge-free greedy subset selection against matched nulls:
  the true generator needs **more** parameters than a non-symmetry target
  (19 vs 3.2 for 10% error; z = −16). In a near-equivariant network `g` is a
  small unstructured residual spread across the tangent space, while a generic
  target loads onto the dominant singular directions.
- Both routes agree, which is the thesis.

**4.5 The positive result: delocalisation over training.** Matched null
`E(M) = ‖f(Mx) − Mf(x)‖²/‖f‖²` for random non-orthogonal `M` — same functional
as the true rotation, defined without a group average.

| | equivariance error | concentration z vs null | ρ(∇symm, ∇task) |
|---|---|---|---|
| untrained | 1.50 | **+3.00** | −0.06 |
| trained | 8.2×10⁻⁵ | **+0.24** | +0.30 |

At initialisation symmetry control is concentrated in a symmetry-specific way
and is decoupled from task control. After training to equivariance it is
indistinguishable from the null and substantially coextensive with the task.
Note the *function* is emphatically symmetric (E under the true rotation is
8×10⁻⁵ against 8×10⁻² for random `M`): it is the localisation of control, not
the symmetry, that dissolves. **Interpretation:** training on a symmetric
system does not carve out a symmetry subnetwork — it absorbs symmetry into
general task competence.

## 5. What to report instead (0.75pp)

- **`E_i`** — exactly invariant; the published Fig. 1b stands without
  qualification where Fig. 1a does not.
- **Function-space quantities**: `P_T g`, the representation residual
  `‖P_{T⊥}g‖/‖g‖` (currently unreported, and it *degrades* to 0.12–0.24 in the
  low-equivariance regime), and `‖g‖`, `‖δ‖` as order parameters.
- **rank(J)** as the well-posed concentration statement — gauge-,
  reparametrisation-, permutation-, and grid-invariant.
- **Hidden-unit granularity**, via set overlap rather than correlation. (Module-
  level *correlation* is saturated — a null-vs-null control scores 0.995 — and
  must not be over-claimed. The genuine signal is unit-level Jaccard: 0.514 vs
  null 0.397, and 0.65 vs 0.09 under grid refinement.)
- **A matched null in every figure** — the difference between a claim about
  symmetry and a claim about conditioning.

## 6. Discussion (0.5pp)

- The general lesson for this venue: a quantity can look basis-free (an SVD, a
  pseudo-inverse, a projection onto a tangent space) while silently encoding a
  metric on parameter space that the theory never fixes. This failure mode
  should be expected wherever a generator is projected onto an
  overparameterised, rank-deficient tangent space — i.e. essentially always.
- Limitations: two systems, one architecture family, few seeds; the causal
  effect sizes are noisy (see open items); two-body covers rotation only.
- Future work, explicitly deferred: a subspace-valued attribution reporting an
  equivalence class rather than a point; disentangling symmetry-specific from
  general sensitivity beyond the matched-null approach; whether SGD's implicit
  bias, not merely convergence, drives §4.5.

---

## Optional §5.5 — a bounded, explicitly-scoped demonstration

*(Include only if space permits; one figure, one paragraph, placed after the
negative results.)* In one trained instance at one fixed gauge, an explicit
20-parameter subset (of 4514) shows a causally-verified positive effect on the
symmetry defect beyond a sensitivity-matched control. Framed as an existence
proof — such subsets exist and can be verified — with the identity explicitly
disclaimed as non-invariant under §3.3–3.5. **Requires** the second-seed
replication showing a *different* index set with the same causal signature;
without that pairing it reads as the very claim §3–4 refute.

---

## Claims we make / do not make

| We claim | We do not claim |
|---|---|
| `\|c_i\|`'s concentration is reproduced by nulls and by conditioning | that `c_i` is uninformative about the network in every respect |
| the parameter-level *identity* is gauge-dependent | that no parameter subset has causal effect |
| alignment-selected subsets causally affect equivariance | that they are unique, or reproducible across seeds |
| symmetry-specific concentration falls as equivariance improves | that the symmetry itself weakens (it strengthens by 3 orders) |
| `E_i`, rank, `P_T g` are invariant | that they answer "which parameters" |

## Figures (5)

1. The factorisation: `log|c_i|` vs `log‖S_i‖`, with `|a_i|` and `|ã_i|` —
   slopes −0.99 / +1.01 / −0.000.
2. Lorenz curves: true generator vs matched null vs `1/‖S_i‖²` alone.
3. Cross-gauge overlap matrix (8 gauges), with the disjoint pair highlighted.
4. Causal: matched-control excess by score, with the published score at zero.
5. **Headline** — delocalisation: concentration z and ρ(symm,task) against
   equivariance error across training.

## Claim → evidence → script

| Claim | Evidence | Script |
|---|---|---|
| conditioning factorisation is exact | residual 6×10⁻¹⁶ | `conditioning_decomposition.py` |
| null calibration cancels it exactly | residual 4×10⁻¹⁶ | `causal_symmetry_control.py` |
| gauge dependence | solution-set dim, cross-gauge Jaccard | `gauge_dependence.py` |
| covariance breaks, 3 sources | departure table | `reparametrisation_covariance.py` |
| `E_i` invariance | 2.8×10⁻¹⁵ | `reparametrisation_covariance.py` |
| identifiability vs null floor | Jaccard table, §4.3b | `attribution_stability.py` |
| rank stable under grid refinement | 96→1792 rows, rank 22 | `probe_grid_sweep.py` |
| causal excess, matched control | Δlog D excess by score | `causal_matched_ablation.py` |
| controlling set non-unique | disjoint-block excess | `disjoint_sets_control.py` |
| influence diffuse; sufficiency worse than null | n_eff; OMP k*(ε) vs nulls | `symmetry_concentration.py`, `symmetry_subspace_sparsity.py` |
| delocalisation over training | z, ρ(symm,task) vs equivariance | `delocalisation_sweep.py` |
| Prop. 0 (⟨δ,g⟩=0) | cos ≈ 10⁻⁸, both systems | `pi_delta_split.py`, `pi_delta_split_two_body.py` |

Proofs and verification residuals: `NOTES_gauge_and_conditioning.md`.

## Open items before submission

1. **Fig. 5 needs the full checkpoint sweep** — currently two endpoints; the
   13-checkpoint run is in progress.
2. **Error bars on the causal excess** — current SEs cover matched-control
   variation only; the same top-20 gave +0.58 and +0.14 on two runs. Needs
   variance over perturbation draws and seeds.
3. **Second seed** for §5.5, or drop §5.5.
4. Decide whether two-body goes in the main text or appendix.
