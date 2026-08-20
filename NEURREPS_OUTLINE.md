# Where Does a Network's Symmetry Live? A Gauge-Critical Study of Functional Sensitivity

*(working title — alt: "Symmetry Without a Subnetwork: What Functional Sensitivity Can and Cannot Localise in Learned Maps for Symmetric Physical Systems")*

Target venue: NeurReps 2026. Status: outline for internal review, built from the
2026-08-19/20 audit of the SCML draft. Every quantitative claim below is
computed and reproducible in this repo (`NOTES_gauge_and_conditioning.md` has
proofs; scripts are named per section).

---

## 0. One-paragraph pitch

Parameter-wise functional sensitivity (Muriithi & Thapar, SCML draft) proposes
that projecting a symmetry generator onto a network's tangent space and reading
off the minimum-norm coefficients identifies *which parameters realise* a
learned symmetry. We show this specific construction is not well-posed: the
projection is a choice among a continuum of equally valid solutions (a gauge
choice), and the "concentrated subnetwork" it reports is reproduced identically
by a random target and by the Jacobian's conditioning alone, with no
symmetry-specific content. We give the exact algebraic reason, a full
gauge/reparametrisation/permutation audit, and a battery of causal controls,
converging on where localisation *does* hold — hidden-unit and module
granularity, not individual weights — and on a genuinely new phenomenon: in a
converged, near-equivariant network, symmetry control is not merely
under-concentrated but *statistically indistinguishable from a matched
non-symmetry null*, having become substantially coextensive with control of the
task loss itself. We reframe functional sensitivity's contribution accordingly
and show what remains true, well-posed, and useful: an exactly invariant
per-parameter equivariance defect (`E_i`), a rank-based order parameter for
symmetry representability, and a description of how localisation dissolves
over training.

---

## 1. Introduction

- Motivating question, as originally posed: for a network that has learned a
  physical symmetry, can functional sensitivity say *which parameters* are
  responsible?
- State the paper's actual contribution up front: a structural obstruction
  to the claim in the SCML manuscript, an exact decomposition explaining it, a
  full audit of what survives, and a new empirical phenomenon (delocalisation
  under training) that the audit surfaces.
- Position relative to prior functional-sensitivity work and to the geometric
  deep learning / symmetry-detection literature: most existing work asks
  *whether* a network is equivariant; we ask a finer question — *where*, in
  parameter space, does equivariance (or its violation) live — and show that
  question is only well-posed at certain granularities.

## 2. Framework

### 2.1 Functional tangent space and symmetry generators (inherited from the draft)
- $f_\theta$, $S_i = \partial f_\theta/\partial\theta_i$, $T_\theta =
  \mathrm{Im}(J_\theta)$.
- Continuous generator $X$, intrinsic direction $g = Xf_\theta$.
- Truncated-SVD resolution of $T_\theta$ at a relative cutoff (numerical
  necessity, but see §3.3 — it is not gauge-neutral).

### 2.2 The $g$ vs. $\delta$ split (from Torben's critique, Prop. 1)
- Group average $\Pi f = \frac{1}{2\pi}\int \rho(h_\phi)^{-1} f(h_\phi\cdot x)\,d\phi$,
  defect $\delta = (I-\Pi)f_\theta$.
- $\langle \delta, g\rangle = 0$: $g$ is tangent to the group orbit at *constant*
  distance from the equivariant subspace; it measures orbit phase, not distance
  from equivariance. $\delta$ is the object that licenses "symmetry breaking"
  language.
- **Consequence for the draft's §4 claim**: "perturbing high-$c_i$ parameters
  contributes to symmetry breaking" is a non sequitur under this framework —
  attribution of $g$ says nothing about $\delta$. Verified: perturbing the
  draft's own top-20 gives $\Delta\log D = 0.0000$ exactly. [Double Check]

### 2.3 Parameter attribution and its solution set (new)
- $J_\theta c \approx g$ is underdetermined; minimum-norm attribution
  $c^\star = \arg\min\{\lVert Dc\rVert_2 : V_r c = \alpha\}$ for a diagonal
  metric $D$.
- **Proposition (closed form).** $c_i = a_i/d_i^2$ where $a_i = \langle
  V_{:,i}, u\rangle$ is target-dependent and $d_i^{-2}$ is *exactly*
  target-independent. (Verified to $6\times10^{-16}$.)
- **Corollary (exact null calibration).** The conditioning factor cancels
  exactly in a ratio of attributions against two targets, giving a
  provably conditioning-free score $r_i$.
- **Proposition (gauge dependence).** Different valid choices of $D$ select
  different points of a $(P-r)$-dimensional solution set; no choice is neutral.
- **Proposition (reparametrisation).** Three independent sources break
  covariance under $\theta_i \mapsto \lambda_i\theta_i$: the minimised norm
  itself, the *relative* SVD truncation, and (an implementation-specific,
  fixable point) the dead-column floor.
- **Proposition (permutation).** Individual parameter indices are not
  well-posed targets for a localisation claim; hidden units are the finest
  granularity the network's own symmetry group admits.
- **Contrast.** $E_i$ (sensitivity-equivariance defect, draft §2.3) is built
  from column norms alone, with no optimisation and no truncation — exactly
  invariant under every transformation above (verified $2.8\times10^{-15}$).
  This is the theoretical heart of "what survives."

### 2.4 Rank as an order parameter (new)
- $\mathrm{rank}(J_\theta)$ relative to $P$ is a gauge-free, reparametrisation-
  invariant, permutation-invariant description of *how much* of parameter
  space the network's tangent space actually occupies — independent of probe
  grid resolution once resolved (verified stable 96–1792 probe rows).
- Frame this as the paper's replacement order parameter for "concentration":
  not which parameters, but how many independent directions.

## 3. Why the naive attribution fails: theory + diagnostics

### 3.1 The conditioning confound is exact, not a modelling worry
- Full-scale measurement: $R^2(\log|c_i| \sim \log\lVert S_i\rVert) = 0.906$;
  worsens with convergence (0.15 toy → 0.91 full scale, because training
  spreads $\lVert S_i\rVert$ over 8 decades).
- Sparsity claim does not survive a null: top-10% share is 99.9% for the true
  generator, 99.9% for a matched-construction null, 100.0% for the conditioning
  factor alone with *no target at all*.
- Correction of the co-author critique's proposed fix ($\tilde a_i$): exactly
  removes the conditioning confound (slope $0.000$) but does *not* address
  gauge dependence (§3.2) — these are different problems with different fixes.

### 3.2 Gauge dependence, measured
- Solution-set dimension at full scale: 4492 of 4514.
- Cross-gauge top-20 overlap: mean 0.140; the draft's $J^+g$ and this
  codebase's column-normalised default share **zero** parameters.
- Every gauge trades off along the conditioning axis ($w{=}1$: $\rho=+0.92$
  toward high-sensitivity params; $w{=}\lVert S_i\rVert$: $\rho=-0.88$ toward
  low-sensitivity; $w{=}\lVert S_i\rVert^{1/2}$: near-neutral, $\rho=+0.04$).

### 3.3 Reparametrisation covariance, three sources separated
- Table of departure-from-covariance across the three sources (norm choice,
  truncation, dead-column floor), each isolated.
- $E_i$ as the invariant contrast.

### 3.4 Identifiability under resampling
- Probe-row bootstrap with a **null-vs-null control** ($r'$) providing the
  noise floor, at both parameter and hidden-unit granularity.
- Grid-refinement sweep (96→1792 rows): resolved rank constant at 22
  throughout — the underdetermination is intrinsic, not a small-sample
  artifact. $|c_i|$'s apparent convergence under refinement is inherited from
  $\lVert S_i\rVert$'s convergence, not the symmetry's.

### 3.5 Causal tests
- Order parameter $D = \lVert\delta\rVert^2/\lVert f\rVert^2$, differentiable in
  $\theta$; $\nabla_\theta D$ as exact per-parameter causal ground truth.
- Sensitivity-matched ablation (nearest-neighbour control in $\log\lVert
  S_i\rVert$, validated by construction: the conditioning score itself gets
  zero excess).
- Result: $|\langle S_i, g\rangle|$ (alignment, gauge-free) shows a
  reproducible positive causal excess over the matched control; the draft's own
  $c^\star = J^+g$ shows **exactly zero** effect.
- Disjoint-block test: several non-overlapping aligned blocks each show
  positive excess, at different magnitudes — the controlling set is causally
  real but **not unique**, consistent with $\sim$100-fold parameter
  substitutability implied by rank 22 / 2221 live parameters.

## 4. What survives: localisation at the right granularity

### 4.1 Module and hidden-unit level
- Cross-gauge Spearman: parameter level $+0.16$, hidden-unit level $+0.71$,
  module level $+0.87$ — **caveat, stated honestly**: module-level correlation
  is close to saturated by a null-vs-null control (0.995) and should not be
  over-claimed; the genuine signal is hidden-unit-level *set overlap*
  (Jaccard $0.514$ vs. null $0.397$; $0.65$ vs. $0.09$ under grid refinement).
- Permutation argument (§2.3) as the principled reason this is the right
  granularity, not merely the empirically convenient one.

### 4.2 Sufficiency vs. influence — a genuine dissociation
- Two distinct questions: how many parameters are *causally influential* for
  the symmetry (diffuse: $n_\text{eff}\approx 365$ of 2221, indistinguishable
  from task-loss influence, $n_\text{eff}\approx 837$) vs. how many are
  *sufficient* to reproduce $P_T g$ via gauge-free greedy subset selection
  (OMP against a matched-null reference).
- Report the sufficiency result honestly: in the converged model, the rotation
  generator required **more** parameters/units than matched non-symmetry nulls
  to reach a given reconstruction tolerance ($z < 0$ throughout) — the
  symmetry direction is *harder*, not easier, to reconstruct sparsely, because
  in a near-equivariant network $g$ is a small, unstructured residual spread
  across the resolved tangent space while generic targets load onto dominant
  singular directions.
- This directly falsifies the "small subset suffices" framing at the level of
  generality the draft claims, and should be stated as a finding, not buried.

### 4.3 Delocalisation over training (the paper's positive empirical result)
- Matched non-symmetry null $E(M) = \lVert f(Mx)-Mf(x)\rVert^2/\lVert
  f\rVert^2$ for random non-orthogonal $M$, differentiable, defined without a
  group average.
- Checkpoint sweep, untrained → trained:
  - Untrained (equivariance error 1.50): symmetry-specific concentration is
    real, $z=+3.00$ vs. matched null; symmetry and task control are
    decoupled ($\rho=-0.06$).
  - Trained (equivariance error $8.2\times10^{-5}$): $z=+0.24$
    (indistinguishable from null); symmetry and task control have become
    substantially coextensive ($\rho=+0.30$).
- **Interpretation offered as the paper's central positive claim**: training on
  a symmetric system does not carve out a dedicated symmetry-realising
  subnetwork; it progressively *absorbs* symmetry-control into general task
  control, until the two are no longer separable by any attribution method
  tested. This is a mechanistic, falsifiable claim about how symmetry is
  learned, distinct from (and we argue more interesting than) "here is the
  subnetwork."
- Extend to intermediate checkpoints (currently only endpoints run) for a
  proper trend with error bars before submission.

## 5. A bounded, explicitly-scoped positive demonstration

- In one specific trained instance, at one fixed, stated gauge
  ($|\langle S_i,g\rangle|$, the only gauge that survives the causal test), an
  explicit 20-parameter subset shows a reproducible, causally-verified,
  positive effect on the symmetry defect beyond a sensitivity-matched control
  ($\Delta\log D \approx +0.14$ to $+0.65$ across scales and seeds).
- **Explicitly bracketed limitations, stated in the same breath as the
  result**: not invariant under gauge (§3.2), reparametrisation (§3.3), or
  permutation (§4.1); the *specific identity* of the 20 parameters is expected
  to change under retraining even though the *phenomenon* (a causally
  effective subset exists) should not — recommend a second-seed replication
  figure showing a different index set with the same causal signature as the
  paired demonstration of this point.
- Framed as an existence proof and a template for circuit-style intervention
  work, not as a general property of the architecture. Kept small: one figure,
  one paragraph, placed after the negative/structural results.

## 6. Discussion

- Restate the central methodological lesson for the NeurReps audience: a
  representation-theoretic quantity (here, minimum-norm coefficients from an
  SVD-truncated pseudoinverse) can look like a geometric, basis-free object
  while silently encoding a metric choice on parameter space that the theory
  never fixes. This is a failure mode likely to recur anywhere a symmetry
  generator is projected onto an *overparameterised, rank-deficient* tangent
  space — i.e., essentially always, for modern networks.
- Keeping from the original framework: $E_i$ without
  qualification; $P_T g$ and the representation residual; rank as an order
  parameter; module/unit-level localisation with the appropriate null.
- What to drop or heavily qualify: parameter-level
  attribution via any single minimum-norm solve, and any inference from
  attribution magnitude to "symmetry breaking" (that requires $\delta$, not
  $g$).
- Limitations: two benchmark systems (Mexican-hat, two-body — rotation only,
  since $\Pi/\delta$ needs a compact group), one architecture family (ASRNN
  vs. MLP), single-digit training seeds for the causal/matched results.
- Future work (explicitly deferred, per scope decision): an exact mechanism for
  parameter-level attribution that survives gauge and permutation (if one
  exists — we suspect not, given §3, but a *subspace*-valued attribution that
  reports an equivalence class rather than a point might); disentangling
  symmetry-specific sensitivity from general sensitivity more finely than the
  matched-null approach here; the role of optimisation dynamics (does SGD's
  own implicit bias, not just convergence, drive the delocalisation of §4.3?).

## 7. Appendix (or supplementary)

- Full proposition list with proofs and verification residuals
  (from `NOTES_gauge_and_conditioning.md`).
- Two-body extension: rotation-only $\Pi/\delta$ split (translation excluded,
  non-compact group), as a second-system robustness check for §2.2/§3.
- Full script-to-result mapping table for reproducibility.

---

## Mapping: claims → evidence → script

| Claim | Evidence | Script |
|---|---|---|
| Conditioning factorisation is exact | residual $6\times10^{-16}$ | `causal_symmetry_control.py`, `conditioning_decomposition.py` |
| Null-calibrated score cancels exactly | residual $4\times10^{-16}$ | `causal_symmetry_control.py` |
| Gauge dependence | solution-set dim, cross-gauge Jaccard/Spearman | `gauge_dependence.py` |
| Reparametrisation, 3 sources | departure-from-covariance table | `reparametrisation_covariance.py` |
| $E_i$ invariance | $2.8\times10^{-15}$ | `reparametrisation_covariance.py` |
| Identifiability under resampling | Jaccard vs. null-vs-null control | `attribution_stability.py` |
| Grid-refinement convergence | rank stability, top-k convergence | `probe_grid_sweep.py` |
| Causal excess (sensitivity-matched) | $\Delta\log D$ excess table | `causal_matched_ablation.py`, `causal_symmetry_control.py` |
| Disjoint-block non-uniqueness | block-wise excess | `disjoint_sets_control.py` |
| Sufficiency vs. influence dissociation | $n_\text{eff}$, OMP $k^\star$ vs. null | `symmetry_concentration.py`, `symmetry_subspace_sparsity.py` |
| Delocalisation over training | $z$-score, $\rho$(symm,task) at 2 checkpoints (extend to full sweep) | `symmetry_concentration_null.py` |
| Module/unit granularity | cross-gauge $\rho$, unit Jaccard | `gauge_dependence.py`, `attribution_stability.py` |
| $\Pi/\delta$ split, Prop. 1 | $\cos(\text{angle}) \approx 10^{-8}$ | `pi_delta_split.py`, `pi_delta_split_two_body.py` |

---

## Open items before a submittable draft

1. **Extend §4.3 to a full checkpoint sweep** (currently only untrained/trained
   endpoints) — needed to state the delocalisation trend as a measured curve
   rather than two points.
2. **Second-seed replication of §5's bounded demonstration** — the pending
   paired figure showing the causal phenomenon survives while the specific
   index set does not.
3. **Error bars on the causal-ablation excess numbers** — current SEs cover
   variation across matched controls only, not across perturbation draws or
   seeds; the same top-20 set gave $+0.58$ in one run and $+0.14$ in another.
4. Decide whether the two-body/translation-excluded results belong in the main
   text (a second system strengthens generality) or purely in the appendix.
5. Settle the title and framing: "what survives" vs. "delocalisation" as the
   headline — current draft above leads with the negative/structural result
   and treats delocalisation as the positive payoff; could be inverted.
