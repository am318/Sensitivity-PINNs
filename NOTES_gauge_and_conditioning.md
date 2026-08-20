# What parameter-wise symmetry attribution can and cannot identify

Companion notes to the NeurReps retarget. Every proposition below is stated for
the setting of the SCML draft (§2.2, §2.4) and is verified numerically by a
script in this repo; the verification residual is quoted with each one.

**All numbers below are full scale**: the Mexican-hat ASRNN, $P=4514$ (2221 of
which can affect the force at all — $K_\theta$ cannot), trained 8000 steps to
trajectory loss $6.7\times10^{-5}$, equivariance residual $9.1\times10^{-3}$ and
$\lVert\delta\rVert/\lVert f\rVert = 6.6\times10^{-3}$. This is squarely the
"network with low equivariance error" regime the claim is stated for. A second,
non-augmented model at matched convergence (loss $6.4\times10^{-5}$,
equivariance residual $4.2\times10^{-3}$) reproduces every qualitative finding.

A note on scale-dependence, because it matters for how these results should be
read: the conditioning confound is **much worse after convergence**. On a
lightly-trained toy model the column norm explains 15% of the variance of
$\log|c_i|$; on the converged full-scale model it explains **91%**. Convergence
spreads $\lVert S_i\rVert$ over eight orders of magnitude, and that spread is
what the attribution ends up reporting.

Throughout: $f_\theta:\mathcal X\to\mathbb R^m$, sensitivities $S_i=\partial
f_\theta/\partial\theta_i$ evaluated on a probe set and flattened, Jacobian
$J\in\mathbb R^{N\times P}$ with columns $J_{:,i}=S_i$, truncated SVD
$J\approx U_r\Sigma_r V_r^{\!\top}$ at relative cutoff $\tau$, and a symmetry
generator direction $g=Xf_\theta$. Write $d_i=\lVert S_i\rVert$, $D=\mathrm{diag}(d)$,
and $\alpha=\Sigma_r^{-1}U_r^{\!\top}g$, so that the constraint $Jc=P_{T_\theta}g$
is equivalent to $V_r c=\alpha$.

---

## 0. The solution set

**Proposition 0 (underdetermination).** The set $\{c: Jc = P_{T_\theta}g\}$ is an
affine subspace of dimension $P-r$.

In these models $r$ is a few tens to a few hundred while $P$ is thousands, so
the solution set has dimension in the thousands. *Every* point in it reproduces
the identical function-space direction. Selecting one requires minimising
something over it, and the data does not say what.

This is the root of everything below: "which parameters realise the symmetry"
is not a question about $f_\theta$ alone until a selection rule is fixed.

---

## 1. The conditioning factor

**Proposition 1 (conditioning–alignment factorisation).** The minimiser of
$\lVert Dc\rVert_2$ subject to $V_rc=\alpha$ is

$$c_i \;=\; \frac{a_i}{d_i^{2}}, \qquad a_i \;=\; \langle V_{:,i},\,u\rangle,\qquad u=(AA^{\!\top})^{-1}\alpha,\quad A=V_rD^{-1}.$$

*Proof.* Substituting $\tilde c=Dc$, the problem is $\min\lVert\tilde c\rVert_2$
subject to $A\tilde c=\alpha$, whose solution is $\tilde c=A^{\!\top}(AA^{\!\top})^{-1}\alpha$.
Then $\tilde c_i=(V_{:,i}\cdot u)/d_i$ and $c_i=\tilde c_i/d_i$. $\square$

The factor $d_i^{-2}=\lVert S_i\rVert^{-2}$ is **target-independent**: identical
for the true generator, for a random target, for anything. Only $a_i$ ever knew
about the symmetry. So a spread in $|c_i|$ is evidence about symmetry only to
the extent that $a_i$, not $d_i$, produces it.

*Verified:* residual $6.4\times10^{-16}$ at full scale (`causal_symmetry_control.py`;
the check must replicate the solve's zeroing of dead columns, or it compares
against a different estimator).

**Measured at full scale.** $\lVert S_i\rVert$ spans **8.0 decades** over live
columns. Regressing $\log|c_i|$ on $\log\lVert S_i\rVert$:

| quantity | slope | $R^2$ |
|---|---|---|
| $\log|c_i|$ | $-0.99$ | **0.906** |
| $\log|a_i|$ (alignment) | $+1.01$ | 0.910 |
| $\log|\tilde a_i|$ | $-0.000$ | **0.000** |

So 91% of the variance of the attribution ranking is predicted by a
target-blind quantity. (The variance decomposition into alignment/conditioning
shares is *not* a useful summary here — the cross term is $-374\%$, because
$a_i$ and $\lVert S_i\rVert$ are themselves strongly correlated. $R^2$ and the
slope are the honest statistics.)

**Correction to §2 of the review notes.** $\tilde a_i$ is measured here to be
*exactly* conditioning-neutral (slope $-0.000$, $R^2 = 0.000$), not partially so.
It fully fixes the conditioning confound. It does not fix gauge dependence
(Prop. 6), which is a different problem.

**The sparsity claim does not survive.** Share of total attribution carried by
the top 10% of parameters:

| | share |
|---|---|
| $|c_i|$, true generator | 99.9% |
| $|c_i|$, matched-construction null | 99.9% |
| $1/\lVert S_i\rVert^2$, conditioning alone | 100.0% |

The draft's headline structure — a small subset carrying essentially all the
attribution — is reproduced exactly by a random target and by the conditioning
factor on its own. And the ranking is largely target-independent:
Spearman$(|c(g)|, |c(\text{null})|) = 0.75$–$0.92$ across five null draws.

**Corollary 1.1 (exact null calibration).** For two targets $g,h$ solved on the
same $J$ with the same retained subspace, $d_i^{-2}$ cancels identically:

$$r_i \;=\; \frac{|c_i(g)|}{|c_i(h)|} \;=\; \frac{|a_i(g)|}{|a_i(h)|}.$$

Taking $h$ over a family of matched-construction nulls and using the geometric
mean gives a **provably conditioning-free attribution** — no regression, no
matching argument, no distributional assumption.

*Verified:* cancellation to $4\times10^{-16}$.

---

## 2. Gauge dependence

Let $\Lambda=\mathrm{diag}(\lambda)$, $\lambda_i>0$, and reparametrise
$\theta_i\mapsto\lambda_i\theta_i$. This renames coordinates and changes nothing
about the learned function: $S_i\mapsto S_i/\lambda_i$, $J\mapsto J\Lambda^{-1}$,
and the *true* coefficients must transform covariantly, $c\mapsto\Lambda c$.

**Proposition 2 (the draft's choice is not covariant).** The Moore–Penrose
selection $c^\star=J^{+}g$, i.e. $D=I$, does not commute with $\Lambda$: it
minimises $\lVert c\rVert_2$, and $\lVert\Lambda c\rVert_2$ is a different
objective, so the selected point is not $\Lambda$ times the original.

*Measured (full scale, two-decade change of units):* relative departure from
covariance $0.998$ for the draft's $J^{+}g$ and $0.954$ for the code default at
the working cutoff.

**Proposition 3 (column normalisation is covariant — as an objective).** With
$D$ the column norms, $D\mapsto D\Lambda^{-1}$ and $\lVert Dc\rVert_2$ is
invariant, so the objective is covariant by construction.

**Proposition 4 (but the truncation is not).** The retained subspace is defined
by discarding singular values below a *relative* cutoff. Column scaling changes
the singular spectrum of $J$, so a different subspace is retained and
$P_{T_\theta}$ itself moves. Covariance therefore fails even for the covariant
objective.

*Measured at full scale* (on columns clear of the floor of Prop. 4b, so this
effect is seen alone):

| cutoff | rank$(J)$ | rank$(J\Lambda^{-1})$ | $\lVert\Delta P_Tg\rVert/\lVert g\rVert$ | departure |
|---|---|---|---|---|
| $10^{-3}$ (working) | 22 | 19 | $2.0\times10^{-1}$ | 1.13 |
| $10^{-5}$ | 40 | 38 | $1.3\times10^{-2}$ | 1.63 |
| $10^{-8}$ | 101 | 86 | $1.5\times10^{-5}$ | 0.61 |
| $10^{-12}$ | 336 | 336 | $1.9\times10^{-15}$ | $2.6\times10^{-7}$ |

Covariance returns only at $10^{-12}$, where nothing is truncated — precisely the
regime the truncation exists to prevent ($\Sigma_r^{-1}$ amplifying numerically
meaningless directions).

**Proposition 4b (an implementation source, worth fixing regardless).** The
dead-column rule floors column norms at $10^{-8}$ times the *largest* column.
That floor is relative to a quantity that itself changes under $\Lambda$, so it
does not commute with a change of units. It only bites when the sensitivity
spectrum is wide enough for live columns to sit near it — which is exactly what
convergence produces. Here 13 live columns lie within $10\times$ of the floor,
and their presence alone moves the departure from $2.6\times10^{-7}$ (columns
clear of the floor) to $0.978$ (all live columns). An absolute-scale-free
criterion, or carrying the floor through the reparametrisation, would remove
this.

**The dilemma.** Covariant-but-unstable, or stable-but-gauge-dependent. There is
no cutoff at which both hold.

**Proposition 5 (no neutral gauge, and the least-bad one).** Under
$\lVert D^{s}c\rVert_2$ the explicit factor is $d_i^{-2s}$, so every choice
commits to a preference along the conditioning axis:

| gauge | $\rho(|c_i|,\lVert S_i\rVert)$ | prefers |
|---|---|---|
| $s=0$, the draft's $J^{+}g$ | $+0.918$ | most sensitive parameters |
| $s=1$, column normalisation | $-0.883$ | **least** sensitive parameters |
| $s=1/2$ | $+0.039$ | ~neutral |
| elastic net / $L_1$ | $+0.117$ / $+0.102$ | ~neutral |

If a min-norm attribution is wanted at all, $s=1/2$ is the choice to defend.

**Consequence.** At full scale the solution set has dimension **4492** (rank 22
of 4514). Mean cross-gauge overlap of the top-20 parameters is **0.140**, and the
column-normalised default has overlap **0.000 with every one of the seven other
gauges** — disjoint sets, from the same network, reproducing the same
function-space direction to within the truncation tolerance.

**Proposition 6 (the proposed fix is orthogonal to the problem).** The review
notes' $\tilde a_i=c_i\langle S_i,g\rangle/\lVert P_Tg\rVert^2$ is invariant
under $\theta_i\mapsto\lambda_i\theta_i$, as claimed. But gauge choice is not
reparametrisation: different gauges select genuinely different points of the
$(P-r)$-dimensional solution set, and $\tilde a$ inherits that ambiguity.
*Measured at full scale:* cross-gauge top-20 Jaccard $0.187$ for $\tilde a_i$
versus $0.140$ for $|c_i|$ — no repair. $\tilde a_i$ is the right fix for the
conditioning confound (Prop. 1) and no fix at all for this one.

*Verified:* `gauge_dependence.py`, `reparametrisation_covariance.py`.

---

## 3. Permutation

**Proposition 7 (parameter indices are not well-defined targets).** For a hidden
layer, permuting units and correspondingly permuting the next layer's columns
leaves $f_\theta$ pointwise unchanged for any activation. Any quantity offered
as "which parameters carry the symmetry" must therefore be
permutation-equivariant, so only its orbit — the multiset of values — is
well-defined. Individual weight indices are not comparable across training runs
and are not intrinsic within one.

The permutation acts on **hidden units** by relabelling, so a unit-level score
*is* permutation-equivariant and its top-$k$ multiset is invariant. The hidden
unit is therefore the finest granularity at which localisation of a symmetry
can be stated; modules are coarser and also safe.

---

## 4. What is well posed

**Proposition 8 ($E_i$ is invariant).** $E_i=\lVert S_i(h\cdot x)-\rho(h)S_i(x)\rVert
/\lVert S_i(x)\rVert$ is built from norms of the same column of two Jacobians,
with no optimisation and no truncation, so numerator and denominator both scale
by $\lambda_i^{-1}$ and $E_i$ is exactly invariant.

*Measured at full scale:* $2.8\times10^{-15}$, under the same reparametrisation
that moves $c_i$ by $0.95$–$1.00$. **$E_i$ is the well-posed half of the
framework** — the draft's Fig. 1b stands where Fig. 1a does not.

**Proposition 9 (coarse-graining restores gauge-invariance).** Aggregating
$|c_i|$ to a permutation-safe granularity recovers most of what the parameter
level loses. Mean cross-gauge Spearman:

| granularity | cross-gauge $\rho$ |
|---|---|
| individual parameters | $+0.163$ |
| hidden units | $+0.711$ |
| modules | $+0.873$ |

This is the constructive form of the localisation claim: **symmetry
localisation in parameter space is well posed at module level, partly at
hidden-unit level, and not at the level of individual weights.**

Also well posed, because they are function-space quantities and never reference
a metric on parameter space:

- $P_{T_\theta}g$ and the representation residual $\lVert P_{T^\perp}g\rVert/\lVert g\rVert$;
- the order parameters $\lVert g\rVert$ and $\lVert\delta\rVert$;
- **gauge-free subset selection**: $\min\lVert c\rVert_0$ s.t. $Jc\approx g$ —
  "which $k$ parameters *suffice* to reproduce the symmetry direction" contains
  no metric on parameter space. $L_1$/elastic net are its standard convex
  relaxations, consistent with those two agreeing (top-20 Jaccard $0.90$) while
  disagreeing with every $L_2$ gauge;
- causal interventions on a given support.

---

## 5. The regime problem

For an exactly equivariant force field the rotational Lie derivative vanishes
identically, so $g=Xf_\theta\to0$ as $f_\theta\to\ker X$; so does $\delta=Qf_\theta$.
Both are order parameters for symmetry breaking and they vanish *together*
(Prop. 1 of the review notes says only that they are orthogonal while doing so).

The attribution is therefore a direction estimated from a vanishing quantity,
exactly in the "low equivariance error" regime the claim is stated for. The
prediction is that identifiability of the symmetry-specific ranking degrades as
equivariance improves, while $|c_i|$ continues to look clean and sparse —
because its apparent structure is carried by $d_i^{-2}$, which neither vanishes
nor cares about the symmetry.

**Measured** (`identifiability_vs_equivariance.py`, 11 checkpoints, augmented model,
equivariance residual falling by a factor of 250 across training):

| step | equivariance | $\lVert g\rVert/\lVert f\rVert$ | repr. error | $r-r'$ (unit) |
|---|---|---|---|---|
| 0 | $1.2\times10^{0}$ | $1.1\times10^{0}$ | 0.001 | $+0.116$ |
| 160 | $5.9\times10^{-2}$ | $5.7\times10^{-2}$ | 0.017 | $+0.192$ |
| 1600 | $2.3\times10^{-2}$ | $2.6\times10^{-2}$ | 0.079 | $+0.106$ |
| 2800 | $1.5\times10^{-2}$ | $1.6\times10^{-2}$ | 0.154 | $+0.091$ |
| 5600 | $6.4\times10^{-3}$ | $7.9\times10^{-3}$ | 0.236 | $+0.016$ |
| 8000 | $9.1\times10^{-3}$ | $9.5\times10^{-3}$ | 0.121 | $-0.011$ |

The strong, significant result is **representation quality**: the relative error of
$P_{T_\theta}g$ rises from $0.001$ at initialisation to $0.12$–$0.24$ at convergence,
$\mathrm{corr}(\log\varepsilon_{\text{equi}}, \text{repr. error}) = -0.835$ ($p=0.0014$).
By the time the network is equivariant, a fifth of the generator direction lies
*outside* the resolved tangent space — and that residual is what $c_i$ is computed
from.

Secondary readings, with their limits:

- Parameter level, the symmetry-specific margin $r - r'$ averages $-0.007$ over all
  checkpoints ($0.066$ against control $0.074$): no signal at any stage of training.
- Unit level it is consistently positive, $+0.085$ ($0.491$ against $0.407$).
- The margin's decline as equivariance improves is in the predicted direction but
  **underpowered** at eleven checkpoints ($r = +0.565$, $p = 0.07$). Read as consistent
  with the mechanism, not established by it.
- $|c_i|$'s own stability is flat against equivariance ($p = 0.61$) — as expected of a
  quantity tracking conditioning rather than symmetry.

---

## 6. Scripts

| script | establishes |
|---|---|
| `conditioning_decomposition.py` | Prop. 1, Cor. 1.1; variance share of conditioning |
| `gauge_dependence.py` | Props. 0, 5, 6; cross-gauge overlap; module/unit survival |
| `reparametrisation_covariance.py` | Props. 2, 3, 4, 8 |
| `attribution_stability.py` | identifiability under probe resampling, with a null-vs-null noise floor |
| `probe_grid_sweep.py` | whether the subnetwork converges under grid refinement |
| `causal_matched_ablation.py` | causal effect at *fixed conditioning*, via a sensitivity-matched control |
| `compare_runs.py` | side-by-side headline table across runs |
| `identifiability_vs_equivariance.py` | §5, the regime problem |
| `causal_symmetry_control.py` | whether any score causally controls the symmetry, against conditioning-matched baselines |


---

## 7. Identifiability, measured

Fixing a gauge leaves a separate question: is the top-$k$ set a property of the
network or of the probe grid? Resampling 80% of probe rows, 40 draws, with two
reference points — $\lVert S_i\rVert$ (target-blind, trivially stable, the "easy"
level) and $r'$ (null-vs-null, same estimator and same denominator noise as $r$
but no symmetry signal, hence the noise floor).

| score | top-20 Jaccard, parameters | top-19 Jaccard, hidden units |
|---|---|---|
| $\lVert S_i\rVert$ (conditioning reference) | 0.776 | 0.935 |
| $|c_i| = |J^{+}g|$ (the draft's score) | **0.748** | 0.881 |
| $r_i$ (conditioning-free) | 0.072 | **0.514** |
| $r'_i$ (null-vs-null control) | 0.063 | 0.397 |
| elastic-net support (gauge-free) | 0.179 | 0.329 |

$|c_i|$ looks robust and lands within noise of the conditioning reference it is
largely made of; the conditioning-free part sits at its own noise floor. At unit
granularity $r_i$ separates from its control.

**Grid refinement settles the "use more probe points" objection**
(`probe_grid_sweep.py`, 96 → 1792 rows). The resolved rank is **22 at every
resolution** — more probe points add no tangent-space dimensions, so the
underdetermination is intrinsic rather than undersampling. $|c_i|$'s top-20 does
converge toward its finest-grid value ($0.67 \to 1.00$) — but so does
$\lVert S_i\rVert$, and $|a_i|$ is flat at $1.00$, so that convergence belongs to
the conditioning factor. $r_i$ never converges at parameter level ($0.00$ at every
resolution below the finest) while at unit level it reaches $0.65$ against $0.09$
for its control.

## 8. Causation, measured

$D = \lVert\delta\rVert^2/\lVert f\rVert^2$ is zero exactly when the model is
SO(2)-equivariant on the grid and is differentiable in $\theta$, so
$\nabla_\theta D$ is the exact per-parameter causal influence on the symmetry.

- **Conditioning predicts it almost completely**:
  $\rho(\lVert S_i\rVert, |\nabla_i D|) = +0.934$. Which parameters causally affect
  the symmetry is, to first order, just which parameters are sensitive at all.
- **The draft's own top-20 has no causal effect whatsoever**: perturbing it gives
  $\Delta\log D = 0.0000$ and $\Delta\log L = 0.0000$, because the column-normalised
  gauge preferentially selects near-dead parameters ($\rho(|c_i|,\lVert S_i\rVert) = -0.88$).
- **After controlling for $\log\lVert S_i\rVert$**, the null-calibrated $r_i$ retains a
  small but significant partial correlation with causal influence: $+0.131$
  ($p = 2\times10^{-6}$) on the augmented model, $+0.242$ ($p = 0.005$) on the plain one.
  Small, replicated, and in the right direction.

A caution on the selectivity ratio $\Delta\log D/\Delta\log L$: it is not evidence on
its own, because a score selecting near-dead parameters makes both terms $\approx 0$
and their ratio noise. `causal_matched_ablation.py` replaces it with the excess over
a sensitivity-matched control.
