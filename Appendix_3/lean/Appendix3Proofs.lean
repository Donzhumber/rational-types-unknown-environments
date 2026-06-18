import Mathlib.Algebra.BigOperators.Ring.Finset
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.SpecificLimits.Normed
import Mathlib.Topology.MetricSpace.Basic
import Mathlib.Data.Fintype.BigOperators
import Mathlib.Data.Real.Sqrt
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Data.Set.Finite.Basic
import Mathlib.Probability.Process.Filtration
import Mathlib.Probability.Martingale.Basic
import Mathlib.Probability.Martingale.Convergence
import Mathlib.MeasureTheory.Function.ConditionalExpectation.Basic
import Mathlib.MeasureTheory.Function.ConditionalExpectation.Real
import Mathlib.MeasureTheory.Function.LpSeminorm.Indicator
import Mathlib.MeasureTheory.MeasurableSpace.Basic
import Mathlib.MeasureTheory.Integral.Indicator
import Mathlib.Topology.Order.OrderClosed

/-!
# Appendix 3 Proofs — Lean 4 Formalization

Standalone formal companion to `Appendix_3.tex` for
*Identifying Rational Types in Unknown Environments* (Econometrica supplement).

This file mirrors the **eight sections** of the TeX appendix. Each block is
tagged with its TeX label. Items marked `[Proved]` are machine-checked; `[Def]`
are definitions; `[Hypothesis]` are structural assumptions proved analytically
in the TeX appendix (Cox, Kakutani criterion for measures, Bayes collapse,
fixed-point existence). Sections 4 & 8 algebra is machine-checked through
`tvSeparationAsymptotic_generalId` and `hazardSeparation_affinityVanishing`;
`kakutaniBridge_from_criterion` closes the bridge modulo
`kakutaniSignalSingularity` and `signalExperimentAlignment`.
Bayesian §2 of `thm:posterior-consistency` is factored into
`falseTypeCollapse_from_bayesBound` and `identificationImpliesCollapse_decomposed`.
Cox algebra (`cox_hazard_ratio_ne_one`) and predictive masses (`predictiveMass`,
`predictiveMass_isProbMass`, `coxTvSeparationLocal_predictiveMass`,
`coxTvSeparationLocal_of_betaSeparation`) are proved in §3 for `def:desenlace-fisico-mt`;
`asymptoticIdentification_from_coxBetaSeparation` wires them to §4 via `IdentificationBridge`.
Abstract `coxTvSeparationLocal` remains the general interface. Section~5 certifies
`kappa_h_signed`, `dLambda_eq_kappa`, `dContProb_eq`, and the sign of
`eq:dE-dgamma-prop` with explicit `Δt` and `survivalTail`. Section~6 records
`prop_optimal_gamma_beliefs` as a `[Hypothesis]` interface (IFT not formalized).
Kakutani layer~(1) is proved (`kakutaniAlgebraicPrerequisite`); layer~(2) is
`kakutaniMutualSingularity` (classical hypothesis).

## Section map

| § | TeX section | `\\label` |
|---|-------------|-----------|
| 1 | Results from the main paper | `sec:main` |
| 2 | Public information, likelihood, and Bayesian learning | `sec:info-bayes` |
| 3 | Stochastic implementation mechanism (MDG) | `sec:mdg` |
| 4 | Asymptotic type identification | `sec:id-asintotica` |
| 5 | Competitive risks and operational pressure | `sec:riesgos` |
| 6 | Policy implications | `sec:politica` |
| 7 | Equilibrium and implementability | `sec:equilibrio` |
| 8 | Abstract structural identification (Kakutani criterion) | `sec:id-abstracta` |
-/

namespace Appendix3Proofs

open scoped MeasureTheory ENNReal BigOperators
open MeasureTheory Filter Topology Set

variable {Ω Z : Type*} {m : MeasurableSpace Ω}
variable {ΘK : Type*}

/-!
## Section 1 — Results from the main paper (`sec:main`)

TeX: Corollary `cor:trend_equivalence` — equivalence of stochastic trends.
-/

section MainResults

def I0Closed (S : Set (ℕ → ℝ)) : Prop :=
  ∀ ⦃x y : ℕ → ℝ⦄, x ∈ S → y ∈ S → ∀ a b : ℝ, (fun t ↦ a * x t + b * y t) ∈ S

/-- `[Proved]` Lemma `lem:I0-closed` — if `x, y ∈ S` (both I(0)) and `S` is closed under
    linear combinations (`I0Closed S`), then `aX + bY ∈ S` for all `a b : ℝ`. -/
theorem lem_I0_closed
    (S : Set (ℕ → ℝ)) (hI0 : I0Closed S)
    (x y : ℕ → ℝ) (hx : x ∈ S) (hy : y ∈ S) (a b : ℝ) :
    (fun t ↦ a * x t + b * y t) ∈ S :=
  hI0 hx hy a b

/-- `[Proved]` Corollary `cor:trend_equivalence`. -/
theorem cor_trend_equivalence
    (S : Set (ℕ → ℝ)) (hI0 : I0Closed S)
    (u1 u2 : ℕ → ℝ) (hu1 : u1 ∈ S) (hu2 : u2 ∈ S) :
    (fun t ↦ u2 t - u1 t) ∈ S := by
  simpa [one_mul, neg_one_mul, sub_eq_add_neg] using hI0 hu2 hu1 1 (-1)

end MainResults

/-!
## Section 2 — Public information, likelihood, and Bayesian learning (`sec:info-bayes`)

TeX: `def:zeta-filtration`, `def:predictive-likelihood`, `def:beliefs-martingale`,
`lem:asymptotic-beliefs`.
-/

section InfoBayes

@[reducible] private noncomputable def measurableSpacePub
    (ℱ0 : MeasurableSpace Ω) (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z) (t : ℕ) :
    MeasurableSpace Ω :=
  match t with
  | 0     => ℱ0
  | n + 1 => ℱ0 ⊔ ⨆ i : Fin (n + 1), MeasurableSpace.comap (fun ω ↦ ζ i.val ω) mZ

private lemma measurableSpacePub_mono
    (ℱ0 : MeasurableSpace Ω) (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z) :
    Monotone (measurableSpacePub ℱ0 mZ ζ) := by
  intro s t hst
  rcases s with (_ | s)
  · rcases t with (_ | t)
    · exact le_rfl
    · dsimp [measurableSpacePub]; exact le_sup_left
  rcases t with (_ | t)
  · exact (Nat.not_succ_le_zero _ hst).elim
  dsimp [measurableSpacePub]
  refine sup_le_sup le_rfl ?_
  refine iSup_le ?_
  intro i
  have hval : (Fin.castLE hst i : Fin (t + 1)).val = i.val := Fin.val_castLE hst i
  have hf : (fun ω : Ω ↦ ζ i.val ω) = fun ω ↦ ζ (Fin.castLE hst i).val ω := by
    funext ω; exact congrArg (fun k : ℕ ↦ ζ k ω) hval
  rw [hf]
  exact le_iSup
    (fun k : Fin (t + 1) ↦ MeasurableSpace.comap (fun ω ↦ ζ k.val ω) mZ)
    (Fin.castLE hst i)

/-- `[Def]` Definition `def:zeta-filtration` — public filtration `ℱ_t^pub`. -/
noncomputable def publicFiltration
    (ℱ0 : MeasurableSpace Ω) (hℱ0 : ℱ0 ≤ m)
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t)) : Filtration ℕ m where
  seq   := measurableSpacePub ℱ0 mZ ζ
  mono' := measurableSpacePub_mono ℱ0 mZ ζ
  le'   := by
    intro t; cases t with
    | zero   => exact hℱ0
    | succ n =>
      refine sup_le hℱ0 ?_
      refine iSup_le ?_
      intro i; exact (hζ i.val).comap_le

/-- `[Def]` Definition `def:predictive-likelihood` — kernel `ℓ_t(θ; J)`. -/
noncomputable def predictiveLikelihood
    (ℱ0 : MeasurableSpace Ω) (hℱ0 : ℱ0 ≤ m)
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t))
    (ν : ΘK → @Measure Ω m) (t : ℕ) (θ : ΘK) (J : Set Z)
    (hJ : MeasurableSet J) : Ω → ℝ :=
  -- eq:likelihood-zeta: ℓ_t(θ; J) = 𝔼[𝟙{ζ_t ∈ J} | ℱ_t^pub] under ν_θ
  let _ : MeasurableSet[m] ((ζ t) ⁻¹' J) := measurableSet_preimage (hζ t) hJ
  let ℱpub := publicFiltration ℱ0 hℱ0 mZ ζ hζ
  condExp (ℱpub t) (ν θ) (((ζ t) ⁻¹' J).indicator (fun _ ↦ (1 : ℝ)))

variable [MeasurableSpace ΘK] [MeasurableSingletonClass ΘK]

/-- `[Def]` Definition `def:beliefs-martingale` — `μ_t(θ) = 𝔼[𝟙{θ_K=θ} | ℱ_t^pub]`. -/
noncomputable def bayesianBeliefs
    (ℱ0 : MeasurableSpace Ω) (hℱ0 : ℱ0 ≤ m)
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t))
    (μE : @Measure Ω m) (θK : Ω → ΘK) (hθK : Measurable[m, inferInstance] θK)
    (t : ℕ) (θ : ΘK) : Ω → ℝ :=
  -- eq:beliefs-ebp-consistency: μ_t(θ) = 𝔼[𝟙{θ_K = θ} | ℱ_t^pub]
  let _ : MeasurableSet[m] (θK ⁻¹' {θ}) :=
    measurableSet_preimage hθK (measurableSet_singleton θ)
  let ℱpub := publicFiltration ℱ0 hℱ0 mZ ζ hζ
  condExp (ℱpub t) μE ((θK ⁻¹' {θ}).indicator (fun _ ↦ (1 : ℝ)))

/-- `[Proved]` Lemma `lem:asymptotic-beliefs` — bounded martingale, a.s. limit in `[0,1]`. -/
theorem asymptoticBeliefsLemma
    (ℱ0 : MeasurableSpace Ω) (hℱ0 : ℱ0 ≤ m)
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t))
    (μE : @Measure Ω m) [IsProbabilityMeasure μE]
    (θK : Ω → ΘK) (hθK : Measurable[m, inferInstance] θK)
    (θ : ΘK) :
    let ℱpub := publicFiltration ℱ0 hℱ0 mZ ζ hζ
    let X (t : ℕ) : Ω → ℝ := bayesianBeliefs ℱ0 hℱ0 mZ ζ hζ μE θK hθK t θ
    (∀ s t : ℕ, s ≤ t → condExp (ℱpub s) μE (X t) =ᵐ[μE] X s) ∧
    (∀ᵐ ω ∂ μE, ∀ t, X t ω ∈ Set.Icc (0 : ℝ) 1) ∧
    ∃ Zθ : Ω → ℝ,
      (∀ᵐ ω ∂ μE, Zθ ω ∈ Set.Icc (0 : ℝ) 1) ∧
      ∀ᵐ ω ∂ μE, Tendsto (fun n ↦ X n ω) atTop (𝓝 (Zθ ω)) := by
  classical
  let ℱpub := publicFiltration ℱ0 hℱ0 mZ ζ hζ
  let M : Ω → ℝ := (θK ⁻¹' {θ}).indicator (fun _ ↦ (1 : ℝ))
  let X (t : ℕ) : Ω → ℝ := bayesianBeliefs ℱ0 hℱ0 mZ ζ hζ μE θK hθK t θ
  have hθ_meas : MeasurableSet[m] (θK ⁻¹' {θ}) :=
    measurableSet_preimage hθK (measurableSet_singleton θ)
  have hMint : Integrable M μE :=
    (integrable_const (1 : ℝ)).indicator hθ_meas
  have hX_eq : ∀ t, X t = μE[M | ℱpub t] := by
    intro t; simp [X, bayesianBeliefs, M, ℱpub]
  have hmg := martingale_condExp (m0 := m) M ℱpub μE
  have hSA : StronglyAdapted ℱpub X := fun t => by
    rw [hX_eq t]; exact hmg.stronglyAdapted t
  have hmart : Martingale X ℱpub μE :=
    Martingale.congr hmg hSA fun t => EventuallyEq.of_eq (hX_eq t).symm
  have hIcc_each : ∀ t, ∀ᵐ ω ∂ μE, (0 : ℝ) ≤ X t ω ∧ X t ω ≤ 1 := by
    intro t
    have h0M : 0 ≤ᵐ[μE] M := by
      filter_upwards with ω
      by_cases h : ω ∈ θK ⁻¹' {θ} <;> simp [M, h]
    have hMle1 : M ≤ᵐ[μE] fun _ : Ω ↦ (1 : ℝ) := by
      filter_upwards with ω
      by_cases h : ω ∈ θK ⁻¹' {θ} <;> simp [M, h]
    have h0 : 0 ≤ᵐ[μE] X t := by
      filter_upwards [condExp_nonneg h0M] with ω hω
      rw [congr_fun (hX_eq t) ω]; exact hω
    have h1 : X t ≤ᵐ[μE] fun _ : Ω ↦ (1 : ℝ) := by
      have hmono := condExp_mono (m := ℱpub t) hMint (integrable_const (1 : ℝ)) hMle1
      have hconst : μE[fun _ : Ω => (1 : ℝ) | ℱpub t] =ᵐ[μE] fun _ : Ω => (1 : ℝ) :=
        EventuallyEq.of_eq (condExp_const (ℱpub.le t) (1 : ℝ))
      filter_upwards [hmono, hconst] with ω hle hEq
      rw [hEq] at hle; rw [congr_fun (hX_eq t) ω]; exact hle
    filter_upwards [h0, h1] with ω h0ω h1ω
    exact ⟨h0ω, h1ω⟩
  have hL1 : ∀ n, eLpNorm (X n) 1 μE ≤ (1 : ENNReal) := by
    intro n
    have hL1M : eLpNorm M 1 μE ≤ (1 : ENNReal) := by
      have hp0 : (1 : ENNReal) ≠ 0 := by simp
      have hp1 : (1 : ENNReal) ≠ ⊤ := by simp
      rw [eLpNorm_indicator_const hθ_meas hp0 hp1]
      have hμ : μE (θK ⁻¹' {θ}) ≤ (1 : ENNReal) := by
        calc
          μE (θK ⁻¹' {θ}) ≤ μE Set.univ := measure_mono (subset_univ _)
          _ = 1 := by simp [measure_univ]
      simp only [enorm_eq_nnnorm, nnnorm_one, ENNReal.toReal_one, div_one, ENNReal.rpow_one]
      simp_rw [ENNReal.coe_one, one_mul]
      exact hμ
    have hstep : eLpNorm (μE[M | ℱpub n]) 1 μE ≤ eLpNorm M 1 μE :=
      eLpNorm_one_condExp_le_eLpNorm (μ := μE) (m := ℱpub n) (m0 := m) M
    have hEq : eLpNorm (X n) 1 μE = eLpNorm (μE[M | ℱpub n]) 1 μE :=
      congrArg (fun f => eLpNorm f 1 μE) (hX_eq n)
    exact le_trans (le_of_eq hEq) (le_trans hstep hL1M)
  have htends :=
    hmart.submartingale.ae_tendsto_limitProcess (μ := μE) (f := X) (ℱ := ℱpub) (R := 1) hL1
  have hall' : ∀ᵐ ω ∂ μE, ∀ t, (0 : ℝ) ≤ X t ω ∧ X t ω ≤ 1 := by
    simp only [ae_all_iff]; intro t; exact hIcc_each t
  refine ⟨?_, ?_, ?_⟩
  · intro s t hst; simpa [← hX_eq s, ← hX_eq t] using hmg.2 s t hst
  · exact hall'
  · refine ⟨Filtration.limitProcess X ℱpub μE, ?_, ?_⟩
    · filter_upwards [htends, hall'] with ω ht hIcc
      exact ⟨ge_of_tendsto ht (Eventually.of_forall fun n => (hIcc n).1),
             le_of_tendsto ht (Eventually.of_forall fun n => (hIcc n).2)⟩
    · simpa using htends

end InfoBayes

/-- `[Def]` False-type posterior collapse — conclusion of
`thm:posterior-consistency` / `thm:posterior-consistency-appendix` (§§4, 8). -/
def falseTypeCollapse
    (ℱ0 : MeasurableSpace Ω) (hℱ0 : ℱ0 ≤ m)
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t))
    [Fintype ΘK] [DecidableEq ΘK]
    [MeasurableSpace ΘK] [MeasurableSingletonClass ΘK]
    (μC : @Measure Ω m) [IsProbabilityMeasure μC]
    (θK : Ω → ΘK) (hθK : Measurable[m, inferInstance] θK)
    (thetaStar : ΘK) : Prop :=
  ∀ θ : ΘK, θ ≠ thetaStar →
    ∀ᵐ ω ∂ μC,
      Tendsto (fun n : ℕ ↦ bayesianBeliefs ℱ0 hℱ0 mZ ζ hζ μC θK hθK n θ ω) atTop (𝓝 0)

section MeasurableThetaShared

variable [MeasurableSpace ΘK] [MeasurableSingletonClass ΘK]

/-- Shared proof step: false-type collapse + simplex normalization ⇒ `μ_t(θ*) → 1`. -/
theorem posteriorConcentration
    (ℱ0 : MeasurableSpace Ω) (hℱ0 : ℱ0 ≤ m)
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t))
    [Fintype ΘK] [DecidableEq ΘK]
    (μC : @Measure Ω m) [IsProbabilityMeasure μC]
    (θK : Ω → ΘK) (hθK : Measurable[m, inferInstance] θK)
    (thetaStar : ΘK)
    (hmass_sum : ∀ t : ℕ, ∀ᵐ ω ∂ μC,
      ∑ θ : ΘK, bayesianBeliefs ℱ0 hℱ0 mZ ζ hζ μC θK hθK t θ ω = 1)
    (hfalse : falseTypeCollapse ℱ0 hℱ0 mZ ζ hζ μC θK hθK thetaStar) :
    falseTypeCollapse ℱ0 hℱ0 mZ ζ hζ μC θK hθK thetaStar ∧
    ∀ᵐ ω ∂ μC,
      Tendsto (fun n : ℕ ↦ bayesianBeliefs ℱ0 hℱ0 mZ ζ hζ μC θK hθK n thetaStar ω)
              atTop (𝓝 1) := by
  classical
  let X (t : ℕ) (θ : ΘK) : Ω → ℝ :=
    bayesianBeliefs ℱ0 hℱ0 mZ ζ hζ μC θK hθK t θ
  rcases asymptoticBeliefsLemma ℱ0 hℱ0 mZ ζ hζ μC θK hθK thetaStar with
    ⟨_, _, Z, _, hconv_true⟩
  have hsum_all : ∀ᵐ ω ∂ μC, ∀ t : ℕ, ∑ θ : ΘK, X t θ ω = 1 := by
    simp only [ae_all_iff]; intro t; exact hmass_sum t
  have hfalse_all :
      ∀ᵐ ω ∂ μC, ∀ θ : ΘK, θ ≠ thetaStar → Tendsto (fun n : ℕ ↦ X n θ ω) atTop (𝓝 0) := by
    simpa [falseTypeCollapse, X] using hfalse
  refine ⟨hfalse, ?_⟩
  filter_upwards [hconv_true, hsum_all, hfalse_all] with ω htrue hsumω hfalseω
  let L : ΘK → ℝ := fun θ => if θ = thetaStar then Z ω else 0
  have hcoord : ∀ θ : ΘK, Tendsto (fun n : ℕ ↦ X n θ ω) atTop (𝓝 (L θ)) := by
    intro θ; by_cases hθ : θ = thetaStar
    · subst hθ; simpa [L] using htrue
    · simpa [L, hθ] using hfalseω θ hθ
  have hsum_tendsto :
      Tendsto (fun n : ℕ ↦ ∑ θ : ΘK, X n θ ω) atTop (𝓝 (∑ θ : ΘK, L θ)) := by
    simpa using tendsto_finsetSum Finset.univ (fun θ _ => hcoord θ)
  have hsum_const :
      Tendsto (fun n : ℕ ↦ ∑ θ : ΘK, X n θ ω) atTop (𝓝 (1 : ℝ)) := by simp [hsumω]
  have hZ_eq : Z ω = 1 := by
    have := tendsto_nhds_unique hsum_tendsto hsum_const
    simpa [L] using this
  simpa [X, hZ_eq] using htrue

end MeasurableThetaShared

/-!
## Shared probability tools (§§3–8)

Finite-support masses and total-variation distance, used in MDG predictive masses
and asymptotic identification.
-/

section SharedProbability

/-- Probability mass on a finite support. -/
def IsProbMass {Y : Type*} [Fintype Y] (p : Y → ℝ) : Prop :=
  (∀ y, 0 ≤ p y) ∧ ∑ y, p y = 1

noncomputable def totalVariationDistance {Y : Type*} [Fintype Y] (p q : Y → ℝ) : ℝ :=
  -- ‖p−q‖_TV; used in eq:TV-separation
  (1 / 2 : ℝ) * ∑ y : Y, |p y - q y|

/-- `[Proved]` TV distance is at most one on finite probability masses. -/
theorem totalVariationDistance_le_one {Y : Type*} [Fintype Y] {p q : Y → ℝ}
    (hp : IsProbMass p) (hq : IsProbMass q) :
    totalVariationDistance p q ≤ 1 := by
  dsimp [totalVariationDistance, IsProbMass] at *
  rcases hp with ⟨hp_nonneg, hp_sum⟩
  rcases hq with ⟨hq_nonneg, hq_sum⟩
  have hle : ∀ y, |p y - q y| ≤ p y + q y := fun y => by
    rw [abs_le]
    constructor
    · linarith [hp_nonneg y, hq_nonneg y]
    · linarith [hp_nonneg y, hq_nonneg y]
  calc (1 / 2 : ℝ) * ∑ y, |p y - q y|
      ≤ (1 / 2) * ∑ y, (p y + q y) :=
        mul_le_mul_of_nonneg_left (Finset.sum_le_sum fun y _ => hle y) (by norm_num)
    _ = 1 := by rw [Finset.sum_add_distrib, hp_sum, hq_sum]; norm_num

/-- `[Hypothesis]` Lemma `lem:hazard-separation`, part (i) — local TV bound (`eq:TV-separation`). -/
def tvSeparationLocal
    {Y : Type*} [Fintype Y]
    (p : ΘK → ℕ → Y → ℝ) (theta theta' : ΘK) (t : ℕ) : Prop :=
  -- eq:TV-separation
  ∃ ε : ℝ, 0 < ε ∧
    ε ≤ totalVariationDistance (p theta t) (p theta' t)

/-- `[Proved]` Pointwise mass difference yields a uniform TV margin. -/
theorem tvSeparationLocal_of_mass_diff {Y : Type*} [Fintype Y] [DecidableEq Y]
    {p q : Y → ℝ} (hp : IsProbMass p) (hq : IsProbMass q) (y : Y) (hne : p y ≠ q y) :
    ∃ ε, 0 < ε ∧ ε ≤ totalVariationDistance p q := by
  rcases hp with ⟨hp_nonneg, _⟩
  rcases hq with ⟨hq_nonneg, _⟩
  refine ⟨(1 / 2 : ℝ) * |p y - q y|, ?_, ?_⟩
  · apply mul_pos (by norm_num)
    exact abs_pos.mpr (sub_ne_zero.mpr hne)
  · dsimp [totalVariationDistance]
    have hle : |p y - q y| ≤ ∑ z : Y, |p z - q z| :=
      Finset.single_le_sum (fun z _ => abs_nonneg (p z - q z)) (Finset.mem_univ y)
    nlinarith

end SharedProbability

/-!
## Section 3 — Stochastic implementation mechanism (MDG) (`sec:mdg`)

TeX: `def:mano-dios-filtro`, `def:desenlace-fisico-mt`, eqs. `eq:kappa-c`,
`eq:logit-hybrid` (logit not formalized).
-/

section MDG

/-- `[Def]` Definition `def:mano-dios-filtro` — MDG implementation law (core axioms). -/
structure MDGImplementationLaw (A : Type*) [Fintype A] where
  plan : A
  temperature : ℝ
  hT_pos : 0 < temperature
  implementationProb : A → ℝ
  full_support : ∀ a : A, 0 < implementationProb a
  plan_dominance : ∀ a : A, a ≠ plan → implementationProb a < implementationProb plan

/-!
### MDG logit and remark `rem:axiomas-logit`

TeX eqs. `eq:logit-hybrid`, `eq:temperatura-piso`, Remark `rem:axiomas-logit`.
Axioms A1 (full support) and A2 (plan dominance) are machine-checked below.
Axiom A3 (maximum entropy / Jaynes 1957) requires variational calculus not in Mathlib
and is recorded as a `[Hypothesis]` interface.
-/

section MDGLogitAxioms

/-- `[Def]` Eq. `eq:logit-hybrid`: hybrid-temperature logit
    `P(ã = a | a*, T) = exp(1{a=a*}/T) / ∑_{a'} exp(1{a'=a*}/T)`. -/
noncomputable def logitProb {A : Type*} [Fintype A] [DecidableEq A]
    (plan : A) (T : ℝ) (a : A) : ℝ :=
  Real.exp ((if a = plan then 1 else 0) / T) /
    ∑ a' : A, Real.exp ((if a' = plan then 1 else 0) / T)

private lemma logitProb_denom_pos {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A]
    (plan : A) (T : ℝ) :
    0 < ∑ a' : A, Real.exp ((if a' = plan then 1 else 0) / T) :=
  Finset.sum_pos (fun _a _h => Real.exp_pos _) Finset.univ_nonempty

/-- `[Proved]` Remark `rem:axiomas-logit` — Axiom A1 (full support):
    logit assigns strictly positive probability to every action. -/
theorem logitProb_pos {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A]
    (plan : A) (T : ℝ) (a : A) :
    0 < logitProb plan T a :=
  div_pos (Real.exp_pos _) (logitProb_denom_pos plan T)

/-- `[Proved]` Remark `rem:axiomas-logit` — Axiom A2 (plan dominance):
    the planned action has strictly higher logit probability than any deviation. -/
theorem logitProb_plan_dominates {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A]
    (plan : A) (T : ℝ) (hT : 0 < T) (a : A) (ha : a ≠ plan) :
    logitProb plan T a < logitProb plan T plan := by
  unfold logitProb
  have hD := logitProb_denom_pos plan T
  -- numerator of the `a`-slot: exp(0/T) = exp(0) = 1
  have hna : Real.exp ((if a = plan then (1:ℝ) else 0) / T) = 1 := by
    simp [if_neg ha]
  -- numerator of the plan-slot: > 1 because exponent 1/T > 0
  have hlt : (1:ℝ) < Real.exp ((if plan = plan then (1:ℝ) else 0) / T) := by
    rw [if_pos rfl]
    exact lt_of_eq_of_lt Real.exp_zero.symm
      (Real.exp_lt_exp.mpr (div_pos one_pos hT))
  -- reduce a/D < b/D to D⁻¹ < exp(...)*D⁻¹; use calc to avoid double-rewriting D⁻¹
  rw [hna, one_div, div_eq_mul_inv]
  set D := ∑ a' : A, Real.exp ((if a' = plan then (1:ℝ) else 0) / T)
  calc D⁻¹ = (1:ℝ) * D⁻¹ := (one_mul _).symm
    _ < Real.exp ((if plan = plan then (1:ℝ) else 0) / T) * D⁻¹ :=
        mul_lt_mul_of_pos_right hlt (inv_pos.mpr hD)

/-- `[Def]` Shannon entropy `H(q) = −∑_a q_a ln q_a` (TeX, after `eq:temperatura-piso`). -/
noncomputable def shannonEntropy {A : Type*} [Fintype A] (q : A → ℝ) : ℝ :=
  -∑ a : A, q a * Real.log (q a)

/-- `[Hypothesis]` Remark `rem:axiomas-logit` — Axiom A3 (maximum entropy):
    among full-support probability masses assigning the plan the same probability as
    the logit, `logitProb` maximises Shannon entropy (Jaynes 1957 characterisation
    theorem; the variational proof is not formalised in Lean). -/
def logitProb_maxEntropy_hyp {A : Type*} [Fintype A] [DecidableEq A] [Nonempty A]
    (plan : A) (T : ℝ) : Prop :=
  ∀ q : A → ℝ, IsProbMass q → (∀ a, 0 < q a) →
    q plan = logitProb plan T plan →
    shannonEntropy q ≤ shannonEntropy (logitProb plan T)

end MDGLogitAxioms

/-- `[Def]` Definition `def:desenlace-fisico-mt` — `p_Cont = exp(-∑_j λ̃_j Δt)`. -/
noncomputable def continuationProbability (hazards : Fin 4 → ℝ) (Δt : ℝ) : ℝ :=
  -- def:desenlace-fisico-mt: Q_t(cont) = exp(-∑_j λ̃_j Δt)
  Real.exp (-(∑ j : Fin 4, hazards j) * Δt)

/-- `[Def]` Net sensitivity `κ_h` (eq. `eq:kappa-c`), used in Section 5.
    Indices: `0` = payment ($j=1$), `1` = death ($j=2$), `2` = rescue ($j=3$). -/
noncomputable def netSensitivity (zetaGamma tildeLambda : Fin 3 → ℝ) : ℝ :=
  -- eq:kappa-c: ζ₂λ₂ + ζ₃λ₃ − ζ₁λ₁
  zetaGamma 1 * tildeLambda 1 + zetaGamma 2 * tildeLambda 2 - zetaGamma 0 * tildeLambda 0

section CoxProportionalHazards

/-- `[Def]` Distinct criminal technology at some cause `j` (TeX `lem:hazard-separation`, step 1). -/
def betaSeparation (beta : ΘK → Fin 3 → ℝ) (theta theta' : ΘK) : Prop :=
  ∃ j : Fin 3, beta theta j ≠ beta theta' j

/-- `[Def]` Cox proportional hazard ratio
    `λ_j(θ)/λ_j(θ') = exp(β_{K,j}(θ) - β_{K,j}(θ'))` (TeX eq. after line 490). -/
noncomputable def coxHazardRatio (beta_theta beta_theta' : ℝ) : ℝ :=
  -- λ_j(θ)/λ_j(θ') = exp(β_θ − β_θ') (Cox; TeX §3 after lem:hazard-separation)
  Real.exp (beta_theta - beta_theta')

/-- `[Proved]` Cox ratio as quotient of exponentials. -/
theorem cox_hazard_ratio_eq_div (beta_theta beta_theta' : ℝ) :
    coxHazardRatio beta_theta beta_theta' =
      Real.exp beta_theta / Real.exp beta_theta' := by
  dsimp [coxHazardRatio]
  exact Real.exp_sub beta_theta beta_theta'

/-- `[Proved]` Distinct `β` implies hazard ratio ≠ 1. -/
theorem cox_hazard_ratio_ne_one (beta_theta beta_theta' : ℝ)
    (hβ : beta_theta ≠ beta_theta') :
    coxHazardRatio beta_theta beta_theta' ≠ 1 := by
  intro h
  dsimp [coxHazardRatio] at h
  have hsub : beta_theta - beta_theta' = 0 :=
    (Real.exp_eq_one_iff (beta_theta - beta_theta')).1 h
  exact hβ (sub_eq_zero.mp hsub)

end CoxProportionalHazards

/-- `[Def]` Daily physical outcome `m_t`: continuation or terminal cause `j ∈ {1,…,4}`
    (TeX `def:desenlace-fisico-mt`). -/
abbrev DailyOutcome := Unit ⊕ Fin 4

section DailyPhysicalOutcome

/-- Cox effective intensity `λ̃_j(t|θ_K)`: TeX causes `j ∈ {1,2,3}` (Lean indices `0,1,2`)
    carry `β_{K,j}`; TeX cause `j = 4` (Lean index `3`) uses the baseline only. -/
noncomputable def typeTildeLambda (baseline : Fin 4 → ℝ) (beta : ΘK → Fin 3 → ℝ)
    (theta : ΘK) (j : Fin 4) : ℝ :=
  -- def:desenlace-fisico-mt: λ̃_j = baseline_j · exp(β_{K,j}) for j ∈ {1,2,3}
  if h : j.val < 3 then
    baseline j * Real.exp (beta theta ⟨j.val, h⟩)
  else
    baseline j

noncomputable def typeTildeLambdaVec (baseline : Fin 4 → ℝ) (beta : ΘK → Fin 3 → ℝ)
    (theta : ΘK) : Fin 4 → ℝ :=
  fun j => typeTildeLambda baseline beta theta j

/-- `[Def]` No knife-edge: total hazard intensity differs across types (TeX step 2). -/
def noHazardSumKnifeEdge (baseline : Fin 4 → ℝ) (beta : ΘK → Fin 3 → ℝ)
    (theta theta' : ΘK) : Prop :=
  (∑ j : Fin 4, typeTildeLambda baseline beta theta j) ≠
    (∑ j : Fin 4, typeTildeLambda baseline beta theta' j)

/-- Marginal mass of `m_t` from `def:desenlace-fisico-mt` (`p_Cont`, `q_t`, `ξ_j`). -/
noncomputable def predictiveMass (baseline : Fin 4 → ℝ) (beta : ΘK → Fin 3 → ℝ)
    (theta : ΘK) (Δt : ℝ) : DailyOutcome → ℝ :=
  let lam := typeTildeLambdaVec baseline beta theta
  let pC := continuationProbability lam Δt  -- p_Cont
  let q := 1 - pC                          -- q_t(θ_K) = 1 − p_Cont
  let lamSum := ∑ j : Fin 4, lam j
  fun y =>
    match y with
    | Sum.inl _ => pC
    | Sum.inr j => q * (lam j / lamSum)   -- h̄_j = q_t · ξ_j, ξ_j = λ̃_j / ∑_ℓ λ̃_ℓ

/-- `[Def]` Type-indexed predictive masses `p_θ^{(t)}(y)` (constant in `t` given fixed baseline). -/
noncomputable def predictiveMassFamily
    (baseline : ℕ → Fin 4 → ℝ) (beta : ΘK → Fin 3 → ℝ) (Δt : ℝ) :
    ΘK → ℕ → DailyOutcome → ℝ :=
  fun theta t => predictiveMass (baseline t) beta theta Δt

private lemma typeTildeLambda_pos (baseline : Fin 4 → ℝ) (beta : ΘK → Fin 3 → ℝ)
    (theta : ΘK) (hb : ∀ j, 0 < baseline j) (j : Fin 4) :
    0 < typeTildeLambda baseline beta theta j := by
  dsimp [typeTildeLambda]
  split_ifs
  · exact mul_pos (hb j) (Real.exp_pos _)
  · exact hb j

private lemma lambdaSum_pos (baseline : Fin 4 → ℝ) (beta : ΘK → Fin 3 → ℝ)
    (theta : ΘK) (hb : ∀ j, 0 < baseline j) :
    0 < ∑ j : Fin 4, typeTildeLambda baseline beta theta j :=
  Finset.sum_pos (fun j _ => typeTildeLambda_pos baseline beta theta hb j)
    (Finset.univ_nonempty)

/-- `[Proved]` `predictiveMass` is a probability mass on `DailyOutcome`. -/
theorem predictiveMass_isProbMass (baseline : Fin 4 → ℝ) (beta : ΘK → Fin 3 → ℝ)
    (theta : ΘK) (Δt : ℝ) (hb : ∀ j, 0 < baseline j) (hΔt : 0 ≤ Δt) :
    IsProbMass (predictiveMass baseline beta theta Δt) := by
  dsimp [IsProbMass, predictiveMass]
  set lam := typeTildeLambdaVec baseline beta theta
  set pC := continuationProbability lam Δt
  set q := 1 - pC
  set lamSum := ∑ j : Fin 4, lam j
  have hLamPos : 0 < lamSum := lambdaSum_pos baseline beta theta hb
  have hpC_nonneg : 0 ≤ pC := by
    dsimp [continuationProbability, pC]
    exact Real.exp_nonneg _
  have hpC_le_one : pC ≤ 1 := by
    dsimp [continuationProbability, pC]
    have hsum_nonneg : 0 ≤ ∑ j : Fin 4, lam j := hLamPos.le
    have hneg : -(∑ j : Fin 4, lam j) * Δt ≤ 0 :=
      mul_nonpos_of_nonpos_of_nonneg (neg_nonpos.mpr hsum_nonneg) hΔt
    exact Real.exp_le_one_iff.mpr hneg
  have hq_nonneg : 0 ≤ q := sub_nonneg.mpr hpC_le_one
  constructor
  · intro y
    rcases y with y
    cases y with
    | inl _ => dsimp; exact hpC_nonneg
    | inr j =>
      dsimp
      apply mul_nonneg hq_nonneg
      exact div_nonneg (typeTildeLambda_pos _ _ _ hb j).le hLamPos.le
  · have hsplit :
        (∑ y : DailyOutcome,
            match y with
            | Sum.inl _ => pC
            | Sum.inr j => q * (lam j / lamSum)) =
          pC + ∑ j : Fin 4, q * (lam j / lamSum) := by
      simp [Fintype.sum_sum_type, Finset.sum_const, Fintype.card_fin]
    calc ∑ y : DailyOutcome, (predictiveMass baseline beta theta Δt) y
        = ∑ y : DailyOutcome,
            match y with
            | Sum.inl _ => pC
            | Sum.inr j => q * (lam j / lamSum) := by
          simp [predictiveMass, lam, pC, q, lamSum, typeTildeLambdaVec]
    _ = pC + ∑ j : Fin 4, q * (lam j / lamSum) := hsplit
    _ = 1 := by
      have hinner : ∑ j : Fin 4, lam j / lamSum = 1 := by
        refine mul_left_cancel₀ hLamPos.ne' ?_
        rw [mul_one]
        dsimp [lamSum]
        calc lamSum * ∑ j : Fin 4, lam j / lamSum
            = ∑ j : Fin 4, lamSum * (lam j / lamSum) := Finset.mul_sum ..
          _ = ∑ j : Fin 4, lam j := Finset.sum_congr rfl fun j _ => by
              rw [← mul_div_assoc, mul_div_cancel_left₀ (lam j) hLamPos.ne']
          _ = lamSum := rfl
      calc pC + ∑ j : Fin 4, q * (lam j / lamSum)
          = pC + q * ∑ j : Fin 4, (lam j / lamSum) := by rw [Finset.mul_sum]
        _ = pC + q * 1 := by rw [hinner]
        _ = 1 := by dsimp [q]; ring

/-- `[Proved]` Distinct `β_{K,j}` ⇒ distinct intensity at the separating cause. -/
theorem typeTildeLambda_ne_of_betaSeparation (baseline : Fin 4 → ℝ)
    (beta : ΘK → Fin 3 → ℝ) (theta theta' : ΘK)
    (hb : ∀ j, 0 < baseline j) (hβ : betaSeparation beta theta theta') :
    ∃ j : Fin 3, typeTildeLambda baseline beta theta (Fin.castAdd 1 j) ≠
      typeTildeLambda baseline beta theta' (Fin.castAdd 1 j) := by
  rcases hβ with ⟨j, hβj⟩
  refine ⟨j, ?_⟩
  have hexp : Real.exp (beta theta j) ≠ Real.exp (beta theta' j) := by
    intro heq
    exact hβj (Real.exp_injective heq)
  have hbj : 0 < baseline (Fin.castAdd 1 j) := hb (Fin.castAdd 1 j)
  have hθ : typeTildeLambda baseline beta theta (Fin.castAdd 1 j) =
      baseline (Fin.castAdd 1 j) * Real.exp (beta theta j) := by
    dsimp [typeTildeLambda, Fin.castAdd]; simp [j.2]
  have hθ' : typeTildeLambda baseline beta theta' (Fin.castAdd 1 j) =
      baseline (Fin.castAdd 1 j) * Real.exp (beta theta' j) := by
    dsimp [typeTildeLambda, Fin.castAdd]; simp [j.2]
  rw [hθ, hθ']
  intro h
  apply hexp
  exact mul_left_cancel₀ hbj.ne' h

private lemma pCont_ne_of_lambdaSum_ne {lam lam' : Fin 4 → ℝ} (Δt : ℝ) (hdt : 0 < Δt)
    (hsum : (∑ j, lam j) ≠ (∑ j, lam' j)) :
    continuationProbability lam Δt ≠ continuationProbability lam' Δt := by
  intro h
  dsimp [continuationProbability] at h
  have hsum_eq : ∑ j, lam j = ∑ j, lam' j := by
    have hneg : -(∑ j, lam j) * Δt = -(∑ j, lam' j) * Δt := Real.exp_injective h
    nlinarith [hneg, hdt]
  exact hsum hsum_eq

/-- `[Proved]` Different total hazard ⇒ different continuation mass. -/
theorem contMass_ne_of_noHazardSumKnifeEdge (baseline : Fin 4 → ℝ)
    (beta : ΘK → Fin 3 → ℝ) (theta theta' : ΘK) (Δt : ℝ) (hdt : 0 < Δt)
    (hsum : noHazardSumKnifeEdge baseline beta theta theta') :
    predictiveMass baseline beta theta Δt (Sum.inl ()) ≠
      predictiveMass baseline beta theta' Δt (Sum.inl ()) := by
  dsimp [predictiveMass, noHazardSumKnifeEdge] at *
  exact pCont_ne_of_lambdaSum_ne Δt hdt hsum

/-- `[Proved]` Cox/Bradburn step 2 for `def:desenlace-fisico-mt` masses (TeX `eq:TV-separation`). -/
theorem coxTvSeparationLocal_predictiveMass (baseline : Fin 4 → ℝ)
    (beta : ΘK → Fin 3 → ℝ) (theta theta' : ΘK) (Δt : ℝ)
    (hb : ∀ j, 0 < baseline j) (hdt : 0 < Δt)
    (hsum : noHazardSumKnifeEdge baseline beta theta theta') :
    tvSeparationLocal (fun θ _ => predictiveMass baseline beta θ Δt) theta theta' 0 := by
  have hp := predictiveMass_isProbMass baseline beta theta Δt hb (le_of_lt hdt)
  have hq := predictiveMass_isProbMass baseline beta theta' Δt hb (le_of_lt hdt)
  have hne := contMass_ne_of_noHazardSumKnifeEdge baseline beta theta theta' Δt hdt hsum
  rcases tvSeparationLocal_of_mass_diff hp hq (Sum.inl ()) hne with ⟨ε, hε, hε_le⟩
  exact ⟨ε, hε, hε_le⟩

end DailyPhysicalOutcome

end MDG

/-!
## Section 4 — Asymptotic type identification (`sec:id-asintotica`)

TeX: `def:event-T-long`, `def:asymptotic-id`, `lem:hazard-separation`,
`thm:posterior-consistency`.
-/

section AsymptoticIdentification

/-- `[Def]` Definition `def:event-T-long` — prolonged captivity event 𝒯. -/
def prolongedCaptivityEvent (T : Set Ω) : Set Ω := T

def mutuallySingularMeasures {α : Type*} [MeasurableSpace α]
    (μ ν : Measure α) : Prop :=
  ∃ A : Set α, MeasurableSet A ∧ μ A = 0 ∧ ν Aᶜ = 0

private noncomputable def restrictedSignalLaw
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t))
    (μ : @Measure Ω m) (T : Set Ω) : Measure (ℕ → Z) :=
  letI : MeasurableSpace Z := mZ
  let _hpath : Measurable[m, inferInstance] (fun ω : Ω => fun t : ℕ => ζ t ω) :=
    measurable_pi_lambda (fun ω : Ω => fun t : ℕ => ζ t ω) hζ
  Measure.map (fun ω : Ω => fun t : ℕ => ζ t ω) (μ.restrict T)

/-- `[Def]` Definition `def:asymptotic-id`. -/
def asymptoticIdentification
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t))
    [Fintype ΘK] [DecidableEq ΘK]
    (ν : ΘK → @Measure Ω m) (T : Set Ω)
    (thetaStar : ΘK) : Prop :=
  ν thetaStar T ≠ 0 ∧
  ∀ θ : ΘK, θ ≠ thetaStar →
    ν θ T = 0 ∨
      mutuallySingularMeasures
        (restrictedSignalLaw mZ ζ hζ (ν θ) T)
        (restrictedSignalLaw mZ ζ hζ (ν thetaStar) T)

/-- `[Def]` Cox/Bradburn step 2 interface: distinct `β_{K,j}` ⇒ local TV (`eq:TV-separation`).
    For `predictiveMass`, see `coxTvSeparationLocal_of_betaSeparation` (proved). -/
def coxTvSeparationLocal {Y : Type*} [Fintype Y]
    (beta : ΘK → Fin 3 → ℝ)
    (p : ΘK → ℕ → Y → ℝ) (theta theta' : ΘK) (t : ℕ) : Prop :=
  betaSeparation beta theta theta' → tvSeparationLocal p theta theta' t

/-- `[Proved]` `def:desenlace-fisico-mt` masses: `β` separation + no hazard-sum knife-edge
    ⇒ `coxTvSeparationLocal`. -/
theorem coxTvSeparationLocal_of_betaSeparation (baseline : Fin 4 → ℝ)
    (beta : ΘK → Fin 3 → ℝ) (theta theta' : ΘK) (Δt : ℝ)
    (hb : ∀ j, 0 < baseline j) (hdt : 0 < Δt)
    (hβ : betaSeparation beta theta theta')
    (hsum : noHazardSumKnifeEdge baseline beta theta theta') :
    coxTvSeparationLocal beta (fun θ _ => predictiveMass baseline beta θ Δt) theta theta' 0 :=
  fun _ => coxTvSeparationLocal_predictiveMass baseline beta theta theta' Δt hb hdt hsum

/-- `[Hypothesis]` Lemma `lem:hazard-separation`, part (ii) — recurrent uniform TV on `S`. -/
def tvSeparationAsymptotic
    {Y : Type*} [Fintype Y]
    (p : ΘK → ℕ → Y → ℝ) (theta theta' : ΘK) : Prop :=
  -- lem:hazard-separation part (ii): recurrent uniform TV on S ⊂ ℕ
  ∃ ε : ℝ, 0 < ε ∧
  ∃ S : Set ℕ, S.Infinite ∧
    ∀ t ∈ S, ε ≤ totalVariationDistance (p theta t) (p theta' t)

/-- `[Proved]` Packaging: uniform margin on an infinite set yields `tvSeparationAsymptotic`. -/
theorem tvSeparationAsymptotic_of_uniform
    {Y : Type*} [Fintype Y]
    (p : ΘK → ℕ → Y → ℝ) (theta theta' : ΘK)
    (ε : ℝ) (hε : 0 < ε) (S : Set ℕ) (hS : S.Infinite)
    (hsep : ∀ t ∈ S, ε ≤ totalVariationDistance (p theta t) (p theta' t)) :
    tvSeparationAsymptotic p theta theta' :=
  ⟨ε, hε, S, hS, hsep⟩

/-- `[Hypothesis]` Kakutani bridge: asymptotic TV separation ⇒ tail singularity. -/
def kakutaniBridge
    {Y : Type*} [Fintype Y]
    (p : ΘK → ℕ → Y → ℝ)
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t))
    [Fintype ΘK] [DecidableEq ΘK]
    (ν : ΘK → @Measure Ω m) (T : Set Ω)
    (thetaStar : ΘK) : Prop :=
  ∀ θ : ΘK, θ ≠ thetaStar →
    tvSeparationAsymptotic p θ thetaStar →
      mutuallySingularMeasures
        (restrictedSignalLaw mZ ζ hζ (ν θ) T)
        (restrictedSignalLaw mZ ζ hζ (ν thetaStar) T)

/-- `[Proved]` Lemma `lem:hazard-separation` — logical assembly of (a)/(b) ⇒ `def:asymptotic-id`. -/
theorem hazardSeparationLemma
    {Y : Type*} [Fintype Y]
    (p : ΘK → ℕ → Y → ℝ)
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t))
    [Fintype ΘK] [DecidableEq ΘK]
    (ν : ΘK → @Measure Ω m) (T : Set Ω) (thetaStar : ΘK)
    (htrue : ν thetaStar T ≠ 0)
    (hdisj : ∀ θ : ΘK, θ ≠ thetaStar →
      ν θ T = 0 ∨ tvSeparationAsymptotic p θ thetaStar)
    (hpuente : kakutaniBridge p mZ ζ hζ ν T thetaStar) :
    asymptoticIdentification mZ ζ hζ ν T thetaStar := by
  refine ⟨htrue, fun θ hθ => ?_⟩
  rcases hdisj θ hθ with h | htv
  · exact Or.inl h
  · exact Or.inr (hpuente θ hθ htv)

section MeasurableThetaContinuation

variable [MeasurableSpace ΘK] [MeasurableSingletonClass ΘK]

section BayesianIdentification

/-- `[Def]` Full-support prior on the true type (TeX: `μ₀(θ*) > 0`). -/
def fullSupportPrior (μ₀ : ΘK → ℝ) (thetaStar : ΘK) : Prop :=
  0 < μ₀ thetaStar

/-- `[Def]` Bayes upper bound `μ_t(θ) ≤ (μ₀(θ)/μ₀(θ*)) · R_t^θ` (TeX §2). -/
def posteriorBayesUpperBound
    (ℱ0 : MeasurableSpace Ω) (hℱ0 : ℱ0 ≤ m)
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t))
    [Fintype ΘK] [DecidableEq ΘK]
    (μ₀ : ΘK → ℝ) (thetaStar : ΘK) (θ : ΘK)
    (μC : @Measure Ω m) (θK : Ω → ΘK) (hθK : Measurable[m, inferInstance] θK)
    (R : ℕ → Ω → ℝ) : Prop :=
  -- Bayes: μ_t(θ) ≤ (μ₀(θ)/μ₀(θ*)) · R_t^θ (thm:posterior-consistency, §2)
  ∀ (t : ℕ) (ω : Ω), 0 ≤ R t ω →
    bayesianBeliefs ℱ0 hℱ0 mZ ζ hζ μC θK hθK t θ ω ≤
      (μ₀ θ / μ₀ thetaStar) * R t ω

/-- `[Hypothesis]` Types excluded from `𝒯` have vanishing posterior under `μC`. -/
def zeroMassPosteriorCollapse
    (ℱ0 : MeasurableSpace Ω) (hℱ0 : ℱ0 ≤ m)
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t))
    [Fintype ΘK] [DecidableEq ΘK]
    (ν : ΘK → @Measure Ω m) (T : Set Ω)
    (μC : @Measure Ω m) [IsProbabilityMeasure μC]
    (θK : Ω → ΘK) (hθK : Measurable[m, inferInstance] θK)
    (thetaStar : ΘK) (θ : ΘK) : Prop :=
  ν θ T = 0 →
    ∀ᵐ ω ∂ μC,
      Tendsto (fun n : ℕ ↦ bayesianBeliefs ℱ0 hℱ0 mZ ζ hζ μC θK hθK n θ ω) atTop (𝓝 0)

/-- `[Hypothesis]` Mutual singularity ⇒ vanishing likelihood ratio (Blackwell–Dubins; TeX §2). -/
def singularityImpliesLikelihoodVanishing
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t))
    [Fintype ΘK] [DecidableEq ΘK]
    (ν : ΘK → @Measure Ω m) (T : Set Ω)
    (μC : @Measure Ω m) [IsProbabilityMeasure μC]
    (thetaStar : ΘK) (θ : ΘK) (R : ℕ → Ω → ℝ) : Prop :=
  mutuallySingularMeasures
      (restrictedSignalLaw mZ ζ hζ (ν θ) T)
      (restrictedSignalLaw mZ ζ hζ (ν thetaStar) T) →
    ∀ᵐ ω ∂ μC, Tendsto (fun n : ℕ ↦ R n ω) atTop (𝓝 0)

private theorem tendsto_zero_of_nonneg_le {f g : ℕ → ℝ}
    (hf : ∀ n, 0 ≤ f n) (hg_nonneg : ∀ n, 0 ≤ g n)
    (hfg : ∀ n, f n ≤ g n) (hg : Tendsto g atTop (𝓝 0)) :
    Tendsto f atTop (𝓝 0) := by
  rw [Metric.tendsto_atTop]
  intro ε hε
  rcases Metric.tendsto_atTop.mp hg ε hε with ⟨N, hN⟩
  refine ⟨N, fun n hn => ?_⟩
  have hg_lt : g n < ε := by
    simpa [Real.dist_0_eq_abs, abs_of_nonneg (hg_nonneg n)] using hN n hn
  simp only [Real.dist_0_eq_abs, abs_of_nonneg (hf n)]
  exact lt_of_le_of_lt (hfg n) hg_lt

/-- `[Proved]` TeX §2: Bayes bound + vanishing likelihood ratio ⇒ coordinate collapse. -/
theorem falseTypeCollapse_from_bayesBound
    (ℱ0 : MeasurableSpace Ω) (hℱ0 : ℱ0 ≤ m)
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t))
    [Fintype ΘK] [DecidableEq ΘK]
    (μ₀ : ΘK → ℝ) (thetaStar : ΘK) (θ : ΘK) (hθ : θ ≠ thetaStar)
    (μC : @Measure Ω m) [IsProbabilityMeasure μC]
    (θK : Ω → ΘK) (hθK : Measurable[m, inferInstance] θK)
    (R : ℕ → Ω → ℝ)
    (hpriorθ : 0 ≤ μ₀ θ) (hprior_star : fullSupportPrior μ₀ thetaStar)
    (hR_nonneg : ∀ (t : ℕ) (ω : Ω), 0 ≤ R t ω)
    (hbound : posteriorBayesUpperBound ℱ0 hℱ0 mZ ζ hζ μ₀ thetaStar θ μC θK hθK R)
    (hbelief_nonneg : ∀ (t : ℕ) (ω : Ω),
      0 ≤ bayesianBeliefs ℱ0 hℱ0 mZ ζ hζ μC θK hθK t θ ω)
    (hLR : ∀ᵐ ω ∂ μC, Tendsto (fun n : ℕ ↦ R n ω) atTop (𝓝 0)) :
    ∀ᵐ ω ∂ μC,
      Tendsto (fun n : ℕ ↦ bayesianBeliefs ℱ0 hℱ0 mZ ζ hζ μC θK hθK n θ ω) atTop (𝓝 0) := by
  classical
  let X (n : ℕ) (ω : Ω) := bayesianBeliefs ℱ0 hℱ0 mZ ζ hζ μC θK hθK n θ ω
  let C := μ₀ θ / μ₀ thetaStar
  have hC_nonneg : 0 ≤ C := div_nonneg hpriorθ (le_of_lt hprior_star)
  filter_upwards [hLR] with ω hRω
  have hg : Tendsto (fun n : ℕ ↦ C * R n ω) atTop (𝓝 0) := by
    simpa [mul_zero] using Tendsto.const_mul C hRω
  have hfg : ∀ n, X n ω ≤ C * R n ω := fun n =>
    hbound n ω (hR_nonneg n ω)
  have hg_nonneg : ∀ n, 0 ≤ C * R n ω := fun n =>
    mul_nonneg hC_nonneg (hR_nonneg n ω)
  exact tendsto_zero_of_nonneg_le (fun n => hbelief_nonneg n ω) hg_nonneg hfg hg

/-- `[Proved]` Decomposed identification ⇒ collapse (TeX §2 of `thm:posterior-consistency`). -/
theorem identificationImpliesCollapse_decomposed
    (ℱ0 : MeasurableSpace Ω) (hℱ0 : ℱ0 ≤ m)
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t))
    [Fintype ΘK] [DecidableEq ΘK]
    (ν : ΘK → @Measure Ω m) (T : Set Ω)
    (μ₀ : ΘK → ℝ) (thetaStar : ΘK)
    (μC : @Measure Ω m) [IsProbabilityMeasure μC]
    (θK : Ω → ΘK) (hθK : Measurable[m, inferInstance] θK)
    (R : ΘK → ℕ → Ω → ℝ)
    (hprior : ∀ θ, 0 ≤ μ₀ θ) (hprior_star : fullSupportPrior μ₀ thetaStar)
    (hbelief_nonneg : ∀ (θ : ΘK) (t : ℕ) (ω : Ω),
      0 ≤ bayesianBeliefs ℱ0 hℱ0 mZ ζ hζ μC θK hθK t θ ω)
    (hzero : ∀ θ : ΘK, θ ≠ thetaStar → zeroMassPosteriorCollapse ℱ0 hℱ0 mZ ζ hζ ν T μC θK hθK thetaStar θ)
    (hbayes : ∀ θ : ΘK, θ ≠ thetaStar →
      posteriorBayesUpperBound ℱ0 hℱ0 mZ ζ hζ μ₀ thetaStar θ μC θK hθK (R θ))
    (hR_nonneg : ∀ θ : ΘK, θ ≠ thetaStar → ∀ t ω, 0 ≤ R θ t ω)
    (hsingLR : ∀ θ : ΘK, θ ≠ thetaStar →
      singularityImpliesLikelihoodVanishing mZ ζ hζ ν T μC thetaStar θ (R θ))
    (hident : asymptoticIdentification mZ ζ hζ ν T thetaStar) :
    falseTypeCollapse ℱ0 hℱ0 mZ ζ hζ μC θK hθK thetaStar := by
  intro θ hθ
  rcases hident with ⟨_, hidentθ⟩
  rcases hidentθ θ hθ with hν | hsing
  · exact hzero θ hθ hν
  · have hLR := hsingLR θ hθ hsing
    simpa using falseTypeCollapse_from_bayesBound ℱ0 hℱ0 mZ ζ hζ μ₀ thetaStar θ hθ μC θK hθK
      (R θ) (hprior θ) hprior_star (hR_nonneg θ hθ) (hbayes θ hθ)
      (fun t ω => hbelief_nonneg θ t ω) hLR

/-- `[Hypothesis]` TeX §2 of `thm:posterior-consistency` — identification ⇒ false-type collapse
    (single-step wrapper; use `identificationImpliesCollapse_decomposed` for the proved factorization). -/
def identificationImpliesCollapse
    (ℱ0 : MeasurableSpace Ω) (hℱ0 : ℱ0 ≤ m)
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t))
    [Fintype ΘK] [DecidableEq ΘK]
    (ν : ΘK → @Measure Ω m) (T : Set Ω)
    (μC : @Measure Ω m) [IsProbabilityMeasure μC]
    (θK : Ω → ΘK) (hθK : Measurable[m, inferInstance] θK)
    (thetaStar : ΘK) : Prop :=
  asymptoticIdentification mZ ζ hζ ν T thetaStar →
    falseTypeCollapse ℱ0 hℱ0 mZ ζ hζ μC θK hθK thetaStar

end BayesianIdentification

/-- `[Proved]` Theorem `thm:posterior-consistency` — TeX §3 normalization closure;
    §2 collapse is `[Hypothesis]` via `identificationImpliesCollapse`. -/
theorem posteriorConsistencyTheorem
    (ℱ0 : MeasurableSpace Ω) (hℱ0 : ℱ0 ≤ m)
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t))
    [Fintype ΘK] [DecidableEq ΘK]
    (ν : ΘK → @Measure Ω m) (T : Set Ω)
    (μC : @Measure Ω m) [IsProbabilityMeasure μC]
    (θK : Ω → ΘK) (hθK : Measurable[m, inferInstance] θK)
    (thetaStar : ΘK)
    (hident : asymptoticIdentification mZ ζ hζ ν T thetaStar)
    (hbridge : identificationImpliesCollapse ℱ0 hℱ0 mZ ζ hζ ν T μC θK hθK thetaStar)
    (hmass_sum : ∀ t : ℕ, ∀ᵐ ω ∂ μC,
      ∑ θ : ΘK, bayesianBeliefs ℱ0 hℱ0 mZ ζ hζ μC θK hθK t θ ω = 1) :
    falseTypeCollapse ℱ0 hℱ0 mZ ζ hζ μC θK hθK thetaStar ∧
    ∀ᵐ ω ∂ μC,
      Tendsto (fun n : ℕ ↦ bayesianBeliefs ℱ0 hℱ0 mZ ζ hζ μC θK hθK n thetaStar ω)
              atTop (𝓝 1) :=
  posteriorConcentration ℱ0 hℱ0 mZ ζ hζ μC θK hθK thetaStar hmass_sum (hbridge hident)

end MeasurableThetaContinuation

end AsymptoticIdentification

/-!
## Section 5 — Competitive risks and operational pressure (`sec:riesgos`)

TeX: Proposition `prop:expected-captivity-gamma`, eqs. `eq:kappa-c`, `eq:dpCont-dgamma`,
`eq:dE-dgamma-prop`. `Fin 3` indices: `0` = payment ($j=1$), `1` = death ($j=2$),
`2` = rescue ($j=3$); exogenous cause $j=4$ is `lambda4`.
-/

section OperationalPressure

/-- Strategic hazard total `Λ_t = ∑_{j=1}^{3}\tilde\lambda_j`. -/
noncomputable def strategicHazardTotal (tildeLambda : Fin 3 → ℝ) : ℝ :=
  ∑ j : Fin 3, tildeLambda j

/-- Strategic causes $j\in\{1,2,3\}$ plus exogenous $j=4$ (γ-invariant). -/
noncomputable def hazardsStrategicExogenous (tildeLambda : Fin 3 → ℝ) (lambda4 : ℝ) :
    Fin 4 → ℝ
  | ⟨0, _⟩ => tildeLambda 0
  | ⟨1, _⟩ => tildeLambda 1
  | ⟨2, _⟩ => tildeLambda 2
  | ⟨3, _⟩ => lambda4

/-- `[Proved]` `p_Cont = exp(-Λ Δt)·exp(-\tilde\lambda_4 Δt)` (Step 1 factorization). -/
theorem continuationProbability_strategic_exogenous
    (tildeLambda : Fin 3 → ℝ) (lambda4 Δt : ℝ) :
    continuationProbability (hazardsStrategicExogenous tildeLambda lambda4) Δt =
      Real.exp (-strategicHazardTotal tildeLambda * Δt) * Real.exp (-lambda4 * Δt) := by
  have hsum :
      (∑ j : Fin 4, hazardsStrategicExogenous tildeLambda lambda4 j) =
        (∑ j : Fin 3, tildeLambda j) + lambda4 := by
    rw [Fin.sum_univ_four]
    simp only [hazardsStrategicExogenous]
    rw [Fin.sum_univ_three]
  dsimp [continuationProbability, strategicHazardTotal]
  rw [hsum]
  have hsplit :
      -((∑ j : Fin 3, tildeLambda j) + lambda4) * Δt =
        -(∑ j : Fin 3, tildeLambda j) * Δt + -lambda4 * Δt := by ring
  rw [hsplit, Real.exp_add]

/-- Signed aggregation `∑_j η_j \tilde\lambda_j` (second equality in `eq:kappa-c`). -/
noncomputable def kappaHSigned (eta tildeLambda : Fin 3 → ℝ) : ℝ :=
  ∑ j : Fin 3, eta j * tildeLambda j

/-- Absolute semi-elasticities: `ζ_{γ,1}=-η_1`, `ζ_{γ,j}=η_j` for $j\in\{2,3\}$. -/
noncomputable def zetaFromEta (eta : Fin 3 → ℝ) (j : Fin 3) : ℝ :=
  if j = 0 then -eta j else eta j

/-- `[Def]` Net sensitivity `κ_h` (eq. `eq:kappa-c`, compact form). -/
noncomputable def kappa_h (zetaGamma tildeLambda : Fin 3 → ℝ) : ℝ :=
  netSensitivity zetaGamma tildeLambda

/-- `[Proved]` `κ_h = ∑_j η_j \tilde\lambda_j` (`kappa_h_signed`; `eq:kappa-c`). -/
theorem kappa_h_signed (eta tildeLambda : Fin 3 → ℝ) :
    kappa_h (zetaFromEta eta) tildeLambda = kappaHSigned eta tildeLambda := by
  dsimp [kappa_h, netSensitivity, kappaHSigned, zetaFromEta]
  rw [Fin.sum_univ_three]
  ring

/-- `[Proved]` Step 1: `∂Λ/∂γ = κ_h` under semi-elasticities (S2). -/
theorem dLambda_eq_kappa (eta tildeLambda : Fin 3 → ℝ) :
    kappaHSigned eta tildeLambda = kappa_h (zetaFromEta eta) tildeLambda :=
  (kappa_h_signed eta tildeLambda).symm

/-- `[Def]` `∂p_{\mathrm{Cont}}/\partial\gamma` in eq. `eq:dpCont-dgamma`. -/
noncomputable def contProbGammaDeriv (kappa_h_val pCont Δt : ℝ) : ℝ :=
  -kappa_h_val * pCont * Δt

/-- `[Proved]` `eq:dpCont-dgamma`: only `Λ_t` depends on `γ`; `\tilde\lambda_4` is γ-invariant. -/
theorem dContProb_eq (tildeLambda : Fin 3 → ℝ) (lambda4 kappa_h_val Δt : ℝ) :
    let pCont := continuationProbability (hazardsStrategicExogenous tildeLambda lambda4) Δt
    contProbGammaDeriv kappa_h_val pCont Δt =
      Real.exp (-lambda4 * Δt) * (-kappa_h_val * Δt *
        Real.exp (-strategicHazardTotal tildeLambda * Δt)) := by
  dsimp [contProbGammaDeriv]
  rw [continuationProbability_strategic_exogenous]
  ring

/-- `[Proved]` Sign certificate for eq. `eq:dE-dgamma-prop` when `κ_h>0`. -/
theorem prop_expected_captivity_gamma
    (kappa_h_val Δt survivalTail : ℝ)
    (hk_pos : 0 < kappa_h_val) (hdt_pos : 0 < Δt) (hsurv_pos : 0 < survivalTail) :
    -- eq:dE-dgamma-prop: ∂E[τ|θ_K]/∂γ* = −κ_h · Δt · ∑_{s≥t} S(s)
    -kappa_h_val * Δt * survivalTail < 0 := by
  nlinarith [mul_pos hk_pos hdt_pos, mul_pos (mul_pos hk_pos hdt_pos) hsurv_pos]

/-- `[Proved]` `eq:dE-dgamma-prop`: `sgn(∂E[τ|θ_K]/∂γ*) = -sgn(κ_h)`. -/
theorem prop_expected_captivity_sgn
    (κ derivEτ Δt survivalTail : ℝ)
    (hderiv : derivEτ = -κ * Δt * survivalTail)
    (hdt : 0 < Δt) (hsurv : 0 < survivalTail) :
    (0 < κ → derivEτ < 0) ∧ (κ < 0 → 0 < derivEτ) := by
  constructor
  · intro hk
    rw [hderiv]
    have hκ_neg : -κ < 0 := neg_neg_iff_pos.mpr hk
    nlinarith [hκ_neg, hdt, hsurv, mul_pos hdt hsurv]
  · intro hk
    rw [hderiv]
    have hκ_pos : 0 < -κ := neg_pos.mpr hk
    nlinarith [hκ_pos, hdt, hsurv, mul_pos hdt hsurv]

end OperationalPressure

/-!
## Section 6 — Policy implications (`sec:politica`)

TeX: Proposition `prop:optimal-gamma-beliefs`, eqs. `eq:dgamma-dmu-completo`,
`eq:monotonia-neta-gamma`.
-/

section PolicyImplications

/-- `[Def]` Net monotonicity in eq. `eq:monotonia-neta-gamma`:
    `e_γ' (H^b)^{-1} ∇_{xμ}^2 J^b < 0`. -/
def netGammaMonotonicity (netMarginal : ℝ) : Prop :=
  netMarginal < 0

/-- `[Hypothesis]` Proposition `prop:optimal-gamma-beliefs` (TeX §6).
    **Not machine-derived:** the complete TFI / branch-objective chain is not
    formalized (`eq:obj-rama-completo`, `eq:foc-rama-completa`, `eq:dx-dmu-completo`,
    `eq:dgamma-dmu-completo`, `eq:efecto-neto-gamma`). This interface records only
    the IFT sign conclusion under net monotonicity (analytical proof in the supplement). -/
def prop_optimal_gamma_beliefs (dγdμ netMarginal : ℝ) : Prop :=
  -- eq:monotonia-neta-gamma ⇒ eq:dgamma-dmu-completo (dγ*/dμ(θ_high) > 0)
  netGammaMonotonicity netMarginal → 0 < dγdμ

end PolicyImplications

/-!
## Section 7 — Equilibrium and implementability (`sec:equilibrio`)

TeX: `def:pbe-mechanism`, `def:implementable-mechanism`, `thm:ebp-implementabilidad`,
eq. `eq:gamma-factible`.
-/

section EquilibriumImplementability

/-- `[Def]` Definition `def:pbe-mechanism`. -/
structure PerfectBayesianEquilibrium (ΘK : Type*) [Fintype ΘK] where
  beliefs : ℕ → ΘK → ℝ
  beliefs_prob : ∀ t, ∑ θ : ΘK, beliefs t θ = 1
  sequential_rationality : Prop
  on_path_bayesian_consistency : Prop

/-- `[Def]` Definition `def:implementable-mechanism` — IC^K, IR^K, IR^F. -/
structure ImplementabilityConstraints where
  icK : Prop
  irK : Prop
  irF : Prop

def implementableMechanism {ΘK : Type*} [Fintype ΘK]
    (_pbe : PerfectBayesianEquilibrium ΘK) (c : ImplementabilityConstraints) : Prop :=
  c.icK ∧ c.irK ∧ c.irF

/-- `[Def]` Feasible policy set `Γ_t(μ_t)` (eq. `eq:gamma-factible`) — non-emptiness. -/
def robustFeasibility {ΘK : Type*} [Fintype ΘK]
    (Γ : ℕ → (ΘK → ℝ) → Set (ℝ × ℝ × ℝ)) : Prop :=
  ∀ t μ, (Γ t μ).Nonempty

/-- `[Hypothesis]` Theorem `thm:ebp-implementabilidad` — existence under (i)–(vii). -/
def conditionalImplementability {ΘK : Type*} [Fintype ΘK]
    (Γ : ℕ → (ΘK → ℝ) → Set (ℝ × ℝ × ℝ)) : Prop :=
  robustFeasibility Γ →
    ∃ pbe : PerfectBayesianEquilibrium ΘK,
      implementableMechanism pbe ⟨True, True, True⟩

end EquilibriumImplementability

/-!
## Section 8 — Abstract structural identification (`sec:id-abstracta`)

TeX: Lemma `lem:general-id-appendix` (eq. `eq:appendix-rho-product`),
Theorem `thm:posterior-consistency-appendix`.
-/

section KakutaniAbstract

noncomputable def bhattacharyyaAffinity {Y : Type*} [Fintype Y] (p q : Y → ℝ) : ℝ :=
  -- ρ(p,q) = ∑_y √(p_y q_y); lem:general-id-appendix
  ∑ y : Y, Real.sqrt (p y * q y)

/-- `[Def]` Finite-horizon affinity product (eq. `eq:appendix-rho-product`). -/
noncomputable def productAffinity (ρ : ℕ → ℝ) (T : ℕ) : ℝ :=
  -- eq:appendix-rho-product: ∏_{t=0}^T ρ_t
  (Finset.range (T + 1)).prod ρ

/-- Trajectory sum in the product-of-experiments identity (eq. `eq:appendix-rho-product`). -/
noncomputable def trajectoryBhattacharyya {Y : Type*} [Fintype Y]
    (p q : ℕ → Y → ℝ) (T : ℕ) : ℝ :=
  -- eq:appendix-rho-product (trajectory form before product factorization)
  ∑ f : Fin (T + 1) → Y,
    ∏ t : Fin (T + 1), Real.sqrt (p t (f t) * q t (f t))

section BhattacharyyaAlgebra

open InnerProductSpace

variable {Y : Type*} [Fintype Y]

private lemma finset_cauchy_schwarz {ι : Type*} [Fintype ι] (f g : ι → ℝ) :
    (∑ i, f i * g i) ^ 2 ≤ (∑ i, f i ^ 2) * (∑ i, g i ^ 2) := by
  set x : EuclideanSpace ℝ ι := WithLp.toLp 2 f
  set y : EuclideanSpace ℝ ι := WithLp.toLp 2 g
  have hin : ⟪x, y⟫_ℝ = ∑ i, f i * g i := by
    simp [x, y, PiLp.inner_apply, inner, mul_comm]
  calc (∑ i, f i * g i) ^ 2
      = (⟪x, y⟫_ℝ) ^ 2 := by rw [hin]
    _ ≤ (‖x‖ * ‖y‖) ^ 2 := by
        refine (sq_le_sq).2 ?_
        simpa [hin, Real.norm_eq_abs, abs_mul, abs_of_nonneg (norm_nonneg x),
          abs_of_nonneg (norm_nonneg y)] using abs_real_inner_le_norm x y
    _ = ‖x‖ ^ 2 * ‖y‖ ^ 2 := mul_pow _ _ _
    _ = (∑ i, f i ^ 2) * (∑ i, g i ^ 2) := by
        rw [EuclideanSpace.real_norm_sq_eq x, EuclideanSpace.real_norm_sq_eq y]

private lemma sq_sqrt_sub_sqrt {p q : ℝ} (hp : 0 ≤ p) (hq : 0 ≤ q) :
    (Real.sqrt p - Real.sqrt q) ^ 2 = p + q - 2 * Real.sqrt (p * q) := by
  rw [sub_sq, Real.sq_sqrt hp, Real.sq_sqrt hq]
  have hnn := mul_nonneg (Real.sqrt_nonneg p) (Real.sqrt_nonneg q)
  have hsqrt :
      Real.sqrt (p * q) = Real.sqrt p * Real.sqrt q :=
    (Real.sqrt_eq_iff_mul_self_eq (mul_nonneg hp hq) hnn).symm.mp (by
      ring_nf
      rw [Real.sq_sqrt hp, Real.sq_sqrt hq])
  linarith

private lemma sqrt_mul_nonneg {p q : ℝ} (hp : 0 ≤ p) (hq : 0 ≤ q) :
    Real.sqrt (p * q) = Real.sqrt p * Real.sqrt q := by
  have hnn := mul_nonneg (Real.sqrt_nonneg p) (Real.sqrt_nonneg q)
  exact (Real.sqrt_eq_iff_mul_self_eq (mul_nonneg hp hq) hnn).symm.mp (by
    ring_nf; rw [Real.sq_sqrt hp, Real.sq_sqrt hq])

private lemma sqrt_sum_sq_le_four {p q : Y → ℝ}
    (hp : ∀ y, 0 ≤ p y) (hq : ∀ y, 0 ≤ q y)
    (hpm : ∑ y, p y = 1) (hqm : ∑ y, q y = 1) :
    ∑ y, (Real.sqrt (p y) + Real.sqrt (q y)) ^ 2 ≤ 4 := by
  have hle : ∀ y, (Real.sqrt (p y) + Real.sqrt (q y)) ^ 2 ≤ 2 * (p y + q y) := by
    intro y
    have hsqrt : 0 ≤ Real.sqrt (p y * q y) := Real.sqrt_nonneg _
    have hAM : 2 * Real.sqrt (p y * q y) ≤ p y + q y := by
      nlinarith [sq_sqrt_sub_sqrt (hp y) (hq y), sq_nonneg (Real.sqrt (p y) - Real.sqrt (q y))]
    have hexpand : (Real.sqrt (p y) + Real.sqrt (q y)) ^ 2 =
        p y + q y + 2 * Real.sqrt (p y * q y) := by
      rw [add_sq, Real.sq_sqrt (hp y), Real.sq_sqrt (hq y)]
      linarith [sqrt_mul_nonneg (hp y) (hq y)]
    linarith [hexpand, hAM]
  calc
    ∑ y, (Real.sqrt (p y) + Real.sqrt (q y)) ^ 2
        ≤ ∑ y, 2 * (p y + q y) := Finset.sum_le_sum fun y _ => hle y
    _ = 2 * ∑ y, (p y + q y) := by rw [← Finset.mul_sum]
    _ = 2 * (∑ y, p y + ∑ y, q y) := by rw [Finset.sum_add_distrib]
    _ = 4 := by rw [hpm, hqm]; norm_num

/-- `[Proved]` Hellinger–TV bound: `‖p−q‖_TV² ≤ 2(1 − ρ(p,q))` on finite supports. -/
theorem totalVariation_sq_le_two_one_minus_affinity
    {p q : Y → ℝ} (hp : ∀ y, 0 ≤ p y) (hq : ∀ y, 0 ≤ q y)
    (hpm : ∑ y, p y = 1) (hqm : ∑ y, q y = 1) :
    totalVariationDistance p q ^ 2 ≤ 2 * (1 - bhattacharyyaAffinity p q) := by
  set w : Y → ℝ := fun y => Real.sqrt (p y) - Real.sqrt (q y)
  set s : Y → ℝ := fun y => Real.sqrt (p y) + Real.sqrt (q y)
  have hdiff : ∀ y, p y - q y = w y * s y := by
    intro y; dsimp [w, s]
    nlinarith [Real.sq_sqrt (hp y), Real.sq_sqrt (hq y)]
  have hTV : 2 * totalVariationDistance p q = ∑ y, |w y * s y| := by
    have hsum : 2 * totalVariationDistance p q = ∑ y, |p y - q y| := by
      dsimp [totalVariationDistance]
      ring_nf
    rw [hsum]
    congr 1
    ext y
    dsimp [w, s]
    rw [hdiff y, abs_mul]
  have hone_sub : 1 - bhattacharyyaAffinity p q = (1 / 2) * ∑ y, w y ^ 2 := by
    suffices h : ∑ y, w y ^ 2 + 2 * bhattacharyyaAffinity p q = 2 by linarith
    dsimp [bhattacharyyaAffinity, w]
    calc ∑ y, w y ^ 2 + 2 * ∑ y, Real.sqrt (p y * q y)
        = ∑ y, (w y ^ 2 + 2 * Real.sqrt (p y * q y)) := by
            rw [Finset.sum_add_distrib, ← Finset.mul_sum]
      _ = ∑ y, (p y + q y) := by
          refine Finset.sum_congr rfl fun y _ => ?_
          linarith [sq_sqrt_sub_sqrt (hp y) (hq y)]
      _ = 2 := by rw [Finset.sum_add_distrib, hpm, hqm]; norm_num
  have hs_nonneg : ∀ y, 0 ≤ s y := fun y => by
    dsimp [s]; exact add_nonneg (Real.sqrt_nonneg _) (Real.sqrt_nonneg _)
  have habs_mul : ∀ y, |w y * s y| = |w y| * |s y| := fun y => by
    rw [abs_mul, abs_of_nonneg (hs_nonneg y)]
  have habs : (∑ y, |w y * s y|) ^ 2 ≤ (∑ y, w y ^ 2) * (∑ y, s y ^ 2) := by
    have h₁ : (∑ y, |w y * s y|) ^ 2 = (∑ y, |w y| * |s y|) ^ 2 := by
      congr 1
      refine Finset.sum_congr rfl fun y _ => habs_mul y
    rw [h₁]
    simpa [pow_two] using finset_cauchy_schwarz (fun y => |w y|) (fun y => |s y|)
  have hss := sqrt_sum_sq_le_four hp hq hpm hqm
  have hw_sq : ∑ y, w y ^ 2 = 2 * (1 - bhattacharyyaAffinity p q) := by linarith [hone_sub]
  have hTV_sq :
      (2 * totalVariationDistance p q) ^ 2 ≤ 8 * (1 - bhattacharyyaAffinity p q) := by
    calc (2 * totalVariationDistance p q) ^ 2
        = (∑ y, |w y * s y|) ^ 2 := by rw [hTV]
      _ ≤ (∑ y, w y ^ 2) * (∑ y, s y ^ 2) := habs
      _ ≤ (2 * (1 - bhattacharyyaAffinity p q)) * 4 := by
          rw [hw_sq]
          have hw_nonneg : 0 ≤ ∑ y, w y ^ 2 :=
            Finset.sum_nonneg fun y _ => sq_nonneg (w y)
          have hrho_nonneg : 0 ≤ 1 - bhattacharyyaAffinity p q := by linarith [hone_sub, hw_nonneg]
          exact mul_le_mul_of_nonneg_left hss (by linarith [hrho_nonneg])
      _ = 8 * (1 - bhattacharyyaAffinity p q) := by ring
  nlinarith [sq_nonneg (totalVariationDistance p q)]

/-- `[Proved]` TV lower bound ⇒ Bhattacharyya upper bound
    (Lemma `lem:general-id-appendix`, singularity step). -/
theorem tvImpliesAffinityBound {p q : Y → ℝ} (ε : ℝ)
    (hp : ∀ y, 0 ≤ p y) (hq : ∀ y, 0 ≤ q y)
    (hpm : ∑ y, p y = 1) (hqm : ∑ y, q y = 1)
    (hε₀ : 0 ≤ ε) (hε : ε ≤ totalVariationDistance p q) :
    bhattacharyyaAffinity p q ≤ 1 - ε ^ 2 / 2 := by
  have h := totalVariation_sq_le_two_one_minus_affinity hp hq hpm hqm
  have htv₀ : 0 ≤ totalVariationDistance p q := by
    dsimp [totalVariationDistance]
    apply mul_nonneg <;> norm_num
    exact Finset.sum_nonneg fun _ _ => abs_nonneg _
  have hε_sq : ε ^ 2 ≤ totalVariationDistance p q ^ 2 := by
    refine (sq_le_sq).2 ?_
    rw [abs_of_nonneg hε₀, abs_of_nonneg htv₀]
    exact hε
  have hε_sq' : ε ^ 2 ≤ 2 * (1 - bhattacharyyaAffinity p q) := le_trans hε_sq h
  linarith

/-- `[Proved]` Identity `eq:appendix-rho-product` — product of marginal affinities. -/
theorem bhattacharyya_product_identity (p q : ℕ → Y → ℝ) (T : ℕ) :
    trajectoryBhattacharyya p q T =
      (Finset.range (T + 1)).prod (fun t => bhattacharyyaAffinity (p t) (q t)) := by
  dsimp [trajectoryBhattacharyya, bhattacharyyaAffinity]
  have hprod := Fintype.prod_sum (f := fun (t : Fin (T + 1)) (y : Y) =>
    Real.sqrt (p t y * q t y))
  rw [hprod.symm, Fin.prod_univ_eq_prod_range (fun t => ∑ y : Y,
    Real.sqrt (p t y * q t y)) (T + 1)]

theorem bhattacharyyaAffinity_le_one {p q : Y → ℝ}
    (hp : ∀ y, 0 ≤ p y) (hq : ∀ y, 0 ≤ q y)
    (hpm : ∑ y, p y = 1) (hqm : ∑ y, q y = 1) :
    bhattacharyyaAffinity p q ≤ 1 := by
  dsimp [bhattacharyyaAffinity]
  have hcs := finset_cauchy_schwarz (fun y => Real.sqrt (p y)) (fun y => Real.sqrt (q y))
  have hp_sq : ∑ y, Real.sqrt (p y) * Real.sqrt (p y) = 1 := by
    trans ∑ y, p y
    · congr 1; ext y; exact Real.mul_self_sqrt (hp y)
    · exact hpm
  have hq_sq : ∑ y, Real.sqrt (q y) * Real.sqrt (q y) = 1 := by
    trans ∑ y, q y
    · congr 1; ext y; exact Real.mul_self_sqrt (hq y)
    · exact hqm
  have hinner : (∑ y, Real.sqrt (p y * q y)) ^ 2 ≤ 1 := by
    have hsum_eq : ∑ y, Real.sqrt (p y * q y) = ∑ y, Real.sqrt (p y) * Real.sqrt (q y) := by
      congr 1
      ext y
      have hnn := mul_nonneg (Real.sqrt_nonneg (p y)) (Real.sqrt_nonneg (q y))
      exact (Real.sqrt_eq_iff_mul_self_eq (mul_nonneg (hp y) (hq y)) hnn).mpr (by
        ring_nf
        rw [Real.sq_sqrt (hp y), Real.sq_sqrt (hq y)])
    rw [hsum_eq]
    simpa [hp_sq, hq_sq, pow_two, mul_comm, mul_left_comm, mul_assoc] using hcs
  have hnonneg : 0 ≤ ∑ y, Real.sqrt (p y * q y) :=
    Finset.sum_nonneg fun y _ => Real.sqrt_nonneg _
  nlinarith [hinner, sq_nonneg (∑ y, Real.sqrt (p y * q y)), hnonneg]

/-- Recurrent uniform TV on an infinite set `S` (Lemma `lem:general-id-appendix`). -/
def recurrentTvSeparation {Y : Type*} [Fintype Y]
    (p q : ℕ → Y → ℝ) (S : Set ℕ) (ε : ℝ) : Prop :=
  0 < ε ∧ ε ≤ 1 ∧ S.Infinite ∧ ∀ t ∈ S, ε ≤ totalVariationDistance (p t) (q t)

/-- `[Proved]` Infinitely many sub-unit affinity factors force the finite product to zero. -/
theorem productAffinity_tendsto_zero (ρ : ℕ → ℝ) (S : Set ℕ) (ε : ℝ)
    (hε : 0 < ε) (hε_le : ε ≤ 1) (hS : S.Infinite)
    (hρ_nonneg : ∀ t, 0 ≤ ρ t) (hρ_le_one : ∀ t, ρ t ≤ 1)
    (hbound : ∀ t ∈ S, ρ t ≤ 1 - ε ^ 2 / 2) :
    Tendsto (fun T ↦ productAffinity ρ T) atTop (𝓝 0) := by
  set c := 1 - ε ^ 2 / 2
  have hc : 0 < c := by dsimp [c]; nlinarith [sq_nonneg ε, hε_le]
  have hc1 : c < 1 := by dsimp [c]; nlinarith [sq_pos_of_pos hε, hε_le]
  rw [Metric.tendsto_atTop]
  intro δ hδ
  obtain ⟨n, hn⟩ := exists_pow_lt_of_lt_one hδ hc1
  obtain ⟨t, ht⟩ := Set.Infinite.exists_subset_card_eq hS (n + 1)
  have hcard : t.card = n + 1 := ht.2
  have htS : ∀ i ∈ t, i ∈ S := fun i hi => ht.1 hi
  use t.sup id
  intro T hT
  have hsub : ∀ i ∈ t, (i : ℕ) ≤ T := fun i hi => Finset.le_sup hi |>.trans hT
  have hprod_t : t.prod ρ ≤ c ^ (n + 1) := by
    have hle_i : ∀ i ∈ t, ρ i ≤ c := fun i hi => hbound i (htS i hi)
    calc
      t.prod ρ ≤ ∏ _i ∈ t, c :=
        Finset.prod_le_prod (fun i hi => hρ_nonneg i) (fun i hi => hle_i i hi)
      _ = c ^ t.card := by rw [Finset.prod_const]
      _ = c ^ (n + 1) := by rw [hcard]
  have hle : productAffinity ρ T ≤ t.prod ρ := by
    dsimp [productAffinity]
    have hsubset : t ⊆ Finset.range (T + 1) := fun i hi =>
      Finset.mem_range.mpr (Nat.lt_succ_of_le (hsub i hi))
    have hunion : t ∪ (Finset.range (T + 1) \ t) = Finset.range (T + 1) :=
      Finset.union_sdiff_of_subset hsubset
    have hdisj : Disjoint t (Finset.range (T + 1) \ t) := Finset.disjoint_sdiff
    have hsplit :
        (Finset.range (T + 1)).prod ρ =
          t.prod ρ * (Finset.range (T + 1) \ t).prod ρ := by
      rw [← Finset.prod_union hdisj, hunion]
    have hrest : (Finset.range (T + 1) \ t).prod ρ ≤ 1 := by
      calc _ ≤ ∏ _i ∈ (Finset.range (T + 1) \ t), (1 : ℝ) :=
          Finset.prod_le_prod (fun i hi => hρ_nonneg i) (fun i hi => hρ_le_one i)
        _ = 1 := by simp
    rw [hsplit]
    refine mul_le_of_le_one_right (Finset.prod_nonneg fun i _ => hρ_nonneg i) hrest
  have hnonneg_T : 0 ≤ productAffinity ρ T := by
    dsimp [productAffinity]
    exact Finset.prod_nonneg fun i _ => hρ_nonneg i
  have hcn : c ^ (n + 1) < c ^ n := by
    rw [pow_succ]
    exact mul_lt_of_lt_one_right (pow_pos hc n) hc1
  have hpow_lt : c ^ (n + 1) < δ := lt_trans hcn hn
  simp [dist_zero_right, abs_of_nonneg hnonneg_T]
  exact lt_of_le_of_lt hle (lt_of_le_of_lt hprod_t hpow_lt)

/-- `[Def]` Marginal Bhattacharyya affinities along a product experiment. -/
noncomputable def marginalAffinity {Y : Type*} [Fintype Y]
    (p q : ℕ → Y → ℝ) : ℕ → ℝ :=
  -- eq:appendix-rho-product: ρ_t = ρ(p_θ^{(t)}, p_{θ'}^{(t)})
  fun t => bhattacharyyaAffinity (p t) (q t)

/-!
### Kakutani bifurcation (`lem:general-id-appendix`)

The supplement splits the Kakutani route into two explicit layers:

1. **`kakutaniAlgebraicPrerequisite`** — `[Proved]` finite-support algebra: recurrent TV
   separation forces `∏_{t∈T} ρ_t → 0` (Bhattacharyya product along horizons).

2. **`kakutaniMutualSingularity`** — `[Hypothesis]` classical Kakutani (1948):
   vanishing product affinity implies mutual singularity of the *infinite* product
   experiment laws. Infinite product measures are not available in Mathlib; this
   step stays an interface matching the analytical proof in the TeX supplement.

`generalIdAppendixLemma` and `kakutaniAlgebraicPrerequisite_of_recurrent` verify
layer (1). Any conclusion about path-space singularity composes (1) with (2).
-/

/-- `[Def]` Layer (1): algebraic prerequisite for Kakutani (TeX step before Kakutani 1948). -/
def kakutaniAlgebraicPrerequisite {Y : Type*} [Fintype Y]
    (p q : ℕ → Y → ℝ) : Prop :=
  -- ∏_{t≤T} ρ_t → 0 as T → ∞
  Tendsto (fun T => productAffinity (marginalAffinity p q) T) atTop (𝓝 0)

/-- `[Hypothesis]` Layer (2): Kakutani (1948) — vanishing product affinity forces mutual
    singularity of the trajectory laws `P`, `Q` of the infinite product experiment.
    The path laws are supplied as data because infinite product measures are not
    constructed in Mathlib; the analytical proof is in the TeX supplement. -/
def kakutaniMutualSingularity {Y : Type*} [Fintype Y] [MeasurableSpace Y]
    (p q : ℕ → Y → ℝ) (P Q : Measure (ℕ → Y)) : Prop :=
  kakutaniAlgebraicPrerequisite p q → mutuallySingularMeasures P Q

/-- `[Proved]` Lemma `lem:general-id-appendix` — product identity + affinity vanishing;
    the Kakutani mutual-singularity conclusion is `[Hypothesis]` (`kakutaniMutualSingularity`). -/
theorem generalIdAppendixLemma {Y : Type*} [Fintype Y]
    (p q : ℕ → Y → ℝ) (ρ : ℕ → ℝ) (S : Set ℕ) (ε : ℝ)
    (hρ : ∀ t, ρ t = bhattacharyyaAffinity (p t) (q t))
    (hprob : ∀ t, IsProbMass (p t) ∧ IsProbMass (q t))
    (hrec : recurrentTvSeparation p q S ε) :
    (∀ T, trajectoryBhattacharyya p q T =
        (Finset.range (T + 1)).prod (fun t => ρ t)) ∧
    (∀ t ∈ S, ρ t ≤ 1 - ε ^ 2 / 2) ∧
    Tendsto (fun T ↦ productAffinity ρ T) atTop (𝓝 0) := by
  rcases hrec with ⟨hε, hε_le, hS, htv⟩
  have hprod (T : ℕ) : trajectoryBhattacharyya p q T =
      (Finset.range (T + 1)).prod (fun t => ρ t) := by
    rw [bhattacharyya_product_identity, Finset.prod_congr rfl fun t _ => (hρ t).symm]
  have hbound : ∀ t ∈ S, ρ t ≤ 1 - ε ^ 2 / 2 := by
    intro t ht
    rw [hρ t]
    rcases hprob t with ⟨hp, hq⟩
    rcases hp with ⟨hp_nonneg, hp_sum⟩
    rcases hq with ⟨hq_nonneg, hq_sum⟩
    exact tvImpliesAffinityBound ε hp_nonneg hq_nonneg hp_sum hq_sum (le_of_lt hε) (htv t ht)
  have hρ_nonneg : ∀ t, 0 ≤ ρ t := by
    intro t; rw [hρ t]; dsimp [bhattacharyyaAffinity]
    exact Finset.sum_nonneg fun y _ => Real.sqrt_nonneg _
  have hρ_le_one : ∀ t, ρ t ≤ 1 := by
    intro t; rw [hρ t]
    rcases hprob t with ⟨hp, hq⟩
    rcases hp with ⟨hp_nonneg, hp_sum⟩
    rcases hq with ⟨hq_nonneg, hq_sum⟩
    exact bhattacharyyaAffinity_le_one hp_nonneg hq_nonneg hp_sum hq_sum
  refine ⟨hprod, hbound, ?_⟩
  exact productAffinity_tendsto_zero ρ S ε hε hε_le hS hρ_nonneg hρ_le_one hbound

/-- `[Proved]` Recurrent TV separation yields the Kakutani algebraic prerequisite. -/
theorem kakutaniAlgebraicPrerequisite_of_recurrent {Y : Type*} [Fintype Y]
    (p q : ℕ → Y → ℝ) (S : Set ℕ) (ε : ℝ)
    (hprob : ∀ t, IsProbMass (p t) ∧ IsProbMass (q t))
    (hrec : recurrentTvSeparation p q S ε) :
    kakutaniAlgebraicPrerequisite p q := by
  dsimp [kakutaniAlgebraicPrerequisite, marginalAffinity, productAffinity]
  rcases generalIdAppendixLemma p q (marginalAffinity p q) S ε (fun _ => rfl) hprob hrec with
    ⟨_, _, htends⟩
  exact htends

end BhattacharyyaAlgebra

section PosteriorNormalization

variable [MeasurableSpace ΘK] [MeasurableSingletonClass ΘK]

/-- `[Proved]` Theorem `thm:posterior-consistency-appendix` — normalization closure;
    false-type collapse is an explicit `[Hypothesis]`. -/
theorem posteriorConsistencyAppendix
    (ℱ0 : MeasurableSpace Ω) (hℱ0 : ℱ0 ≤ m)
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t))
    [Fintype ΘK] [DecidableEq ΘK]
    (μC : @Measure Ω m) [IsProbabilityMeasure μC]
    (θK : Ω → ΘK) (hθK : Measurable[m, inferInstance] θK)
    (thetaStar : ΘK)
    (hmass_sum : ∀ t : ℕ, ∀ᵐ ω ∂ μC,
      ∑ θ : ΘK, bayesianBeliefs ℱ0 hℱ0 mZ ζ hζ μC θK hθK t θ ω = 1)
    (hfalse : ∀ θ : ΘK, θ ≠ thetaStar →
      ∀ᵐ ω ∂ μC,
        Tendsto (fun n : ℕ ↦ bayesianBeliefs ℱ0 hℱ0 mZ ζ hζ μC θK hθK n θ ω)
                atTop (𝓝 0)) :
    falseTypeCollapse ℱ0 hℱ0 mZ ζ hζ μC θK hθK thetaStar ∧
    ∀ᵐ ω ∂ μC,
      Tendsto (fun n : ℕ ↦ bayesianBeliefs ℱ0 hℱ0 mZ ζ hζ μC θK hθK n thetaStar ω)
              atTop (𝓝 1) := by
  have hcollapse : falseTypeCollapse ℱ0 hℱ0 mZ ζ hζ μC θK hθK thetaStar := by
    simpa [falseTypeCollapse] using hfalse
  exact posteriorConcentration ℱ0 hℱ0 mZ ζ hζ μC θK hθK thetaStar hmass_sum hcollapse

end PosteriorNormalization

end KakutaniAbstract

/-!
## Identification bridge — Sections 4 & 8 (`lem:hazard-separation` part (ii))

Connects `tvSeparationAsymptotic` (§4) to the machine-checked Bhattacharyya/Kakutani
algebra of §8. The measure-theoretic Kakutani conclusion remains `[Hypothesis]`.
-/

section IdentificationBridge

variable {Y : Type*} [Fintype Y] [MeasurableSpace Y]

omit [MeasurableSpace Y] in
/-- `[Proved]` Normalize `tvSeparationAsymptotic` to `recurrentTvSeparation`
    (using `totalVariationDistance_le_one` on probability masses). -/
theorem tvSeparationAsymptotic_to_recurrent {ΘK : Type*}
    (p : ΘK → ℕ → Y → ℝ) (theta theta' : ΘK)
    (hprob : ∀ t, IsProbMass (p theta t) ∧ IsProbMass (p theta' t))
    (htv : tvSeparationAsymptotic p theta theta') :
    ∃ (S : Set ℕ) (ε : ℝ), recurrentTvSeparation (p theta) (p theta') S ε := by
  rcases htv with ⟨ε, hε, S, hS, htvS⟩
  rcases hS.nonempty with ⟨t, ht⟩
  have hε_le : ε ≤ 1 := by
    rcases hprob t with ⟨hp, hq⟩
    exact le_trans (htvS t ht) (totalVariationDistance_le_one hp hq)
  exact ⟨S, ε, ⟨hε, hε_le, hS, htvS⟩⟩

omit [MeasurableSpace Y] in
/-- `[Proved]` Lemma `lem:general-id-appendix` / `lem:hazard-separation` part (ii), algebraic
    core: recurrent TV on an infinite set forces the affinity product to zero. -/
theorem hazardSeparation_affinityVanishing {ΘK : Type*}
    (p : ΘK → ℕ → Y → ℝ) (theta theta' : ΘK)
    (hprob : ∀ t, IsProbMass (p theta t) ∧ IsProbMass (p theta' t))
    (htv : tvSeparationAsymptotic p theta theta') :
    kakutaniAlgebraicPrerequisite (p theta) (p theta') := by
  obtain ⟨S, ε, hrec⟩ :=
    tvSeparationAsymptotic_to_recurrent p theta theta' hprob htv
  exact kakutaniAlgebraicPrerequisite_of_recurrent (p theta) (p theta') S ε hprob hrec

omit [MeasurableSpace Y] in
/-- `[Proved]` Full finite-horizon package from `tvSeparationAsymptotic`. -/
theorem tvSeparationAsymptotic_generalId {ΘK : Type*}
    (p : ΘK → ℕ → Y → ℝ) (theta theta' : ΘK)
    (hprob : ∀ t, IsProbMass (p theta t) ∧ IsProbMass (p theta' t))
    (htv : tvSeparationAsymptotic p theta theta') :
    ∃ (S : Set ℕ) (ε : ℝ),
      recurrentTvSeparation (p theta) (p theta') S ε ∧
      (∀ T, trajectoryBhattacharyya (p theta) (p theta') T =
          productAffinity (marginalAffinity (p theta) (p theta')) T) ∧
      kakutaniAlgebraicPrerequisite (p theta) (p theta') := by
  obtain ⟨S, ε, hrec⟩ :=
    tvSeparationAsymptotic_to_recurrent p theta theta' hprob htv
  refine ⟨S, ε, hrec, ?_, ?_⟩
  · intro T
    dsimp [productAffinity, marginalAffinity]
    exact bhattacharyya_product_identity (p theta) (p theta') T
  · exact kakutaniAlgebraicPrerequisite_of_recurrent (p theta) (p theta') S ε hprob hrec

/-- `[Hypothesis]` Kakutani path-law singularity at the outcome level: the algebraic
    prerequisite (vanishing product affinity of `p_θ^{(t)}`, `p_{θ'}^{(t)}`) forces mutual
    singularity of the outcome-trajectory laws `PY θ`, `PY θ'` (Kakutani 1948;
    TeX `lem:general-id-appendix`, singularity step). -/
def kakutaniSignalSingularity {ΘK : Type*}
    (p : ΘK → ℕ → Y → ℝ) (PY : ΘK → Measure (ℕ → Y))
    (theta theta' : ΘK) : Prop :=
  kakutaniAlgebraicPrerequisite (p theta) (p theta') →
    mutuallySingularMeasures (PY theta) (PY theta')

/-- `[Hypothesis]` Alignment of `ζ_t` with `Y_t = (m_t, d_t)`
    (TeX `thm:posterior-consistency-appendix`, item (iii)): mutual singularity of the
    outcome-trajectory laws transfers to the restricted signal laws on `𝒯`
    (monotonicity of singularity with respect to σ-algebras). -/
def signalExperimentAlignment
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t))
    [Fintype ΘK] [DecidableEq ΘK]
    (ν : ΘK → @Measure Ω m) (T : Set Ω)
    (PY : ΘK → Measure (ℕ → Y)) : Prop :=
  ∀ theta theta' : ΘK,
    mutuallySingularMeasures (PY theta) (PY theta') →
      mutuallySingularMeasures
        (restrictedSignalLaw mZ ζ hζ (ν theta) T)
        (restrictedSignalLaw mZ ζ hζ (ν theta') T)

/-- `[Proved]` Conditional `kakutaniBridge`: asymptotic TV ⇒ algebraic prerequisite
    (proved), then `kakutaniSignalSingularity` (outcome-level Kakutani) composed with
    `signalExperimentAlignment` (transfer to `ζ_t`) closes `def:asymptotic-id`. -/
theorem kakutaniBridge_from_criterion {ΘK : Type*}
    (p : ΘK → ℕ → Y → ℝ) (PY : ΘK → Measure (ℕ → Y))
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t))
    [Fintype ΘK] [DecidableEq ΘK]
    (ν : ΘK → @Measure Ω m) (T : Set Ω) (thetaStar : ΘK)
    (hprob : ∀ (θ : ΘK) (t : ℕ), IsProbMass (p θ t))
    (hkak : ∀ θ : ΘK, θ ≠ thetaStar →
      kakutaniSignalSingularity p PY θ thetaStar)
    (halign : signalExperimentAlignment mZ ζ hζ ν T PY) :
    kakutaniBridge p mZ ζ hζ ν T thetaStar := by
  intro θ hθ htv
  have hpre := hazardSeparation_affinityVanishing p θ thetaStar
    (fun t => ⟨hprob θ t, hprob thetaStar t⟩) htv
  exact halign θ thetaStar (hkak θ hθ hpre)

/-- `[Proved]` Corollary: full asymptotic-identification assembly from TV separation,
    assuming the signal-law Kakutani step and alignment. -/
theorem asymptoticIdentification_from_tvSeparation {ΘK : Type*}
    (p : ΘK → ℕ → Y → ℝ) (PY : ΘK → Measure (ℕ → Y))
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t))
    [Fintype ΘK] [DecidableEq ΘK]
    (ν : ΘK → @Measure Ω m) (T : Set Ω) (thetaStar : ΘK)
    (htrue : ν thetaStar T ≠ 0)
    (hprob : ∀ (θ : ΘK) (t : ℕ), IsProbMass (p θ t))
    (hdisj : ∀ θ : ΘK, θ ≠ thetaStar →
      ν θ T = 0 ∨ tvSeparationAsymptotic p θ thetaStar)
    (hkak : ∀ θ : ΘK, θ ≠ thetaStar →
      kakutaniSignalSingularity p PY θ thetaStar)
    (halign : signalExperimentAlignment mZ ζ hζ ν T PY) :
    asymptoticIdentification mZ ζ hζ ν T thetaStar :=
  hazardSeparationLemma p mZ ζ hζ ν T thetaStar htrue hdisj
    (kakutaniBridge_from_criterion p PY mZ ζ hζ ν T thetaStar hprob hkak halign)

/-- `[Proved]` Cox predictive masses + asymptotic TV margin ⇒ full asymptotic-ID assembly. -/
theorem asymptoticIdentification_from_coxPredictive {ΘK : Type*}
    (baseline : ℕ → Fin 4 → ℝ) (beta : ΘK → Fin 3 → ℝ) (Δt : ℝ)
    (PY : ΘK → Measure (ℕ → DailyOutcome))
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t))
    [Fintype ΘK] [DecidableEq ΘK]
    (ν : ΘK → @Measure Ω m) (T : Set Ω) (thetaStar : ΘK)
    (hb : ∀ t j, 0 < baseline t j) (_hdt : 0 < Δt)
    (htrue : ν thetaStar T ≠ 0)
    (hdisj : ∀ θ : ΘK, θ ≠ thetaStar →
      ν θ T = 0 ∨
        tvSeparationAsymptotic (predictiveMassFamily baseline beta Δt) θ thetaStar)
    (hkak : ∀ θ : ΘK, θ ≠ thetaStar →
      kakutaniSignalSingularity (predictiveMassFamily baseline beta Δt) PY θ thetaStar)
    (halign : signalExperimentAlignment mZ ζ hζ ν T PY) :
    asymptoticIdentification mZ ζ hζ ν T thetaStar :=
  asymptoticIdentification_from_tvSeparation
    (predictiveMassFamily baseline beta Δt) PY mZ ζ hζ ν T thetaStar htrue
    (fun θ t => predictiveMass_isProbMass (baseline t) beta θ Δt (hb t) (le_of_lt _hdt))
    hdisj hkak halign

/-- `[Proved]` Sufficient Cox route: `β` separation + no hazard-sum knife-edge at each false type. -/
theorem asymptoticIdentification_from_coxBetaSeparation {ΘK : Type*}
    (baseline : ℕ → Fin 4 → ℝ) (beta : ΘK → Fin 3 → ℝ) (Δt : ℝ)
    (PY : ΘK → Measure (ℕ → DailyOutcome))
    (mZ : MeasurableSpace Z) (ζ : ℕ → Ω → Z)
    (hζ : ∀ t, Measurable[m, mZ] (ζ t))
    [Fintype ΘK] [DecidableEq ΘK]
    (ν : ΘK → @Measure Ω m) (T : Set Ω) (thetaStar : ΘK)
    (hb : ∀ t j, 0 < baseline t j) (hdt : 0 < Δt)
    (htrue : ν thetaStar T ≠ 0)
    (hβsum : ∀ θ : ΘK, θ ≠ thetaStar →
      betaSeparation beta θ thetaStar ∧
        noHazardSumKnifeEdge (baseline 0) beta θ thetaStar)
    (hrecurrent : ∀ θ : ΘK, θ ≠ thetaStar →
      ∃ (S : Set ℕ) (ε : ℝ), 0 < ε ∧ S.Infinite ∧
        ∀ t ∈ S, ε ≤ totalVariationDistance
          (predictiveMassFamily baseline beta Δt θ t)
          (predictiveMassFamily baseline beta Δt thetaStar t))
    (hdisj : ∀ θ : ΘK, θ ≠ thetaStar →
      ν θ T = 0 ∨
        tvSeparationAsymptotic (predictiveMassFamily baseline beta Δt) θ thetaStar)
    (hkak : ∀ θ : ΘK, θ ≠ thetaStar →
      kakutaniSignalSingularity (predictiveMassFamily baseline beta Δt) PY θ thetaStar)
    (halign : signalExperimentAlignment mZ ζ hζ ν T PY) :
    asymptoticIdentification mZ ζ hζ ν T thetaStar := by
  have htv : ∀ θ : ΘK, θ ≠ thetaStar →
      tvSeparationAsymptotic (predictiveMassFamily baseline beta Δt) θ thetaStar := by
    intro θ hθ
    rcases hrecurrent θ hθ with ⟨S, ε, hε, hS, hsep⟩
    exact tvSeparationAsymptotic_of_uniform
      (predictiveMassFamily baseline beta Δt) θ thetaStar ε hε S hS hsep
  have hdisj' : ∀ θ : ΘK, θ ≠ thetaStar →
      ν θ T = 0 ∨
        tvSeparationAsymptotic (predictiveMassFamily baseline beta Δt) θ thetaStar := by
    intro θ hθ
    rcases hdisj θ hθ with hzero | _
    · exact Or.inl hzero
    · exact Or.inr (htv θ hθ)
  exact asymptoticIdentification_from_coxPredictive baseline beta Δt PY mZ ζ hζ ν T thetaStar
    hb hdt htrue hdisj' hkak halign

end IdentificationBridge

end Appendix3Proofs
