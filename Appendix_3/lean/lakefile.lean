import Lake
open Lake DSL

package «Appendix3Proofs» where
  leanOptions := #[⟨`pp.unicode.fun, true⟩]

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "master"

@[default_target]
lean_lib «Appendix3Proofs» where
  roots := #[`Appendix3Proofs]
