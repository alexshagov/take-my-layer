# LIMITATIONS.md — intended use and measured limits

**Intended use**: decision *support* (triage, routing, prioritization,
done-yet checks) on a local machine, returning full probability
distributions for a human or agent harness in the loop. **Out of scope**:
autonomous high-stakes decisions (fraud adjudication, HR actions,
moderation enforcement, insurance/claims outcomes, or any medical, legal,
financial or safety-critical call) without human oversight. Known limits,
concretely measured on this artifact (details in docs/RESULTS.md):

- **Confidence gating does not transfer across domains.** On foreign live
  traffic the seed-domain-fit τ=0.8 curve gave 37% routed error at 59% coverage
  with ECE 0.238: the documented failure mode of threshold gating under
  distribution shift (Ovadia et al., NeurIPS 2019, arXiv:1906.02530).
  This is why the proxy is shadow-only; re-read τ per surface before any
  routing use.
- **Sharp overconfident misses exist**: ~one 0.9+ false certainty per ~100
  questions on niche factual or policy-judgment items. L1 temperature
  calibration is the targeted in-distribution fix, and temperature
  scaling itself degrades under shift, so it calibrates the tail but does
  not license autonomy.
- **Small + quantized**: quantization hits small models harder than large
  ones (Dettmers & Zettlemoyer, arXiv:2212.09720). Every number is
  measured on the exact served 4-bit artifact ("fit on what you serve"),
  never an fp16 stand-in.
- **No human ground truth** anywhere in the shipped evaluations; all
  agreement numbers are fidelity-to-reference (Evaluation disclosure in
  docs/RESULTS.md).
