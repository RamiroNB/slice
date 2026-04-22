# 20 paths forward for SLICE/GLIMPSE: sharpening gradient-surgery LoRA init

**SLICE's G1 regression and G2 success point to the same diagnosis: hard orthogonal projection is too aggressive when <G_A, G_P>_F > 0 is both common and small.** On NI-Seq-G1 the surgery triggers frequently but the conflict magnitudes are modest, so the PCGrad subtraction wipes out useful descent directions along G_P that also help the adapt task — this is the classic "tragic triad" failure mode that later gradient-surgery work (GradVac, CAGrad, Aligned-MTL) was designed to fix. On NI-Seq-G2, where forgetting in LoRA-GA is catastrophic (0.1185), even a blunt projection buys huge stability (forgetting drops 11×, FP +8 pts), at only modest AP cost. The switch from PCGrad to OGD on G1 didn't help because OGD's subspace projection is an even harder kill than PCGrad's vector projection — the problem is not *which* orthogonality, it is that full orthogonality is the wrong target. A soft, thresholded, magnitude-preserving, and conflict-scaled alternative is likely to win both benchmarks. Two additional forces should shape the redesign: recent evidence (Zhang et al., NeurIPS 2025 spotlight, "The Primacy of Magnitude in Low-Rank Adaptation") that spectral LoRA init gains come from *magnitude amplification* — meaning SLICE's SVD is probably only half the story — and the observation that most CL benchmarks show positive but small inner products, pushing toward cosine-threshold rather than sign-threshold conflict detection.

Below: 20 concrete ideas grouped by category, each with an expected effect on G1/G2 metrics, prior-art grounding, and difficulty. Then a top-5 synthesis, a gradient-surgery literature review, a focused discussion of threshold hyperparameterization, and a critical look at the LoRA-GA SVD selection rule.

---

## A. Projection mechanism improvements

### 1. Soft CAGrad-style interpolation between G_A and projection

**What.** Replace the hard PCGrad projection with a CAGrad-style worst-case update (Liu et al., NeurIPS 2021, arXiv:2110.14048). Solve a 2-vector QP: find d minimizing ‖d − G_A‖ subject to ⟨d, G_P⟩ ≥ −c·‖G_A‖·‖G_P‖ for a constant c ∈ [0,1]. With c=0 you recover vanilla (no projection); c=1 recovers full orthogonal projection (SLICE/PCGrad behavior). **Why it helps.** G1's regression suggests SLICE over-projects; interpolating lets you dial back the "forget-avoidance tax" on conflict-positive but signal-rich directions. G2 is fine with aggressive c. A single scalar c becomes a principled handle linking the two regimes. **Difficulty:** low — the 2-vector CAGrad has a closed form.

### 2. GradVac-style cosine-target projection with EMA

**What.** Instead of forcing ⟨G_A, G_P⟩=0, rotate G_A so that cos(G_A, G_P) equals an *EMA-tracked target* φ^T per layer (Wang et al., ICLR 2021, GradVac, arXiv:2010.05874). The closed-form λ such that G_A + λG_P hits φ^T is in the GradVac paper. **Why it helps.** The authors note that almost all CL task pairs have positive dot products. PCGrad's zero-cosine target is wrong for that regime — related tasks *should* have positive cosine. GradVac would actively preserve the shared-signal component that SLICE currently discards, plausibly closing the 0.005 AP gap on G1 while keeping the forgetting gain on G2. **Difficulty:** low-medium.

### 3. Cosine-threshold instead of dot-product threshold for conflict detection

**What.** Apply the PCGrad subtraction only when cos(G_A, G_P) < τ where τ ∈ {−0.1, −0.05, 0, 0.05, 0.1}. Cosine normalizes away the magnitude imbalance that makes the raw dot-product threshold uninformative. **Why it helps.** Raw Frobenius inner products depend on layer-wise norm scales (attention vs MLP, early vs late), so "dot product > 0" triggers unevenly. A cosine cutoff makes the threshold scale-invariant and gives a meaningful hyperparameter sweep. Given the authors' plan to explore τ, cosine is the right unit. **Difficulty:** trivial.

### 4. Per-layer learned conflict threshold

**What.** Treat τ_ℓ as a per-layer scalar hyperparameter fit by a small grid or a one-dimensional bandit on a validation split. Alternatively, parameterize τ_ℓ = σ(w_ℓ) and learn via a meta-objective (preserve-loss minus adapt-loss) with a cheap inner-loop. **Why it helps.** Attention Q/V layers and late MLP blocks have very different gradient geometries; a single global threshold is coarse. Modest complexity, potentially large effect on G1 where over-projection in some layers is the suspect. **Difficulty:** medium.

### 5. Null-space projection against the preserve subspace (Adam-NSCL style)

**What.** Instead of projecting only against one gradient vector G_P, compute the SVD of the preserve-task feature-covariance matrix X_P^T X_P layer-wise and project G_A into its *approximate null space* — the subspace spanned by singular vectors whose singular values are below a threshold (Wang et al., CVPR 2021 oral, arXiv:2103.07113). This generalizes OGD and is what OWM (Zeng et al., 2019), GPM (Saha et al., ICLR 2021), and OPLoRA use. **Why it helps.** Feature-covariance null-space is a *structural* invariant of the preserve task, not a single stochastic gradient estimate. The SLICE authors already have the G_P mini-batches; add a covariance accumulation pass for negligible cost. This is the single biggest theoretical upgrade available. **Difficulty:** medium.

### 6. Magnitude-preserving (Procrustes) rotation projection

**What.** After computing G̃_A via PCGrad, rescale to **‖G̃_A‖_F = ‖G_A‖_F**. Equivalently, solve the Procrustes problem: find the rotation R minimizing ‖RG_A − G̃_A‖ subject to R∈SO(·). **Why it helps.** Zhang et al. (NeurIPS 2025 spotlight, arXiv:2507.06558, "The Primacy of Magnitude in Low-Rank Adaptation") show that spectral LoRA init gains are primarily *magnitude amplification*. PCGrad projection reduces ‖G̃_A‖, so SLICE discards the very lever that makes LoRA-GA work. Norm-restoring after surgery should close the G1 AP gap directly. **Difficulty:** low (just rescale); medium if you do the full Procrustes rotation.

### 7. Multi-step iterative projection with a trust region

**What.** Iterate: at step k, project G_A^(k) against G_P with shrinking strength η_k, then re-measure conflict; stop when cos < ε. Like proximal gradient surgery. **Why it helps.** A single projection step over-corrects when conflicts are small but frequent (the G1 regime). An iterative scheme finds the minimum-distortion G̃_A that reduces conflict below ε, preserving more signal than PCGrad's one-shot zeroing. **Difficulty:** medium.

### 8. Nash-bargaining weighted combination of G_A and G_P

**What.** Replace projection with a 2-task Nash-MTL (Navon et al., ICML 2022, arXiv:2202.01017) aggregator: find α∈Δ^2 maximizing log⟨g_A,d⟩+log⟨g_P,d⟩ subject to d=α_A G_A + α_P G_P. Decompose SVD of this aggregate instead of G̃_A. **Why it helps.** Produces a unique, scale-invariant descent direction that respects both tasks' dot-product utility. Unlike orthogonalization, it keeps shared signal; unlike naive averaging, it balances magnitude. The 2-task case is cheap (one scalar CCP iteration). **Difficulty:** medium.

### 9. GradDrop-style stochastic per-coordinate masking

**What.** For each scalar entry in G_A and G_P, compute sign-purity P and sample a Bernoulli sign; zero the entries whose signs disagree (Chen et al., NeurIPS 2020, arXiv:2010.06808). SVD the masked G_A. **Why it helps.** Preserves agreement coordinates (where the two tasks point the same way — common in G1) while suppressing disagreement coordinates. Unbiased in expectation under agreement. The authors would discover which coordinates are truly conflict-prone rather than applying a full-vector zero. **Difficulty:** low.

### 10. Aligned-MTL SVD of the [G_A, G_P] stack

**What.** Stack G_A and G_P as columns of a matrix G∈ℝ^{p×2}, compute SVD G=UΣV^T, replace Σ by σ_min·I (condition number κ=1), yielding orthogonalized, equal-norm task-gradient columns (Senushkin et al., CVPR 2023, "Independent Component Alignment", arXiv:2305.19000). Use the aligned adapt-task column as the SVD input for LoRA init. **Why it helps.** This simultaneously fixes direction conflict *and* magnitude imbalance in a basis-independent way, and provably converges to a preference-controlled Pareto point. Directly addresses both forgetting (G2) and signal preservation (G1). **Difficulty:** medium.

---

## B. Gradient estimation improvements

### 11. Fisher-preconditioned gradients (K-FAC lite)

**What.** Approximate the Fisher F_ℓ per layer using the same mini-batches, then project in the natural-gradient metric: G̃_A = G_A − (⟨G_A, G_P⟩_F / ⟨G_P, G_P⟩_F) G_P where the inner product is now ⟨·,·⟩_{F^{-1}}. Even a diagonal Fisher (EWC-style) is enough. **Why it helps.** Raw gradients confuse "large because important" with "large because poorly normalized." Fisher preconditioning rescales so orthogonality reflects function-space interference, which is what actually causes forgetting. Echoes the Fisher-information-aware LoRA unlearning work (VILA, arXiv:2508.21300) and LoRA-DA (arXiv:2510.24561), the latter explicitly arguing that raw single-step gradients are inadequate for LoRA init. **Difficulty:** medium.

### 12. Variance-reduced preserve gradient via more accumulation steps

**What.** Increase the number of mini-batches used to estimate G_P (e.g., 32 → 256) and optionally apply SVRG-style control variates. **Why it helps.** On G1, the preserve-gradient estimate is almost certainly noisy enough that the "conflict > 0" test fires partly on sampling noise, causing spurious projection. A lower-variance G_P makes the threshold more meaningful. Cheap and likely to shift G1 metrics by 1–2 points. **Difficulty:** low.

### 13. Representation-gradient surgery instead of loss-gradient surgery

**What.** Project gradients with respect to *intermediate representations* (MGDA-UB style; Sener & Koltun, NeurIPS 2018) or on *task-embedding* gradients rather than per-parameter loss gradients. **Why it helps.** Loss gradients conflate the functional change we care about with the parameterization. Representation gradients capture what each task actually wants the *features* to do, which is the right invariance for multi-task continual learning. Also cheaper (one backward pass shared across layers). Caveat: Navon et al. (2022) report that MGDA-UB *hurts* Nash-MTL, so evaluate carefully. **Difficulty:** medium.

### 14. Per-sample preserve gradients with task clustering

**What.** Instead of one averaged G_P, compute K per-sample preserve gradients, cluster them (k-means on ℝ^p with k=3–5), and project against each cluster centroid. This yields multiple "preserve directions" — robust to the case where preserve data is heterogeneous. **Why it helps.** Super-NaturalInstructions preserve tasks (e.g., the 15-task sequences used in O-LoRA/InfLoRA) are themselves multi-skill; a single G_P smears them together. On G2, where the preserve tasks are more diverse, this should further reduce forgetting. **Difficulty:** medium.

---

## C. SVD / decomposition improvements

### 15. Joint / generalized SVD of G_A and G_P

**What.** Compute the generalized SVD (GSVD): find U_A, U_P, V, Σ_A, Σ_P such that G_A = U_A Σ_A V^T, G_P = U_P Σ_P V^T with shared right singular vectors. Use V columns where σ_A is large but σ_P is small (i.e., directions that matter for adapt but not preserve) to build LoRA init. **Why it helps.** This is the multi-task analogue of CorDA (Yang et al., NeurIPS 2024), which does context-oriented SVD on one task. GSVD gives you the *best* rank-r subspace that is simultaneously adapt-useful and preserve-safe, without any gradient subtraction. Much cleaner than PCGrad+SVD. **Difficulty:** medium-high.

### 16. Revisit the B=U[:,:r], A=V[r:2r,:]^T selection — use σ-weighted split

**What.** The LoRA-GA trick of using *different* singular-vector index ranges for B and A is a hack to avoid BA=0 at init while staying rank-r. But it discards σ_i and wastes the top-r right singular vectors. A cleaner alternative (concurrent to LoRA-One, Zhang et al. 2025): set B = U[:,:r] √Σ[:r,:r] and A = (√Σ[:r,:r] V[:,:r]^T)·scale, then subtract the product from the frozen weight W₀ (PiSSA-style "residual"). This gives BA = first-r SVD rank-1 approximation of G̃_A, and BA≠0 is absorbed into the residual. **Why it helps.** Uses the *right* top-r singular pair with proper scaling, aligning with the LoRA-One / LoRA-DA finding that the optimal B initialization is the top-r left singular vectors of G. Removes the arbitrary V[r:2r,:] choice. **Difficulty:** low-medium.

### 17. LoRAM magnitude-driven hybrid init

**What.** Borrowing from Zhang et al. (NeurIPS 2025 spotlight, arXiv:2507.06558): use deterministic orthogonal bases (QR of random or of G̃_A), but scale them by pretrained weight magnitudes ‖W₀‖_ℓ to simulate spectral amplification — no SVD needed. Combine with a conflict-aware re-weighting of the adapt-vs-preserve direction. **Why it helps.** Zhang's paper empirically shows LoRAM matches or exceeds spectral init (PiSSA, LoRA-GA). If magnitude is the real driver, SLICE's SVD machinery may be orthogonal to (or conflicting with) what actually matters. Bake magnitude amplification into the init pipeline and a lot of gradient-surgery complexity becomes optional. **Difficulty:** medium.

---

## D. Training-time integration (hybrid init + training methods)

### 18. SLICE init + O-LoRA training-time orthogonality constraint

**What.** Use SLICE (or its improved variants) to initialize, then during training add an O-LoRA-style loss term λ ‖B_{A,t}^T B_{prev}‖_F^2 keeping new task's B orthogonal to cached preserve-task LoRA bases (Wang et al., EMNLP Findings 2023, arXiv:2310.14152). **Why it helps.** G2's 0.1185 → 0.0105 forgetting improvement from SLICE shows that init alone moves the needle, but AP still slips (0.324 → 0.296). O-LoRA's training regularizer catches the drift that init can't prevent over many epochs. The two mechanisms are complementary: init handles the subspace choice; training keeps it there. **Difficulty:** low.

### 19. Periodic re-SLICE during training (InfLoRA-style)

**What.** Every K epochs, pause, recompute G_A and G_P with the current adapter weights, redo the surgery+SVD, and warm-start the next phase. Effectively applies InfLoRA's interference-free subspace update (Liang & Li, CVPR 2024, arXiv:2404.00228) at init cadence. **Why it helps.** A one-shot init is unavoidably committed to the initial G_A estimate, which drifts as training progresses. Periodic re-projection matches the subspace to the *current* state, preventing both under-projection (G1) and delayed forgetting (G2). Addresses the fundamental limitation that LoRA-One and LoRA-DA critique: "single-step reliance limits effectiveness" (LoRA-DA, arXiv:2510.24561). **Difficulty:** medium.

### 20. SLICE as warm-start for training-time gradient surgery (PCGrad-train)

**What.** Use SLICE for init, then apply PCGrad or CAGrad *during training* on each mini-batch by computing small G_A, G_P estimates per step and projecting the actual training gradient. **Why it helps.** Right now SLICE's projection is frozen after step 0; but the preserve task keeps getting violated during training. A lightweight per-step surgery (with cached/EMA G_P to avoid double backprop) turns SLICE from an init-only method into a continual-learning optimizer with a good starting point. Makes the paper's scope broader and lets it compete directly with O-LoRA rather than just LoRA-GA. **Difficulty:** medium-high.

---

## Top-5 ranking: what to try first for NeurIPS

Given the specific observations (positive dot products common, G1 regression, G2 win, OGD failure, magnitude-amplification literature), the most likely-to-pay-off experiments, in order:

1. **Magnitude-preserving PCGrad (#6)** — Fixes the most plausible G1 failure mode (norm shrinkage) with a one-line code change. If Zhang 2025's "magnitude primacy" thesis is correct, this alone might close the G1 gap.

2. **Cosine-threshold τ hyperparameter sweep (#3, #4)** — The authors already plan this; frame it as cosine not raw dot, sweep τ ∈ {−0.1, 0, 0.1, 0.2}, and report a per-layer optional variant. Cheapest route to a positive ablation table.

3. **GradVac with EMA target (#2)** — Directly addresses the "most dot products are positive" observation. Replaces SLICE's zero-cosine target with a data-driven per-layer target, turning the hypothesis about shared signal into the mechanism.

4. **Null-space projection on preserve-feature covariance (#5)** — Theoretically principled, works with existing cached mini-batches, and is the natural bridge to the Adam-NSCL/OPLoRA/InfLoRA literature. Likely biggest G2 win; risk is under-plasticity on G1 unless you threshold the null-space rank.

5. **Periodic re-SLICE during training (#19) or SLICE+O-LoRA reg (#18)** — Moves from init-only to "init + training", which is where most recent CL LoRA methods live. Without this, reviewers will ask why SLICE is init-only when O-LoRA and InfLoRA integrate both.

Honorable mentions: **CAGrad interpolation (#1)** if you want a single clean scalar knob that subsumes both vanilla and full-projection as endpoints, and **GSVD/joint decomposition (#15)** for an elegant re-derivation that avoids gradient surgery entirely.

---

## Literature review: gradient surgery beyond PCGrad

**PCGrad** (Yu et al., NeurIPS 2020, arXiv:2001.06782) projects g_i onto the normal plane of g_j when cos(g_i, g_j) < 0. It has only local convergence guarantees in the 2-task convex case, is order-dependent, and — as multiple critiques note (Kurin et al., NeurIPS 2022, arXiv:2201.04122; Xin et al., NeurIPS 2022, arXiv:2209.11379) — is often matched by plain scalarization once regularization is properly tuned. Its most cited failure mode for SLICE's setting is **norm shrinkage**: the orthogonal component is strictly smaller than the original, so repeated projection against a dominant G_P erases subordinate signal.

**MGDA** (Sener & Koltun, NeurIPS 2018, arXiv:1810.04650) finds the min-norm convex combination of task gradients. Principled but biased toward the easiest task; MGDA-UB's feature-gradient approximation hurts downstream methods (Navon 2022).

**CAGrad** (Liu et al., NeurIPS 2021, arXiv:2110.14048) maximizes the worst-case per-task descent inside a ball of radius c around the average gradient. c=0 → vanilla GD, c=1 → MGDA, with provable convergence to the average-loss minimum (not just Pareto stationarity). **Key insight for SLICE:** c is exactly the kind of soft knob missing from PCGrad; most CL wins come at c≈0.4, not c=1.

**GradVac** (Wang et al., ICLR 2021, arXiv:2010.05874) generalizes PCGrad by rotating gradients toward an EMA-tracked cosine target φ^T instead of zero. Directly addresses the "related tasks should have positive cosine" critique — precisely SLICE's G1 regime.

**GradDrop** (Chen et al., NeurIPS 2020, arXiv:2010.06808) operates per-coordinate via sign-purity sampling; unbiased in expectation and converges to *joint* minima rather than arbitrary Pareto points. Good fit when per-coordinate conflicts matter more than full-vector ones.

**RotoGrad** (Javaloy & Valera, ICLR 2022 spotlight, arXiv:2103.02631) learns per-task SO(d) rotations plus GradNorm-style magnitude balancing, separating direction and norm. Heaviest but captures both failure modes of PCGrad.

**Nash-MTL** (Navon et al., ICML 2022, arXiv:2202.01017) frames aggregation as a Nash bargaining game; unique axiomatic solution, scale-invariant, SOTA on NYU-v2 / CityScapes / QM9. The 2-task case (adapt vs preserve) is especially cheap.

**IMTL** (Liu et al., ICLR 2021) enforces equal *unit-vector projections* of the aggregated gradient onto each task, eliminating magnitude bias without hyperparameters.

**Aligned-MTL** (Senushkin et al., CVPR 2023, arXiv:2305.19000) replaces singular values of the stacked gradient matrix G=[g_1, ..., g_K] with σ_min·I, guaranteeing κ(G)=1 and Pareto convergence under user-specified weights. Captures conflict and magnitude imbalance in a single scalar (condition number).

**FAMO** (Liu et al., NeurIPS 2023, arXiv:2306.03792) achieves similar effects with O(1) memory via log-loss adaptive weighting — no per-task backward passes.

**Critiques to address.** Kurin et al. (2022) and Xin et al. (2022) empirically show that unitary scalarization often ties or beats these methods once the baseline is properly regularized, and that apparent MTO wins reflect different points on a trade-off curve rather than Pareto improvement. For SLICE, this argues for reporting multiple weighting regimes (plain sum, weighted sum, CAGrad-c sweep) on the same benchmarks and demonstrating that the surgery-based method does not just move along an existing frontier. Failure modes common across these methods: sensitivity to gradient-norm balance, O(K) backward-pass cost, and brittleness when tasks are poorly scaled — all of which SLICE inherits.

---

## Threshold hyperparameterization: why, what range, how

The authors observe that in their SuperNaturalInstructions CL runs the Frobenius inner product ⟨G_A, G_P⟩_F is **almost always positive**, so the current "project only when > 0" rule means *almost every* layer gets projected. Since G1 regresses while G2 wins, the natural hypothesis is that projection strength should be modulated by *how severe* the conflict actually is, not just its sign.

**Why a threshold helps.** Three regimes exist: (a) cos≈1 (tasks already aligned — projection erases useful shared direction; this is the G1 pathology), (b) cos≈0 (mild conflict — projection is roughly a no-op), (c) cos<0 (true conflict — projection is beneficial). PCGrad's threshold at 0 triggers in case (a) unnecessarily. A positive cosine threshold skips projection in the aligned case; a slightly negative threshold also lets through mild conflicts.

**Recommended parameterization.** Use cosine similarity rather than raw dot product — the raw dot depends on layer-wise scale, so a single global threshold is effectively different per layer. Concretely: sweep τ ∈ {−0.1, −0.05, 0.0, 0.05, 0.1, 0.2}, project only when cos(G_A, G_P) < τ. τ > 0 means "project even in mild alignment"; τ < 0 means "only project true conflicts." Given the observation that ⟨G_A, G_P⟩>0 almost everywhere, τ around 0.1–0.2 is the regime most likely to *reduce* projection frequency on G1 while keeping it on G2.

**Per-layer vs global.** Empirically (GradVac EMA per-pair, IMTL-G unit-vector balance), per-layer thresholds help when gradient geometries vary across depth. A simple per-layer approach: compute the median cosine across a validation batch, set τ_ℓ = median_ℓ − δ. A learned approach: parameterize τ_ℓ = σ(w_ℓ) and optimize on a held-out preserve-loss / adapt-loss trade-off objective via zero-order or a short meta-inner-loop. Given NeurIPS reviewer pressure, first show the fixed global cosine sweep (cheap, interpretable), then include per-layer in an ablation.

**Conflict magnitude for strength modulation.** Even better than a binary trigger: scale projection strength by conflict severity, e.g., β = clip(−cos(G_A, G_P), 0, 1), giving G̃_A = G_A − β·proj_{G_P}(G_A). At cos≈0, β≈0 (no projection); at cos=−1, β=1 (full PCGrad). This is essentially a continuous CAGrad-lite and avoids the cliff at the threshold.

---

## Revisiting the LoRA-GA SVD selection B = U[:, :r], A = V[r:2r, :]^T

The rule exists for one reason: if both matrices come from the *same* rank-r singular vectors (B=U[:,:r]Σ^{1/2}, A=Σ^{1/2}V[:,:r]^T), then at init BA = U[:,:r]Σ_{:r}V[:,:r]^T = top-r reconstruction of G, which is **not zero**. A nonzero BA at init corrupts the frozen W₀ and breaks LoRA's core invariant that fine-tuning starts from the pretrained state. LoRA-GA avoids this by using *disjoint* singular-vector slices so that BA = U[:,:r]·V[r:2r,:]^T — a product of orthogonal slices, generically rank-0 in aggregate.

**Is this optimal?** Not clearly. The LoRA-GA derivation (Theorem 3.1) optimizes gradient alignment subject to rank-r and BA=0 constraints, and shows that *any* pair of orthogonal rank-r matrices drawn from non-overlapping singular-vector indices gives the same first-step gradient direction. The choice of specifically [:, :r] and [r:2r, :] is therefore arbitrary among such pairs. Two alternatives worth evaluating:

1. **LoRA-One / LoRA-DA style (Zhang et al. 2025; arXiv:2510.24561).** Use B = U[:,:r]·√Σ[:r,:r] and A = √Σ[:r,:r]·V[:,:r]^T (i.e., the *correct* top-r SVD pair), and compensate BA≠0 by subtracting it from W₀ (PiSSA-style residual: W₀ ← W₀ − BA). This preserves the LoRA identity-at-init property and uses the actual best rank-r approximation. Strictly stronger than LoRA-GA's hack under the gradient-approximation objective.

2. **PiSSA/CorDA-style "BA subtracted from W₀".** Same move, but applied to gradient SVD instead of weight SVD. Already supported by the HuggingFace PEFT library for PiSSA and OLoRA; the infrastructure exists.

**For SLICE specifically**, the singular values of G̃_A carry important magnitude information (per Zhang 2025); discarding them via the LoRA-GA trick and using orthogonal vectors only throws away magnitude amplification. Switching to a LoRA-One-style decomposition with W₀-residual compensation is likely to improve SLICE on both G1 and G2 independent of any projection-mechanism change. This is probably the single highest-leverage architectural tweak available.

**Recommendation.** Run an ablation: (a) SLICE with LoRA-GA selection, (b) SLICE with LoRA-One selection + W₀ residual, (c) SLICE with LoRAM magnitude-driven init. Expect (b) or (c) to dominate, and this single change may explain more of the G1 gap than any surgery modification.

---

## Conclusion: the critical bet

The experimental pattern points to a clean story: **SLICE works when forgetting is catastrophic (G2), and hurts when forgetting is manageable (G1), because hard orthogonal projection is a blunt instrument that destroys useful shared-task signal whenever the adapt and preserve tasks are mostly aligned.** The OGD negative result reinforces this — moving to a subspace projection doesn't fix a sharpness problem, it amplifies it. The fix is *softness in three dimensions:* conflict-magnitude-scaled projection strength (items 1, 6), cosine-based thresholding with per-layer calibration (items 3, 4), and magnitude-preserving post-projection rescaling (item 6, item 17). Combined with a switch to the LoRA-One-style SVD selection rule (revisiting the LoRA-GA [:, :r] / [r:2r, :] convention), these changes would position GLIMPSE as a principled superset of LoRA-GA rather than a preserve-task ablation of it, and provide clean ablations reviewers will expect: one for threshold, one for projection interpolation, one for magnitude preservation, one for SVD selection. The G2 win is already the paper's strongest result; the above changes should let G1 match or exceed LoRA-GA without sacrificing it.