# Book plan

*Probabilistic deep learning: from maximum likelihood to diffusion models* — a Quarto book (same format as `../DeepReinforcementLearning`) explaining the probabilistic concepts behind modern machine learning. The common thread: virtually every learning algorithm fits a parameterized distribution $p_\theta$ to data sampled from an unknown distribution $p$, i.e. minimizes a divergence between them, and most of the difficulty comes from intractable expectations (integrals), which are handled either by **sampling** or by **optimization** (variational inference).

Legend: `[x]` written · `[~]` in progress · `[ ]` planned · ★ priority.

---

## Workflow

For every new chapter:

1. A detailed outline is written in this file and agreed on (scope, notation, figures).
2. One section at a time is written in the `.qmd`, rendered, and **waits for approval** before the next one is started.
3. Figures are proposed in the outline and generated with matplotlib scripts in `src/img/` (`<figure-name>.py` → `<figure-name>.png`), reproducible (fixed seed).

## Conventions

- **Layout:** one subfolder of `src/` per part, chapters numbered inside it (`src/<part>/<n>-<Name>.qmd`); all images in `src/img/` with descriptive kebab-case names, referenced as `../img/…`. Parts and chapter order are defined in `_quarto.yml`.
- **Build:** `just` copies `~/Articles/bibtex/Library.bib` and runs `quarto render .` into `docs/` (gitignored).
- **Labels:** every chapter has a `{#sec-…}` label, figures `{#fig-…}`; refer to other chapters with `@sec-…` rather than "see above/below". Citations with `@Key` from `Library.bib`.
- **Math:** KaTeX. Display math in `$$` blocks indented by 4 spaces, **no blank lines inside `$$`**; multi-line derivations with `\begin{aligned}`, comments as `& \text{(…)}`. `\text{KL}(p || q)`, `\mathbb{E}_{x \sim p(x)}[\cdot]`, `\mathcal{L}` for losses, `\mathcal{N}(\mu, \sigma)` parameterized by the standard deviation.
- **Callouts:** `:::{.callout-tip|note|warning icon="false"}` with a `##` title; proofs in collapsed `callout-note`.
- **Diagrams:** `{mermaid}` with HTML labels (`<i>x</i>`, `<sub>θ</sub>`), as mermaid does not render `$…$`.
- **Style:** step-by-step derivations, bold key terms at their definition, bullet lists to interpret results (limit cases), a sentence linking each chapter to the previous ones. Figures from other sources carry "Source: …" in the caption.

---

## Current and planned structure

### Preface — `index.qmd` [x]

Rewritten: purpose, prerequisites, figures, feedback, license; the motivation and the common thread moved to the introduction.

### Introduction — `src/introduction.qmd` [x]

New chapter before Part I, `{#sec-introduction}`: why probabilistic rather than deterministic deep learning, with applications. Outline below ("Chapter plan: Introduction"); all four sections written. Borrowed figures: `gencast.png` (Fig. 2 of @Price2024, arXiv HTML), `alphafold3.png` (Fig. 1d of @Abramson2024, CC-BY 4.0), `ocean-currents-forecast.gif` (eOdyn, from the IntroAI lecture). Title changed to *Probabilistic deep learning: from maximum likelihood to diffusion models* (repository renamed `probabilisticdl`).

### Part I — Basics (`src/basics/`)

- [x] **1. Information theory** `{#sec-informationtheory}` — self-information, entropy (discrete / differential, Bernoulli example), joint and conditional entropy, mutual information, cross-entropy `{#sec-crossentropy}`, KL divergence `{#sec-kl}` (relation to cross-entropy, asymmetry). Rewritten intro (Bell Labs, information as surprise, compression, predicting = compressing) [@Nyquist1924; @Hartley1928; @Shannon1948]; collapsed note on differential entropy; $p$, $q$ notation for cross-entropy and KL.
- [x] **2. Maximum likelihood estimation** `{#sec-mle}` — likelihood vs. joint probability, i.i.d. assumption, log-likelihood and NLL; worked derivation of MLE for the univariate normal (empirical mean and variance).
- [x] **3. Bayesian estimation** `{#sec-bayesian}` — central limit theorem (frequentist view); MAP `{#sec-map}`: Bayes' rule on the parameters (prior, likelihood, evidence, posterior), MLE as MAP with a uniform prior, MAP for the mean of a normal with a normal prior (weighted average of $\bar{x}$ and $\mu_0$, limit cases, conjugate posterior variance, figure `map-posterior.png`), MAP as L2 / L1 regularization, limitations (point estimate, posterior predictive, intractable evidence → VI).
- Notation unified: cross-entropy and KL use distributions $p$, $q$ ($H(p, q)$, $\text{KL}(p || q)$, normals $\mathcal{N}(x; \mu_p, \sigma_p)$) in `basics/1-InformationTheory.qmd` and `statisticalml/2-KL.qmd`; $X$, $Y$ kept for entropy / joint / conditional entropy / mutual information; `cross-entropy.svg` replaced by `cross-entropy.png` (`src/img/cross-entropy.py`).
- [x] ★ **Forward vs. reverse KL** `{#sec-forwardreverse}` — new last section of `statisticalml/2-KL.qmd`, approved (figure `forward-reverse-kl.png`, modes at ±2 as the reverse KL also covers both modes at ±1.5).

### Part II — Statistical machine learning (`src/statisticalml/`)

- [x] **1. Learning paradigms** `{#sec-paradigms}` — polished: one figure `learning-paradigms.png` (`src/img/learning-paradigms.py`, four panels replacing the mermaid diagrams); discriminative $p_\theta(y | x)$ (classification, regression with fixed or predicted variance, MDN), generative $p_\theta(x)$ (density estimation vs. generation), autoregressive models as generative models through the chain rule (LLMs, self-supervised, masked modeling), summary table.
- [x] **2. KL divergence in machine learning** `{#sec-klml}` — closed form between two Gaussians `{#sec-klnormal}` (full derivation, special case $\mathcal{N}(0,1)$); equivalence of KL minimization, cross-entropy minimization and MLE.
- [x] **3. Softmax classifier** `{#sec-softmax}` — one-hot targets as a multinomial distribution, softmax, cross-entropy loss $-\mathbf{t} \cdot \log \mathbf{y}$, gradient $-(\mathbf{t} - \mathbf{y})$ w.r.t. the logits (collapsed proof). Rewritten: classification as density estimation, the softmax function (ranking, differences, temperature), the cross-entropy loss (NLL of the correct class, limit cases), the gradient (prediction error), using the predicted distribution (argmax, sampling, LLMs).
- [x] **4. Mixture density networks** `{#sec-mdn}` — mean-variance estimation (NLL of a Gaussian, regression as a special case), multimodal targets (inverse problems, half-moons), mixture of Gaussians output, NLL loss, two-step sampling.

### Part III — Sampling methods (`src/sampling/`) — new

- [x] ★ **1. Monte Carlo and importance sampling** `{#sec-montecarlo}` — why sampling (expectations everywhere, curse of dimensionality `{#sec-curse}`, $\pi$ example), the Monte Carlo estimator `{#sec-mcestimator}` (unbiased, $\sigma_f / \sqrt{N}$, KL estimate), sampling from simple distributions `{#sec-simplesampling}` (inverse transform, normal, categorical, ancestral), rejection sampling `{#sec-rejection}`, importance sampling `{#sec-importance}` (self-normalized, ESS, PPO), limits `{#sec-mclimits}` (ESS collapse, idea of MCMC, summary table). Detailed plan below.
- [x] ★ **2. Markov chain Monte Carlo** `{#sec-mcmc}` — Markov chains `{#sec-markovchains}` (weather example, stationary distribution, time averages), detailed balance `{#sec-detailedbalance}` (global vs. pairwise balance of flows), Metropolis-Hastings `{#sec-metropolishastings}`, using MCMC in practice `{#sec-mcmcpractice}` (step size, burn-in, ESS, multiple chains), Gibbs sampling `{#sec-gibbs}`, posterior example `{#sec-mcmcposterior}` (posterior predictive), limits `{#sec-mcmclimits}` (gradients, score, summary table). Detailed plan below.
- [x] **3. Langevin dynamics and Hamiltonian Monte Carlo** `{#sec-hmc}` — the score `{#sec-score}`, Langevin dynamics `{#sec-langevin}` (MALA, SGLD), Hamiltonian Monte Carlo `{#sec-hmcalgorithm}` (leapfrog, NUTS), limits and the bridge to diffusion models `{#sec-hmclimits}` (scaling, multimodality, learned score, noise levels, summary table). Detailed plan below.

### Part IV — Variational inference (`src/variationalinference/`)

- [x] **1. Latent variable models** `{#sec-latentvariables}` — inverse problems section `{#sec-inverseproblems}` (encoder as the inverse of the decoder, link to the MDN inverse problem; restoring degraded observations $y = A(x) + $ noise with a generative prior, figure `inverse-problem.png` from `src/img/inverse-problem.py`, [@Song2022a; @Rozet2023]). Rewritten: generative modeling as MLE (back-reference to @sec-klml), images live on a manifold (figure `image-manifold.png`, `src/img/image-manifold.py`: rotated / scaled "7"s as a 2D surface in pixel space, PCA projection, and the 2D latent grid; manifold hypothesis), latent variables (prior, decoder, ancestral sampling, hidden), the marginal likelihood is intractable (Monte Carlo over the prior fails like a bad importance sampling proposal → posterior / encoder → VI).
- [x] **2. Variational inference** `{#sec-variationalinference}` — merged with the former chapter "Why maximizing the ELBO works": the intractable posterior, the evidence lower bound (single derivation $\log p_\theta(x) = \text{ELBO} + \text{KL}$; Jensen / importance sampling view in a collapsed callout), why maximizing the ELBO works `{#sec-elbo}` (gradient w.r.t. $\phi$, decoder, EM, figure `elbo-gap.png` from `src/img/elbo-gap.py`) [@Luo2022a], a tractable loss (reconstruction + regularization), summary.
- [x] **3. Variational autoencoders** `{#sec-vae}` — intro continues the VI chapter with three open questions (distributions, gradient, latent space); choosing the distributions (prior, encoder outputs $\mu_x, \sigma_x$, decoder), the loss (BCE / MSE reconstruction, closed-form KL, log-variance), the reparameterization trick `{#sec-reparameterization}` (gradient of an expectation over $q_\phi$, $z = \mu + \sigma \odot \xi$, REINFORCE mentioned), a smooth latent space (why autoencoders cannot generate, noise → local smoothness, KL → no holes / overlaps / interpolation, MNIST 2D manifold, link to `image-manifold.png`, blurriness and $\beta$-VAE). Figures from the Neurocomputing slides (towardsdatascience, Keras blog).
- [x] **4. Gradient estimators** `{#sec-gradientestimators}` — the gradient of an expectation, reparameterization (pathwise) estimator, score-function estimator (log-derivative trick, two scores, baselines; figure `gradient-estimators.png`), application to reinforcement learning `{#sec-reinforcementlearning}` (figures `rl-agent.png`, `policy-search.svg` from the Deep RL book; REINFORCE, PPO, RLHF), summary table.
- [x] **5. Tighter and weighted bounds** `{#sec-tighterbounds}` — IWAE `{#sec-iwae}` (short, no figure) [@Burda2016], $\beta$-VAE `{#sec-betavae}` (disentanglement, figure `betavae-results.png` [@Higgins2016], posterior collapse, KL annealing), summary of Part IV (transition to diffusion as hierarchical VAEs).

### Part V — Deep generative models (`src/generative/`)

- [ ] Normalizing flows — change of variables, exact likelihood; continuous normalizing flows already introduced in @sec-flowmatching.
- [ ] Energy-based models — score matching is now covered in @sec-diffusion (denoising score matching, Tweedie); a possible short chapter on energy-based models alone (unnormalized densities $e^{-E_\theta(x)}$, training with MCMC samples / contrastive divergence), or drop.
- [x] ★ **1. Denoising diffusion models** `{#sec-diffusion}` and **2. Conditional and latent diffusion** `{#sec-conditionaldiffusion}` — written (detailed plan below, "Chapter plan: Diffusion models"). Notes: port `~/Documents/Obsidian/Notes/Machine Learning/Diffusion/` (~850 lines; its images `langevin.gif`, `kerneldensityestimation.png`, `manifold.png` are in the Obsidian `Inference/img/` folder); hierarchical-VAE derivation of @Luo2022a.
    - Planned paragraph: **diffusion and flow matching in the forward / reverse KL dichotomy** (link to @sec-forwardreverse).
        - Rule of thumb: an expectation over the **data** is mass-covering (MLE, VAE, diffusion, flow matching); over the **model's own samples** it is mode-seeking (VI, mode collapse of GAN generators).
        - Diffusion is trained by maximizing an ELBO on $\log p_\theta(x)$ (hierarchical VAE, @Luo2022a); the encoder is the fixed noising process, and each term $\text{KL}(q(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t))$ has the data-driven distribution first → forward KL / maximum likelihood. With the right weighting, score matching also bounds $\text{KL}(p_\text{data} || p_\theta)$ [@Song2021].
        - Not blurry because the model family is flexible: each reverse step is nearly Gaussian for small steps, so a Gaussian per step is almost exact, and the chain of many steps gives an arbitrarily multimodal marginal. The forward KL only averages between modes when the family cannot represent the target (single normal of `forward-reverse-kl.png`).
        - The averaging happens at the right place: the MSE denoiser predicts $\mathbb{E}[x_0 | x_t]$, blurry at high noise but exactly the score of the noisy distribution (Tweedie's formula); the modes are separated progressively as the noise decreases (the noise-levels argument of @sec-hmclimits, seen from the training side).
        - Flow matching: same picture — MSE regression of conditional velocities over data samples (covering), learned field $\mathbb{E}[u_t(x | x_0) | x_t]$; not literally a KL (closer to transport distances).
        - Mode-seeking comes back at sampling time: classifier-free guidance samples roughly from $p(x | c) \, p(c | x)^w$, sharpening towards the modes (diversity traded for quality).
- [x] **3. Optimal transport** `{#sec-ot}` — written (detailed plan below, "Chapter plan: Optimal transport"); Jensen-Shannon defined in @sec-kl.
- [x] **4. Flow matching** `{#sec-flowmatching}` — written (detailed plan below, "Chapter plan: Flow matching"). Original entry: after optimal transport: continuous normalizing flows, velocity field regression on conditional paths [@Lipman2023], link to the probability-flow ODE of the diffusion chapter, OT couplings / rectified flows (straighter paths, fewer sampling steps). Detailed outline to write when we get there.
    - (The draft text moved from the OT chapter is now in section 6 of the flow matching chapter.)
- [ ] GANs as divergence minimization (f-divergences) — groundwork in place: Jensen-Shannon defined in @sec-kl, Wasserstein GANs in @sec-ot.

### Part VI — Uncertainty and Bayesian deep learning — optional

- [ ] Aleatoric vs. epistemic uncertainty (MVE / MDN only capture aleatoric uncertainty).
- [ ] Bayesian neural networks, MC dropout, deep ensembles.
- [ ] Gaussian processes (mentioned in the MDN chapter, never explained).

### Part VII — Inference over time and for control — optional

- [ ] State-space models: Kalman filter, particle filters (sequential importance sampling); background for world models.
- [ ] Control as inference [@Levine2018], maximum-entropy RL.
- [ ] Active inference: variational free energy = negative ELBO, expected free energy, action-perception loop.

### References — `src/references.qmd` [x]

---

## Open items

- [x] GitHub repository `git@github.com:vitay/probabilisticdl.git` (renamed from `probabilisticinference`); CI `.github/workflows/quarto-publish.yml` renders and publishes to the `gh-pages` branch on every push to `main` (as in the Deep RL book); site <https://julien-vitay.net/probabilisticdl/>.

---

## Chapter plan: Monte Carlo and importance sampling

File: `src/sampling/1-MonteCarlo.qmd`, label `{#sec-montecarlo}`, first chapter of a new part **Sampling methods** placed between *Statistical machine learning* and *Variational inference*.

**Goal of the chapter:** explain how to estimate expectations $\mathbb{E}_{x \sim p}[f(x)]$ with samples, why this works in high dimensions when numerical integration does not, and what to do when we cannot sample from $p$ directly. It ends on the limits of these methods, which motivate MCMC.

**Notation** (to agree on):

- $p(x)$: the target distribution; $\tilde{p}(x)$ its unnormalized version, $p(x) = \tilde{p}(x) / Z$.
- $q(x)$: a proposal distribution we can sample from.
- $f(x)$: the function whose expectation we want.
- $\hat{f}_N = \frac{1}{N} \sum_{i=1}^N f(x_i)$: the Monte Carlo estimator with $N$ samples.
- $w(x) = p(x) / q(x)$: importance weights.

### Sections

1. [x] **Why sampling?** — approved; $\pi$ example added.
    - Nearly everything in the book is an expectation: entropy / cross-entropy / KL, loss functions, the evidence $p(x) = \mathbb{E}_{z}[p(x|z)]$, the posterior predictive, the ELBO.
    - Closed forms exist only in special cases (KL between Gaussians, conjugate priors).
    - Numerical integration on a grid needs $M^d$ evaluations: impossible beyond a few dimensions. Label `{#sec-curse}`, figure `curse-of-dimensionality.png` (moved here from the latent variables chapter).
    - Sampling replaces the integral by an average. Example callout: computing $\pi$ with uniform samples in a square (indicator function $\mathbb{I}$), figure `montecarlo-pi.png` ($N$ = 100, 1000, 10000). Requirements: evaluate $f$, sample from $p$. Outline of the chapter.
2. [x] **The Monte Carlo estimator** `{#sec-mcestimator}` — approved. Figure `montecarlo-convergence.png` (runs + standard-error band; RMSE vs. $1.64/\sqrt{N}$ on log-log).
    - Definition of $\hat{f}_N$.
    - Unbiasedness: $\mathbb{E}[\hat{f}_N] = \mathbb{E}_p[f]$ (derivation).
    - Variance: $\text{Var}(\hat{f}_N) = \text{Var}_p(f) / N$ (derivation, uses i.i.d.).
    - Consequences: error $\propto 1 / \sqrt{N}$ (CLT, link to @sec-bayesian), **independent of the dimension** $d$; halving the error costs 4× more samples; the constant $\text{Var}_p(f)$ is what matters.
    - Back to the $\pi$ example of section 1: variance of $4 \, \mathbb{I}(||x|| \leq 1)$ is $\pi (4 - \pi)$, hence the standard error $\approx 1.64 / \sqrt{N}$.
    - Example: Monte Carlo estimate of the KL divergence $\text{KL}(p || q) \approx \frac{1}{N} \sum_i - \log \frac{q(x_i)}{p(x_i)}$ with $x_i \sim p$ (moved here from the KL chapter; can reuse `cross-entropy.svg`).
    - *Figure (proposed):* `montecarlo-convergence.png` — running estimate of $\pi$ vs. $N$ (log-scale) for a few independent runs, with the $\pm$ standard-error band.
3. [x] **Sampling from simple distributions** `{#sec-simplesampling}` — approved.
    - Pseudo-random number generators, seeds; affine transformation of uniforms.
    - Inverse transform sampling $x = F^{-1}(u)$ (collapsed proof; exponential distribution). Limits: needs the inverse CDF, 1D only.
    - Normal distributions: Box-Muller (no proof), affine / Cholesky transformation → the VAE reparameterization trick.
    - Categorical distributions: segments of $[0, 1]$ (softmax, MDN components, LLM tokens).
    - Ancestral sampling: $z \sim p(z)$ then $x \sim p(x|z)$; discarding $z$ samples the marginal without the integral (MDN, VAE generation, autoregressive models).
    - The sampling / evaluation asymmetry; unnormalized densities $\tilde{p}(x) = Z \, p(x)$ (posterior) → rejection and importance sampling.
    - Figure `inverse-transform.png` (`src/img/inverse-transform.py`) — CDF of the exponential with uniform $u$ on the y-axis mapped to $x$, histogram of the samples against the pdf.
4. [x] **Rejection sampling** `{#sec-rejection}` — approved.
    - Setting: evaluate $\tilde{p}(x)$, proposal $q$, envelope $M q(x) \geq \tilde{p}(x)$; algorithm; geometric interpretation (uniform points under the envelope); collapsed proof; $Z$ never needed.
    - Acceptance rate $Z / M$; choosing $M$ (too small → silently wrong samples); proposal tails at least as heavy as the target.
    - Failure in high dimensions: two isotropic Gaussians, $M = (\sigma_q / \sigma_p)^d$, $1.01^{-1000} \approx 5 \times 10^{-5}$.
    - Figure `rejection-sampling.png` (`src/img/rejection-sampling.py`): bimodal $\tilde{p}$ with $Z = 3$, proposal $\mathcal{N}(0, 2)$, $M \approx 8.2$; accepted / rejected points under the envelope; histogram of accepted samples vs. $p(x)$.
5. [x] **Importance sampling** `{#sec-importance}` — approved.
    - Intuition first (figure `importance-sampling-principle.png`): plain average of samples from $q$ estimates the mean of $q$; weighting by $p / q$ recovers the mean of $p$ (seed 2, typical run).
    - Principle: $\mathbb{E}_p[f] = \mathbb{E}_q[w f]$, $w = p / q$; estimator, support condition, unbiased, variance $\text{Var}_q(w f) / N$.
    - Choosing the proposal (no formula): weights close to 1 when $q \approx p$, huge weights where $q \ll p$; $q$ close to $p$ with heavy enough tails. (Variance integral, optimal $q^* \propto |f| p$ and the rare-event example dropped: not needed later.)
    - Failures of a too narrow proposal: missed regions (no sample) vs. huge weights (one sample dominates).
    - Self-normalized importance sampling (short): normalized weights $\bar{w}_i$, $Z$ cancels in the normalization, slightly biased; by-product $Z \approx \frac{1}{N} \sum \tilde{w}_i$ (needed for IWAE / evidence estimation).
    - Effective sample size $1 / \sum \bar{w}_i^2$; warning callout: the ESS cannot detect missed modes.
    - Pointers: PPO in RL (robotics, RLHF of ChatGPT) [@Schulman2017a; @Ouyang2022], which clips the weights near 1; IWAE, particle filters.
    - Figure `importance-sampling.png` (`src/img/importance-sampling.py`): bimodal target of the rejection figure, $N = 50$; good proposal $\mathcal{N}(0, 2)$ vs. two runs (seeds 5 and 3) of $\mathcal{N}(-1.5, 0.8)$: missed mode (ESS 46, estimate $-1.66$) and degenerate weights (ESS 1.8); estimates of $\mathbb{E}_p[x] = -0.3$ in the titles.
6. [x] **Limits: towards MCMC** `{#sec-mclimits}` — approved.
    - Importance sampling in high dimensions: weights are products of $d$ factors ($1.1^{100}$ vs. $0.9^{100}$); table of ESS/N for $\sigma_q = 1.2 \, \sigma_p$ (analytical $(\sqrt{1.88}/1.44)^d$, checked by simulation).
    - Fixed proposals must be good everywhere, but the mass is concentrated in an unknown tiny region.
    - Idea of MCMC: local proposals, accept with a probability depending on $\tilde{p}(x') / \tilde{p}(x)$ ($Z$ cancels); correlated samples → larger variance.
    - Summary table of the chapter: method / requires / produces / fails when.

---

## Chapter plan: Markov chain Monte Carlo

File: `src/sampling/2-MCMC.qmd`, label `{#sec-mcmc}`, second chapter of *Sampling methods*.

**Goal of the chapter:** show how a Markov chain that only ever compares $\tilde{p}$ at two points can produce samples of $p$, why Metropolis-Hastings works, how to use it in practice (step size, burn-in, correlated samples), and Gibbs sampling as the other classical method. Theory is kept light, as in the previous chapter: results on Markov chains are stated with intuition, the only proof (detailed balance → stationarity) is short and collapsed.

**Notation** (to agree on):

- $p(x) = \tilde{p}(x) / Z$: the target, as in @sec-montecarlo.
- $x_0, x_1, \ldots, x_t, \ldots$: the chain; $t$ is the step.
- $T(x' | x)$: the transition kernel (probability of moving from $x$ to $x'$ in one step).
- $p_t(x)$: the distribution of the chain at step $t$ (over many independent runs).
- $q(x' | x)$: the proposal of Metropolis-Hastings — keeps the name "proposal" of the previous chapter, now conditioned on the current sample. 
- $\alpha(x, x')$: the acceptance probability.

**References** (in `Library.bib`, see Decisions): Metropolis et al. (1953), Hastings (1970), Geman & Geman (1984) for Gibbs sampling, Roberts, Gelman & Gilks (1997) for the optimal acceptance rate; textbooks: Bishop (2006, ch. 11), MacKay (2003, ch. 29-30).

### Sections

1. [x] **Markov chains** `{#sec-markovchains}` — approved.
    - Markov property, transition kernel, simulation; weather example as a table.
    - Evolution of the distribution (law of total probability), worked numbers $p_1, p_2, \ldots$.
    - Stationary distribution $(6/13, 4/13, 3/13)$, checked by hand; irreducible + aperiodic → unique, convergence from any start (stated, @Bishop2006, @MacKay2003).
    - Time averages of a single chain converge to $\mathbb{E}_p[f]$ (correlated Monte Carlo estimator).
    - Turning the problem around: design $T$ for a given $p$.
    - Figure `markov-chain.png` (`src/img/markov-chain.py`): graph, $p_t$ from two initial states, and (added) the time fractions of one chain of 10000 steps.
2. [x] **Detailed balance** `{#sec-detailedbalance}` — approved.
    - Probability flows $p(x) T(x'|x)$; stationarity as global balance (inflow = outflow for each state), which allows one-way loops A → B → C → A (crowd analogy).
    - Detailed balance $p(x) T(x'|x) = p(x') T(x|x')$: stronger, pairwise condition; summing over $x$ gives global balance (inline one-line proof); requires every move to be possible backwards (reversible chains).
    - The weather chain satisfies it (table of the three pairs of flows).
    - Sufficient, not necessary: the loop (extreme, periodic) and the rotating 3-state chain (0.6 / 0.4), uniform stationary distribution with a circulating flow.
    - Ratio form $T(x'|x) / T(x|x') = \tilde{p}(x') / \tilde{p}(x)$: $Z$ cancels → Metropolis-Hastings.
    - No figure.
3. [x] **The Metropolis-Hastings algorithm** `{#sec-metropolishastings}` — approved.
    - Algorithm with $q(x'|x)$ and $\alpha = \min(1, \frac{\tilde{p}(x') q(x_t|x')}{\tilde{p}(x_t) q(x'|x_t)})$ [@Metropolis1953; @Hastings1970]; $Z$ cancels (posteriors).
    - Symmetric proposal $\mathcal{N}(x, s)$ → Metropolis rule, random-walk Metropolis: always uphill, sometimes downhill (otherwise an optimizer).
    - Rejected candidates are repeated: repetitions give each state its weight.
    - Collapsed proof of detailed balance (flow $= \min(p(x) q(x'|x), p(x') q(x|x'))$, symmetric), read as slowing down the larger flow.
    - Figure `metropolis-hastings.png` (`src/img/metropolis-hastings.py`, seed 1): bimodal target, $s = 1$, $x_0 = 0$; trace of 200 steps with rejected candidates, histogram of 10000 steps; acceptance 65%, mean $-0.34$ (true $-0.3$), ~25 steps per mode on average.
4. [x] **Using MCMC in practice** `{#sec-mcmcpractice}` — approved.
    - Step size: too small (random walk, high acceptance), too large (rejections), good; acceptance 25-50%, 23% in high dimensions [@Roberts1997]; $s = 1$ of section 3 was a bit small.
    - Burn-in: discard the steps that still depend on $x_0$, chosen from the trace.
    - Correlated samples in words (autocorrelation, ESS with the same meaning as in importance sampling, thinning); formulas of $\rho_k$ and the ESS in a collapsed callout.
    - Multimodality: stuck chains look converged (missed mode); several chains from different initial states as a diagnostic, no proof of convergence. No numbers.
    - Figure `metropolis-stepsize.png` (`src/img/metropolis-stepsize.py`, seed 0): $s = 0.1, 3, 50$ from $x_0 = 8$; trace with burn-in, histogram, autocorrelation; acceptance 96 / 42 / 3%, ESS 8 / 1584 / 193 out of 9500.
5. [x] **Gibbs sampling** `{#sec-gibbs}` — approved.
    - One coordinate at a time from its full conditional, sweeps, no acceptance step [@Geman1984]; special case of MH with $\alpha = 1$ (collapsed proof).
    - Useful when the conditionals are simple: conjugate models, latent variable models; image restoration.
    - Example: bivariate normal, conditionals $\mathcal{N}(\rho x_2, \sqrt{1 - \rho^2})$ (admitted); independent vs. strongly correlated coordinates; blocked Gibbs in one sentence.
    - Figure `gibbs-sampling.png` (`src/img/gibbs-sampling.py`, seed 0): Gibbs on $\rho = 0$ vs. $\rho = 0.95$, 30 sweeps from $(-3, -3)$ (instead of Gibbs vs. random-walk Metropolis).
6. [x] **Example: sampling a posterior** `{#sec-mcmcposterior}` — approved (no subsections).
    - Back to @sec-map: $\mu$ and $\sigma$ unknown, 20 samples of $\mathcal{N}(3, 2)$, priors $\mathcal{N}(0, 5)$ and half-normal (not conjugate); unnormalized posterior = likelihood × prior.
    - Random-walk Metropolis on $(\mu, \sigma)$ directly, negative $\sigma$ rejected; acceptance test in the log domain.
    - Whole posterior instead of a point estimate; posterior predictive as an average of $\mathcal{N}(x; \mu_t, \sigma_t)$ over the chain.
    - Figure `mcmc-posterior.png` (`src/img/mcmc-posterior.py`, seed 0): joint samples with marginals, posterior predictive with a few individual densities and the data.
7. [x] **Limits: towards gradient-based samplers** `{#sec-mcmclimits}` — approved (no subsections, no figure).
    - Random-walk moves are blind; in high dimensions the step size must shrink and exploration slows down (distance $\propto \sqrt{t}$); Gibbs: correlated coordinates.
    - Gradient $\nabla_x \log \tilde{p}(x) = \nabla_x \log p(x)$ ($Z$ disappears); Langevin dynamics, HMC → next chapter; the score → diffusion models.
    - Summary table: Metropolis-Hastings, Gibbs sampling (requires / produces / fails when).

### Decisions

- Proposal notation: $q(x'|x)$.
- Section 6 (posterior example): kept for now.
- Figures: all five.
- References: all in `Library.bib` as `Metropolis1953`, `Hastings1970`, `Geman1984`, `Roberts1997` (plus `Bishop2006`, `MacKay2003`):
    - Metropolis et al. (1953), *Equation of State Calculations by Fast Computing Machines*, J. Chem. Phys. — doi:10.1063/1.1699114
    - Hastings (1970), *Monte Carlo sampling methods using Markov chains and their applications*, Biometrika — doi:10.1093/biomet/57.1.97
    - Geman & Geman (1984), *Stochastic Relaxation, Gibbs Distributions, and the Bayesian Restoration of Images*, IEEE TPAMI — doi:10.1109/TPAMI.1984.4767596
    - Roberts, Gelman & Gilks (1997), *Weak convergence and optimal scaling of random walk Metropolis algorithms*, Ann. Appl. Probab. — doi:10.1214/aoap/1034625254
    - Bishop (2006), *Pattern Recognition and Machine Learning*, Springer (ch. 11) — <https://www.microsoft.com/en-us/research/publication/pattern-recognition-machine-learning/>
    - MacKay (2003), *Information Theory, Inference, and Learning Algorithms*, Cambridge University Press (ch. 29-30) — <https://www.inference.org.uk/itila/book.html>

---

## Chapter plan: Langevin dynamics and Hamiltonian Monte Carlo

File: `src/sampling/3-HMC.qmd`, label `{#sec-hmc}`, third and last chapter of *Sampling methods*.

**Goal of the chapter:** show how the gradient of the log-density (the score) lets a sampler make large moves that are still accepted, through two methods: Langevin dynamics (gradient ascent plus noise) and Hamiltonian Monte Carlo (a ball rolling without friction on the landscape of the target). The chapter ends with the bridge to score-based diffusion models: if the score can be learned from data, Langevin dynamics generates new samples. As in the previous chapters, the main text keeps to the main ideas; the physics (Hamiltonian dynamics, leapfrog) is kept minimal and the convergence results are stated, not proved.

**Notation** (to agree on):

- $p(x) = \tilde{p}(x) / Z$: the target, as in @sec-montecarlo and @sec-mcmc.
- $\nabla_x \log p(x)$: the **score**, always written out (no symbol: $s$ was the step size of random-walk Metropolis); $s_\theta(x)$ only for a learned score at the end of the chapter, as in the diffusion literature.
- $\epsilon$: the step size of Langevin dynamics and of the leapfrog integrator (standard in the literature, replaces $s$).
- $U(x) = - \log \tilde{p}(x)$: the **potential energy** (HMC only).
- $v$: the momentum (instead of the usual $p$ of physics, which would clash with the densities); $K(v) = \frac{1}{2} ||v||^2$ the kinetic energy; $H(x, v) = U(x) + K(v)$ the Hamiltonian.
- $L$: the number of leapfrog steps per HMC iteration.

**References** (all in `Library.bib`):

- `Duane1987` — Hybrid Monte Carlo (origin of HMC)
- `Roberts1996` — MALA
- `Neal2012` — MCMC using Hamiltonian dynamics
- `Welling2011` — SGLD
- `Hoffman2011` — the No-U-Turn Sampler
- `Betancourt2017` — conceptual introduction to HMC
- `Song2020a` — score-based generative modeling

### Sections

1. [x] **The score** `{#sec-score}` — approved (no subsections).
    - Recap of @sec-mcmclimits: random-walk proposals are blind; the gradient of the log-density points uphill, and does not depend on $Z$.
    - Examples: the score of $\mathcal{N}(\mu, \sigma)$ is $-(x - \mu) / \sigma^2$ (a spring pulling back to the mean, stronger for narrow distributions); for the bimodal target, the score points towards the nearest mode and vanishes at the modes and at the bottom of the valley.
    - Computing it: by hand for simple models, by automatic differentiation otherwise (the same backpropagation as for training neural networks); the target must be differentiable (continuous $x$ only).
    - Figure `score.png` (`src/img/score.py`, redone in the style of the book after @Song2020a's blog post): left, the 1D bimodal target of the previous chapters and its score; right, a 2D mixture of two Gaussians with the score as a vector field.
2. [x] **Langevin dynamics** `{#sec-langevin}` — approved (no subsections; MALA [@Roberts1996], SGLD [@Welling2011]).
    - Update $x_{t+1} = x_t + \epsilon \, \nabla_x \log \tilde{p}(x_t) + \sqrt{2 \epsilon} \, \xi_t$, $\xi_t \sim \mathcal{N}(0, 1)$.
    - Limit cases: without noise, gradient ascent on $\log p$ → converges to a mode (the MAP of @sec-map); without gradient, a random walk. The $\sqrt{2 \epsilon}$ factor balances the two so that $p$ is stationary (stated, for $\epsilon \rightarrow 0$; no proof).
    - With a finite $\epsilon$, the samples are slightly biased. **MALA** (Metropolis-adjusted Langevin algorithm) uses the Langevin step as the proposal of Metropolis-Hastings; this proposal is not symmetric, which is a use case for the Hastings correction $q(x_t | x') / q(x' | x_t)$ of @sec-metropolishastings.
    - Pointer (one paragraph): stochastic gradient Langevin dynamics for Bayesian deep learning, where the gradient is estimated on minibatches.
    - Figure `langevin.png` (`src/img/langevin.py`, redone after the blog post's `langevin.gif`): 400 Langevin chains on the 2D mixture of `score.png`, starting on a regular grid, snapshots at $t$ = 0, 10, 100, 1000 (the 1D histogram panel was dropped). Note for section 4: the chains first split about 50/50 between the modes and only slowly reach the 60/40 weights of the target.
3. [x] **Hamiltonian Monte Carlo** `{#sec-hmcalgorithm}` — approved (subsections: ball on the landscape, algorithm, tuning).
    - Physical analogy: $U(x) = - \log \tilde{p}(x)$ is a landscape whose valleys are the modes; a ball given a random push rolls without friction, travels far, and spends more time in the valleys.
    - Auxiliary momentum $v \sim \mathcal{N}(0, 1)$; joint $p(x, v) \propto \exp(- H(x, v))$ whose marginal on $x$ is the target: sampling $(x, v)$ and dropping $v$ samples $p(x)$ (ancestral-sampling argument of @sec-simplesampling in reverse).
    - The dynamics conserves $H$: the ball moves along a level set of $H$, i.e. can travel far while staying in regions of high probability.
    - Leapfrog integrator (three lines of updates using the score, in a collapsed callout); $H$ is only approximately conserved → Metropolis-Hastings correction with $\min(1, \exp(H(x, v) - H(x', v')))$, close to 1 for a small $\epsilon$.
    - Algorithm: resample $v$, $L$ leapfrog steps, accept / reject.
    - Tuning: $\epsilon$ (acceptance) and $L$ (length of the trajectory: too short → random walk, too long → U-turns back); NUTS chooses $L$ automatically, the default in Stan, PyMC and NumPyro.
    - Figure `hmc.png` (own matplotlib script `src/img/hmc.py`, no equivalent in the blog post) — the correlated 2D Gaussian of @fig-gibbssampling ($\rho = 0.95$), same starting point: random-walk Metropolis, Gibbs, and HMC (with the leapfrog trajectories drawn) for the same number of iterations.
4. [x] **Limits and the bridge to diffusion models** `{#sec-hmclimits}` — approved (subsections: limits, learning the score, summary).
    - One sentence on scaling (merged former section 4): the number of steps needed grows roughly like $d$ for random-walk Metropolis, $d^{1/3}$ for MALA and $d^{1/4}$ for HMC; no figure.
    - Still local: the score points to the nearest mode, so crossing the valleys between distant modes remains hard (one sentence on tempering / annealing).
    - Requires a differentiable $\tilde{p}(x)$: no discrete variables.
    - The key observation: Langevin dynamics only needs the score, not the density. If we only have **data** and no $\tilde{p}$, but can **learn** the score $s_\theta(x) \approx \nabla_x \log p(x)$ with a neural network, Langevin dynamics generates new samples: this is the idea of score-based generative models [@Song2020a]. Adding noise at several levels makes the score easier to learn and solves the multimodality problem → diffusion models (Part V).
    - Summary table of the chapter: Langevin / MALA, HMC (requires / produces / fails when), same format as the previous chapters.
    - Figures from the blog post: `score-matching-langevin.jpg` (data samples → score matching → scores → Langevin dynamics → new samples) and `score-pitfalls.jpg` (the learned score is only accurate near the modes, motivating the noise levels of diffusion models).

### Decisions

- Title and label: "Langevin dynamics and Hamiltonian Monte Carlo", `{#sec-hmc}`.
- Former section 4 (scaling with the dimension) merged into the limits section as one sentence; its figure `sampler-dimension.png` is dropped.
- Momentum $v$.
- Figures: `score.png`, `langevin.png` and `hmc.png` from own scripts (house style, same targets as the previous chapters); `score-matching-langevin.jpg` and `score-pitfalls.jpg` from Yang Song's blog post, captions with "Source: <https://yang-song.net/blog/2021/score/>".

---

## Section plan: Forward vs. reverse KL

File: `src/statisticalml/2-KL.qmd`, new last section `## Forward vs. reverse KL {#sec-forwardreverse}`, after "Equivalence between KL and cross-entropy" (which ends on MLE = minimizing $\text{KL}(p || p_\theta)$).

**Goal:** the KL is not symmetric (@sec-kl only states it); show that the two directions lead to very different fits, and that the choice is usually imposed by what we can compute: samples of the target (forward KL, maximum likelihood) or its unnormalized density (reverse KL, variational inference). Main ideas only, few formulas.

**Notation:** generic $p$ (the target) and $q$ (the approximation we fit), with the mapping given once: MLE fits $q = p_\theta$ to the data $p$; VI fits $q = q_\phi(z|x)$ to the posterior $p = p_\theta(z|x)$.

### Content

1. **Two directions.**
    - Forward KL $\text{KL}(p || q) = \mathbb{E}_{x \sim p}[- \log \frac{q(x)}{p(x)}]$: expectation under the target, needs **samples of $p$** → maximum likelihood (previous section).
    - Reverse KL $\text{KL}(q || p) = \mathbb{E}_{x \sim q}[- \log \frac{p(x)}{q(x)}]$: expectation under the approximation, needs samples of $q$ and the **density of $p$, up to its normalization constant** ($\log Z$ is a constant) → variational inference (@sec-variationalinference), where the posterior can be evaluated up to the evidence but not sampled.
2. **Where the penalty is.**
    - Forward: the log-ratio is weighted by $p$; large penalty where $p(x) > 0$ but $q(x) \approx 0$ → $q$ must put mass **everywhere $p$ has mass** (mass-covering, "mean-seeking"); no penalty for putting mass where $p$ has none.
    - Reverse: weighted by $q$; large penalty where $q(x) > 0$ but $p(x) \approx 0$ → $q$ must **avoid the regions where $p$ is small** (zero-forcing, mode-seeking); no penalty for ignoring parts of $p$.
3. **Example: fitting a normal distribution to the bimodal target** of @sec-montecarlo.
    - Forward KL: a single broad normal covering both modes, with its mass in the valley between them; for a normal $q$, the optimum simply matches the mean and variance of $p$ (stated, collapsed one-line proof optional).
    - Reverse KL: two local optima, each a narrow normal on one of the modes; which one is found depends on the initialization.
    - Figure (proposed) `forward-reverse-kl.png` (`src/img/forward-reverse-kl.py`): left, forward-KL fit; right, the two reverse-KL fits (numerical minimization over $(\mu, \sigma)$), with the value of each KL in the legend.
4. **Consequences** (short, bullet list).
    - Maximum likelihood (forward): models try to cover all the data and may put probability between the modes → blurry or "averaged" samples (link to the blurriness of VAEs, @sec-vae).
    - Variational inference (reverse): the approximate posterior is too confident: it underestimates the uncertainty and can miss whole modes of the posterior — the same failure as a too narrow importance sampling proposal (@sec-importance) or a stuck MCMC chain (@sec-mcmcpractice), and a limitation for the multimodal posteriors of inverse problems (@sec-inverseproblems).
    - One sentence: other divergences interpolate between the two behaviors or are symmetric (Jensen-Shannon, already mentioned in @sec-kl; GANs in Part V).

### Related edits

- `sec-kl` (information theory): the sentence "In practice, it does not matter much" contradicts this section → replace by a pointer to @sec-forwardreverse.
- VI summary ("this KL is taken in the opposite direction…"): add the reference @sec-forwardreverse.

### Open questions

- Placement as a section of `2-KL.qmd` (proposed) or a short chapter of its own?
- Figure: 1D bimodal only (proposed), or also a 2D correlated Gaussian fitted by a factorized (diagonal) Gaussian, the classical illustration where reverse KL underestimates the variance?

---

## Chapter plan: Gradient estimators

File: `src/variationalinference/4-GradientEstimators.qmd`, label `{#sec-gradientestimators}`, after the VAE chapter.

**Goal:** the VAE needed the gradient of an expectation whose distribution depends on the parameters, $\nabla_\phi \, \mathbb{E}_{z \sim q_\phi(z)}[f(z)]$, and solved it with the reparameterization trick (@sec-reparameterization). This chapter presents the problem in general and the two families of estimators, and applies the second one to reinforcement learning, introduced from a high-level point of view (the book stays standalone: no reference to the Deep RL book, only figures borrowed from it). Main ideas, few formulas.

**Notation:** $q_\phi(z)$ the distribution whose parameters we optimize, $f(z)$ the function whose expectation we want to maximize or minimize; $\xi$ the parameter-free noise of the reparameterization; for RL: state $s_t$, action $a_t$, reward $r_t$, policy $\pi_\theta(a | s)$, trajectory $\tau$, return $R(\tau)$, trajectory distribution $\rho_\theta(\tau)$.

**References:** in `Library.bib`: `Williams1992` (REINFORCE), `Sutton1998` (RL), `Kingma2013`, `Schulman2017a` (PPO), `Ouyang2022` (RLHF). To add if wanted: Mohamed et al. (2020), *Monte Carlo Gradient Estimation in Machine Learning*, JMLR (review of both estimators).

### Sections

1. [x] **The gradient of an expectation** — approved.
    - The problem $\nabla_\phi \, \mathbb{E}_{z \sim q_\phi(z)}[f(z)]$: the parameters are in the distribution, the gradient cannot be moved inside (recap of @sec-reparameterization).
    - Where it appears: the reconstruction loss of VAEs, discrete latent variables, black-box variational inference (any model, no closed form), and reinforcement learning, where the agent controls the distribution of its own experiences.
2. [x] **The reparameterization estimator** (pathwise gradient) — approved.
    - General form: $z = g_\phi(\xi)$ with $\xi \sim p(\xi)$ fixed, $\nabla_\phi \, \mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{\xi}[\nabla_\phi f(g_\phi(\xi))]$.
    - Requirements: $f$ differentiable with respect to $z$, and $q_\phi$ reparameterizable (location-scale families like the normal, inverse transform sampling of @sec-simplesampling); continuous $z$ only.
    - Low variance: uses the gradient of $f$, i.e. knows in which direction to move each sample.
3. [x] **The score-function estimator** (log-derivative trick, REINFORCE) — approved (offset of 10 added to $f$ in the figure to show the effect of the baseline).
    - Log-derivative trick $\nabla_\phi q_\phi(z) = q_\phi(z) \, \nabla_\phi \log q_\phi(z)$ → $\nabla_\phi \, \mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{q_\phi}[f(z) \, \nabla_\phi \log q_\phi(z)]$ (short derivation in the main text).
    - Intuition: increase the probability of the samples with a high $f(z)$, decrease it for a low $f(z)$; only $f(z)$ is needed, never its gradient: $f$ can be non-differentiable, a black box, and $z$ discrete.
    - Terminology warning: $\nabla_\phi \log q_\phi(z)$ is called the score in statistics (gradient w.r.t. the **parameters**), not to be confused with the score $\nabla_x \log p(x)$ of @sec-score (gradient w.r.t. the **variable**).
    - Unbiased but high variance: $f(z)$ multiplies the score, so an offset of $f$ changes every single estimate.
    - Variance reduction with a **baseline**: $\mathbb{E}_{q_\phi}[\nabla_\phi \log q_\phi(z)] = 0$ (one-line proof), so subtracting a constant $b$ from $f$ does not change the expectation but can reduce the variance a lot.
    - *Figure (proposed):* `gradient-estimators.png` (own script) — distribution of single-sample estimates of $\frac{d}{d\mu} \mathbb{E}_{z \sim \mathcal{N}(\mu, 1)}[f(z)]$ for a simple quadratic $f$: reparameterization, score function, score function with baseline; true gradient as a vertical line.
4. [x] **Application: reinforcement learning** `{#sec-reinforcementlearning}` — approved.
    - The RL problem at a high level: an agent observes a state, takes an action, receives a reward and a new state; the goal is to maximize the sum of rewards (return) of its trajectories [@Sutton1998]. Figure `rl-agent.png` (from the Deep RL book, agent-environment loop).
    - A stochastic policy $\pi_\theta(a | s)$ (a neural network outputting a distribution over actions, e.g. a softmax, @sec-softmax) defines a distribution over trajectories $\rho_\theta(\tau)$; objective $\mathcal{J}(\theta) = \mathbb{E}_{\tau \sim \rho_\theta}[R(\tau)]$: exactly the problem of section 1. Figure `policysearch.svg` (from the Deep RL book: trajectories with high return).
    - The environment is unknown and not differentiable, and actions are often discrete → no reparameterization; the score-function estimator works, and the unknown dynamics disappear from $\nabla_\theta \log \rho_\theta(\tau) = \sum_t \nabla_\theta \log \pi_\theta(a_t | s_t)$ (short derivation): the **REINFORCE** algorithm [@Williams1992], policy gradient methods.
    - Baselines again (the average return); pointers in one sentence: PPO reuses old trajectories with importance sampling (@sec-importance) [@Schulman2017a], and RLHF of language models, where the "actions" are discrete tokens [@Ouyang2022].
5. [x] **Summary** — table: estimator / requires / variance / typical uses — approved.

---

## Chapter plan: Tighter and weighted bounds

File: `src/variationalinference/5-TighterBounds.qmd`, label `{#sec-tighterbounds}`, last chapter of Part IV.

**Goal:** two modifications of the ELBO: making the bound tighter with several importance samples (IWAE), and weighting its two terms differently ($\beta$-VAE). Short chapter.

**References:** in `Library.bib`: `Higgins2016` ($\beta$-VAE). `Burda2016` (IWAE).

### Sections

1. [x] **Importance-weighted autoencoders** `{#sec-iwae}` — approved, short (no figure: little practical interest) [@Burda2016].
    - Recap of the callout of @sec-variationalinference: the ELBO is the logarithm of a single importance weight $p_\theta(x, z) / q_\phi(z | x)$, averaged; Jensen's inequality creates the gap.
    - IWAE bound: average $K$ importance weights **inside** the logarithm: $\mathcal{L}_K = \mathbb{E}[\log \frac{1}{K} \sum_k \frac{p_\theta(x, z_k)}{q_\phi(z_k | x)}]$; $\mathcal{L}_1$ is the ELBO, $\mathcal{L}_K$ increases with $K$ and tends to $\log p_\theta(x)$ (stated) — the average of the weights is the importance sampling estimate of the evidence (@sec-importance), more accurate with more samples.
    - Consequence: the encoder no longer needs to match the posterior exactly; the gradient weights the $K$ samples by their normalized importance weights (self-normalized importance sampling).
    - Cost: $K$ forward passes through the decoder per input; known side effect: the encoder can become worse while the bound gets tighter (one sentence).
    - *Figure (proposed):* `iwae-bound.png` (own script) — on a toy latent variable model where $\log p_\theta(x)$ is computable (linear-Gaussian), $\mathcal{L}_K$ as a function of $K$ (log scale) approaching $\log p_\theta(x)$, for a fixed, imperfect encoder.
2. [x] **$\beta$-VAE** `{#sec-betavae}` — approved (with posterior collapse and KL annealing).
    - Loss: reconstruction $+ \, \beta \, \text{KL}(q_\phi(z|x) || p_\theta(z))$; $\beta = 1$ is the ELBO.
    - $\beta > 1$: stronger pressure towards the factorized prior → more regular latent space, **disentangled** factors of variation [@Higgins2016], blurrier reconstructions; $\beta < 1$: sharper reconstructions, less regular latent space (link to the trade-off of @sec-vae).
    - For $\beta \neq 1$ the loss is no longer a bound on the evidence: it trades likelihood for structure.
    - Figure `betavae-results.png` (from the Neurocomputing slides, source @Higgins2016): traversals of single latent dimensions controlling skin colour, age/gender, saturation.
3. [x] **Summary of Part IV** — approved. — short: latent variable models, VI, VAE, gradient estimators, bounds; transition to the deep generative models of Part V (diffusion models as hierarchical VAEs).

### Decisions

- Two chapters.
- IWAE figure dropped: a trained VAE encoder closes the gap of the toy model, and IWAE has little practical interest; short section without figure (bound, when it helps, evaluation).
- Posterior collapse included in the $\beta$-VAE section: with a powerful decoder, the KL term can drive $q_\phi(z|x)$ to the prior for every input, the latent variables are ignored; KL annealing (increasing $\beta$ from 0 to 1 during training) as a remedy.

---

## Chapter plan: Diffusion models

Files: `src/generative/1-Diffusion.qmd` (`{#sec-diffusion}`, sections 1-6) and `src/generative/2-ConditionalDiffusion.qmd` (`{#sec-conditionaldiffusion}`, sections 7-8), first chapters of a new part **Deep generative models** (`src/generative/`, Part V in `_quarto.yml`). Source material: `~/Documents/Obsidian/Notes/Machine Learning/Diffusion/Diffusion tutorial.md` (DDPM, improved DDPM, DDIM, score-based models, guidance, latent diffusion).

**Goal of the chapter:** present denoising diffusion models as the meeting point of the two halves of the book: they are hierarchical VAEs trained with the ELBO of Part IV, whose network learns the score of Part III, and they are sampled with (a discretization of) Langevin dynamics. Keep the main text on the ideas; the long ELBO derivation and the posterior of the forward process in collapsed callouts.

**Notation** (to agree on):

- $x_0$ a data sample, $p(x_0)$ the data distribution (as in the rest of the book); $x_1, \ldots, x_T$ the noisy versions, $t$ the diffusion step, $T$ the number of steps.
- $q(x_t | x_{t-1})$ the forward (noising) process: it plays the role of the encoder $q$ of Part IV, but has no parameters.
- $\beta_t$ the variance schedule, $\alpha_t = 1 - \beta_t$, $\bar{\alpha}_t = \prod_{s \leq t} \alpha_s$.
- $\epsilon \sim \mathcal{N}(0, I)$ the noise and $\epsilon_\theta(x_t, t)$ the denoising network, as in the literature. $\epsilon$ was the step size of Langevin dynamics in @sec-hmc: in this chapter, step sizes are written $\eta$.
- $p_\theta(x_{t-1} | x_t)$ the learned reverse process; $s_\theta(x_t, t)$ the learned score.

**References** (all in `Library.bib`): `Sohl-Dickstein2015` (origin), `Ho2020` (DDPM), `Nichol2021` (improved DDPM, cosine schedule), `Song2022` (DDIM), `Song2020a` (NCSN, score-based), `Song2021` (maximum likelihood training of score-based models), `Luo2022a` (unified perspective), `Karras2022` (design space), `Dhariwal2021` (classifier guidance), `Ho2022` (classifier-free guidance), `Radford2021` (CLIP), `Rombach2022` (latent diffusion), `Vahdat2021`, `Ronneberger2015` (U-Net), `Lipman2023` (flow matching, pointer only). Optional, not in the library: Hyvärinen (2005) score matching, Vincent (2011) denoising score matching.

### Sections

1. [x] **From VAEs to diffusion models** — approved.
    - The weakness of VAEs (@sec-tighterbounds summary): a single decoder maps a simple distribution onto the data in one step. Diffusion: many small steps, each easy to invert.
    - Idea: destroy the data progressively with noise (fixed encoder), learn to undo each step (decoder); generation starts from pure noise. A hierarchical VAE with $T$ latent layers of the same dimension as the data [@Sohl-Dickstein2015; @Ho2020].
    - Figure (from the notes, source towardsdatascience): `diffusion-steps.webp`, an image progressively noised and denoised.
2. [x] **The forward process** — approved.
    - $q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1 - \beta_t} \, x_{t-1}, \sqrt{\beta_t})$, i.e. $x_t = \sqrt{1 - \beta_t} \, x_{t-1} + \sqrt{\beta_t} \, \epsilon$; the shrinking factor keeps the variance bounded.
    - Closed form in one step (sum of Gaussians): $x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon$; for large $t$, $q(x_T) \approx \mathcal{N}(0, I)$ whatever the data; schedules (linear, cosine [@Nichol2021]).
    - Figure (own, `diffusion-forward.png`): the 2D mixture of @fig-score noised at a few steps $t$ (samples), ending as a standard normal.
3. [x] **The reverse process** — approved.
    - The true reverse step $q(x_{t-1} | x_t)$ is intractable (it depends on the whole data distribution), but nearly Gaussian when $\beta_t$ is small; conditioned on $x_0$ it is Gaussian with a closed form (collapsed derivation, Bayes' rule).
    - Learned reverse step $p_\theta(x_{t-1} | x_t) = \mathcal{N}(\mu_\theta(x_t, t), \sigma_t)$, with the mean parameterized through a network $\epsilon_\theta(x_t, t)$ predicting the noise contained in $x_t$.
    - Network: a U-Net taking $x_t$ and $t$ [@Ronneberger2015] (one paragraph, figure from the notes optional).
4. [x] **Training: the ELBO of a hierarchical VAE** — approved.
    - The ELBO of @sec-variationalinference with the forward process as encoder: a sum of KL divergences between Gaussians, $\text{KL}(q(x_{t-1} | x_t, x_0) || p_\theta(x_{t-1} | x_t))$, in closed form (@sec-klnormal); full derivation in a collapsed callout [@Luo2022a].
    - Simplification [@Ho2020]: each term is a squared error between noises → the simple loss $\mathbb{E}_{x_0, t, \epsilon}[|| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, t) ||^2]$: a regression, no sequence to simulate; training algorithm.
    - The decoder is trained with a forward KL, i.e. maximum likelihood (paragraph planned in the Part V notes above): mass-covering, but not blurry, as each reverse step is nearly Gaussian and the chain of many steps is arbitrarily multimodal; [@Song2021].
5. [x] **Sampling** — approved; learned variances [@Nichol2021] added before DDIM (static + GIF tabset; DDIM checked with 50 steps on the exact denoiser).
    - Ancestral sampling of the reverse chain from $x_T \sim \mathcal{N}(0, I)$ (@sec-simplesampling); algorithm.
    - Figure (own, `diffusion-reverse.png`): reverse process on the 2D mixture with the **exact** optimal denoiser (computable in closed form for a Gaussian mixture, i.e. what a perfect network would learn): samples at a few steps, from noise to the two modes.
    - Speed: $T \approx 1000$ network evaluations; DDIM [@Song2022]: deterministic sampling with the same trained network in far fewer steps (short).
6. [x] **Diffusion models learn the score** — approved (figure `diffusion-score.png`; continuous time [@Song2021a]).
    - The noise predicted by the network is proportional to the score of the noisy distribution: $s_\theta(x_t, t) = - \epsilon_\theta(x_t, t) / \sqrt{1 - \bar{\alpha}_t}$ (one line from the Gaussian score of @sec-score; Tweedie's formula in words: the denoiser predicts $\mathbb{E}[x_0 | x_t]$).
    - The reverse step is a Langevin step with a learned, noise-dependent score: diffusion = score-based generative models with decreasing noise levels [@Song2020a], which solves the problems of @sec-hmclimits (score learned everywhere, modes correctly weighted).
    - Averaging at the right place: blurry predictions of $x_0$ at high noise are exactly the right score; modes separated progressively.
    - One paragraph: continuous-time view (forward SDE, reverse SDE, deterministic probability-flow ODE behind DDIM) and pointer to flow matching [@Lipman2023], which learns such a deterministic flow directly.
7. [x] **Conditional generation** `{#sec-guidance}` (chapter 2, section 1) — approved (own exact CFG figure on overlapping classes; MRI figure from the notes, source Song2022a; intro figure `conditional-generation.png`: 1D cat/dog modes, unconditional vs. conditional trajectories, Twemoji icons CC-BY 4.0).
    - Conditioning the network on a class or a text embedding, $\epsilon_\theta(x_t, c, t)$.
    - Classifier guidance [@Dhariwal2021] (Bayes' rule on scores) and classifier-free guidance [@Ho2022]: extrapolation between conditional and unconditional scores; sharpening, i.e. mode-seeking at sampling time (diversity vs. quality).
    - Text conditioning with CLIP embeddings [@Radford2021], one paragraph.
8. [x] **Latent diffusion models** `{#sec-latentdiffusion}` (chapter 2, section 2) — approved (figures from the notes: `latent-diffusion.png` [@Rombach2022], `lsgm.png` [@Vahdat2021]; summary as section 3).
    - Diffusion in the latent space of a pretrained autoencoder [@Rombach2022]: cheaper, focuses on semantics; the VAE of Part IV comes back (figure from the notes: LDM architecture).
    - Summary of the chapter (table: model / encoder / trained with / samples with?), and the link back to the whole book.

8b. [x] **Modern text-to-image models** `{#sec-imagegenerators}` (chapter 2, section 3) — approved: Stable Diffusion, GLIDE [@Nichol2022], DALL-E 2 / unCLIP [@Ramesh2022] (figure from the notes), Imagen cascade [@Saharia2022] (figure from <https://imagen.research.google>), current trends (transformers [@Peebles2023], flow matching [@Esser2024]).
9. [x] **Summary** (chapter 2, section 4) — approved: table of the ingredients of the book used by diffusion models, five-bullet recap, transition to optimal transport and flow matching.

### Decisions

- Two chapters: "Denoising diffusion models" (sections 1-6) and "Conditional and latent diffusion" (sections 7-8, plus the summary).
- Own 2D toy figures with the exact optimal denoiser of a Gaussian mixture.
- $\epsilon$ for the noise and $\epsilon_\theta$ for the network, as in the literature; step sizes written $\eta$ in these chapters.
- Diffusion first in Part V; normalizing flows and GANs later; "Energy-based models and score matching" partly absorbed by section 6.

---

## Chapter plan: Optimal transport

File: `src/generative/3-OptimalTransport.qmd`, label `{#sec-ot}`, third chapter of *Deep generative models*, before flow matching.

**Goal of the chapter:** the whole book compares distributions with the KL divergence, which compares densities point by point and ignores the geometry of the space. Optimal transport compares distributions by the cost of moving one onto the other: it gives distances (Wasserstein) that remain meaningful when distributions do not overlap, and transport maps and paths between distributions, which flow matching will learn. Main ideas, few formulas, own figures; no measure theory.

**Notation** (to agree on):

- $p$ and $q$ the source and target distributions (as in @sec-forwardreverse); $x \sim p$, $y \sim q$.
- $c(x, y) = ||x - y||^2$ the cost of moving a unit of mass from $x$ to $y$.
- $T$ a transport map, with $T(x) \sim q$ when $x \sim p$ (written $T_\# p = q$, the push-forward, defined in words).
- $\pi(x, y)$ a coupling (transport plan): a joint distribution whose marginals are $p$ and $q$.
- $W_2(p, q)$ the 2-Wasserstein distance (and $W_1$ for the WGAN pointer).

**References** (all in `Library.bib`): `Villani2009` (reference book), `Peyre2020` (Peyré & Cuturi, *Computational Optimal Transport*), `Cuturi2013` (Sinkhorn), `Arjovsky2017` (WGAN), `Tong2024` (minibatch OT for flow matching), `Liu2022e` (rectified flow), `Bortoli2023` (diffusion Schrödinger bridge).

### Sections

1. [x] **Comparing distributions without overlap** — approved (JS defined in @sec-kl).
    - The KL compares densities at the same points: it is infinite as soon as $q$ misses part of the support of $p$, and it does not depend on **how far** the two distributions are; the Jensen-Shannon divergence is constant ($\log 2$) for disjoint supports → no gradient (the training problem of the original GANs).
    - Data on low-dimensional manifolds (@sec-latentvariables) typically have such non-overlapping supports.
    - Idea: measure the **effort** needed to move the mass of one distribution onto the other (earth mover's distance).
    - *Figure (proposed, own):* `ot-vs-kl.png` — two narrow distributions shifted by $d$; left: the distributions for a few shifts; right: KL / Jensen-Shannon and $W_2$ as functions of $d$ (KL infinite or saturating, $W_2 = d$).
2. [x] **Monge's problem** — approved.
    - Moving a pile of sand into a hole: transport map $T$ moving each $x$ to $T(x)$, pushing $p$ onto $q$, minimizing the average cost $\mathbb{E}_{x \sim p}[c(x, T(x))]$.
    - Limits: a map may not exist (one point mass cannot be split into two), and the constraint $T_\# p = q$ is hard to handle.
3. [x] **Kantorovich's relaxation and the Wasserstein distance** — approved (optimal pairing: 12 crossings vs. 129 for random pairs, so the text says 'cross much less often', not 'do not cross').
    - Couplings: instead of a map, a joint distribution $\pi(x, y)$ saying how much mass goes from $x$ to $y$, with marginals $p$ and $q$; the independent coupling $p(x) \, q(y)$ always exists.
    - Optimal coupling and Wasserstein distance: $W_2(p, q)^2 = \min_\pi \mathbb{E}_{(x, y) \sim \pi}[||x - y||^2]$; a true distance (symmetric, triangle inequality), unlike the KL.
    - Samples: for two sets of $N$ samples with equal weights, the optimal coupling is an **assignment** (a permutation), solvable exactly for small $N$.
    - *Figure (proposed, own):* `ot-coupling.png` — 2D samples of a standard normal and of the two-mode mixture of @fig-score, paired by the independent (random) coupling vs. the optimal assignment: random pairs cross, optimal pairs do not.
4. [x] **Closed forms** — approved (also: the affine map between normals is the reparameterization trick).
    - 1D: the optimal map sorts the samples, $T = F_q^{-1} \circ F_p$ — inverse transform sampling of @sec-simplesampling is optimal transport from the uniform distribution.
    - Normal distributions: $W_2^2 = (\mu_p - \mu_q)^2 + (\sigma_p - \sigma_q)^2$ in 1D (multivariate formula stated), to compare with the closed-form KL of @sec-klnormal.
5. [x] **Interpolating between distributions** — approved (claims 'never at the same place at the same time', not 'non-crossing'; Benamou-Brenier [@Benamou2000]).
    - Displacement interpolation: move each sample along a straight line, $x_t = (1 - t) \, x + t \, T(x)$: the intermediate distributions slide from $p$ to $q$, whereas the mixture $(1 - t) \, p + t \, q$ makes the mass disappear and reappear.
    - Dynamic view (Benamou-Brenier, stated): optimal transport is the flow of minimal kinetic energy, with straight, non-crossing trajectories → the velocity field that flow matching will learn.
    - *Figure (proposed, own):* `ot-interpolation.png` — 1D: mixture interpolation vs. displacement interpolation between two distributions.
6. [x] **Entropic regularization and the Sinkhorn algorithm** — approved.
    - Exact OT is expensive for large sample sets; adding an entropy term $- \epsilon \, H(\pi)$ (@sec-informationtheory) makes the problem smooth, and solvable by alternately rescaling the rows and columns of a matrix (Sinkhorn); $\epsilon \rightarrow 0$: exact OT, $\epsilon \rightarrow \infty$: independent coupling.
    - Pointer: the dynamic version of entropic OT is the Schrödinger bridge, a diffusion process between two arbitrary distributions (link to @sec-diffusion).
7. [x] **Wasserstein GANs** and **Summary** (two sections) — approved: dual of $W_1$, critic; summary table; the subsection "Optimal transport for flow matching" was moved to the flow matching chapter (text kept in its plan entry). — short pointers and summary.
    - Wasserstein GANs [@Arjovsky2017]: $W_1$ through its dual form (1-Lipschitz critic), useful gradients even without overlap.
    - Flow matching with OT couplings [@Tong2024] and rectified flows: straighter paths, fewer sampling steps (next chapter).
    - Summary table: KL vs. Wasserstein (requires, symmetric, sensitive to geometry, defined without overlap, cost).

### Decisions

- All seven sections, including Sinkhorn.
- All three own figures.
- Notation $p$ / $q$ for source / target.

---

## Chapter plan: Flow matching

File: `src/generative/4-FlowMatching.qmd`, label `{#sec-flowmatching}`, fourth chapter of *Deep generative models*, after optimal transport.

**Goal of the chapter:** learn directly a velocity field that transports noise onto data, by a simple regression, without any noising process or likelihood computation. Show that diffusion models are a particular case (curved Gaussian paths), and that optimal transport couplings straighten the paths. As for diffusion, all toy figures use the exact velocity field of a Gaussian mixture (no training).

**Notation** (to agree on):

- **Time runs from noise to data**, as in the flow matching literature: $t = 0$ noise, $t = 1$ data — the opposite of the diffusion chapters ($t = 0$ data, $t = T$ noise); stated explicitly at the beginning.
- $x_0 \sim p_0 = \mathcal{N}(0, I)$ the noise, $x_1 \sim p_1 = p$ the data (consistent with $p$ / $q$ = source / target of @sec-ot, here $p_0$ / $p_1$).
- $p_t$ the distribution at time $t$; $v_t(x)$ (learned: $v_\theta(x, t)$) the velocity field; $u_t(x | x_0, x_1)$ the conditional velocity.

**References** (all in `Library.bib`): `Lipman2023`, `Liu2022e` (rectified flow), `Albergo2023` (stochastic interpolants), `Tong2024` (minibatch OT), `Esser2024` (Stable Diffusion 3), `Song2021a` (probability flow), `Chen2018` (neural ODEs, continuous normalizing flows), `Lipman2024` (*Flow Matching Guide and Code*).

### Sections

1. [x] **Flows** — approved (added schematic `flow.png`: 1D field in the (t, x) plane, exact, functions reused in section 6).
    - A velocity field $v_t(x)$ and the ODE $dx/dt = v_t(x)$ transport samples of $p_0$ into samples of $p_1$ (the dynamic view of @sec-ot, the probability flow of @sec-diffusion).
    - Continuity equation stated in words (mass conservation: the density changes only because mass flows).
    - Continuous normalizing flows (Chen et al.) learn $v_\theta$ by maximum likelihood, which requires simulating the ODE during training: expensive. Flow matching learns it by regression, without simulation.
2. [x] **Conditional paths and the marginal velocity** — approved (static + GIF tabset; the GIF shows both clouds have the same p_t).
    - Pick a noise sample $x_0$ and a data sample $x_1$ (independent coupling) and join them by a straight line, $x_t = (1 - t) \, x_0 + t \, x_1$, with constant velocity $u_t = x_1 - x_0$.
    - The distribution $p_t$ of all these interpolated points goes from noise to data; the velocity field that generates it, the **marginal velocity**, is the average of the conditional velocities of all the lines passing through a point: $v_t(x) = \mathbb{E}[x_1 - x_0 | x_t = x]$ (stated, collapsed justification).
    - The lines cross, so the marginal velocity averages over different directions: the marginal trajectories are **curved**, even though each conditional path is straight, and never cross.
    - *Figure (own, exact):* `flow-matching-paths.png` — 2D mixture of @fig-score: left, straight conditional paths between random pairs (noise, data); right, trajectories of the exact marginal flow from the same noise samples (curved, non-crossing).
3. [x] **The flow matching loss** — approved.
    - The marginal velocity is unknown, but regressing on the conditional velocity gives the same gradients: $\mathcal{L}(\theta) = \mathbb{E}_{t, x_0, x_1}[|| v_\theta(x_t, t) - (x_1 - x_0) ||^2]$ (collapsed proof: the two losses differ by a constant) [@Lipman2023; @Liu2022e].
    - Training algorithm (sample $t$, $x_0$, $x_1$, interpolate, regress): as simple as the diffusion loss, and the same denoising flavor (predicting the direction from noise to data).
4. [x] **Sampling** — approved (static + GIF over the number of steps).
    - Integrate the ODE from $x_0 \sim \mathcal{N}(0, I)$ with a few Euler steps $x_{t + h} = x_t + h \, v_\theta(x_t, t)$; error due to the curvature of the trajectories.
    - *Figure (own, exact):* `flow-matching-sampling.png` — samples generated with 1, 2, 5 and 50 Euler steps on the exact field: few steps fail where trajectories are curved.
5. [x] **Diffusion models are flow matching models** — approved.
    - Gaussian paths $x_t = \alpha_t \, x_1 + \sigma_t \, x_0$: the straight line is $\alpha_t = t$, $\sigma_t = 1 - t$; diffusion (variance-preserving) corresponds to $\alpha_t^2 + \sigma_t^2 = 1$ (curved schedule).
    - For Gaussian paths, velocity, noise prediction and score are linear functions of each other (one line, Tweedie): a diffusion model's probability flow is a flow matching model with a particular path; flow matching mostly changes the path and the regression target.
    - Why straight paths: constant conditional velocity, simpler regression target, fewer sampling steps.
6. [x] **Straightening the flow: optimal transport couplings and rectified flows** — approved (1D static figure only, the GIF was dropped; 2x2 grid: minibatch conditional paths on top (random pairs cross, sorted pairs do not), resulting flows below; 1D end points of the marginal flow = OT map, checked numerically). — draft text moved from the OT chapter (above, in the Part V entry): independent vs. minibatch OT couplings [@Tong2024] (reuse @fig-otcoupling), rectified flows / reflow [@Liu2022e].
    - *Figure (own, exact):* `flow-matching-1d.png` — 1D: trajectories of the marginal flow with the independent coupling (curved) vs. the optimal transport (straight lines from quantile matching, @sec-ot), same endpoints (in 1D the marginal flow already realizes the monotone OT map, but along curved paths).
7. [x] **Summary** — approved.
    - Table: diffusion vs. flow matching (path, training target, sampler, steps, likelihood).
    - Modern use: Stable Diffusion 3 and other large models [@Esser2024]; end of Part V for now (normalizing flows, GANs as possible later chapters).

### Decisions

- Flow matching convention: $t = 0$ noise, $t = 1$ data, with an explicit warning.
- All three own figures, each as a static figure + GIF in a tabset (as in @sec-diffusion).
- Keep the 1D figure in section 6, in addition to reusing @fig-otcoupling.

---

## Chapter plan: Introduction

File: `src/introduction.qmd`, label `{#sec-introduction}`, placed in `_quarto.yml` after the preface and before Part I. The preface (`index.qmd`) keeps only the practical information.

**Goal:** motivate the whole book in a few pages, without equations beyond the minimum: deterministic deep learning predicts a single answer, while most interesting problems need a distribution of answers; show where distributions appear in modern applications; state the common thread; give a map of the book.

**References:** in `Library.bib`: `Brown2020` (GPT-3), `Ouyang2022` (RLHF), `Ramesh2022`, `Esser2024` (image generation), `Song2022a` (MRI), `Lam2023` (GraphCast), `Bi2023` (Pangu-Weather), `Price2025` (GenCast, Nature; `Price2024` arXiv), `Rozet2023` (score-based data assimilation), `Huang2024b` (DiffDA), `Andrae2025` (DAISI), `Martin2025` (generative ocean data assimilation), `Chi2023` (diffusion policy). `Abramson2024` (AlphaFold 3).

### Sections

1. [ ] **One answer, or many?**
    - A network trained with the mean square error predicts the **average** of the possible targets (the conditional mean, as for the denoisers of @sec-diffusion): fine when there is a single plausible answer, wrong when there are several. The average of two plausible futures, of a cat and a dog, or of two branches of an inverse problem, is not plausible itself.
    - A probabilistic model predicts a **distribution** of answers instead: it can be sampled, it tells how uncertain it is, and it can be combined with other information (Bayes' rule).
    - Figure `intro-average.png` (`src/img/intro-average.py`): inverse problem $y = \pm \sqrt{x}$ + noise; left, the MSE prediction (conditional mean, $y = 0$) passes between the branches, where there is no data; right, the exact $p(y | x)$ and its samples on both branches.
2. [ ] **Distributions everywhere** — one short paragraph per application, each pointing to the chapters where the tools are introduced:
    - **Generating text and images:** language models are softmax classifiers over tokens, sampled one token at a time [@Brown2020] (@sec-softmax); image generators sample diffusion or flow matching models [@Ramesh2022; @Esser2024] (@sec-diffusion, @sec-flowmatching).
    - **Inverse problems:** reconstructing an image from incomplete or noisy measurements (MRI, inpainting) has many solutions; a generative prior selects the plausible ones [@Song2022a] (@sec-inverseproblems).
    - **Weather forecasting and data assimilation:** deterministic forecasts become blurry at long lead times, as they average over the possible futures [@Lam2023; @Bi2023]; probabilistic forecasts sample an ensemble of possible weathers [@Price2025]. Data assimilation estimates the current state of the atmosphere or of the ocean from sparse, noisy observations: a posterior distribution, now increasingly computed with diffusion models [@Rozet2023; @Huang2024b; @Andrae2025; @Martin2025].
    - **Science:** protein structure prediction with a diffusion module (AlphaFold 3), posterior distributions over the parameters of scientific models (@sec-bayesian).
    - **Decisions:** stochastic policies in reinforcement learning, fine-tuning of language models from human feedback [@Ouyang2022] (@sec-gradientestimators), robot policies generated by diffusion models [@Chi2023].
    - *Figures (optional):* one or two borrowed illustrations, e.g. a GenCast ensemble forecast, or a data assimilation example from your own work.
3. [ ] **The common thread** — moved and expanded from the preface: almost every model fits a parameterized distribution $p_\theta$ to an unknown distribution $p$ known through samples, by minimizing a divergence (the KL, @sec-kl); the expectations and integrals this requires are usually intractable, and there are two ways out: **sampling** (Part III) and **optimization** (Part IV); diffusion models and flow matching (Part V) combine both.
    - Figure `common-thread.png` (`src/img/common-thread.py`, redrawn after the author's IntroAI lecture figure `statisticalml.png`): bimodal data distribution (dogs / cats, Twemoji icons instead of photos), training samples, model $p_\theta$, divergence between both.
4. [ ] **A map of the book** — the five parts in a few sentences, with a small `{mermaid}` diagram of their dependencies, and suggested reading paths (e.g. a short path to diffusion models).

### Open questions

- Section 1: own toy figure (proposed), or reuse the inverse-problem figure of @sec-mdn?
- Section 2: borrowed figures for some applications (which ones? a figure from your data assimilation work?), or text only?

