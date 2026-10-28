# Book plan

*Probabilistic inference for machine learning* — a Quarto book (same format as `../DeepReinforcementLearning`) explaining the probabilistic concepts behind modern machine learning. The common thread: virtually every learning algorithm fits a parameterized distribution $p_\theta$ to data sampled from an unknown distribution $p$, i.e. minimizes a divergence between them, and most of the difficulty comes from intractable expectations (integrals), which are handled either by **sampling** or by **optimization** (variational inference).

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

Goal of the book, the common thread ($p_\theta$ vs. $p$, KL minimization), license.

### Part I — Basics (`src/basics/`)

- [x] **1. Information theory** `{#sec-informationtheory}` — self-information, entropy (discrete / differential, Bernoulli example), joint and conditional entropy, mutual information, cross-entropy `{#sec-crossentropy}`, KL divergence `{#sec-kl}` (relation to cross-entropy, asymmetry). Rewritten intro (Bell Labs, information as surprise, compression, predicting = compressing) [@Nyquist1924; @Hartley1928; @Shannon1948]; collapsed note on differential entropy; $p$, $q$ notation for cross-entropy and KL.
- [x] **2. Maximum likelihood estimation** `{#sec-mle}` — likelihood vs. joint probability, i.i.d. assumption, log-likelihood and NLL; worked derivation of MLE for the univariate normal (empirical mean and variance).
- [x] **3. Bayesian estimation** `{#sec-bayesian}` — central limit theorem (frequentist view); MAP `{#sec-map}`: Bayes' rule on the parameters (prior, likelihood, evidence, posterior), MLE as MAP with a uniform prior, MAP for the mean of a normal with a normal prior (weighted average of $\bar{x}$ and $\mu_0$, limit cases, conjugate posterior variance, figure `map-posterior.png`), MAP as L2 / L1 regularization, limitations (point estimate, posterior predictive, intractable evidence → VI).
- Notation unified: cross-entropy and KL use distributions $p$, $q$ ($H(p, q)$, $\text{KL}(p || q)$, normals $\mathcal{N}(x; \mu_p, \sigma_p)$) in `basics/1-InformationTheory.qmd` and `statisticalml/2-KL.qmd`; $X$, $Y$ kept for entropy / joint / conditional entropy / mutual information; `cross-entropy.svg` replaced by `cross-entropy.png` (`src/img/cross-entropy.py`).
- [x] ★ **Forward vs. reverse KL** `{#sec-forwardreverse}` — new last section of `statisticalml/2-KL.qmd`, approved (figure `forward-reverse-kl.png`, modes at ±2 as the reverse KL also covers both modes at ±1.5).

### Part II — Statistical machine learning (`src/statisticalml/`)

- [x] **1. Learning paradigms** `{#sec-paradigms}` — discriminative (classification, regression, MDN), generative (density estimation vs. generation), autoregressive / self-supervised (LLMs, PixelCNN, masked learning); all of them fit $p_\theta$ to $p$ by minimizing the KL.
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

### Part V — Deep generative models — planned

- [ ] Normalizing flows — change of variables, exact likelihood.
- [ ] Energy-based models and score matching.
- [ ] ★ Diffusion models — port `~/Documents/Obsidian/Notes/Machine Learning/Diffusion/` (~850 lines; its images `langevin.gif`, `kerneldensityestimation.png`, `manifold.png` are in the Obsidian `Inference/img/` folder); hierarchical-VAE derivation of @Luo2022a.
    - Planned paragraph: **diffusion and flow matching in the forward / reverse KL dichotomy** (link to @sec-forwardreverse).
        - Rule of thumb: an expectation over the **data** is mass-covering (MLE, VAE, diffusion, flow matching); over the **model's own samples** it is mode-seeking (VI, mode collapse of GAN generators).
        - Diffusion is trained by maximizing an ELBO on $\log p_\theta(x)$ (hierarchical VAE, @Luo2022a); the encoder is the fixed noising process, and each term $\text{KL}(q(x_{t-1}|x_t, x_0) || p_\theta(x_{t-1}|x_t))$ has the data-driven distribution first → forward KL / maximum likelihood. With the right weighting, score matching also bounds $\text{KL}(p_\text{data} || p_\theta)$ [@Song2021].
        - Not blurry because the model family is flexible: each reverse step is nearly Gaussian for small steps, so a Gaussian per step is almost exact, and the chain of many steps gives an arbitrarily multimodal marginal. The forward KL only averages between modes when the family cannot represent the target (single normal of `forward-reverse-kl.png`).
        - The averaging happens at the right place: the MSE denoiser predicts $\mathbb{E}[x_0 | x_t]$, blurry at high noise but exactly the score of the noisy distribution (Tweedie's formula); the modes are separated progressively as the noise decreases (the noise-levels argument of @sec-hmclimits, seen from the training side).
        - Flow matching: same picture — MSE regression of conditional velocities over data samples (covering), learned field $\mathbb{E}[u_t(x | x_0) | x_t]$; not literally a KL (closer to transport distances).
        - Mode-seeking comes back at sampling time: classifier-free guidance samples roughly from $p(x | c) \, p(c | x)^w$, sharpening towards the modes (diversity traded for quality).
- [ ] GANs as divergence minimization (f-divergences).

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

- [~] GitHub repository `git@github.com:vitay/probabilisticinference.git`; CI `.github/workflows/quarto-publish.yml` renders and publishes to the `gh-pages` branch on every push to `main` (as in the Deep RL book).

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

