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

- [x] **1. Information theory** `{#sec-informationtheory}` — self-information, entropy (discrete / differential, Bernoulli example), joint and conditional entropy, mutual information, cross-entropy `{#sec-crossentropy}`, KL divergence `{#sec-kl}` (relation to cross-entropy, asymmetry).
- [x] **2. Maximum likelihood estimation** `{#sec-mle}` — likelihood vs. joint probability, i.i.d. assumption, log-likelihood and NLL; worked derivation of MLE for the univariate normal (empirical mean and variance).
- [x] **3. Bayesian estimation** `{#sec-bayesian}` — central limit theorem (frequentist view); MAP `{#sec-map}`: Bayes' rule on the parameters (prior, likelihood, evidence, posterior), MLE as MAP with a uniform prior, MAP for the mean of a normal with a normal prior (weighted average of $\bar{x}$ and $\mu_0$, limit cases, conjugate posterior variance, figure `map-posterior.png`), MAP as L2 / L1 regularization, limitations (point estimate, posterior predictive, intractable evidence → VI).
- [ ] ★ **Forward vs. reverse KL** (section of `statisticalml/2-KL.qmd` or a short chapter here): mode-covering vs. mode-seeking; MLE minimizes the forward KL, VI the reverse one.

### Part II — Statistical machine learning (`src/statisticalml/`)

- [x] **1. Learning paradigms** `{#sec-paradigms}` — discriminative (classification, regression, MDN), generative (density estimation vs. generation), autoregressive / self-supervised (LLMs, PixelCNN, masked learning); all of them fit $p_\theta$ to $p$ by minimizing the KL.
- [x] **2. KL divergence in machine learning** `{#sec-klml}` — closed form between two Gaussians `{#sec-klnormal}` (full derivation, special case $\mathcal{N}(0,1)$); equivalence of KL minimization, cross-entropy minimization and MLE.
- [x] **3. Softmax classifier** `{#sec-softmax}` — one-hot targets as a multinomial distribution, softmax, cross-entropy loss $-\mathbf{t} \cdot \log \mathbf{y}$, gradient $-(\mathbf{t} - \mathbf{y})$ w.r.t. the logits (collapsed proof).
- [x] **4. Mixture density networks** `{#sec-mdn}` — mean-variance estimation (NLL of a Gaussian, regression as a special case), multimodal targets (inverse problems, half-moons), mixture of Gaussians output, NLL loss, two-step sampling.

### Part III — Sampling methods (`src/sampling/`) — new

- [x] ★ **1. Monte Carlo and importance sampling** `{#sec-montecarlo}` — why sampling (expectations everywhere, curse of dimensionality `{#sec-curse}`, $\pi$ example), the Monte Carlo estimator `{#sec-mcestimator}` (unbiased, $\sigma_f / \sqrt{N}$, KL estimate), sampling from simple distributions `{#sec-simplesampling}` (inverse transform, normal, categorical, ancestral), rejection sampling `{#sec-rejection}`, importance sampling `{#sec-importance}` (self-normalized, ESS, PPO), limits `{#sec-mclimits}` (ESS collapse, idea of MCMC, summary table). Detailed plan below.
- [~] ★ **2. Markov chain Monte Carlo** `{#sec-mcmc}` — outline below, waiting for approval.
- [ ] **3. Hamiltonian Monte Carlo and Langevin dynamics** — gradient-based samplers; bridge to score-based diffusion models.

### Part IV — Variational inference (`src/variationalinference/`)

- [x] **1. Latent variable models** `{#sec-latentvariables}` — generative modeling as KL minimization, curse of dimensionality (back-reference to `sec-curse`), manifolds, latent variables and the decoder $p_\theta(x|z)$, marginalization.
- [x] **2. Variational inference** `{#sec-variationalinference}` — intractable posterior, approximate posterior $q_\phi(z|x)$, derivation of the ELBO from $\text{KL}(q_\phi(z|x) || p_\theta(z|x))$, reconstruction and regularization losses.
- [x] **3. Why maximizing the ELBO works** `{#sec-elbo}` — ELBO from Jensen's inequality, full decomposition $\log p_\theta(x) = \text{ELBO} + \text{KL}$, why maximizing the ELBO w.r.t. $\phi$ minimizes the gap [@Luo2022a].
- [x] **4. Variational autoencoders** `{#sec-vae}` — $\mathcal{N}(0, 1)$ prior, closed-form regularization loss, log-variance parameterization, reparameterization trick (figure only).
- [ ] **Gradient estimators** — reparameterization trick in detail vs. score-function (REINFORCE) estimator; link to policy gradients in the DeepRL book.
- [ ] **Tighter bounds** — IWAE (importance-weighted ELBO, reuses Part III), β-VAE.

### Part V — Deep generative models — planned

- [ ] Normalizing flows — change of variables, exact likelihood.
- [ ] Energy-based models and score matching.
- [ ] ★ Diffusion models — port `~/Documents/Obsidian/Notes/Machine Learning/Diffusion/` (~850 lines; its images `langevin.gif`, `kerneldensityestimation.png`, `manifold.png` are in the Obsidian `Inference/img/` folder); hierarchical-VAE derivation of @Luo2022a.
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

- [ ] GitHub repository and publishing (e.g. GitHub Pages from `docs/`).

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
5. [ ] **Gibbs sampling**
    - In several dimensions, update one coordinate at a time by sampling from its conditional $p(x_k | x_{-k})$ (all other coordinates fixed).
    - Special case of Metropolis-Hastings where every proposal is accepted.
    - Useful when the conditionals are simple even if the joint is not (conjugate models, latent variable models: sample $z | x$, then parameters).
    - Limitation: strongly correlated coordinates → tiny axis-aligned steps, slow mixing.
    - Worked example: correlated bivariate normal, whose conditionals are 1D normal distributions (formula given).
    - *Figure (proposed):* `gibbs-sampling.png` — Gibbs path (axis-aligned steps) on a correlated 2D Gaussian, compared to a random-walk Metropolis path with the same number of steps.
6. [ ] **Example: sampling a posterior** (optional — to decide)
    - Back to @sec-map: posterior of the mean **and** standard deviation of a normal distribution from data (no simple closed form for the joint posterior with a non-conjugate prior); MH on $(\mu, \log \sigma)$; posterior histograms and the posterior predictive as a Monte Carlo average over the chain.
    - Closes the loop with the limitations of MAP (point estimate vs. full posterior).
    - *Figure (proposed):* `mcmc-posterior.png` — samples of $(\mu, \sigma)$ in 2D with marginal histograms.
7. [ ] **Limits: towards gradient-based samplers**
    - Random-walk proposals ignore the shape of the target: in high dimensions the step size must shrink and exploration becomes very slow.
    - Using the gradient $\nabla_x \log \tilde{p}(x)$ to propose moves towards high probability (Langevin dynamics, Hamiltonian Monte Carlo) → next chapter, and the bridge to score-based diffusion models.

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
