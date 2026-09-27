# Port plan

Source: `~/Documents/Obsidian/Notes/Machine Learning/Inference/*.md` → `src/*.qmd`, same Quarto book format as `../DeepReinforcementLearning`.

Legend: `[ ]` todo · `[x]` done · `[~]` partially done

## Book structure

```
Preface                         index.qmd
Basics                          src/basics/
  Information theory              1-InformationTheory.qmd
  Maximum likelihood estimation   2-MLE.qmd
  Bayesian estimation             3-Bayesian.qmd
Statistical machine learning    src/statisticalml/
  Learning paradigms              1-Paradigms.qmd
  KL divergence in ML             2-KL.qmd
  Softmax classifier              3-Softmax.qmd
  Mixture density networks        4-MDN.qmd
Variational inference           src/5-VariationalInference.qmd
References                      src/references.qmd
```

All images live in `src/img/` (referenced as `../img/` from the part subfolders, `img/` from `src/`).

## Chapters

- [x] `1. Information theory.md` → `src/basics/1-InformationTheory.qmd`
- [x] `2. Maximum Likelihood Estimation (MLE).md` → `src/basics/2-MLE.qmd`
- [x] `3. Statistical machine learning.md` → split into `src/statisticalml/` (one chapter per former `##` section, headings promoted one level; "Three paradigms" renamed "Learning paradigms")
- [x] `4. Bayesian estimation.md` → `src/basics/3-Bayesian.qmd` — the source was a stub; the MAP section was written from scratch. Placed in Basics as it continues MLE (CLT → MAP).
- [x] `5. Variational inference - ELBO.md` → `src/5-VariationalInference.qmd` — standalone for now; could become a "Generative models" part (ELBO, VAE, …) as content grows.

## Conversion conventions

- Obsidian callouts `> [!TIP]` and `---`-delimited boxes → `:::{.callout-* icon="false"}`; proofs are collapsed.
- `mehrmaid` diagrams → `{mermaid}` with HTML labels (mermaid does not render `$...$`).
- Blank lines removed inside `$$` blocks (pandoc breaks display math on them); `align` + `\tag*` → `aligned` with `\text{}` comments.
- Section/figure labels added (`sec-mle`, `sec-kl`, `sec-klnormal`, `sec-mdn`, `sec-vae`, …) and bare "see below"/"explanation 2" replaced by cross-references.
- Bibliography: `Library.bib` (copied by `just`), because `Bayes1763`, `Nix1994`, `Sluijterman2024` are only there. The arXiv link 2208.11970 became `@Luo2022a`; `@Kingma2013` added for VAEs.

## Images (only the ones referenced were copied, renamed)

| Obsidian | Book |
|---|---|
| `img/selfinformation.png` | `self-information.png` |
| `img/entropy-binomial.png` | `entropy-bernoulli.png` |
| `img/crossentropy.svg` | `cross-entropy.svg` |
| `img/MLE2.png` | `mle-normal-fits.png` |
| `img/IllustrationCentralTheorem.png` | `central-limit-theorem.png` |
| `img/softmax-transformation.png` | `softmax-transformation.png` |
| `img/mdn-image.jpg` | `mve-network.jpg` |
| `img/multimodal.png` | `multimodal-distribution.png` |
| `img/mdn.webp` | `mdn-architecture.webp` |
| `attachments/Pasted image 20250206173409.png` | `mdn-inverse-regression.png` |
| `attachments/Pasted image 20250206173608.png` | `mdn-halfmoons-pdf.png` |
| `attachments/Pasted image 20250206213444.png` | `mdn-posterior-density.png` |
| `attachments/Pasted image 20250206213647.png` | `mdn-halfmoons-sampling.png` |
| `img/cursedimensionality.png` | `curse-of-dimensionality.png` |
| `img/latentvariable.png` | `latent-variable-model.png` |
| `img/VAE-graphical-model.png` | `vae-graphical-model.png` |
| `img/vae-reparameterization2.png` | `vae-reparameterization.png` |

Unused in the notes (not copied): `crossentropy-animation.gif`, `kerneldensityestimation.png`, `langevin.gif`, `manifold.png`, `multimodal.webp`, `softmax.svg`, `Pasted image 20250206140910/141227/142244/142344.png`.

## Fixes made during the port (revert any you disagree with)

- [x] Information theory: `H(X|Y) = H(X,Y) - H(Y)` was listed under "if independent"; it holds in general, moved out.
- [x] MLE: NLL line said `argmin NLL = argmax Σ -log p`; now `argmin`. Broken bold `**...++` fixed.
- [x] Softmax / MDN: softmax proof differentiated `log(y_i)` where it meant `y_i`; `t_j/y_i` → `t_j/y_j`; MDN list numbered 4./5. → 1./2.; MDN loss now evaluates each component at `t_i` with `x_i`.
- [x] Variational inference: "latent representation $x$" → $z$; "should be too far from the prior" → "should **not** be"; VAE reconstruction term `-log p(x)` → `-log p(x|z)`; `KL(q, p)` → `KL(q || p)`; Jensen stated for concave `f` (the log); the first explanation's intractability equation (which integrated over $x$ and was trivially $p(z)$) replaced by the evidence integral over $z$.
- [x] Typos: Bernouilli, substractions, familiy, algoroithm, outputing, Euclidian, "less from", etc.

## Todo

- [x] Write the MAP section (with `src/img/map-posterior.py` → `map-posterior.png`).
- [ ] Decide whether the GitHub repo / publishing (gh-pages from `docs/`) should be set up.
