# A Synthetic Testbed for Separating Stimulus-Invariant and Nuisance-Driven Structure in Retinal Population Codes

**Aarush Kandukoori** (Carnegie Mellon University) — Lead Student Author  
**Aditya Kandukoori** (Poolesville High School) — Co-Author

> **Status: early-stage research proposal.** No experimental results
> are claimed. We are actively seeking expert feedback.

## Abstract

Recent work on the retina reports that stimulus-invariant aspects of the population code help discriminate natural scenes. We outline an early-stage, feedback-seeking framework for testing how such invariant structure can be isolated and measured. We model a retinal ganglion cell population as a noisy, nonlinear encoder whose responses depend on scene identity and on nuisance variables such as luminance, contrast and gaze-like shifts. We then ask which linear read-out subspaces maximize scene discriminability while suppressing nuisance-driven variance. The central object is a generalized Rayleigh quotient that trades between-scene signal against noise and nuisance covariance, and it has a closed-form solution via a generalized eigenproblem. We plan to study it on clearly labeled synthetic populations with known ground truth, and to compare it against standard baselines such as PCA, Fisher discriminant analysis and a contrastive encoder. No experiments have been run and no results are reported. We are undergraduate and high school researchers and we seek expert feedback on whether the setup is biologically sensible and which real-data checks would be most informative. AI tools (Claude) were used for literature synthesis, ideation, and draft preparation, an AI review checked the draft before release, and the authors take responsibility for all content.

## Introduction & Motivation

**Status.** This is an early-stage proposal from an undergraduate and a high school student. It is a framework and plan, not a results paper. We are seeking feedback from retinal-coding experts.

**Question.** Retinal ganglion cell (RGC) populations respond to natural scenes with activity that depends both on *what the scene is* and on *how it is presented* (mean luminance, contrast, small shifts of the image on the retina). Hoshal et al. (2024) report that stimulus-invariant aspects of the retinal code contribute to discriminating natural scenes. We want to make the following question precise and testable in simulation: *given a population with known structure, which read-out directions carry scene identity while being insensitive to nuisance variables, and how much discriminability do they retain?*

**Why a synthetic testbed.** In real recordings the true invariant structure is unknown, so any method for extracting it cannot be validated against ground truth. A simulator with planted structure lets us check whether a method recovers it, and how that depends on population size, noise correlations and nuisance strength. This is a methodological stepping stone and does not substitute for real data.

**Contributions we hope to make (to our knowledge, a modest framing).** (1) A clear formal statement of invariant-discriminability as a generalized eigenproblem. (2) An open synthetic benchmark with planted ground truth. (3) A plan for comparison with standard baselines. We are not claiming any new biological finding.

## Related Work

**Population codes and correlations.** Schneidman et al. (2006) showed that weak pairwise correlations can imply strongly correlated network states in retinal populations, and Tkačik et al. (2014) studied collective behavior in a large sensory population with maximum-entropy models. Pillow et al. (2008) modeled spatio-temporal correlations in a complete RGC population with coupled generalized linear models.

**Predictive and efficient coding.** Palmer et al. (2015) analyzed predictive information in a sensory population. The information bottleneck (Tishby et al., 2000) formalizes compressing a response while preserving information about a relevant variable, which is close in spirit to keeping scene identity while discarding nuisance.

**Deep models of retinal responses.** McIntosh et al. (2016) fit convolutional networks to RGC responses to natural scenes, which suggests the kind of nonlinear encoders a more realistic simulator could use later.

**Invariance and discriminability in the retina.** Hoshal et al. (2024) is the direct motivation for this proposal.

**Contrastive representation learning.** Oord et al. (2018) introduced the InfoNCE contrastive objective via contrastive predictive coding, and Chen et al. (2020) (SimCLR) show that learned invariances are shaped by the choice of data augmentations. We treat a contrastive encoder as a baseline that learns invariance to nuisance transformations without an explicit linear structure.

## Proposed Framework

**Setup.** A population of $N$ cells responds to scene $k \in \{1,\dots,K\}$ under nuisance setting $\theta \in \Theta$. Responses are

$$r = f(k,\theta) + \varepsilon, \qquad \varepsilon \sim \mathcal{N}(0,\Sigma),$$

where $f(k,\theta)\in\mathbb{R}^N$ is the mean response and $\Sigma\in\mathbb{R}^{N\times N}$ is a (possibly correlated) noise covariance. For simplicity we start with a Gaussian model with fixed $\Sigma$. Real spike counts are not Gaussian, which we list as a limitation.

**Discriminability.** For two scenes $a,b$ at the same $\theta$ and a linear read-out $w\in\mathbb{R}^N$, the projected responses $w^\top r$ are Gaussian with means $w^\top f(a,\theta)$, $w^\top f(b,\theta)$ and common variance $w^\top \Sigma w$. The sensitivity index is

$$d'^2(w) = \frac{\big(w^\top \Delta f\big)^2}{w^\top \Sigma w}, \quad \Delta f = f(a,\theta)-f(b,\theta).$$

Maximizing over $w$ gives $w^\star \propto \Sigma^{-1}\Delta f$ and $d'^2_{\max} = \Delta f^\top \Sigma^{-1}\Delta f$ (Cauchy-Schwarz in the $\Sigma$ inner product).

**Scene and nuisance covariances.** Let $\bar f_k = \mathbb{E}_\theta f(k,\theta)$ and $\bar f = \frac1K\sum_k \bar f_k$. Define

$$S_B = \frac1K\sum_{k}(\bar f_k-\bar f)(\bar f_k-\bar f)^\top, \qquad S_N = \frac1K\sum_k \mathbb{E}_\theta\big[(f(k,\theta)-\bar f_k)(f(k,\theta)-\bar f_k)^\top\big].$$

$S_B$ is the between-scene covariance of mean responses (scene signal), and $S_N$ is the average within-scene covariance of the mean response caused by nuisance variation.

**Objective.** For a penalty weight $\lambda\ge 0$, choose $w$ to maximize the generalized Rayleigh quotient

$$J_\lambda(w) = \frac{w^\top S_B\, w}{w^\top(\Sigma + \lambda S_N)\,w}.$$

The numerator is the variance of the projection due to scene identity. The denominator is the variance due to trial noise plus $\lambda$ times the variance due to nuisance. Setting the gradient to zero shows that stationary points satisfy $S_B w = \mu (\Sigma+\lambda S_N) w$, so the maximizers are the top generalized eigenvectors of $(S_B, \Sigma+\lambda S_N)$, and the maximal value of $J_\lambda$ equals the largest generalized eigenvalue. This is a standard result (the same structure as Fisher's linear discriminant), and it is checkable numerically against a generic eigen-solver. The matrix $\Sigma+\lambda S_N$ is positive definite whenever $\Sigma$ is.

**Special cases.** With $\lambda=0$ the objective is Fisher-style discriminability with noise only. As $\lambda\to\infty$ the solution is pushed toward the directions in which $S_N$ is small, i.e. the nuisance-invariant subspace. If $S_N=0$ the nuisance is absent and nothing is lost by $\lambda>0$. The trade-off curve $\lambda \mapsto (\text{scene variance retained}, \text{nuisance variance leaked})$ is the main quantity we want to study.

**Planted ground truth.** In the simulator we construct $f(k,\theta) = A\,u_k + B\,g(\theta) + \text{(optional nonlinearity)}$, where $u_k\in\mathbb{R}^p$ are scene codes, $g(\theta)\in\mathbb{R}^q$ are nuisance features, and $A, B \in\mathbb{R}^{N\times p}, \mathbb{R}^{N\times q}$ have a controllable subspace overlap, measured by principal angles between $\mathrm{col}(A)$ and $\mathrm{col}(B)$. We draw $g(\theta)$ independently of the scene index $k$. In the linear case $S_B = A\,\mathrm{Cov}(u)A^\top$ and $S_N = B\,\mathrm{Cov}_\theta(g)B^\top$ are then known exactly, so the recovered $w$ can be compared to the analytic solution. We expect, as a hypothesis and not a result, that recovery degrades as principal angles shrink.

**Nonlinear extension.** A saturating nonlinearity $r = \phi(A u_k + B g(\theta)) + \varepsilon$ breaks the linear analysis. We plan to treat this as a second stage and compare against a contrastive encoder trained with nuisance transformations as the augmentations.

## Methodology

**Synthetic data.** All data in the proposed repository are **clearly labeled synthetic/mock data** generated by a simulator we write. No real recordings, no animal data and no data from Prof. Berry's group are used or implied. Synthetic populations are not claimed to match any real retina.

**Simulator design.** (1) Draw $K$ scene codes $u_k$ from a Gaussian. (2) Draw nuisance features $g(\theta)$ from a low-dimensional distribution (luminance, contrast, shift). (3) Build $A,B$ with a user-set subspace overlap. (4) Add noise from a covariance $\Sigma$ with tunable correlation strength (e.g., a low-rank plus diagonal form). (5) Optionally apply a rectifying nonlinearity and Poisson-like noise for a second regime.

**Estimation.** From $T$ simulated trials per (scene, nuisance) condition, estimate $S_B$, $S_N$, $\Sigma$ with shrinkage (e.g., Ledoit-Wolf style) and solve the generalized eigenproblem. We will study sample-size dependence, since $N$ may be comparable to $T$.

**Protocol knobs.** Population size $N$, number of scenes $K$, noise correlation strength, nuisance strength, subspace overlap, $\lambda$.

**AI Assistance Disclosure.** AI tools (Claude) were used for literature synthesis, ideation, and draft preparation. An AI review checked the draft before release. The authors take responsibility for all content, including any errors.

## Planned Evaluation & Baselines

*This section describes a protocol only. No experiments have been run and no results are claimed.*

**Metrics.** (a) Held-out scene classification accuracy and $d'$ along the learned read-outs, evaluated under nuisance settings not used for fitting. (b) Nuisance leakage: fraction of projected variance explained by $\theta$. (c) In the linear regime, subspace recovery error against the planted ground truth (largest principal angle). (d) The $\lambda$ trade-off curve between (a) and (b).

**Baselines (to be implemented).** PCA (top-variance directions); Fisher linear discriminant on pooled data (the $\lambda=0$ case with no nuisance handling); a variant that regresses out nuisance features before discrimination; and a small contrastive encoder (InfoNCE-style, following Oord et al., 2018 and Chen et al., 2020) trained with nuisance transformations as positive pairs, optimized with Adam (Kingma & Ba, 2015).

**Controls.** Shuffled scene labels as a chance level; $S_N=0$ sanity check where all methods should agree; analytic check of the eigen-solution against a generic solver.

**Statistics.** Multiple random seeds per configuration, with confidence intervals reported across seeds. Hyperparameters ($\lambda$, shrinkage) chosen by validation conditions separate from test conditions.

## Limitations & Open Questions

- **Synthetic only.** Conclusions about simulated populations do not transfer automatically to real retinas. We do not claim they do.
- **Gaussian, fixed-covariance noise** is a strong simplification; real spike counts are discrete and noise covariance may depend on the stimulus.
- **Linear read-outs** may miss nonlinear invariant structure, which the contrastive baseline only partly probes.
- **Definition of 'invariant'.** Our $S_N$-based definition is one choice. It may differ from how Hoshal et al. (2024) define stimulus-invariant aspects of the code, and we would value correction on that.
- **Nuisance choice.** Are luminance, contrast and shift the right nuisance variables for natural-scene experiments? What would be a realistic range?
- **Open question for the professor.** What minimal real-data check (even a public or already-published dataset) would make this testbed credible to the field, and are there known failure modes of linear-discriminant analyses of RGC populations we should build in?

## References

- [hoshal2024] Hoshal et al., *Stimulus-invariant aspects of the retinal code drive discriminability of natural scenes*, Proceedings of the National Academy of Sciences 2024. https://doi.org/10.1073/pnas.2313676121
- [schneidman2006] Schneidman et al., *Weak pairwise correlations imply strongly correlated network states in a neural population*, Nature 2006. https://arxiv.org/abs/q-bio/0512013
- [tkacik2014] Tkačik et al., *Searching for collective behavior in a network of real neurons*, PLoS Computational Biology 2014. https://arxiv.org/abs/1306.3061
- [pillow2008] Pillow et al., *Spatio-temporal correlations and visual signalling in a complete neuronal population*, Nature 2008. https://doi.org/10.1038/nature07140
- [palmer2015] Palmer et al., *Predictive information in a sensory population*, Proceedings of the National Academy of Sciences 2015. https://arxiv.org/abs/1307.0225
- [tishby2000] Tishby et al., *The information bottleneck method*, arXiv preprint (Allerton Conference) 2000. https://arxiv.org/abs/physics/0004057
- [mcintosh2016] McIntosh et al., *Deep Learning Models of the Retinal Response to Natural Scenes*, Advances in Neural Information Processing Systems (NeurIPS) 2016. https://arxiv.org/abs/1702.01825
- [oord2018] Oord et al., *Representation Learning with Contrastive Predictive Coding*, arXiv preprint 2018. https://arxiv.org/abs/1807.03748
- [chen2020] Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations*, International Conference on Machine Learning (ICML) 2020. https://arxiv.org/abs/2002.05709
- [kingma2015] Kingma et al., *Adam: A Method for Stochastic Optimization*, International Conference on Learning Representations (ICLR) 2015. https://arxiv.org/abs/1412.6980
