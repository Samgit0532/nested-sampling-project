# Monte Carlo Project: Nested Sampling

## Overview

This project explores **Nested Sampling (NS)**, a Monte Carlo method used to estimate the **Bayesian marginal likelihood (evidence)**:

[
Z = \int L(\theta),\pi(\theta),d\theta
]

Nested sampling transforms this high-dimensional integral into a one-dimensional integral over the **prior mass**, making estimation more efficient when the likelihood is highly concentrated.

---

## Project Outline

The project is structured in three parts:

### 1. Theory and Intuition

* Key transformation:
  [
  Z = \int_0^1 L(X),dX
  ]
* Interpretation: Nested sampling explores **likelihood levels**
* Role of the **Beta distribution** in shrinking prior mass

---

### 2. Gaussian Special Case (Exact Implementation)

* Simplified model:

  * $\Sigma = I_d$
  * $\bar{y} = 0$
* The likelihood constraint becomes a **ball**
* Enables **exact sampling** (no MCMC required)

👉 Implemented in: `NS_gaussian_special_case.ipynb`

---

### 3. General Case (MCMC Implementation)

* Exact sampling is no longer possible
* Use **Metropolis–Hastings** to sample from:
  [
  \pi(\theta \mid L(\theta) > \ell)
  ]
* MCMC is embedded inside the nested sampling loop

👉 Implemented in: `NS_MCMC_gaussian_general.ipynb`

---

## Additional Material

A detailed explanation of:

* theory
* derivations
* implementation choices

is available in the report:

📄 `report.pdf`

---

## Key Takeaways

* Nested sampling converts a difficult integral into a tractable form
* The main challenge is sampling under a likelihood constraint
* Two approaches:

  * **Exact sampling** (special case)
  * **MCMC sampling** (general case)

---

## Possible Extensions

* Higher-dimensional models
* Alternative MCMC methods (e.g. slice sampling)
* Comparison with standard Monte Carlo or importance sampling
