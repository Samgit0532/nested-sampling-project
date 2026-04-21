# Monte Carlo Project: Nested Sampling

## Overview

This project explores **Nested Sampling (NS)**, a Monte Carlo method used to estimate the **Bayesian marginal likelihood (evidence)**:

$$
Z = \int L(\theta)\,\pi(\theta)\,d\theta.
$$

Nested sampling transforms this high-dimensional integral into a one-dimensional integral over the **prior mass**, allowing for more efficient estimation in problems where the likelihood is highly concentrated.

---

## Project Outline

The project is structured in three parts:

### 1. Theory and Intuition
- Understanding the transformation:
  $$
  Z = \int_0^1 L(X)\,dX
  $$
- Interpretation of nested sampling as exploring **likelihood levels**
- Role of the Beta distribution in the shrinkage of prior mass

---

### 2. Gaussian Special Case (Exact Implementation)
- A simplified model where:
  - $\Sigma = I_d$
  - $\bar y = 0$
- The likelihood constraint becomes a **ball**
- Enables **exact sampling** of new points (no MCMC)

👉 Implemented in: `NS_gaussian_special_case.ipynb`

---

### 3. General Case (MCMC Implementation)
- In the general setting, exact sampling is not possible
- We use a **Metropolis–Hastings algorithm** to sample from:
  $$
  \pi(\theta \mid L(\theta) > \ell)
  $$
- The MCMC sampler is integrated inside the nested sampling loop

👉 Implemented in: `NS_MCMC_gaussian_general.ipynb`


---

## Additional Material

A detailed explanation of:
- the theory,
- the derivations,
- and the implementation choices

is provided in the accompanying **PDF report**: `report.pdf`

---

## Key Takeaways

- Nested sampling converts a difficult integral into a tractable form
- The main challenge is sampling under a likelihood constraint
- Two approaches:
  - **Exact sampling** (special case)
  - **MCMC sampling** (general case)

---
