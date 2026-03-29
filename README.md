# Scaling Span Valuation by Shapley Value Approximation in Named Entity Recognition (SPANVAL)
    This repository includes a ready-to-run Jupyter Notebook example to help you quickly understand how to use the SPANVAL: 
    Example Path: src/example.ipynb   
    Running Instructions: Ensure your local environment has dependencies (e.g., jupyterlab, required libraries) installed. Then open the file directly to execute all code blocks.  
# Appendix: Technical Derivations and Proofs

## A. Derivation of the Gradient for the Objective Function

To further clarify the optimization dynamics, we expand the gradient using the likelihood ratio trick:

$$\nabla_\theta \mathcal{J}(\theta) = \mathbb{E}_{\mathbf{a}} \left[ \sum_{i=1}^M \nabla_\theta \log \pi_\theta(a_i|s_i) \cdot R_t + \nabla_\theta R_t \right]$$

where $R_t = \mathbb{E}_{z_v}\left[\ell\left(\mathbf{w}_{t}, z_v \right)- \ell\left(\mathbf{w}_{t+1} (z_l;\theta_t), z_v\right) \right]$.

Due to $\ell\left(\mathbf{w}_{t}, z_v \right)$ being a constant term, $\nabla_\theta \mathbb{E}_{\mathbf{a}}\left[R_t \right]$ can be further expanded:

$$\nabla_\theta \mathbb{E}_{\mathbf{a}}\left[R_t \right] = \mathbb{E}_{\mathbf{a},z_v}\left[- \nabla_\theta \ell\left(\mathbf{w}_{t+1} (z_l;\theta_t), z_v\right) \right] \tag{1}$$

According to the chain rule, the gradient decomposes as:

$$\nabla_\theta \ell(\mathbf{w}_t, z_v) = \nabla_\mathbf{w} \ell(\mathbf{w}_{t+1}, z_v)^\top \cdot \nabla_\theta \mathbf{w}_{t+1} \tag{2}$$

where $\mathbf{w}_{t+1} = \mathbf{w}_{t} - \eta \nabla_{\mathbf{w}} \sum_j \ell(z_l^{(j)}; \mathbf{w}_{t}) \cdot a_j$.

Since $\mathbf{w}_{t}$ is independent of $\theta$ and $a_j$ are sampled from policy $\pi_{\theta}$:

$$\nabla_\theta \mathbf{w}_{t+1} = - \eta \left[\nabla_{\mathbf{w}} \sum_j \ell(z_l^{(j)}; \mathbf{w}_{t}) \cdot \nabla_{\theta}\mathbb{E}_{\mathbf{a}}\left[a_j\right] \right]$$

where $a_j \sim \text{Bernoulli}(\pi_\theta(a_j|s_j))$ with $\mathbb{E}_{\mathbf{a}}[a_j] = \mathbb{E}_{\mathbf{a}}[\pi_\theta(a_j|s_j)]$.

By the Score Function Theorem, $\nabla_\theta\mathbb{E}_{\mathbf{a}}[a_j] = \mathbb{E}_{\mathbf{a}}[\nabla_\theta \log\pi_\theta(a_j|s_j)\cdot a_j]$. Thus:

$$\nabla_\theta \mathbf{w}_{t+1} = - \eta \nabla_{\mathbf{w}} \sum_j \ell(z_l^{(j)}; \mathbf{w}_{t}) \cdot \mathbb{E}_{\mathbf{a}}\left[a_j\nabla_\theta \log \pi_\theta(a_j|s_j) \right] \tag{3}$$

---

### Taylor Expansion of $\nabla_\mathbf{w} \ell(\mathbf{w}_t, z_v)$

$$\nabla_\mathbf{w} \ell(\mathbf{w}_{t+1}, z_v) = \nabla_\mathbf{w} \ell(\mathbf{w}_{t}, z_v) + \nabla^2_\mathbf{w} \ell \cdot \Delta\mathbf{w} + \mathcal{O} (\|\Delta\mathbf{w}\|^2)+\cdots$$

where $\Delta\mathbf {w} = \mathbf {w}_{t+1} - \mathbf{w}_{t} = -\eta \left[\nabla_\mathbf{w} \sum_j \ell(z_l^{(j)}; \mathbf{w}_{t}) a_j \right]$ and $\nabla^2_\mathbf{w}$ is the Hessian matrix.

**Assumption:** There exists $G > 0$ such that for all training samples $z_l^{(j)}$ and parameters $\mathbf{w}$:

$$\big\| \nabla_\mathbf{w} {\textstyle \sum_{j}} \ell(z_l^{(j)}; \mathbf{w}_{t}) a_j \big\| \leq G$$

Therefore, the parameter increment satisfies:

$$\|\Delta\mathbf {w}\| = \eta \big\| \nabla_\mathbf{w} {\textstyle \sum_{j}} \ell(z_l^{(j)}; \mathbf{w}_{t}) a_j \big\| \leq \eta G \tag{4}$$

Suppose $\nabla_\mathbf{w} \ell$ is Lipschitz smooth with constant $L$:

$$\|\nabla_\mathbf{w} \ell(\mathbf{w}_{t+1}, z_v) - \nabla_\mathbf{w} \ell(\mathbf{w}_{t}, z_v)\| \leq L \| \mathbf {w}_{t+1} - \mathbf{w}_{t} \| \tag{5}$$

By (4) and (5), the gradient variation is bounded:

$$\|\nabla_\mathbf{w} \ell(\mathbf{w}_{t+1}, z_v) - \nabla_\mathbf{w} \ell(\mathbf{w}_{t}, z_v)\| \leq L \|\Delta\mathbf{w}\| \leq L \eta G$$

For sufficiently small $\eta < \frac{\epsilon}{L G}$, this difference is constrained within threshold $\epsilon$. Thus, higher-order terms become negligible:

$$\nabla_\mathbf{w} \ell(\mathbf{w}_{t+1}, z_v) \approx \nabla_\mathbf{w} \ell(\mathbf{w}_{t}, z_v) \tag{6}$$

---

### Final Gradient Expression

Substituting (3) and (6) into (1):

$$\nabla_\theta \mathbb{E}_{\mathbf{a}}\left[R_t \right] \approx \eta \cdot \mathbb{E}_{\mathbf{a},z_v} \left[ \nabla_{\mathbf{w}}\ell(\mathbf{w}_{t},z_v )^\top\cdot \nabla_{\mathbf{w}} \sum_{j=1}^{M} \ell(\mathbf{w}_{t},z_l^{(j)} )\cdot \nabla_\theta \log\pi_\theta(a_j|s_j)\cdot a_j \right]$$

Thus, the complete policy-gradient derivative is:

$$\begin{aligned}
\nabla_\theta \mathcal{J}(\theta) \approx &\ \mathbb{E}_{\mathbf{a}} \left[ \sum_{i=1}^M \nabla_\theta \log \pi_\theta(a_i|s_i) \cdot R_t\right] \\
&+ \eta \cdot \mathbb{E}_{\mathbf{a},z_v} \left[ \nabla_{\mathbf{w}}\ell(\mathbf{w}_{t},z_v )^\top\cdot \nabla_{\mathbf{w}} \sum_{j=1}^{M} \ell(\mathbf{w}_{t},z_l^{(j)} )\cdot a_j\nabla_\theta \log\pi_\theta(a_j|s_j) \right]
\end{aligned}$$

> **Observation:** The gradient signal consists of (i) the traditional policy gradient weighted by $R_t$, and (ii) a gradient correction term $\nabla_\theta R_t$ integrated into the policy update.

---

## B. Proof for Theorem 1

### B.1 Utility Gain Aligns with Shapley

**Proof.** The immediate utility gain $R_t$ is defined as the validation loss reduction induced by selected subset $S$:

$$R_t = \mathbb{E}_{z_v} \left[ \ell(\mathbf{w}_t, z_v) - \ell(\mathbf{w}_{t+1}, z_v) \right] = \nu(S) - \nu(\emptyset)$$

where $\nu(S) = -\mathbb{E}_{z_v} \ell(\mathbf{w}_{t+1}(S), z_v)$ denotes the utility of $S$, and $\nu(\emptyset) = -\mathbb{E}_{z_v} \ell(\mathbf{w}_t, z_v)$ is the baseline value.

By definition, the Shapley value of span $j$ quantifies its expected marginal contribution across all subsets excluding $j$. Taking expectation over $S \sim \pi_\theta$:

$$\mathbb{E}_S \left[ R_t \right] = \mathbb{E}_S \left[ \nu(S) - \nu(\emptyset) \right] = \mathbb{E}_S \left[ \nu(S) \right] - \nu(\emptyset)$$

By the axiomatic properties of Shapley values, any coalition value decomposes into additive contributions and higher-order interactions:

$$\nu(S) = \sum_{j \in S} \phi_j + \sum_{T \subseteq S, |T| \geq 2} \Delta_T$$

where $\Delta_T$ denotes the interaction term of subset $T$, quantifying non-additive synergies or redundancies.

Substituting into the expectation:

$$\mathbb{E}_S \left[ R_t \right] = \sum_{j=1}^M \pi_\theta(a_j=1 \mid s_j) \cdot \phi_j + \mathbb{E}_S \Big[ \sum_{\substack{T \subseteq S, |T| \geq 2}} \Delta_T \Big] - \nu(\emptyset)$$

At convergence, higher-order interaction terms vanish as $|T| \to \infty$ (since significant multi-element synergies are sparse in practice). Hence, $R_t$ approximates the weighted sum of Shapley values. $\blacksquare$

---

### B.2 Error Bounding

**Proof.** Using first-order Taylor expansion of $\ell(\mathbf{w}_{t+1}, z_v)$ around $\mathbf{w}_t$, the approximation error of $R_t$ is bounded by:

$$\left| R_t - \eta \cdot \nabla_{\mathbf{w}} \ell(\mathbf{w}_t, z_v)^\top \sum_{j \in S} \nabla_{\mathbf{w}} \ell(\mathbf{w}_t, z_l^{(j)}) \right| \leq \frac{L \eta^2}{2} \left\| \sum_{j \in S} \nabla_{\mathbf{w}} \ell(\mathbf{w}_t, z_l^{(j)}) \right\|^2$$

where $L$ is the Lipschitz constant of $\ell$. This implies the overall error in $\nabla_\theta \mathcal{J}(\theta)$ is $\mathcal{O}(\eta^2)$. $\blacksquare$

---
