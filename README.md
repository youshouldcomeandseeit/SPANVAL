# Scaling Span Valuation by Shapley Value Approximation in Named Entity Recognition (SPANVAL)
    This repository includes a ready-to-run Jupyter Notebook example to help you quickly understand how to use the SPANVAL: 
    Example Path: src/example.ipynb   
    Running Instructions: Ensure your local environment has dependencies (e.g., jupyterlab, required libraries) installed. Then open the file directly to execute all code blocks.  


# Appendix: Supplementary Materials

## A. Proof for Theorem 1

### A.1 Utility Gain Aligns with Shapley

**Proof.** The immediate utility gain $R_t$ is defined as the validation loss reduction induced by selected subset $S$:

$$R_t = \mathbb{E}_{z_v} \left[ \ell(\mathbf{w}_t, z_v) - \ell(\mathbf{w}_{t+1}, z_v) \right] = \nu(S) - \nu(\emptyset)$$

where $\nu(S) = -\mathbb{E}\_{z_v} \ell(\mathbf{w}\_{t+1}(S), z\_v)$ denotes the utility of $S$, and $\nu(\emptyset) = -\mathbb{E}_{z_v} \ell(\mathbf{w}_t, z_v)$ is the baseline value.

By definition, the Shapley value of span $j$ quantifies its expected marginal contribution across all subsets excluding $j$. Taking expectation over $S \sim \pi_\theta$:

$$\mathbb{E}_S \left[ R_t \right] = \mathbb{E}_S \left[ \nu(S) - \nu(\emptyset) \right] = \mathbb{E}_S \left[ \nu(S) \right] - \nu(\emptyset)$$

By the axiomatic properties of Shapley values, any coalition value decomposes into additive contributions and higher-order interactions:

$$\nu(S) = \sum_{j \in S} \phi_j + \sum_{T \subseteq S, |T| \geq 2} \Delta_T$$

where $\Delta_T$ denotes the interaction term of subset $T$, quantifying non-additive synergies or redundancies.

Substituting into the expectation:

$$\mathbb{E}_S \left[ R_t \right] = \sum_{j=1}^M \pi_\theta(a_j=1 \mid s_j) \cdot \phi_j + \mathbb{E}_S \Big[ \sum_{\substack{T \subseteq S, |T| \geq 2}} \Delta_T \Big] - \nu(\emptyset)$$

At convergence, higher-order interaction terms vanish as $|T| \to \infty$ (since significant multi-element synergies are sparse in practice). Hence, $R_t$ approximates the weighted sum of Shapley values. 

---

### A.2 Error Bounding

**Proof.** Using first-order Taylor expansion of $\ell(\mathbf{w}_{t+1}, z_v)$ around $\mathbf{w}_t$, the approximation error of $R_t$ is bounded by:

$$\left| R_t - \eta \cdot \nabla_{\mathbf{w}} \ell(\mathbf{w}_t, z_v)^\top \sum_{j \in S} \nabla_{\mathbf{w}} \ell(\mathbf{w}_t, z_l^{(j)}) \right| \leq \frac{L \eta^2}{2} \left\| \sum_{j \in S} \nabla_{\mathbf{w}} \ell(\mathbf{w}_t, z_l^{(j)}) \right\|^2$$

where $L$ is the Lipschitz constant of $\ell$. This implies the overall error in $\nabla_\theta \mathcal{J}(\theta)$ is $\mathcal{O}(\eta^2)$. 

---

For more specific details, please refer to the pdf.
