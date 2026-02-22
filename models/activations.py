"""Novel activation functions discovered by evolutionary search.

Implements the four activation functions from:
  Vitvitskyi et al. (2026), "Mining Generalizable Activation Functions",
  arXiv:2602.05688.

The functions were discovered via AlphaEvolve — an evolutionary search pipeline
driven by frontier LLMs — using out-of-distribution (OOD) performance as the
fitness objective to obtain functions with strong generalisation inductive biases.

Functions provided
------------------
GELUSine
    f(x) = GELU(x) + α·sin(x),  α=0.1
    Additive periodic perturbation of GELU.  Appendix C ablations show α=0.1
    is optimal; AlphaEvolve independently discovered this value.

GELUSincPerturbation
    f(x) = GELU(x) + α·sinc(x),  α=0.5,  sinc(x)=sin(x)/x
    Self-decaying oscillatory perturbation via the unnormalized sinc function.
    α=0.5 consistently outperforms random hyperparameter samples in the paper.

GMTU (Gaussian-Modulated Tangent Unit)
    f(x) = p_alpha·tanh(p_beta·x)·exp(−p_gamma·x²) + p_gamma·x
    Combines a Gaussian-gated tanh response with a linear leak that preserves
    gradient flow for large |x|.  Default params: p_alpha=1.0, p_beta=1.5,
    p_gamma=0.2.

Turbulent
    f(x) = sign(x)·log(1+0.5|x|) + sin(x)·exp(−z²/2)
    where z=(x−μ)/σ computed over all elements of the tensor.
    The only non-pointwise function: it computes simple batch statistics to
    introduce distribution-aware, non-monotonic ripples into a log-growth base.

All four functions are pointwise except Turbulent, which uses batch statistics.

Usage
-----
>>> act = build_activation("gelusine")
>>> out = act(x)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Valid activation names (after lowercasing and removing hyphens/underscores)
_VALID_NAMES = frozenset(["gelu", "gelusine", "gelusincperturbation", "gmtu", "turbulent"])


class GELUSine(nn.Module):
    """GELU perturbed by a scaled sine term.

    f(x) = GELU(x) + α · sin(x)

    The sine perturbation introduces smooth periodic non-linearity that
    encourages the network to represent functions with oscillatory components.
    Paper-optimal α = 0.1.

    Args:
        alpha: Amplitude of the sine perturbation (default 0.1).
    """

    def __init__(self, alpha: float = 0.1) -> None:
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x) + self.alpha * torch.sin(x)

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"


class GELUSincPerturbation(nn.Module):
    """GELU perturbed by a scaled unnormalized sinc term.

    f(x) = GELU(x) + α · sinc(x)

    where sinc(x) = sin(x) / x  (unnormalized sinc; limit = 1 at x = 0).

    The sinc factor decays as O(1/|x|), so the perturbation is strongest near
    the origin and vanishes for large activations, giving GELU-like asymptotic
    behaviour while adding near-origin richness.  Paper-optimal α = 0.5.

    Args:
        alpha: Amplitude of the sinc perturbation (default 0.5).
        eps: Threshold below which sinc is approximated as 1 to avoid 0/0.
    """

    def __init__(self, alpha: float = 0.5, eps: float = 1e-8) -> None:
        super().__init__()
        self.alpha = alpha
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Unnormalized sinc: sin(x)/x with the exact limit 1 at x=0
        sinc_x = torch.where(
            x.abs() < self.eps,
            torch.ones_like(x),
            torch.sin(x) / x,
        )
        return F.gelu(x) + self.alpha * sinc_x

    def extra_repr(self) -> str:
        return f"alpha={self.alpha}"


class GMTU(nn.Module):
    """Gaussian-Modulated Tangent Unit.

    f(x) = p_alpha · tanh(p_beta · x) · exp(−p_gamma · x²) + p_gamma · x

    Design rationale (from Vitvitskyi et al., 2026, Appendix A):

    * Primary response — tanh(p_beta·x): provides a localized, non-periodic
      non-linearity around the origin with bounded output.
    * Gaussian gate — exp(−p_gamma·x²): decays the tanh component for large |x|,
      allowing the function to "release" saturated activations.
    * Linear leak — p_gamma·x: ensures the function grows approximately linearly
      for large |x| (where the Gaussian gate → 0), preventing saturation and
      maintaining gradient flow throughout training.

    Near zero: f(x) ≈ (p_alpha·p_beta + p_gamma)·x  (linear regime).
    Large |x|: f(x) ≈ p_gamma·x                      (pure linear leak).

    Default parameters from Appendix A: p_alpha=1.0, p_beta=1.5, p_gamma=0.2.

    Args:
        p_alpha: Amplitude of the tanh component.
        p_beta:  Steepness of the tanh curve.
        p_gamma: Controls both the Gaussian decay rate and the linear leak slope.
    """

    def __init__(
        self,
        p_alpha: float = 1.0,
        p_beta: float = 1.5,
        p_gamma: float = 0.2,
    ) -> None:
        super().__init__()
        self.p_alpha = p_alpha
        self.p_beta = p_beta
        self.p_gamma = p_gamma

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gaussian_gate = torch.exp(-self.p_gamma * x ** 2)
        primary = self.p_alpha * torch.tanh(self.p_beta * x) * gaussian_gate
        linear_leak = self.p_gamma * x
        return primary + linear_leak

    def extra_repr(self) -> str:
        return f"p_alpha={self.p_alpha}, p_beta={self.p_beta}, p_gamma={self.p_gamma}"


class Turbulent(nn.Module):
    """Turbulent activation function.

    f(x) = base(x) + perturbation(x)

    where:
        base(x)         = sign(x) · log(1 + 0.5|x|)   [symmetric log-growth]
        μ, σ            = mean and std of ALL elements of the input tensor
        z               = (x − μ) / (σ + ε)            [globally standardized x]
        perturbation(x) = sin(x) · exp(−z² / 2)        [sine gated by Gaussian of z]

    The perturbation term introduces non-monotonic "ripples" whose magnitude
    depends on how far each activation is from the batch mean in standard
    deviations — activations near the mean receive a full sine perturbation;
    outlier activations are suppressed by the Gaussian gate.

    This is the only non-pointwise function among the four: it uses simple
    batch statistics (mean and std over the entire tensor) to make the function
    distribution-aware.  The paper notes this property makes Turbulent more
    expressive but also more prone to overfitting to synthetic dataset statistics.

    Note: batch statistics are computed from the current tensor at both train
    and eval time (no running-average tracking).

    Args:
        eps: Small constant added to std for numerical stability (default 1e-6).
    """

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Symmetric logarithmic base function
        base = torch.sign(x) * torch.log1p(0.5 * x.abs())

        # Global batch statistics over all elements of the tensor
        mean = x.mean()
        std = x.std() + self.eps

        # Globally standardised input for Gaussian modulation
        z = (x - mean) / std

        # Perturbation: sine wave gated by Gaussian function of z
        perturbation = torch.sin(x) * torch.exp(-0.5 * z ** 2)

        return base + perturbation

    def extra_repr(self) -> str:
        return f"eps={self.eps}"


# ---------------------------------------------------------------------------
# Registry and factory
# ---------------------------------------------------------------------------

def build_activation(act_type: str) -> nn.Module:
    """Instantiate an activation module by name.

    Supported names (case-insensitive, hyphens/underscores ignored):
        'gelu'                 → nn.GELU
        'gelusine'             → GELUSine          (α=0.1)
        'gelusincperturbation' → GELUSincPerturbation (α=0.5)
        'gmtu'                 → GMTU
        'turbulent'            → Turbulent

    Args:
        act_type: Activation function identifier string.

    Returns:
        An nn.Module implementing the specified activation.

    Raises:
        ValueError: If act_type is not recognised.
    """
    key = act_type.lower().replace("-", "").replace("_", "")
    if key == "gelu":
        return nn.GELU()
    elif key == "gelusine":
        return GELUSine()
    elif key in ("gelusincperturbation", "gelusin", "gelusinc"):
        return GELUSincPerturbation()
    elif key == "gmtu":
        return GMTU()
    elif key == "turbulent":
        return Turbulent()
    else:
        available = ["gelu", "gelusine", "gelusincperturbation", "gmtu", "turbulent"]
        raise ValueError(
            f"Unknown activation '{act_type}'. Available: {available}"
        )
