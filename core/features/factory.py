"""Feature transform factory.

Provides a single entry point for creating feature transforms so that
downstream projects (e.g. ML) never need to import directly from other
projects (e.g. DL).  Heavy imports (torch, etc.) are deferred until the
specific transform is actually requested.
"""

from __future__ import annotations

from core.features.transforms import BaseFeatureTransform, PCATransform

# Default refit frequencies by transform type (steps between refits).
REFIT_DEFAULTS: dict[str, int] = {
    "har": 1,
    "raw": 1,
    "pca": 1,
    "ae": 240,
}


def create_feature_transform(
    kind: str,
    *,
    n_components: int = 5,
    n_features: int = 0,
    alpha: float = 0.5,
    hidden_dim: int | None = None,
    epochs: int = 50,
    ae_loss_path: str | None = None,
    ae_weights_path: str | None = None,
) -> BaseFeatureTransform | None:
    """Create a feature transform by name.

    Returns None for feature types that don't require a transform (har, raw).
    """
    if kind == "pca":
        return PCATransform(n_components=n_components)

    if kind == "ae":
        from projects.dl.features import AETransform

        transform = AETransform(
            n_features=n_features,
            n_components=n_components,
            alpha=alpha,
            hidden_dim=hidden_dim,
            epochs=epochs,
            ae_loss_path=ae_loss_path,
        )
        if ae_weights_path is not None:
            transform.load_weights(ae_weights_path)
        return transform

    return None
