"""Scoped inference-only transforms for stock, fixed-weight MACE models."""

from contextlib import ExitStack, contextmanager

BASE_MODES = ("none", "angular", "product", "both")
MODES = (*BASE_MODES, "product_deferred", "both_deferred")


@contextmanager
def optimized_model(model, mode="none", atomic_numbers=None):
    """Evaluate one model with selected transforms and restore it on exit.

    First position derivatives are supported. Use separate model instances for
    concurrent evaluations in different modes or for training.
    """
    if mode not in MODES:
        raise ValueError(f"unknown adjoint mode {mode}")
    with ExitStack() as stack:
        metadata = {"mode": mode, "specialized_product_layers": 0}
        if mode in ("angular", "both", "both_deferred"):
            from .adjoint import install_angular_adjoint

            stack.enter_context(install_angular_adjoint(model))
        if mode in ("product", "both", "product_deferred", "both_deferred"):
            from .frozen_product import install_frozen_products

            for layer in model.products:
                stack.callback(
                    setattr,
                    layer,
                    "symmetric_contractions",
                    layer.symmetric_contractions,
                )
            schedule = "deferred" if mode.endswith("_deferred") else "precompute"
            count = install_frozen_products(model, atomic_numbers, schedule=schedule)
            metadata["product_schedule"] = schedule
            if not count:
                raise ValueError(
                    "this checkpoint has no compatible scalar product blocks"
                )
            metadata["specialized_product_layers"] = count
        yield metadata
