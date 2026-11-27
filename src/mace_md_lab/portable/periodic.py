"""Public fixed-cell NVE interface for MACE, SevenNet and DPA4."""

from contextlib import ExitStack, contextmanager
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from ase.stress import full_3x3_to_voigt_6_stress

from ..dynamics import ArrayVerlet
from .deepmd_eval import deep_eval


def official_calculator(framework, checkpoint):
    if framework == "mace":
        from mace.calculators import MACECalculator

        return MACECalculator(
            model_paths=[str(checkpoint)], device="cuda", default_dtype="float64"
        )
    if framework == "sevennet":
        from sevenn.calculator import SevenNetCalculator

        return SevenNetCalculator(model=str(checkpoint), device="cuda")
    if framework == "dpa4":
        from deepmd.calculator import DP

        return DP(model=str(checkpoint))
    raise ValueError("choose mace, sevennet or dpa4")


def _mace(atoms, calc, stack):
    from ..candidate_replay import explicit_cutoff
    from ..candidate_schedule import ScheduledRunner
    from ..edge_virial import install_edge_virial
    from ..exact_packed_radial import install_exact_packed_radials
    from ..grouped_first import install_grouped_first
    from ..grouped_second import install_second_interaction
    from ..transforms import optimized_model

    model = calc.models[0].eval().requires_grad_(False)
    layout, stress = SimpleNamespace(), {}
    stack.enter_context(optimized_model(model, "both", atoms.numbers))
    stack.enter_context(install_exact_packed_radials(model, tangent=True))
    stack.enter_context(explicit_cutoff(model, calc.r_max))
    stack.enter_context(
        install_grouped_first(model, atoms.numbers, fold_linear=True, layout=layout)
    )
    stack.enter_context(install_second_interaction(model, layout=layout))
    stack.enter_context(install_edge_virial(model, atoms.get_volume(), stress))
    runner = ScheduledRunner(atoms, calc, 0.15, layout, True, 1.0)

    def evaluate(positions):
        energy, forces = runner.evaluate(positions)
        tensor = stress["tensor"].cpu().numpy().reshape(3, 3)
        return energy.item(), forces.cpu().numpy(), full_3x3_to_voigt_6_stress(tensor)

    return evaluate


def _sevennet(atoms, calc, stack):
    from sevenn.nn.convolution import IrrepsConvolution
    from .sevennet_graph import SevenReplay
    from .sevennet_reverse import EdgeAdjoint, use_adjoint

    blocks = [
        (m, EdgeAdjoint(m))
        for m in calc.model.modules()
        if isinstance(m, IrrepsConvolution)
    ]
    if len(blocks) != 5:
        raise ValueError("expected five SevenNet-0 convolutions")
    stack.enter_context(use_adjoint(blocks))
    return SevenReplay(atoms, calc, stack)


def _dpa4(atoms, calc, stack):
    from deepmd.pt.model.descriptor.sezm_nn.so2 import SO2Convolution
    from .dpa4_graph import replay_energy_core
    from .dpa4_reverse import use_adjoint

    modules = list(calc.dp.deep_eval.dp.named_modules())
    blocks = [(n, m) for n, m in modules if isinstance(m, SO2Convolution)]
    (model,) = [
        m
        for _, m in modules
        if hasattr(m, "core_compute") and hasattr(m, "forward_common_lower")
    ]
    stack.enter_context(use_adjoint(blocks, "local"))
    stack.enter_context(replay_energy_core(model, forward_only=False))
    return deep_eval(atoms, calc)


@contextmanager
def periodic_evaluator(framework, checkpoint, atoms):
    """Yield evaluate(positions) -> energy [eV], forces [eV/Å], ASE Voigt stress [eV/Å³]."""
    if (
        not len(atoms)
        or not np.all(atoms.pbc)
        or atoms.cell.rank != 3
        or atoms.constraints
    ):
        raise ValueError(
            "requires an unconstrained, fully periodic structure with a fixed cell"
        )
    from .qualified_models import CHECKPOINTS

    if framework not in CHECKPOINTS:
        raise ValueError("choose mace, sevennet or dpa4")
    with Path(checkpoint).open("rb") as file:
        digest = sha256()
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != CHECKPOINTS[framework]:
        raise ValueError(f"unqualified {framework} checkpoint")
    calc = official_calculator(framework, checkpoint)
    adapters = {"mace": _mace, "sevennet": _sevennet, "dpa4": _dpa4}
    with ExitStack() as stack:
        yield adapters[framework](atoms, calc, stack)


@contextmanager
def periodic_md(framework, checkpoint, atoms, dt_fs=0.5):
    """Yield an independent MD runner. Call advance(n) and read positions, energy, forces, stress."""
    if not np.isfinite(dt_fs) or dt_fs <= 0:
        raise ValueError("dt_fs must be finite and positive")
    with periodic_evaluator(framework, checkpoint, atoms) as evaluate:
        yield ArrayVerlet(atoms, evaluate, dt_fs)
