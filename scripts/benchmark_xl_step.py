"""Test a fixed-cell neighbor-list reuse bound on native Symmetrix-XL 0.2.0.

All paths run unchanged native energy, conservative forces and stress inside ASE
velocity Verlet. The benchmark temporarily replaces the calculator module's
reuse predicate only within an individual, single-threaded trial.

Use the model export and fixture from benchmarks/adjoints.md. Run with
--symmetrix-json runs/compact-small.json --structure fixtures-interface/au-water.extxyz
--output runs/xl-pair-skin-interface.json for the 16-step benchmark. For the
48-step repeat add --builders kokkos --skins 0.30 0.60 1.0 --steps 48 --warmup 5.
"""
import argparse
import json
import math
import os
from contextlib import ExitStack
from itertools import product
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch
from mace_md_lab.bench import ASEPath, file_hash, package_version, starting_structure


def fixed_pair_bound(original, decision_class):
    def reuse(*, reference, atomic_numbers, positions, cell, pbc, cutoff, skin):
        if not (
            skin > 0
            and np.array_equal(cell, reference.cell)
            and np.array_equal(cell, reference.topology_cell)
            and np.array_equal(pbc, reference.pbc)
            and np.array_equal(atomic_numbers, reference.atomic_numbers)
            and positions.shape == reference.topology_reference_positions.shape
        ):
            return original(
                reference=reference,
                atomic_numbers=atomic_numbers,
                positions=positions,
                cell=cell,
                pbc=pbc,
                cutoff=cutoff,
                skin=skin,
            )
        # Unwrapped Cartesian differences are conservative even at an image jump:
        # a jump can cause an extra rebuild. A pair of distinct atoms changes its
        # separation by at most the sum of their two displacement norms. A
        # periodic self-image has fixed separation in a fixed cell.
        displacement = positions - reference.topology_reference_positions
        squared = np.einsum('ij,ij->i', displacement, displacement)
        if len(squared) >= 2:
            first, second = np.partition(squared, -2)[-2:]
            bound = math.sqrt(first) + math.sqrt(second)
        else:
            bound = 0.0
        scale = max(
            1.0,
            float(np.max(np.abs(positions))),
            float(cutoff),
            float(np.max(np.abs(reference.topology_reference_positions))),
        )
        margin = 32.0 * np.finfo(float).eps * scale
        maximum = math.sqrt(float(squared.max(initial=0.0)))
        return decision_class(bound + margin < skin, False, maximum, None, None)

    return reuse


def verify_boundary_cases(original, reference_class, decision_class):
    cell = np.eye(3) * 12.0
    pbc = np.ones(3, dtype=bool)
    positions = np.array([[1., 1., 1.], [3., 1., 1.], [5., 1., 1.]])
    numbers = np.array([1, 1, 1])
    reference = reference_class(numbers, pbc, cell, positions, positions / 12.0, cell, positions)
    reuse = fixed_pair_bound(original, decision_class)
    cases = {
        'small_uniform_translation': ([0.03, 0.03, 0.03], True),
        'one_fast_atom': ([0.08, -0.01, 0.0], True),
        'two_fast_atoms_rebuild': ([0.08, -0.05, 0.0], False),
        'equality_rebuild': ([0.06, -0.06, 0.0], False),
        'full_periodic_image_jump_rebuild': ([12.0, 0.0, 0.0], False),
    }
    observations = {}
    for name, (offsets, expected) in cases.items():
        frame = positions.copy()
        frame[:, 0] += offsets
        arguments = dict(reference=reference, atomic_numbers=numbers, positions=frame,
                         cell=cell, pbc=pbc, cutoff=3., skin=0.12)
        accepted = bool(reuse(**arguments).reusable)
        if accepted != expected:
            raise AssertionError(f'{name}: expected {expected}, got {accepted}')
        observations[name] = {
            'stock_reuse': bool(original(**arguments).reusable),
            'two_largest_reuse': accepted,
        }
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--symmetrix-json', type=Path, required=True)
    parser.add_argument('--structure', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--selector', default='cuda12-sm70')
    parser.add_argument('--builders', nargs='+', choices=['automatic', 'kokkos'], default=['automatic', 'kokkos'])
    parser.add_argument('--skins', type=float, nargs='+', default=[0.12, 0.30])
    parser.add_argument('--steps', type=int, default=16)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=6)
    parser.add_argument('--dt-fs', type=float, default=0.5)
    parser.add_argument('--frame', type=int, default=0)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--temperature', type=float, default=300.0)
    args = parser.parse_args()
    if min(args.steps, args.repeats, args.warmup, *args.skins, args.dt_fs) <= 0:
        parser.error('all step counts, skins and time step must be positive')
    if args.output.resolve() in {args.structure.resolve(), args.symmetrix_json.resolve()}:
        parser.error('output overlaps an input')
    if package_version('symmetrix-xl') != '0.2.0':
        raise RuntimeError('this experimental hook is pinned to Symmetrix-XL 0.2.0')
    from symmetrix import load_backend, selected_backend
    load_backend(args.selector)
    from symmetrix import Symmetrix, calculator
    from symmetrix.neighbor_graph import NeighborReuseDecision, NeighborTopologyReference
    if selected_backend().get('selector') != args.selector or not args.selector.startswith('cuda'):
        raise RuntimeError('wrong backend')
    atoms, velocity_source = starting_structure(args)
    # Below half the shortest perpendicular cell height, every displacement
    # accepted by this certificate uses the same image in native fractional geometry.
    cell_heights = 1.0 / np.linalg.norm(np.linalg.inv(atoms.cell.array), axis=0)
    if max(args.skins) >= 0.5 * cell_heights.min():
        parser.error('use skins below half the shortest perpendicular cell height')
    original = calculator.neighbor_topology_reuse
    pair_bound = fixed_pair_bound(original, NeighborReuseDecision)
    boundary_cases = verify_boundary_cases(original, NeighborTopologyReference, NeighborReuseDecision)
    paths = list(product(args.skins, args.builders, ['stock', 'two_largest']))
    names = {p: f'{p[1]}_skin{p[0]}_{p[2]}' for p in paths}
    # The native workspace is large on sm70. Reuse one evaluator per skin
    # and start every trajectory with an empty candidate-graph cache.
    calculators = {
        skin: Symmetrix(
            str(args.symmetrix_json),
            dtype='float64',
            execution_profile='speed',
            streamed_edges='direct',
            neighbor_skin=skin,
            dispersion=False,
        )
        for skin in args.skins
    }

    def setup(stack, config):
        skin, builder, mode = config
        stack.enter_context(patch.dict(os.environ, SYMMETRIX_NEIGHBOR_BACKEND=builder))
        if mode == 'two_largest':
            stack.enter_context(patch.object(calculator, 'neighbor_topology_reuse', pair_bound))
        engine = calculators[skin]
        engine.reset()
        engine._neighbor_cache = None
        return engine, ASEPath(atoms, engine, args.dt_fs)

    def snap(path):
        return (*path.snapshot(), path.atoms.get_stress().copy())

    def series(config):
        with ExitStack() as stack:
            _, path = setup(stack, config)
            frames = [snap(path)]
            for _ in range(args.warmup + args.steps):
                path.advance(1)
                frames.append(snap(path))
            return frames

    def change_in_energy(sample, start):
        return sample[2] + sample[4] - start[2] - start[4]

    print('Starting moving-trajectory validation.', flush=True)
    reference = series(paths[0])
    if np.max(np.abs(reference[0][0] - reference[-1][0])) < 1e-12:
        raise ValueError('reference trajectory is stationary')
    gates = dict(energy_eV_atom=1e-8, force_eV_A=1e-6, stress_eV_A3=1e-7,
                 position_A=1e-6, velocity_A_fs=1e-7, nve_eV_atom=1e-8)
    records = {}
    for config in paths:
        print('Validating', names[config], flush=True)
        frames = series(config)
        errors = {k: 0.0 for k in gates}
        with ExitStack() as stack:
            engine, _ = setup(stack, config)
            common = atoms.copy()
            common.calc = engine
            for frame, ref in zip(frames, reference):
                common.positions[:] = ref[0]
                nve = change_in_energy(frame, frames[0])
                ref_nve = change_in_energy(ref, reference[0])
                delta = dict(
                    energy_eV_atom=abs(common.get_potential_energy() - ref[2]) / len(atoms),
                    force_eV_A=np.max(np.abs(common.get_forces() - ref[3])),
                    stress_eV_A3=np.max(np.abs(common.get_stress() - ref[5])),
                    position_A=np.max(np.abs(frame[0] - ref[0])),
                    velocity_A_fs=np.max(np.abs(frame[1] - ref[1])),
                    nve_eV_atom=abs(nve - ref_nve) / len(atoms),
                )
                if not np.isfinite(list(delta.values())).all():
                    raise ValueError('nonfinite validation output')
                errors = {k: float(max(errors[k], delta[k])) for k in errors}
            records[names[config]] = dict(
                errors=errors,
                passed=all(errors[k] <= gates[k] for k in gates),
                actual_neighbor_builder=engine.neighbor_graph_backend,
                trials_ms_step=[],
                timed_rebuild_counts=[],
            )
    passed = all(item['passed'] for item in records.values())
    orders = []
    if passed:
        groups = list(product(args.skins, args.builders))
        for trial in range(args.repeats):
            print('Timing trial', trial + 1, flush=True)
            shift = trial % len(groups)
            modes = ['stock', 'two_largest']
            if trial % 2:
                modes.reverse()
            order = [
                (*group, mode) for group in groups[shift:] + groups[:shift] for mode in modes
            ]
            orders.append([names[c] for c in order])
            for config in order:
                with ExitStack() as stack:
                    engine, path = setup(stack, config)
                    path.advance(args.warmup)
                    torch.cuda.synchronize()
                    builds = engine.neighbor_cache_build_count
                    start = perf_counter()
                    path.advance(args.steps)
                    torch.cuda.synchronize()
                    elapsed = (perf_counter() - start) * 1000 / args.steps
                    entry = records[names[config]]
                    entry['trials_ms_step'].append(elapsed)
                    entry['timed_rebuild_counts'].append(engine.neighbor_cache_build_count - builds)
        for entry in records.values():
            entry['median_ms_step'] = median(entry['trials_ms_step'])
    result = dict(
        passed=passed,
        gates=gates,
        results=records,
        trial_orders=orders,
        boundary_cases=boundary_cases,
        symmetrix_version=package_version('symmetrix-xl'),
        selector=args.selector,
        gpu=torch.cuda.get_device_name(),
        dtype='float64',
        streamed_edges='direct',
        execution_profile='speed',
        observables=['energy', 'forces', 'stress'],
        structure_sha256=file_hash(args.structure),
        converted_sha256=file_hash(args.symmetrix_json),
        atoms=len(atoms),
        dt_fs=args.dt_fs,
        warmup=args.warmup,
        steps=args.steps,
        repeats=args.repeats,
        velocity_source=velocity_source,
        timing='Warm full ASE velocity Verlet with synchronized endpoints; setup and validation excluded.',
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print({k: (v['passed'], v.get('median_ms_step')) for k, v in records.items()}, flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
