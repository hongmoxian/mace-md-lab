"""Fixed-capacity replay for the single-frame SevenNet-0 energy and its edge gradient."""

import numpy as np
import torch
from sevenn import _keys as K
from sevenn.nn.convolution import IrrepsConvolution
from sevenn.nn.force_output import ForceStressOutputFromEdge
from sevenn.nn.node_embedding import OnehotEmbedding
from sevenn.train.dataload import unlabeled_atoms_to_graph


class SevenReplay:
    def __init__(self, atoms, calculator, stack):
        self.atoms = atoms.copy()
        self.model = calculator.model
        self.cutoff = float(calculator.cutoff)
        self.volume = atoms.get_volume()
        self.skin = 1.0
        self.reference = None
        self.capacity = 0
        self.captures = 0
        self.graph = None
        children = list(self.model.children())
        if (
            not isinstance(children[-1], ForceStressOutputFromEdge)
            or self.model.eval_modal_map
        ):
            raise ValueError(
                "expected the single-modal SevenNet energy with edge-force output"
            )
        self.numbers = torch.as_tensor(
            atoms.numbers, device=next(self.model.parameters()).device, dtype=torch.long
        )
        # Species are fixed throughout NVE. One-time initialization also keeps
        # the upstream CPU type lookup and one-hot conversion outside capture.
        self.static = {
            K.ATOMIC_NUMBERS: self.numbers,
            self.model.key_node_feature: self.model._atomic_numbers_to_onehot(
                self.numbers
            ),
        }
        self.layers = []
        for layer in children[:-1]:
            if isinstance(layer, OnehotEmbedding):
                self.static = layer(self.static)
            else:
                self.layers.append(layer)

        # Padded edges contribute zero to every message layer and its coordinate VJP.
        def mask_weights(module, args, weights):
            return weights * self.valid[:, None]

        for layer in self.layers:
            if isinstance(layer, IrrepsConvolution):
                stack.callback(
                    layer.weight_nn.register_forward_hook(mask_weights).remove
                )

    def compute(self):
        edge = self.vec.detach().requires_grad_()
        data = dict(self.static)
        data.update(
            {
                K.EDGE_IDX: self.index,
                K.EDGE_VEC: edge,
                K.NUM_ATOMS: self.num_atoms,
                K.CELL_VOLUME: self.vol,
            }
        )
        for layer in self.layers:
            data = layer(data)
        energy = data[K.PRED_TOTAL_ENERGY].sum()
        (grad,) = torch.autograd.grad(energy, edge)
        forces = edge.new_zeros((len(self.atoms), 3))
        forces.index_add_(0, self.index[0], grad)
        forces.index_add_(0, self.index[1], -grad)
        # ASE order xx, yy, zz, yz, xz, xy, with SevenNet's off-diagonal convention.
        diag = (edge * grad).sum(0)
        shear = torch.stack(
            (
                (edge[:, 1] * grad[:, 2]).sum(),
                (edge[:, 2] * grad[:, 0]).sum(),
                (edge[:, 0] * grad[:, 1]).sum(),
            )
        )
        return (
            energy.detach(),
            forces.detach(),
            (torch.cat((diag, shear)) / self.volume).detach(),
        )

    def __call__(self, positions):
        if (
            self.reference is None
            or 2 * np.linalg.norm(positions - self.reference, axis=1).max()
            >= self.skin - 1e-12
        ):
            self.atoms.positions[:] = positions
            graph = unlabeled_atoms_to_graph(
                self.atoms, self.cutoff + self.skin, with_shift=True
            )
            self.ij = graph[K.EDGE_IDX]
            self.shift = graph[K.CELL_SHIFT] @ self.atoms.cell.array
            self.reference = positions.copy()
        vec = positions[self.ij[1]] - positions[self.ij[0]] + self.shift
        active = np.einsum("ij,ij->i", vec, vec) < self.cutoff**2
        n = int(active.sum())
        if n > self.capacity or self.graph is None:
            self.graph = None
            self.capacity = 64 * ((n + max(64, n // 50) + 63) // 64)
            self.index = torch.zeros(
                (2, self.capacity), device="cuda", dtype=torch.long
            )
            self.vec = torch.empty(
                (self.capacity, 3), device="cuda", dtype=torch.float32
            )
            self.valid = self.vec.new_empty(self.capacity)
            self.vol = self.vec.new_tensor(self.volume)
            self.num_atoms = self.numbers.new_tensor([len(self.atoms)])
        ij = np.zeros((2, self.capacity), dtype=np.int64)
        ij[:, :n] = self.ij[:, active]
        vecs = np.zeros((self.capacity, 3), dtype=np.float32)
        vecs[:n] = vec[active]
        vecs[n:, 0] = 1.0
        self.index.copy_(torch.from_numpy(ij))
        self.vec.copy_(torch.from_numpy(vecs))
        self.valid[:n].fill_(1.0)
        self.valid[n:].zero_()
        if self.graph is None:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream), torch.enable_grad():
                self.compute()
            torch.cuda.current_stream().wait_stream(stream)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, stream=stream), torch.enable_grad():
                self.out = self.compute()
            self.captures += 1
        self.graph.replay()
        energy, forces, stress = self.out
        return energy.item(), forces.cpu().numpy(), stress.cpu().numpy()
