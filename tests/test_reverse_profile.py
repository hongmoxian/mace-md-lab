"""Boundary-VJP tests with tuple outputs, unused features and aliased arguments."""

import unittest
from contextlib import ExitStack

import torch

from mace_md_lab.profile import Mark
from mace_md_lab.reverse_profile import (
    fresh_inputs,
    local_vjp,
    maximum_delta,
    observe_calls,
)


class Mix(torch.nn.Module):
    def forward(self, primary, *, other, labels):
        summed = primary + other
        return summed.square(), summed, None, summed.sin()


class CotangentReplay(unittest.TestCase):
    def test_actual_boundary_cotangents_are_partial_derivatives(self):
        x = torch.tensor([-0.8, 0.4, 1.1], dtype=torch.float64, requires_grad=True)
        layer = Mix()
        labels = torch.tensor([1, 3, 5])
        with ExitStack() as stack:
            records = observe_calls({"mix": layer}, stack)
            squared, summed, _, _ = layer(x, other=x, labels=labels)
            # One output depends internally on the other. At separate export
            # boundaries, indirect gradients must be counted only once.
            energy = (squared + 0.3 * summed).sum()
            (observed,) = torch.autograd.grad(energy, x)
        torch.testing.assert_close(observed, 8 * x + 0.6)
        (record,) = records
        torch.testing.assert_close(record.cotangents[0], torch.ones_like(x))
        torch.testing.assert_close(record.cotangents[1], torch.full_like(x, 0.3))
        self.assertIsNone(record.cotangents[2])
        self.assertIsNone(record.cotangents[3])
        device = torch.device("cpu")
        elapsed, grads, error = local_vjp(
            layer, record, device, (Mark(device), Mark(device)), True
        )
        self.assertEqual(len(grads), 1)
        torch.testing.assert_close(grads[0], observed)
        self.assertEqual(error, 0.0)
        self.assertGreaterEqual(elapsed, 0.0)
        (args, kwargs), variables = fresh_inputs(record.arguments)
        self.assertIs(args[0], kwargs["other"])
        self.assertFalse(kwargs["labels"].requires_grad)
        self.assertEqual(len(variables), 1)

    def test_unused_and_zero_vjps_are_equivalent(self):
        err, scale = maximum_delta(
            [None, torch.tensor([2.0])], [torch.tensor([0.0]), torch.tensor([2.5])]
        )
        self.assertEqual(err, 0.5)
        self.assertEqual(scale, 2.0)


if __name__ == "__main__":
    unittest.main()
