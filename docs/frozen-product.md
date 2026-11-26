# Fixed-weight scalar product contraction

For an atom of species $z$, let $x_{ci}$ denote the interaction features entering a MACE product block. The index $c$ runs over channels, and $i$ over angular components. The scalar output in one channel is

```math
P_{zc}(x) = \sum_{q=1}^3 \sum_{i_1,\ldots,i_q} \sum_p
U^{(q)}_{i_1\ldots i_q,p} W^{(q)}_{zpc} \prod_{a=1}^{q} x_{ci_a}.
```

$U$ and the trained $W$ are constant during MD. For each sorted multiset of angular components, the compiler adds the distinct permutations of the corresponding $U$ rows, then contracts with $W$ once when the transform is installed. This evaluates the same polynomial even when the original coupling tensors have different entries for different permutations. All path coefficients must be exactly zero for a term to be dropped.

The MACE-MP-0 small checkpoint used in the benchmarks has 128 channels and 16 angular components in each of two scalar product blocks. Its one-, two- and three-factor terms reduce to 1, 16 and 77 monomials, respectively. This count comes from its own coupling tensors. For a different compatible checkpoint the compiler derives a fresh term list.

A CUDA thread evaluates one atom/channel. The precompute schedule evaluates the polynomial and its feature derivatives together, stores the derivatives, and multiplies by the incoming adjoint during force evaluation. The deferred schedule evaluates only the value in the forward pass. During the backward pass it rereads the features and coefficients, accumulates the derivatives and applies the incoming adjoint in one kernel. Both schedules use the same coefficient table, polynomial order and handling of repeated indices. Precompute can reduce repeated coefficient traffic; deferred avoids writing and rereading a separate derivative array. The existing feature tensor may have to stay alive longer in the deferred case. This keeps all three factors of a cubic term in the chain rule. Coefficients for all checkpoint species are retained. For ordinary atoms the kernel selects the species row, and for padding atoms with all-zero element attributes the value and its feature derivative vanish. Fractional attributes work by linearly combining species coefficients.

On CPU the same polynomial is evaluated with ordinary PyTorch operations. On CUDA the implementation uses an optional CuPy NVRTC kernel on the current PyTorch stream. Select eager e3nn MACE or ScaleShiftMACE for this experiment. It specializes scalar-only product modules with correlation at most three, up to 16 angular components, in FP64 or FP32. Other product modules continue to use their existing implementation. The optimized CUDA path provides first position derivatives for fixed-weight MD. Training, higher derivatives, and gradients with respect to chemical identity require the stock implementation.

The product can be timed by selecting `product` in the adjoint benchmark. Compare it with `none` at identical replay settings to isolate its contribution. Installation cost, compilation/warm-up and production MD cost have different amortization periods.
