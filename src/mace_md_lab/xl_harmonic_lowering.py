"""Cartesian derivatives of XL's retained component-normalized real harmonics.

Rows of the degree-l solid-harmonic derivative contain at most two entries in
the degree-(l-1) basis. Cartesian unit directions are also present in Y[1:4].
This identity is for the fixed real basis and ordering of XL v0.2.0 with l<=3.
"""
import math


# Rows are in original Cartesian x, y, z; at each (axis, lm), tuples contain
# a lower-order harmonic index and the signed square of its coefficient.
# Solid harmonic derivatives use this sparse set before the radial projection.
ROWS = (
    (
        (), ((0, 3),), (), (),
        ((3, 5),), ((2, 5),), ((1, -5/3),), (), ((1, -5),),
        ((8, 10.5),), ((7, 7),), ((8, 0.7), (6, 8.4)), ((5, -4.2),),
        ((4, -0.7),), ((5, -7),), ((4, -10.5),),
    ),
    (
        (), (), ((0, 3),), (),
        (), ((1, 5),), ((2, 20/3),), ((3, 5),), (),
        (), ((4, 7),), ((5, 11.2),), ((6, 12.6),), ((7, 11.2),),
        ((8, 7),), (),
    ),
    (
        (), (), (), ((0, 3),),
        ((1, 5),), (), ((3, -5/3),), ((2, 5),), ((3, 5),),
        ((4, 10.5),), ((5, 7),), ((4, -0.7),), ((7, -4.2),),
        ((8, -0.7), (6, 8.4)), ((7, 7),), ((8, 10.5),),
    ),
)


def _coefficient(signed_square):
    return math.copysign(math.sqrt(abs(signed_square)), signed_square)


def cuda_helper(ambient_only=False):
    """Return C++ after Scalar is defined, before an XL R0/R1 kernel definition.

    Call cooperative_gradients(Y, physical_radius, lane, scratch) across a
    16-lane subgroup or a warp and synchronize before consuming scratch[48].
    ambient_only returns the homogeneous Cartesian derivatives divided by r.
    Its caller must project the accumulated edge derivative onto the tangent
    plane before using it as a Cartesian derivative.
    """
    declarations = []
    reductions = []
    for axis, row in enumerate(ROWS):
        contributions = []
        for tap in range(max(map(len, row))):
            entries = [terms[tap] if tap < len(terms) else (0, 0) for terms in row]
            # Four-bit indexes fit all 16 lanes in a single literal.
            packed_index = sum(index << (4 * lane) for lane, (index, _) in enumerate(entries))
            name = f"mdlab_lower_{axis}_{tap}"
            coeffs = ", ".join(format(_coefficient(value), ".17g") for _, value in entries)
            declarations.append(f"__device__ Scalar {name}[16] = {{{coeffs}}};")
            selector = f"int((0x{packed_index:016x}ULL >> (4*lane)) & 15)"
            contributions.append(f"{name}[lane] * __shfl_sync(mask, y, {selector}, 16)")
        terms = " + ".join(contributions)
        component = f"""
    const Scalar solid_{axis} = {terms};
"""
        if ambient_only:
            component += f"    scratch[{axis*16}+lane] = solid_{axis} * inverse_radius;"
        else:
            component += f"""
    const Scalar unit_{axis} = __shfl_sync(mask, y, {axis+1}, 16) * Scalar(0.57735026918962576451);
    scratch[{axis*16}+lane] = (solid_{axis} - l_y * unit_{axis}) * inverse_radius;"""
        reductions.append(component)
    return "\n".join(declarations) + """
__device__ __forceinline__ void cooperative_gradients(
    const Scalar* Y, double radius, int lane, Scalar* scratch) {
    if (lane >= 16) return;
    const unsigned mask = __activemask();
    const Scalar y = Y[lane];
    const int l = int(lane >= 1) + int(lane >= 4) + int(lane >= 9);
    const Scalar l_y = Scalar(l) * y;
    const Scalar inverse_radius = Scalar(1) / Scalar(radius);
""" + "\n".join(reductions) + "\n}\n"
