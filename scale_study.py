"""Compare coordinate scaling without assuming that a smaller lifted norm wins.

This is a deterministic matrix/resource study, not hardware benchmarking.
It does not allocate the exponentially large product-LCU statevector.
"""
import argparse
import csv
from math import factorial
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from carleman import Lorenz, Carleman, taylor_data, reference


def study(order=3, h=.001, out=Path('results'), radius=44.):
    if order < 1 or h <= 0 or radius < 44:
        raise ValueError('Require order>=1, h>0 and radius>=44')
    out.mkdir(parents=True, exist_ok=True)
    u0 = np.ones(3)
    ref_system = Carleman(Lorenz(scale=64.), order)
    ref_phi = ref_system.lift(ref_system.model.normalize(u0))
    ref_next = ref_system.exact_map(u0, h)
    ode = reference(ref_system.model, u0, [h])[0]
    degrees = np.array([sum(a) for a in ref_system.exponents])
    rows = []
    for S in [16., 24., 32., 38., 40., 44., 48., 64., 96., 128., 256.]:
        model = Lorenz(scale=S)
        system = Carleman(model, order)
        v = model.normalize(u0)
        phi = system.lift(v)
        B = np.linalg.norm(phi)
        E, weights, alpha, tail = taylor_data(system.C, h)
        gamma = float(weights[1:].sum())
        factor = S*B*gamma
        # Finite-order norm bound valid also when radius/S >= 1.
        r = radius/S
        B_bound = np.sqrt(sum(r**(2*k) for k in range(order+1)))
        # Missing coupling from homogeneous degree N+1 into degree N.
        extended = Carleman(model, order+1)
        top = np.flatnonzero(degrees == order)
        next_layer = np.array([i for i, a in enumerate(extended.exponents)
                               if sum(a) == order+1])
        injection = extended.C[np.ix_(top, next_layer)]
        beta_N = np.linalg.norm(injection, 2)
        truncation_bound = (S*beta_N*r**(order+1)*np.exp(alpha*h)
                            *alpha**(order-1)*h**order/factorial(order))
        diagonal = (64./S)**degrees
        similar = diagonal[:, None]*ref_system.C/diagonal[None, :]
        invariant_next = system.exact_map(u0, h)
        rows.append(dict(N=order, h=h, S=S, radius=radius,
            normalized_input_norm=float(np.linalg.norm(v)),
            normalized_domain_radius=r, B=float(B), B_domain_bound=float(B_bound),
            alpha=float(alpha), alpha_h=float(alpha*h), K=len(weights)-1,
            gamma_increment=gamma, gamma_full=1+gamma,
            physical_readout_factor=float(factor),
            domain_readout_factor_bound=float(S*B_bound*gamma),
            base_query_prefactor=float((len(weights)-1)*factor),
            sampling_prefactor=float(factor**2),
            domain_truncation_bound=float(truncation_bound),
            deterministic_error=float(np.linalg.norm(invariant_next-ode)),
            scale_invariance_error=float(np.linalg.norm(invariant_next-ref_next)),
            similarity_error=float(np.linalg.norm(system.C-similar)),
            lift_similarity_error=float(np.linalg.norm(phi-diagonal*ref_phi)),
            taylor_operator_tail_bound=float(tail)))
    with open(out/'scale_study.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    scales = [r['S'] for r in rows]
    for key, label in [('alpha','alpha'),('gamma_increment','Gamma increment')]:
        axes[0].loglog(scales, [r[key] for r in rows], 'o-', label=label)
    axes[0].set(title='Operator normalization', xlabel='S')
    for key, label in [('B','B at input'),('B_domain_bound','finite-N domain bound')]:
        axes[1].loglog(scales, [r[key] for r in rows], 'o-', label=label)
    axes[1].set(title='Lift normalization', xlabel='S')
    for key, label in [('physical_readout_factor','S B Gamma at input'),
                       ('domain_readout_factor_bound','domain bound')]:
        axes[2].loglog(scales, [r[key] for r in rows], 'o-', label=label)
    axes[2].set(title='Physical query prefactor', xlabel='S')
    for ax in axes:
        ax.axvline(radius, color='gray', linestyle=':', label='domain radius' if ax is axes[2] else None)
        ax.grid(True, which='both', alpha=.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out/'scale_study.png', dpi=160)
    plt.close(fig)
    for row in rows:
        print(f"S={row['S']:6.1f} B={row['B']:.5f} alpha={row['alpha']:.5f} "
              f"Gamma={row['gamma_increment']:.6f} F={row['physical_readout_factor']:.5f} "
              f"K={row['K']:2d} invariant_error={row['scale_invariance_error']:.2e}")
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--order', type=int, default=3)
    parser.add_argument('--h', type=float, default=.001)
    parser.add_argument('--out', type=Path, default=Path('results'))
    args = parser.parse_args()
    study(args.order, args.h, args.out)
