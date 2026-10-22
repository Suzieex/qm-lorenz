"""Reproducible single-step and supplementary relifted-trajectory experiments."""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from carleman import (Lorenz, Carleman, DenseLCUCircuit, ProductLCUCircuit, reference,
                      taylor_data, qae_samples)


def save_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(out, seed=20261022, trials=300):
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    model = Lorenz()
    u0 = np.array([1., 1., 1.])
    order, h = 3, 0.001
    system = Carleman(model, order)
    increment = ProductLCUCircuit(system, h, increment=True)
    dense_increment = DenseLCUCircuit(system, h, increment=True)
    full = DenseLCUCircuit(system, h, increment=False)
    p_inc, B, norm_error = increment.hadamard_probabilities(u0)
    p_dense, _, _ = dense_increment.hadamard_probabilities(u0)
    p_full, _, _ = full.hadamard_probabilities(u0)
    recovered = increment.recover(u0, p_inc, B)
    exact_carleman = system.exact_map(u0, h)
    ode = reference(model, u0, [h])[0]
    outstate, _ = increment.output(u0)
    rngstate = rng.normal(size=outstate.size)
    rngstate /= np.linalg.norm(rngstate)
    inverse_error = np.linalg.norm(increment.apply(increment.apply(rngstate), True)-rngstate)
    rows = []
    for mode, circuit, probabilities in [('increment', increment, p_inc),
                                          ('full', full, p_full)]:
        factor = model.scale*B*circuit.gamma
        for phase_bits in [8, 10, 12, 14, 16]:
            repeats = 7
            calls = repeats*(2*(2**phase_bits-1)+1)
            # Same A/A^dagger-call budget per coordinate for shots and QAE.
            sample_p = rng.binomial(calls, probabilities, size=(trials, 3))/calls
            ae_p = np.stack([qae_samples(p, phase_bits, repeats, trials, rng)
                             for p in probabilities], axis=1)
            for method, estimates in [('shots', sample_p), ('qae', ae_p)]:
                physical_errors = factor*2*(estimates-probabilities)
                rows.append(dict(mode=mode, method=method,
                    phase_bits=phase_bits if method == 'qae' else 0,
                    calls_per_coordinate=calls, calls_all_coordinates=3*calls,
                    trials=trials,
                    vector_rmse=float(np.sqrt(np.mean(np.sum(physical_errors**2, axis=1)))),
                    median_vector_error=float(np.median(np.linalg.norm(physical_errors, axis=1))),
                    p95_vector_error=float(np.quantile(np.linalg.norm(physical_errors, axis=1), .95))))
    save_csv(out/'readout.csv', rows)

    conv = []
    hs = [.0005, .001, .002, .004, .008, .016]
    for n in [1, 2, 3, 4, 5]:
        c = Carleman(model, n)
        for step in hs:
            ref = reference(model, u0, [step])[0]
            E, weights, alpha, tail = taylor_data(c.C, step)
            phi = c.lift(model.normalize(u0))
            y = E @ phi
            approximate = model.restore(y[c.linear])
            exact = c.exact_map(u0, step)
            gamma_inc, gamma_full = float(weights[1:].sum()), float(weights.sum())
            conv.append(dict(order=n, h=step, dimension=len(c.C),
                carleman_error=float(np.linalg.norm(exact-ref)),
                taylor_error=float(np.linalg.norm(approximate-exact)),
                combined_error=float(np.linalg.norm(approximate-ref)),
                alpha=alpha, alpha_h=alpha*step, K=len(weights)-1,
                gamma_increment=gamma_inc, gamma_full=gamma_full,
                ideal_query_ratio=gamma_full/gamma_inc,
                taylor_operator_bound=tail))
    save_csv(out/'convergence.csv', conv)

    # Supplemental multi-step test: readout uses the exact binomial law of the
    # validated Hadamard circuit, not statevector evolution for every shot.
    end, shots = .05, 1_000_000
    times = np.arange(round(end/h)+1)*h
    truth = reference(model, u0, times)
    E = increment.E
    D = E-np.eye(len(E))
    deterministic = [u0.copy()]
    trajectories = []
    for _ in range(20):
        path = [u0.copy()]
        for _ in times[1:]:
            v = model.normalize(path[-1])
            phi = system.lift(v)
            norm = np.linalg.norm(phi)
            amplitude = (D @ phi)[system.linear]/(norm*increment.gamma)
            probabilities = (1+amplitude)/2
            if np.any((probabilities < -1e-12) | (probabilities > 1+1e-12)):
                raise RuntimeError('Invalid circuit probability')
            sample = rng.binomial(shots, np.clip(probabilities, 0, 1))/shots
            vnext = v+norm*increment.gamma*(2*sample-1)
            path.append(model.restore(vnext))
        trajectories.append(path)
    for _ in times[1:]:
        phi = system.lift(model.normalize(deterministic[-1]))
        deterministic.append(model.restore((E @ phi)[system.linear]))
    deterministic = np.array(deterministic)
    trajectories = np.array(trajectories)
    np.savez_compressed(out/'trajectory.npz', times=times, reference=truth,
                        ideal_relifted=deterministic, shot_paths=trajectories)

    # Classical timing is a local CPU baseline, never a hardware speedup claim.
    timing = []
    lift0 = system.lift(model.normalize(u0))
    t = time.perf_counter()
    target_rows = D[system.linear].copy()
    precompute_seconds = time.perf_counter()-t
    for label in ['three_rows', 'dense_full']:
        samples = []
        for _ in range(5):
            t = time.perf_counter()
            for _ in range(2000):
                phi = system.lift(model.normalize(u0))
                _ = target_rows @ phi if label == 'three_rows' else D @ phi
            samples.append((time.perf_counter()-t)/2000)
        timing.append(dict(method=label, median_seconds=float(np.median(samples))))
    save_csv(out/'classical_timing.csv', timing)

    target_physical_error = 1e-3
    # Hoeffding: per-coordinate error <= target with total failure <= 0.05.
    # a_hat is a +/-1 sample mean, physical coordinate = S*B*gamma*a_hat.
    delta_a = target_physical_error/(model.scale*B*increment.gamma)
    shot_bound = int(np.ceil(2*np.log(6/.05)/delta_a**2))
    summary = dict(seed=seed, trials=trials, initial_state=u0.tolist(),
        order=order, h=h, scale=model.scale, center=model.center.tolist(),
        compressed_dimension=len(system.C), padded_data_dimension=increment.data_size,
        explicit_hadamard_circuit_qubits=increment.qubits_with_hadamard,
        dense_power_reference_qubits=dense_increment.qubits_with_hadamard,
        product_vs_dense_probability_error=float(np.max(np.abs(p_inc-p_dense))),
        base_block_calls_per_interference_circuit=increment.K,
        qae_extra_phase_qubits=[8, 10, 12, 14, 16],
        alpha=increment.alpha, alpha_h=increment.alpha*h, K=increment.K,
        gamma_increment=increment.gamma, gamma_full=full.gamma,
        B=B, readout_scale_increment=model.scale*B*increment.gamma,
        readout_scale_full=model.scale*B*full.gamma,
        query_ratio_full_over_increment=full.gamma/increment.gamma,
        sampling_variance_ratio_full_over_increment=(full.gamma/increment.gamma)**2,
        hadamard_probabilities_increment=p_inc.tolist(),
        circuit_state_normalization_error=norm_error,
        circuit_forward_inverse_error=float(inverse_error),
        ideal_circuit_next_state=recovered.tolist(),
        exact_carleman_next_state=exact_carleman.tolist(),
        ode_next_state=ode.tolist(),
        circuit_vs_carleman_error=float(np.linalg.norm(recovered-exact_carleman)),
        carleman_vs_ode_error=float(np.linalg.norm(exact_carleman-ode)),
        taylor_operator_tail_bound=increment.tail,
        hoeffding_target_physical_error_per_coordinate=target_physical_error,
        hoeffding_shots_per_coordinate=shot_bound,
        hoeffding_total_failure_probability=.05,
        trajectory_end=end, trajectory_shots_per_coordinate_per_step=shots,
        trajectory_shot_paths=len(trajectories),
        trajectory_ideal_final_error=float(np.linalg.norm(deterministic[-1]-truth[-1])),
        trajectory_shot_final_median_error=float(np.median(np.linalg.norm(trajectories[:, -1]-truth[-1], axis=1))),
        classical_target_rows_copy_seconds=precompute_seconds,
        limitations=['LCU uses products with fresh flags; the base UC and Householder lift are dense reference gates, not efficiently synthesized sparse circuits.',
                     'QAE is exact reduced-invariant-subspace QPE simulation; no quantum hardware was used.',
                     'Oracle call counts are not elementary gate counts or wall-time speedups.',
                     'Only quantum measurement sampling is noisy; gate noise and fault tolerance are excluded.',
                     'Trajectory demonstration is supplemental; no long-time or quantum-advantage claim.'])
    (out/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    for n in [1, 2, 3, 4, 5]:
        data = [r for r in conv if r['order'] == n]
        axes[0].loglog([r['h'] for r in data], [r['carleman_error'] for r in data], 'o-', label=f'N={n}')
    axes[0].set(xlabel='Step h', ylabel='Physical vector error', title='Ideal one-step Carleman vs DOP853')
    axes[0].legend()
    for mode, method, style in [('full','shots','o--'), ('increment','shots','o-'),
                                 ('full','qae','s--'), ('increment','qae','s-')]:
        data = [r for r in rows if r['mode']==mode and r['method']==method]
        axes[1].loglog([r['calls_per_coordinate'] for r in data], [r['vector_rmse'] for r in data], style, label=f'{mode} / {method}')
    axes[1].set(xlabel='A or A-dagger calls per coordinate', ylabel='Physical readout vector RMSE', title=f'Finite readout, {trials} trials')
    axes[1].legend(fontsize=8)
    data = [r for r in conv if r['order']==order]
    axes[2].loglog([r['h'] for r in data], [r['gamma_increment'] for r in data], 'o-', label='increment Gamma')
    axes[2].loglog([r['h'] for r in data], [r['gamma_full'] for r in data], 's-', label='full Gamma')
    axes[2].set(xlabel='Step h', ylabel='LCU normalization', title='Normalization is a real cost')
    axes[2].legend()
    for ax in axes:
        ax.grid(True, which='both', alpha=.25)
    fig.tight_layout()
    fig.savefig(out/'single_step.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for path in trajectories:
        axes[0].plot(times, path[:, 1], color='C1', alpha=.15)
        axes[1].semilogy(times[1:], np.linalg.norm(path-truth, axis=1)[1:], color='C1', alpha=.15)
    axes[0].plot(times, truth[:, 1], 'k-', label='DOP853')
    axes[0].plot(times, deterministic[:, 1], 'b--', label='ideal relifted')
    axes[0].set(xlabel='Time', ylabel='y', title='Supplement: finite-shot relifting')
    axes[0].legend()
    axes[1].semilogy(times[1:], np.linalg.norm(deterministic-truth, axis=1)[1:], 'b--', label='ideal relifted')
    axes[1].set(xlabel='Time', ylabel='Physical vector error', title=f'{shots:,} shots / coordinate / step')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out/'trajectory.png', dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=Path('results'))
    parser.add_argument('--seed', type=int, default=20261022)
    parser.add_argument('--trials', type=int, default=300)
    args = parser.parse_args()
    run(args.out, args.seed, args.trials)
