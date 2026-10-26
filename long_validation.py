"""Long-horizon empirical validation with multiprecision cross-checked references.

No interval enclosure, hardware execution, or quantum speedup is claimed.
Canonical QAE is sampled from its discrete law; no Gaussian surrogate is used.
"""
import argparse
import csv
import json
from pathlib import Path
import time
import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.integrate import solve_ivp
from carleman import Lorenz, Carleman, taylor_data, qae_rejection_samples


def write_csv(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def mp_parameters(model):
    # Same exact binary64 parameter values as the existing implementation.
    # In particular beta is not silently replaced by exact rational 8/3.
    return [mp.mpf(model.sigma), mp.mpf(model.rho), mp.mpf(model.beta)]


def taylor_coefficients(u, degree, parameters):
    sigma, rho, beta = parameters
    x, y, z = [u[0]], [u[1]], [u[2]]
    for k in range(degree):
        xz = sum(x[j]*z[k-j] for j in range(k+1))
        xy = sum(x[j]*y[k-j] for j in range(k+1))
        x.append(sigma*(y[k]-x[k])/(k+1))
        y.append((rho*x[k]-y[k]-xz)/(k+1))
        z.append((xy-beta*z[k])/(k+1))
    return [list(reversed(a)) for a in (x, y, z)]


def mp_reference(model, u0, end, sample_h, segment_steps, degree, digits):
    """Return mp samples from direct ODE Taylor series with dense evaluation."""
    steps = round(end/sample_h)
    result = []
    with mp.workdps(digits):
        dt = mp.mpf(str(sample_h))
        parameters = mp_parameters(model)
        u = [mp.mpf(float(a)) for a in u0]
        result.append(tuple(u))
        for start in range(0, steps, segment_steps):
            length = min(segment_steps, steps-start)
            coefficients = taylor_coefficients(u, degree, parameters)
            for j in range(1, length+1):
                result.append(tuple(mp.polyval(a, j*dt) for a in coefficients))
            u = list(result[-1])
    return result


def first_exit(times, errors, threshold):
    indices = np.flatnonzero(np.asarray(errors) > threshold)
    return float(times[indices[0]]) if len(indices) else None


def ideal_path(model, u0, N, h, end):
    c = Carleman(model, N)
    # Compute expm-I using a direct Taylor sum to avoid cancellation of diagonal
    # increments. This matches the increment readout convention.
    E, weights, alpha, tail = taylor_data(c.C, h, tol=2e-15)
    term = np.eye(len(c.C))
    increment = np.zeros_like(term)
    for k in range(1, len(weights)):
        term = term @ (h*c.C)/k
        increment += term
    rows = increment[c.linear]
    path = [np.asarray(u0, dtype=float)]
    for _ in range(round(end/h)):
        u = path[-1]
        v = model.normalize(u)
        path.append(u+model.scale*(rows @ c.lift(v)))
    return np.array(path), c, rows, weights, alpha, tail


def mp_increment_rows(model, N, h, digits=70, tolerance='1e-65'):
    """Sparse row Taylor recurrence, no dense multiprecision matrix expm.

    Coefficients are assembled from exact representations of the model's
    binary64 parameters, not by converting an already rounded float C_N.
    """
    with mp.workdps(digits):
        system = Carleman(model, N)
        exponents = system.exponents
        index = {a:i for i,a in enumerate(exponents)}
        sigma, rho, beta = mp_parameters(model)
        S = mp.mpf(model.scale)
        polys = [
            {(1,0,0):-sigma, (0,1,0):sigma},
            {(1,0,0):-sigma, (0,1,0):-mp.mpf(1), (1,0,1):-S},
            {(1,1,0):S, (0,0,1):-beta, (0,0,0):-beta*(rho+sigma)/S}]
        entries = {}
        for i,a in enumerate(exponents):
            for j,poly in enumerate(polys):
                if not a[j]: continue
                reduced=list(a); reduced[j]-=1
                for power,coef in poly.items():
                    target=tuple(reduced[k]+power[k] for k in range(3))
                    if target in index:
                        key=(i,index[target])
                        entries[key]=entries.get(key,mp.mpf(0))+a[j]*coef
        row_norm=[mp.mpf(0)]*len(index);col_norm=row_norm.copy()
        for (i,j),v in entries.items():
            row_norm[i]+=abs(v);col_norm[j]+=abs(v)
        alpha=mp.sqrt(max(row_norm)*max(col_norm))
        a=alpha*mp.mpf(str(h));weight=mp.mpf(1)
        term=[[mp.mpf(int(i==j)) for i in range(len(index))] for j in system.linear]
        total=[[mp.mpf(0) for _ in index] for _ in range(3)]
        dt=mp.mpf(str(h));tol=mp.mpf(tolerance)
        for k in range(1,400):
            next_term=[[mp.mpf(0) for _ in index] for _ in range(3)]
            for (i,j),coef in entries.items():
                for coord in range(3):
                    if term[coord][i]: next_term[coord][j]+=term[coord][i]*coef
            for coord in range(3):
                for j in range(len(index)):
                    next_term[coord][j]*=dt/k
                    total[coord][j]+=next_term[coord][j]
            term=next_term;weight*=a/k
            ratio=a/(k+2)
            tail=weight*a/(k+1)/(1-ratio) if ratio<1 else mp.inf
            if tail<tol:
                return total, exponents, k, mp.nstr(tail,20)
        raise RuntimeError('MP Carleman Taylor truncation failed')


def mp_carleman_path(model,u0,N,h,end,digits=70):
    with mp.workdps(digits):
        rows,exponents,K,tail=mp_increment_rows(model,N,h,digits)
        S=mp.mpf(model.scale);center=[mp.mpf(float(a)) for a in model.center]
        u=[mp.mpf(float(a)) for a in u0];path=[tuple(u)]
        for _ in range(round(end/h)):
            v=[(u[j]-center[j])/S for j in range(3)]
            powers=[[mp.mpf(1)] for _ in range(3)]
            for j in range(3):
                for k in range(N):powers[j].append(powers[j][-1]*v[j])
            phi=[powers[0][a]*powers[1][b]*powers[2][c] for a,b,c in exponents]
            u=[u[j]+S*sum(coef*x for coef,x in zip(rows[j],phi)) for j in range(3)]
            path.append(tuple(u))
        return path, K, tail


def validate(end=30.,out=Path('results/longtime'),paths=16,quick=False):
    out.mkdir(parents=True,exist_ok=True)
    start_clock=time.perf_counter()
    h_ref=.001; steps=round(end/h_ref)
    if abs(steps*h_ref-end)>1e-10 or end<=0:
        raise ValueError('horizon must be a positive multiple of .001')
    if not quick and abs(round(end/.01)*.01-end)>1e-10:
        raise ValueError('Full-mode horizon must be a multiple of .01')
    if paths<1:raise ValueError('paths must be positive')
    model=Lorenz()
    initials={'standard':np.ones(3),'attractor':np.array([-8.,8.,27.])}
    if quick:initials={'standard':initials['standard']}
    reference_info=[];rows=[];time_series={};survival_rows=[];resources=[]
    configs=[(N,h) for N in ([3,5] if quick else [3,5,7,9]) for h in ([.001] if quick else [.01,.005,.001])]
    budgets=[16,24] if quick else [16,24,32,40]
    qae_N=5 if quick else 9
    for initial_id,u0 in initials.items():
        print(f'{initial_id}: constructing 70-digit reference...',flush=True)
        coarse=mp_reference(model,u0,end,h_ref,10,50,70)
        print(f'{initial_id}: cross-checking at 90 digits, degree 60, half segment...',flush=True)
        refined=mp_reference(model,u0,end,h_ref,5,60,90)
        with mp.workdps(95):
            differences=[mp.sqrt(sum((a-b)**2 for a,b in zip(x,y))) for x,y in zip(coarse,refined)]
            max_diff=max(differences)
            ref_error=mp.nstr(max_diff,25)
        truth=np.array([[float(x) for x in u] for u in refined])
        times=np.arange(steps+1)*h_ref
        reference_info.append(dict(initial_id=initial_id,initial_state=u0.tolist(),
            digits=[70,90],orders=[50,60],segments=[.01,.005],sample_step=h_ref,
            max_mp_refinement_difference=ref_error,
            exact_float_beta=repr(model.beta),
            beta_exact_mp=mp.nstr(mp.mpf(model.beta),60),
            final_reference=[mp.nstr(x,75) for x in refined[-1]]))
        print(f'{initial_id}: maximum reference refinement difference {ref_error}',flush=True)
        np.savez_compressed(out/f'reference_{initial_id}.npz',times=times,state=truth)
        for max_step in [.01,.005]:
            solver=solve_ivp(model.rhs,(0,end),u0,method='DOP853',t_eval=times,
                              rtol=2.3e-14,atol=2e-14,max_step=max_step)
            if not solver.success:raise RuntimeError(solver.message)
            error=np.linalg.norm(solver.y.T-truth,axis=1)
            label=f'DOP853 maxstep={max_step}'
            time_series[(initial_id,label)]=(times,error)
            rows.append(dict(initial_id=initial_id,method='DOP853',N=0,h=max_step,
                arithmetic='binary64',max_error=float(error.max()),final_error=float(error[-1]),
                **{f'first_exit_{tol:g}':first_exit(times,error,tol) for tol in [1e-10,1e-8,1e-6,1e-3]}))
        for N,h in configs:
            if round(end/h)*h>end+1e-10:continue
            path,c,increment,w,alpha,tail=ideal_path(model,u0,N,h,end)
            stride=round(h/h_ref);sampled=truth[::stride][:len(path)]
            t=np.arange(len(path))*h
            error=np.linalg.norm(path-sampled,axis=1)
            label=f'N={N}, h={h:g}'
            time_series[(initial_id,label)]=(t,error)
            rows.append(dict(initial_id=initial_id,method='ideal_relifted',N=N,h=h,
                arithmetic='binary64',max_error=float(error.max()),final_error=float(error[-1]),
                **{f'first_exit_{tol:g}':first_exit(t,error,tol) for tol in [1e-10,1e-8,1e-6,1e-3]}))
            print(f'{initial_id}: {label}, maximum error {error.max():.3e}',flush=True)
        mp_N=5 if quick else 9
        mp_h=.001 if quick else .002
        print(f'{initial_id}: multiprecision Carleman N={mp_N}, h={mp_h}...',flush=True)
        high,Kmp,tailmp=mp_carleman_path(model,u0,mp_N,mp_h,end,digits=70)
        stride=round(mp_h/h_ref);t=np.arange(len(high))*mp_h
        with mp.workdps(90):
            exact_errors=[mp.sqrt(sum((a-b)**2 for a,b in zip(u,refined[i*stride]))) for i,u in enumerate(high)]
        error=np.array([float(e) for e in exact_errors])
        time_series[(initial_id,'MP70 Carleman')]=(t,error)
        rows.append(dict(initial_id=initial_id,method='ideal_relifted',N=mp_N,h=mp_h,
            arithmetic='mp70',max_error=float(error.max()),final_error=float(error[-1]),
            **{f'first_exit_{tol:g}':first_exit(t,error,tol) for tol in [1e-10,1e-8,1e-6,1e-3]}))
        reference_info[-1]['mp_carleman_K']=Kmp
        reference_info[-1]['mp_carleman_tail_bound']=tailmp
        print(f'{initial_id}: MP70 maximum error {error.max():.3e}',flush=True)
        # QAE noise follows the exact discrete phase-estimation law. Paths stop
        # after first exit from error .1; survival at stricter thresholds remains
        # valid. This is empirical survival, not an analytic confidence guarantee.
        _,c,increment,w,alpha,tail=ideal_path(model,u0,qae_N,h_ref,h_ref)
        exponents=np.array(c.exponents);gamma=float(w[1:].sum());K=len(w)-1
        for bits in budgets:
            print(f'{initial_id}: QAE {bits} bits, {paths} paths...',flush=True)
            rng=np.random.default_rng(20261026+bits+(0 if initial_id=='standard' else 1000))
            states=np.tile(u0,(paths,1));peak=np.zeros(paths);active=np.ones(paths,dtype=bool)
            calls=7*(2*2**bits-1);executed_steps=0
            for n in range(1,steps+1):
                ids=np.flatnonzero(active)
                if ids.size:
                    v=model.normalize(states[ids])
                    phi=np.prod(v[:,None,:]**exponents[None,:,:],axis=2)
                    B=np.linalg.norm(phi,axis=1)
                    amplitude=(phi@increment.T)/(B[:,None]*gamma)
                    p=(1+amplitude)/2
                    if np.any((p < -1e-10)|(p > 1+1e-10)):
                        raise RuntimeError('Invalid Hadamard probability')
                    estimates=qae_rejection_samples(np.clip(p,0,1),bits,7,rng)
                    states[ids]+=model.scale*B[:,None]*gamma*(2*estimates-1)
                    error=np.linalg.norm(states[ids]-truth[n],axis=1)
                    peak[ids]=np.maximum(peak[ids],error)
                    active[ids[peak[ids]>.1]]=False
                    executed_steps+=ids.size
                if n==1 or n%10==0 or n==steps:
                    for tol in [1e-3,1e-2]:
                        survival_rows.append(dict(initial_id=initial_id,phase_bits=bits,
                            time=float(times[n]),threshold=tol,paths=paths,
                            survival_fraction=float(np.mean(peak<=tol)),
                            active_paths=int(active.sum())))
                if not active.any():
                    # All paths already failed both reported thresholds.
                    for nn in range(n+1,steps+1):
                        if nn%10==0 or nn==steps:
                            for tol in [1e-3,1e-2]:
                                survival_rows.append(dict(initial_id=initial_id,phase_bits=bits,
                                    time=float(times[nn]),threshold=tol,paths=paths,
                                    survival_fraction=0.,active_paths=0))
                    break
            resources.append(dict(initial_id=initial_id,phase_bits=bits,paths=paths,
                N=qae_N,h=h_ref,K=K,gamma=gamma,repeats=7,
                A_calls_per_coordinate_step=calls,
                A_calls_per_full_trajectory=3*steps*calls,
                base_calls_per_full_trajectory=K*3*steps*calls,
                executed_path_steps=executed_steps,
                simulated_equivalent_A_calls=3*executed_steps*calls,
                arithmetic='binary64 probability model',stop_error=.1))
            print(f'{initial_id}: QAE {bits} bits complete, active={active.sum()}',flush=True)
        # Save intermediate checkpoints so a long run remains reviewable.
        write_csv(out/'deterministic.csv',rows)
        write_csv(out/'qae_survival.csv',survival_rows)
        write_csv(out/'qae_resources.csv',resources)
        (out/'reference_checks.json').write_text(json.dumps(reference_info,indent=2),encoding='utf-8')
        del coarse,refined
    plot_results(out,time_series,survival_rows,initials)
    summary=dict(horizon=end,reference_sample_step=h_ref,
        initial_ids=list(initials),qae_paths_per_setting=paths,
        qae_order=qae_N,qae_h=h_ref,qae_phase_bits=budgets,
        elapsed_seconds=time.perf_counter()-start_clock,
        caveats=['Empirical refinement checks, not interval-certified enclosures.',
            'Model beta is the exact value of its stored binary64 parameter.',
            'High-precision Carleman is an ideal arithmetic benchmark, not a quantum circuit.',
            'QAE uses exact-law classical rejection sampling; quantum query counts remain exponential in phase_bits.',
            'QAE paths stop after prefix error .1; survival <=.01 remains well defined.',
            'No hardware gate noise, fault tolerance, or quantum speedup is included.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2),flush=True)


def plot_results(out,time_series,survival,initials):
    fig,axes=plt.subplots(len(initials),1,figsize=(12,4*len(initials)),squeeze=False)
    for ax,initial_id in zip(axes[:,0],initials):
        for (name,label),(t,e) in time_series.items():
            if name!=initial_id:continue
            if label.startswith('N=') and ', h=0.001' not in label:continue
            style='--' if label.startswith('DOP') else '-'
            ax.semilogy(t[1:],np.maximum(e[1:],1e-35),style,label=label,linewidth=1.2)
        ax.axhline(1e-8,color='gray',linestyle=':',label='1e-8 threshold')
        ax.set(title=f'{initial_id}: same-initial-state trajectory error',xlabel='Time',ylabel='Physical vector error')
        ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
    fig.tight_layout();fig.savefig(out/'ideal_longtime.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(len(initials),2,figsize=(12,4*len(initials)),squeeze=False)
    for index,name in enumerate(initials):
        for column,tol in enumerate([1e-3,1e-2]):
            ax=axes[index,column]
            for bits in sorted(set(r['phase_bits'] for r in survival)):
                selected=[r for r in survival if r['initial_id']==name and r['phase_bits']==bits and r['threshold']==tol]
                ax.plot([r['time'] for r in selected],[r['survival_fraction'] for r in selected],label=f'QAE {bits} bits')
            ax.set(title=f'{name}: prefix error <= {tol:g}',xlabel='Time',ylabel='Empirical survival fraction',ylim=(-.03,1.03))
            ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'qae_longtime.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--horizon',type=float,default=30.)
    parser.add_argument('--paths',type=int,default=16)
    parser.add_argument('--out',type=Path,default=Path('results/longtime'))
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args()
    validate(args.horizon,args.out,args.paths,args.quick)
