"""Higher-order ideal arithmetic and high-precision QAE windows.

Run after long_validation.py. References are rounded to binary64 for this
extension, so errors near 1e-14 are reference-format limited. No interval or
hardware certification is implied.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import mpmath as mp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from carleman import Lorenz,qae_rejection_samples
from long_validation import mp_carleman_path,ideal_path,write_csv,first_exit


def extend(out=Path('results/longtime'),paths=16):
    model=Lorenz();h=.001
    ids=[p.stem.removeprefix('reference_') for p in sorted(out.glob('reference_*.npz'))]
    if not ids:raise RuntimeError('Run long_validation.py first')
    statefile=out/'extension_state.json'
    previous=json.loads(statefile.read_text()) if statefile.exists() else {}
    deterministic=previous.get('deterministic',[])
    exits=previous.get('exits',[]);settings=previous.get('settings',[]);curves={}
    budgets=[24,32,40,44];thresholds=[1e-8,1e-6]
    for name in ids:
        data=np.load(out/f'reference_{name}.npz');times=data['times'];truth=data['state']
        end=float(times[-1]);u0=truth[0];steps=len(times)-1
        cache=out/f'high_order_{name}.npz'
        if any(r['initial_id']==name and r.get('horizon')==end for r in deterministic) and any(r['initial_id']==name and r['paths']==paths for r in settings) and cache.exists():
            saved=np.load(cache);curves[name]=(saved['times'],saved['errors'])
            print(f'{name}: reusing completed extension checkpoint',flush=True)
            continue
        deterministic=[r for r in deterministic if r['initial_id']!=name]
        exits=[r for r in exits if r['initial_id']!=name]
        settings=[r for r in settings if r['initial_id']!=name]
        print(f'{name}: higher-order MP70 Carleman N=13 h=.002...',flush=True)
        high,K,tail=mp_carleman_path(model,u0,13,.002,end,digits=70)
        with mp.workdps(90):
            errors=np.array([float(mp.sqrt(sum((a-mp.mpf(float(b)))**2 for a,b in zip(u,truth[2*i])))) for i,u in enumerate(high)])
        t=times[::2][:len(high)];curves[name]=(t,errors)
        np.savez_compressed(cache,times=t,errors=errors)
        deterministic.append(dict(initial_id=name,N=13,h=.002,horizon=end,arithmetic='mp70',
            reference_format='binary64-rounded MP90 reference',
            reference_rounding_scale=float(np.sqrt(3)*np.max(np.abs(truth))*np.finfo(float).eps),
            maximum_error=float(errors.max()),final_error=float(errors[-1]),
            first_exit_1e_10=first_exit(t,errors,1e-10),
            first_exit_1e_8=first_exit(t,errors,1e-8),K=K,tail=tail))
        print(f'{name}: N=13 max error {errors.max():.3e}',flush=True)
        _,c,increment,w,alpha,_=ideal_path(model,u0,9,h,h)
        exponents=np.array(c.exponents);gamma=float(w[1:].sum())
        for bits in budgets:
            print(f'{name}: high-precision QAE {bits} bits...',flush=True)
            rng=np.random.default_rng(20261027+bits+(0 if name=='standard' else 1000))
            states=np.tile(u0,(paths,1));active=np.ones(paths,dtype=bool)
            first=np.full((paths,len(thresholds)),np.nan);peak=np.zeros(paths)
            executed=0
            for n in range(1,steps+1):
                indices=np.flatnonzero(active)
                if not indices.size:break
                v=model.normalize(states[indices])
                phi=np.prod(v[:,None,:]**exponents[None,:,:],axis=2)
                B=np.linalg.norm(phi,axis=1)
                p=(1+(phi@increment.T)/(B[:,None]*gamma))/2
                estimates=qae_rejection_samples(np.clip(p,0,1),bits,7,rng)
                states[indices]+=model.scale*B[:,None]*gamma*(2*estimates-1)
                errors=np.linalg.norm(states[indices]-truth[n],axis=1)
                peak[indices]=np.maximum(peak[indices],errors)
                for column,tol in enumerate(thresholds):
                    hit=indices[(peak[indices]>tol)&np.isnan(first[indices,column])]
                    first[hit,column]=times[n]
                active[indices[peak[indices]>max(thresholds)]]=False
                executed+=len(indices)
            for trajectory in range(paths):
                for column,tol in enumerate(thresholds):
                    when=first[trajectory,column]
                    exits.append(dict(initial_id=name,phase_bits=bits,path=trajectory,
                        threshold=tol,first_exit=None if np.isnan(when) else float(when),
                        observation_end=end,censored=bool(np.isnan(when))))
            factor=7*(2*2**bits-1)
            settings.append(dict(initial_id=name,phase_bits=bits,paths=paths,N=9,h=h,
                repeats=7,K=len(w)-1,A_calls_per_coordinate_step=factor,
                A_calls_per_full_trajectory=3*steps*factor,
                base_calls_per_full_trajectory=3*steps*factor*(len(w)-1),
                simulated_path_steps=executed,stop_threshold=1e-6))
            event_times=np.where(np.isnan(first),np.inf,first)
            print(f'{name}: {bits} bits median first exits '+str([
                None if not np.isfinite(np.median(event_times[:,j])) else float(np.median(event_times[:,j]))
                for j in range(len(thresholds))]),flush=True)
        write_csv(out/'high_precision_ideal.csv',deterministic)
        write_csv(out/'high_precision_qae_exits.csv',exits)
        write_csv(out/'high_precision_qae_resources.csv',settings)
        statefile.write_text(json.dumps(dict(deterministic=deterministic,exits=exits,settings=settings),indent=2))
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for name,(t,e) in curves.items():
        axes[0].semilogy(t[1:],e[1:],label=f'{name}, MP70 N=13')
    axes[0].axhline(1e-10,color='gray',linestyle=':',label='1e-10')
    axes[0].set(xlabel='Time',ylabel='Physical vector error',title='Ideal arithmetic; reference rounding floor ~1e-14')
    axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    # Individual first exits are retained in CSV; draw empirical survival at 1e-8.
    for name in ids:
        for bits in [32,40,44]:
            selected=[r for r in exits if r['initial_id']==name and r['phase_bits']==bits and r['threshold']==1e-8]
            events=np.array([np.inf if r['first_exit'] is None else r['first_exit'] for r in selected])
            grid=np.linspace(0,selected[0]['observation_end'],1001)
            axes[1].plot(grid,[np.mean(events>t) for t in grid],label=f'{name}, {bits} bits')
    axes[1].set(xlabel='Time',ylabel='Empirical survival fraction',title='QAE: prefix vector error <=1e-8',ylim=(-.03,1.03))
    axes[1].legend(fontsize=7);axes[1].grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'high_precision.png',dpi=160);plt.close(fig)
    (out/'extension_summary.json').write_text(json.dumps(dict(
        ideal_N=13,ideal_h=.002,ideal_digits=70,qae_N=9,qae_h=.001,
        qae_paths=paths,phase_bits=budgets,thresholds=thresholds,
        interpretation='Empirical same-initial-state trajectories, not rigorous enclosures or quantum hardware.',
        reference_note='MP90 reference rounded to binary64; do not interpret errors below its rounding scale.'),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('results/longtime'))
    parser.add_argument('--paths',type=int,default=16)
    args=parser.parse_args()
    extend(args.out,args.paths)
