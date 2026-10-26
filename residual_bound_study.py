"""Compare certified rational bounds, numerical VOC estimates and observations.

Reference integration and quadrature never enter the certified bound. The
certificates concern ideal algebraic Carleman truncation, not floating matrix
exponentiation, quantum gates or statistical readout.
"""
import argparse,csv,json
from pathlib import Path
from fractions import Fraction as F
from math import factorial
import numpy as np
import mpmath as mp
from scipy.linalg import expm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from carleman import Lorenz,Carleman,reference
from long_validation import taylor_coefficients
from residual_bounds import certified_bounds,outward_decimal,rational_generator


def to_mp(x):return mp.mpf(x.numerator)/x.denominator if isinstance(x,F) else mp.mpf(x)


def numerical_observations(model,N,h,u0,data):
    """Independent high-precision ODE / matrix-series observations and quadrature."""
    with mp.workdps(100):
        dt=to_mp(F(str(h)));S=to_mp(F(float(model.scale)))
        sigma,rho,beta=[to_mp(F(float(x))) for x in [model.sigma,model.rho,model.beta]]
        center=[mp.mpf(0),mp.mpf(0),sigma+rho]
        u=[to_mp(F(float(x))) for x in u0]
        coarse=taylor_coefficients(u,80,[sigma,rho,beta])
        fine=taylor_coefficients(u,100,[sigma,rho,beta])
        truth=[mp.polyval(a,dt) for a in fine]
        ref_check=mp.sqrt(sum((mp.polyval(a,dt)-b)**2 for a,b in zip(coarse,truth)))
        v=[(a-b)/S for a,b in zip(u,center)]
        phi=[mp.fprod(v[j]**p for j,p in enumerate(a)) for a in data['exponents']]
        C=[{j:to_mp(x) for j,x in row.items()} for row in data['C']]
        term=[{j:mp.mpf(1)} for j in data['linear']]
        increment=[mp.mpf(0)]*3;alpha=to_mp(data['infinity_norm']);weight=mp.mpf(1)
        for k in range(1,400):
            next_term=[]
            for row in term:
                new={}
                for i,a in row.items():
                    for j,b in C[i].items():new[j]=new.get(j,mp.mpf(0))+a*b*dt/k
                next_term.append(new)
            term=next_term
            for j,row in enumerate(term):increment[j]+=sum(a*phi[i] for i,a in row.items())
            weight*=alpha*dt/k;ratio=alpha*dt/(k+2)
            tail=weight*alpha*dt/(k+1)/(1-ratio) if ratio<1 else mp.inf
            if tail<mp.mpf('1e-85'):break
        carleman=[a+S*b for a,b in zip(u,increment)]
        error_vector=[a-b for a,b in zip(truth,carleman)]
        observed=mp.sqrt(sum(a*a for a in error_vector))
        H=[[{a:to_mp(v) for a,v in row.items()} for row in H] for H in data['H']]
        def quadrature(order):
            nodes,weights=mp.gauss_quadrature(order,'legendre')
            signed=[mp.mpf(0)]*3;absolute=signed.copy()
            for node,weight in zip(nodes,weights):
                s=dt*(node+1)/2;tau=dt-s
                state=[(mp.polyval(a,s)-c)/S for a,c in zip(fine,center)]
                monomials=[mp.fprod(state[j]**p for j,p in enumerate(a)) for a in data['shell']]
                for j in range(3):
                    integrand=S*sum(tau**k/factorial(k)*sum(v*monomials[a] for a,v in Hk[j].items()) for k,Hk in enumerate(H))
                    signed[j]+=weight*dt/2*integrand
                    absolute[j]+=weight*dt/2*abs(integrand)
            return signed,mp.sqrt(sum(x*x for x in absolute))
        small,triangle16=quadrature(16);large,triangle32=quadrature(32)
        return dict(observed_ideal_mp_error=float(observed),
            observed_ideal_mp_error_decimal=mp.nstr(observed,40),
            reference_refinement_indicator=float(ref_check),
            voc_reconstruction_error=float(mp.sqrt(sum((a-b)**2 for a,b in zip(error_vector,large)))),
            numerical_triangle_estimate=float(triangle32),
            numerical_quadrature_change=float(abs(triangle32-triangle16)),
            observed_vector=[mp.nstr(x,30) for x in error_vector],
            matrix_series_degree=k)


def study(out=Path('results/residual_bounds'),quick=False):
    out.mkdir(parents=True,exist_ok=True)
    model=Lorenz();rows=[];certificates=[];details=[]
    initial_states={'standard':np.ones(3),'attractor':np.array([-8.,8.,27.]),
                    'off_attractor':np.array([20.,20.,30.])}
    orders=[3] if quick else [1,2,3,4,5]
    steps=[.001] if quick else [.0005,.001,.002,.005,.01]
    if quick:initial_states={'standard':np.ones(3)}
    for name,u0 in initial_states.items():
        for N in orders:
            original=Carleman(model,N);extended=Carleman(model,N+1)
            top=[i for i,a in enumerate(original.exponents) if sum(a)==N]
            shell=[i for i,a in enumerate(extended.exponents) if sum(a)==N+1]
            beta=np.linalg.norm(extended.C[np.ix_(top,shell)],2)
            alpha=np.sqrt(np.linalg.norm(original.C,1)*np.linalg.norm(original.C,np.inf))
            for h in steps:
                bounds,certificate,data=certified_bounds(model,N,h,u0)
                observed=numerical_observations(model,N,h,u0,data)
                old=(model.scale*beta*(44/model.scale)**(N+1)*np.exp(alpha*h)
                     *alpha**(N-1)*h**N/factorial(N))
                float_next=original.exact_map(u0,h)
                float_ref=reference(model,u0,[h])[0]
                float_error=np.linalg.norm(float_next-float_ref)
                label=f'{name}_N{N}_h{h:g}'
                row=dict(case=label,initial_id=name,N=N,h=h,
                    old_formula_float_evaluation=float(old),
                    **{f'certified_{key}':outward_decimal(value) for key,value in bounds.items()},
                    observed_ideal_mp_error=observed['observed_ideal_mp_error_decimal'],
                    observed_float_implementation_error=float(float_error),
                    numerical_triangle_estimate=observed['numerical_triangle_estimate'],
                    quadrature_change_indicator=observed['numerical_quadrature_change'],
                    voc_reconstruction_error=observed['voc_reconstruction_error'],
                    reference_refinement_indicator=observed['reference_refinement_indicator'],
                    certified_to_observed_ratio=float(bounds['path_local'])/observed['observed_ideal_mp_error'],
                    float_implementation_exceeds_ideal_bound=bool(float_error>float(bounds['path_local'])))
                rows.append(row);certificate['case']=label;certificates.append(certificate)
                details.append(dict(case=label,**observed))
                # A numerical consistency check, not the logical reason why the
                # exact-arithmetic bound is valid.
                if observed['observed_ideal_mp_error']>float(bounds['path_local'])*(1+1e-12):
                    raise AssertionError(f'Numerical ideal error exceeds certified bound: {label}')
                print(label,'actual',f"{observed['observed_ideal_mp_error']:.5e}",
                      'certified',outward_decimal(bounds['path_local'],8),flush=True)
    with open(out/'comparison.csv','w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'certificates.json').write_text(json.dumps(certificates,indent=2),encoding='utf-8')
    (out/'numerical_observations.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    default=next(r for r in rows if r['case']=='standard_N3_h0.001')
    summary=dict(cases=len(rows),all_numerical_ideal_checks_passed=True,
        default_case=default,
        min_certified_to_observed_ratio=min(r['certified_to_observed_ratio'] for r in rows),
        max_certified_to_observed_ratio=max(r['certified_to_observed_ratio'] for r in rows),
        float_implementation_exceeds_ideal_bound_count=sum(r['float_implementation_exceeds_ideal_bound'] for r in rows),
        classes=dict(certified='Fraction interval tube + signed path series + rational majorant tail; ideal exact-generator truncation.',
            numerical_estimate='Numerical reference/16-32 point quadrature; no integration-error enclosure.',
            observation='High precision direct ODE and Carleman series, or binary64 DOP853/expm; not certificates.'))
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    if not quick:plot(out,rows)
    print(json.dumps(summary,indent=2),flush=True)


def plot(out,rows):
    names=['standard','attractor','off_attractor'];fig,axes=plt.subplots(1,3,figsize=(16,4.5))
    for ax,name in zip(axes,names):
        selected=[r for r in rows if r['initial_id']==name and r['N']==3]
        for key,label in [('old_formula_float_evaluation','old global-norm formula'),
                          ('certified_block_local','certified local degree blocks'),
                          ('certified_path_local','certified local signed paths'),
                          ('observed_ideal_mp_error','observed ideal MP error')]:
            ax.loglog([r['h'] for r in selected],[float(r[key]) for r in selected],'o-',label=label)
        ax.set(title=f'{name}, N=3',xlabel='Step h',ylabel='Physical vector error/bound')
        ax.grid(True,which='both',alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(out/'bounds.png',dpi=160);plt.close(fig)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',type=Path,default=Path('results/residual_bounds'))
    parser.add_argument('--quick',action='store_true')
    args=parser.parse_args();study(args.out,args.quick)
