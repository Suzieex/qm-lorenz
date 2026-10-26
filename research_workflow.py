"""Run research priorities 1->5; separate certificates, models and observations.

python research_workflow.py --stage 1 (then 2,3,4,5)
Each stage consumes saved preceding work. No timing or oracle-count ratio is
called a hardware speedup. All QAE costs refer to this particular construction.
"""
import argparse,csv,json,math
from fractions import Fraction as Q
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.stats import binom
from carleman import Lorenz
from residual_bounds import (local_enclosure,norm_infinity,path_bound,sqrt_upper,
    outward_decimal)
from research_bounds import (AffineLorenz,affine_bound,affine_field,generator,
    predictor_bounds,select_taylor,exact,product,propagation_data,chebyshev_bound)
from sparse_compiler import compile_block_encoding,gate_resources
from weighted_compiler import compile_weighted,coefficient_row_hadamard

OUT=Path('results/research_sequence')
INITIAL={'standard':(1.,1.,1.),'attractor':(-8.,8.,27.)}
STEPS=[.001,.005,.01,.02]
TARGETS=[1e-4,1e-6,1e-8]


def save(name,rows):
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def read(name):
    with (OUT/name).open() as f:return list(csv.DictReader(f))


def json_save(name,obj):
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/name).write_text(json.dumps(obj,indent=2))


def alpha_exact(model,N):
    """Normalization of the existing grouped threshold-counting compiler."""
    diagonal=[];other=Q(0)
    for axis,poly in enumerate(affine_field(model)):
        for power,coefficient in poly.items():
            shift=list(power);shift[axis]-=1
            if not any(shift):diagonal.append(coefficient)
            else:other+=max(0,N-max(sum(shift),0))*abs(coefficient)
    if all(c<=0 for c in diagonal) or all(c>=0 for c in diagonal):
        other+=N*max([abs(c) for c in diagonal]+[Q(0)])
    else:other+=N*sum(abs(c) for c in diagonal)
    return other


def qae_budget(Fs,vector_budget,steps=1,failure=.05,K=1):
    """Canonical QAE sufficient counts, median-amplified; not a lower bound."""
    delta=vector_budget/math.sqrt(3)
    p_fail=1-8/math.pi**2;repeats=1
    while binom.sf(repeats//2,repeats,p_fail)>failure/(3*steps):repeats+=2
    resolutions=[]
    for f in Fs:
        required=max(1.,4*math.pi*f/delta,2*math.pi*math.sqrt(f/delta))
        resolutions.append(2**math.ceil(math.log2(required)))
    calls=repeats*sum(2*m-1 for m in resolutions)
    return dict(qae_A_calls=calls,base_calls=K*calls,state_preparation_uses=calls,
                repeats=repeats,max_phase_bits=max(m.bit_length()-1 for m in resolutions))


def model_for(row):
    return AffineLorenz(scale=float(row['S0']),pivot=tuple(float(row[f'c{j}']) for j in range(3)),
                        scales=tuple(float(row[f'S{j}']) for j in range(3)))


def selected_resources():
    """Gate counts for actual generated encodings, not a dense norm proxy."""
    rows=[];cache={}
    for r in read('01_selected.csv'):
        model=model_for(r);N=int(r['N']);key=(model,N)
        if key not in cache:cache[key]=gate_resources(compile_block_encoding(model,N))
        resource=cache[key]
        rows.append(dict(initial=r['initial'],h=r['h'],epsilon=r['epsilon'],N=N,
            alpha=float(alpha_exact(model,N)),base_qubits=resource['qubits'],
            base_cx=resource['elementary_operations'].get('cx',0),
            base_u=resource['elementary_operations'].get('u',0),
            base_calls=r['base_calls'],
            uncontrolled_base_cx_product=int(r['base_calls'])*resource['elementary_operations'].get('cx',0),
            classification='resource product, not full controlled-QAE gate count'))
    save('01_selected_gate_resources.csv',rows)
    from sparse_workflow import build_sparse_hadamard
    from qiskit import transpile
    selected=next(r for r in read('01_selected.csv') if r['initial']=='standard'
                  and float(r['h'])==.001 and float(r['epsilon'])==1e-8)
    qc,meta=build_sparse_hadamard(model_for(selected),int(selected['N']),.001,
        np.ones(3),1,K=int(selected['K']))
    gates=transpile(qc,basis_gates=['u','cx'],optimization_level=1)
    json_save('01_representative_circuit.json',dict(**meta,qubits=qc.num_qubits,
        gates=dict(gates.count_ops()),classification='full generated Hadamard A, excludes QAE reflections/controls',
        primitive_basis=['u','cx'],coordinate=1,N=int(selected['N']),h=.001,
        epsilon=float(selected['epsilon']),artifact='rebuild from selected_resources(); no recursively expanded QPY stored'))


def sparse_matrix(rows):
    rr=[];cc=[];vv=[]
    for i,row in enumerate(rows):
        for j,v in row.items():rr.append(i);cc.append(j);vv.append(float(v))
    return csr_matrix((vv,(rr,cc)),shape=(len(rows),len(rows)))


def numerical_step(bound,u):
    data=bound['data'];C=sparse_matrix(data['C'])
    state=expm_multiply(float(bound['h'])*C,np.array(list(map(float,bound['phi']))))
    return np.array([float(Q(c)) for c in bound['certificate']['pivot']])+np.array(list(map(float,bound['factors'])))*state[data['linear']]


def stage1():
    rows=[];certificates=[];failures=[]
    for name,u in INITIAL.items():
        tubes={}
        for h in STEPS:
            try:tubes[h]=local_enclosure(Lorenz(),u,Q(str(h)))[0]
            except ValueError as e:failures.append(dict(initial=name,h=h,error=str(e)))
        for center_name,center in [('energy',(0.,0.,38.)),('rho',(0.,0.,28.)),('initial',u)]:
            for S in [16.,32.,64.]:
                model=AffineLorenz(pivot=center,scales=(S,)*3)
                for N in range(1,7):
                    alpha=alpha_exact(model,N)
                    for h,tube in tubes.items():
                        b=affine_bound(model,N,h,u,tube=tube)
                        for eps in TARGETS:
                            feasible=b['truncation']<=Q(str(eps))/3
                            K,w,tail=select_taylor(b,alpha,Q(str(eps))/6)
                            gamma=sum(w[1:]);Fs=[float(f*b['B']*gamma) for f in b['factors']]
                            cost=qae_budget(Fs,eps/3,K=K)
                            rows.append(dict(initial=name,center=center_name,
                                **{f'c{j}':center[j] for j in range(3)},
                                **{f'S{j}':S for j in range(3)},N=N,h=h,epsilon=eps,
                                D=len(b['phi']),alpha=float(alpha),B=float(b['B']),K=K,
                                gamma=float(gamma),Fmax=max(Fs),
                                truncation_upper=outward_decimal(b['truncation']),
                                taylor_upper=outward_decimal(tail),deterministic_feasible=feasible,
                                **cost,calls_per_unit_time=cost['base_calls']/h,
                                resource_class='canonical_QAE_sufficient_count_model'))
        print('stage1',name,'configurations',len(rows),flush=True)
    save('01_parameter_scan.csv',rows)
    selected=[]
    for name in INITIAL:
        for h in STEPS:
            for eps in TARGETS:
                candidates=[r for r in rows if r['initial']==name and r['h']==h
                            and r['epsilon']==eps and r['deterministic_feasible']]
                if not candidates:continue
                best=min(candidates,key=lambda r:r['base_calls']);selected.append(best)
                model=model_for(best);b=affine_bound(model,best['N'],h,INITIAL[name])
                certificates.append(dict(stage=1,initial=name,epsilon=eps,**b['certificate']))
    save('01_selected.csv',selected);json_save('01_certificates.json',certificates)
    json_save('01_summary.json',dict(configurations=len(rows),selected=len(selected),
        feasible=sum(r['deterministic_feasible'] for r in rows),tube_failures=failures,
        status='single-step truncation/Taylor certificates; QAE counts assume exact logical gates',
        objectives='same h,initial,epsilon comparison; no reference trajectory selects feasibility'))


@lru_cache(maxsize=32)
def classical_options(name,h):
    return predictor_bounds(INITIAL[name],h,max_order=12)


def stage2():
    rows=[];classical=[]
    selected=read('01_selected.csv')
    eligible={(r['initial'],float(r['h']),float(r['epsilon'])) for r in selected}
    for name in INITIAL:
        for h in STEPS:
            for eps in TARGETS:
                if (name,h,eps) not in eligible:continue
                options=classical_options(name,h)
                good=[p for p in options if p['error']<=Q(str(eps))]
                best=good[0] if good else None
                classical.append(dict(initial=name,h=h,epsilon=eps,
                    certified_order=best['p'] if best else '',
                    certified_remainder=outward_decimal(best['error']) if best else '',
                    convolution_products=best['convolution_products'] if best else '',
                    online_certificate_terms=best['certificate_terms'] if best else '',
                    certificate_precompute_products=best['certificate_precompute_products'] if best else '',
                    certificate_coefficient_max_bits=best['certificate_coefficient_max_bits'] if best else '',
                    guarantee='exact Taylor predictor; coefficient rounding excluded'))
    for r in selected:
        name=r['initial'];h=float(r['h']);eps=float(r['epsilon']);N=int(r['N'])
        model=model_for(r);b=affine_bound(model,N,h,INITIAL[name]);alpha=alpha_exact(model,N)
        K,w,tail=select_taylor(b,alpha,Q(str(eps))/6)
        C=b['data']['C'];nnz=sum(map(len,C));phi=b['phi']
        term=[{j:Q(1)} for j in b['data']['linear']];powers=[term]
        from residual_bounds import row_times
        for k in range(1,K+1):
            term=[{j:v*b['h']/k for j,v in row_times(row,C).items()} for row in term]
            powers.append(term)
        options=classical_options(name,h)
        for p in range(min(N-1,K-1,8)+1):
            # Degree of the p-th derivative is <=p+1: require p<=N-1.
            gamma=sum(w[p+1:]);Fs=[float(f*b['B']*gamma) for f in b['factors']]
            cost=qae_budget(Fs,eps/3,K=K)
            remainder=[]
            for j in range(3):
                row={}
                for power in powers[p+1:]:
                    for i,v in power[j].items():row[i]=row.get(i,Q(0))+v
                remainder.append(row)
            row_norms=[sqrt_upper(sum(v*v for v in row.values())) for row in remainder]
            Fs_row=[float(f*b['B']*rho) for f,rho in zip(b['factors'],row_norms)]
            rowcost=qae_budget(Fs_row,eps/3,K=1)
            predictor=options[p-1] if p else None
            stop=bool(predictor and predictor['error']<=Q(str(eps)))
            rows.append(dict(initial=name,h=h,epsilon=eps,N=N,p=p,K=K,
                gamma=float(gamma),Fmax=max(Fs),**cost,
                row_Fmax=max(Fs_row),row_A_calls=rowcost['qae_A_calls'],
                row_classical_precompute_sparse_products=3*K*nnz,
                row_preparation_grid_size=2**(3*math.ceil(math.log2(N+2))),
                predictor_convolution_products=p*(p+1),
                predictor_certified_error=outward_decimal(predictor['error']) if predictor else '',
                classical_stop_certified=stop,
                entire_task_classical_stop=any(q['error']<=Q(str(eps)) for q in options),
                preprocessing_scope='row computation per center,N,h,p; reusable only if unchanged',
                basis_input=all(v==0 for v in phi[1:])))
    joint=[];best_joint=[]
    # Re-optimize the sparse-LCU resource model over ALL certified feasible
    # stage1 configurations, rather than freezing the p=0 winners.
    for r in read('01_parameter_scan.csv'):
        if r['deterministic_feasible']!='True':continue
        name=r['initial'];h=float(r['h']);eps=float(r['epsilon'])
        N=int(r['N']);K=int(r['K']);a=float(r['alpha'])*h
        weights=[1.]
        for k in range(1,K+1):weights.append(weights[-1]*a/k)
        options=classical_options(name,h)
        for p in range(min(N-1,K-1)+1):
            gamma=sum(weights[p+1:]);F=float(r['S0'])*float(r['B'])*gamma
            stop=bool(p and options[p-1]['error']<=Q(str(eps)))
            cost=qae_budget([F]*3,eps/3,K=K)
            joint.append(dict(**{k:r[k] for k in ['initial','center','c0','c1','c2','S0','S1','S2','N','h','epsilon','alpha','B','K','truncation_upper','taylor_upper']},
                p=p,F=F,**cost,classical_stop_certified=stop,
                predictor_convolution_products=p*(p+1),
                classification='resource model using saved floating alpha,B; truncation/Taylor acceptance inherited from exact certificates'))
    for name,h,eps in sorted(eligible):
        candidates=[r for r in joint if r['initial']==name and float(r['h'])==h
                    and float(r['epsilon'])==eps and not r['classical_stop_certified']]
        if candidates:best_joint.append(min(candidates,key=lambda r:r['base_calls']))
    save('02_joint_remainder_scan.csv',joint);save('02_joint_selected.csv',best_joint)
    save('02_remainder_readout.csv',rows);save('02_classical_predictors.csv',classical)
    json_save('02_summary.json',dict(candidates=len(rows),
        certified_classical_stop=sum(r['classical_stop_certified'] for r in rows),
        all_tasks_have_certified_classical_alternative=all(r['certified_order']!='' for r in classical),
        max_classical_order=max(r['certified_order'] or 0 for r in classical),
        joint_candidates=len(joint),joint_nonstop_selected=len(best_joint),
        warning='row norm has an implemented preparation backend, but its classical precompute is charged; counts are not a speed ratio'))


def stage3():
    rows=[];observations=[]
    horizons=[.001,.005,.01,.02,.05,.1]
    for name,u in INITIAL.items():
        for N in [2,3,4,5,6]:
            for H in horizons:
                model=AffineLorenz()
                try:b=affine_bound(model,N,H,u)
                except ValueError as e:
                    rows.append(dict(initial=name,N=N,H=H,valid=False,reason=str(e),
                        truncation_upper='',epsilon='',K='',base_calls_per_T5='',segments=''))
                    continue
                for eps in TARGETS:
                    K,w,tail=select_taylor(b,alpha_exact(model,N),Q(str(eps))/6)
                    gamma=sum(w[1:]);segments=math.ceil(5/H)
                    Fs=[float(f*b['B']*gamma) for f in b['factors']]
                    # Deliberately optimistic: no accumulated-error penalty.
                    cost=qae_budget(Fs,eps/3,steps=segments,K=K)
                    rows.append(dict(initial=name,N=N,H=H,valid=True,reason='',
                        truncation_upper=outward_decimal(b['truncation']),epsilon=eps,K=K,
                        base_calls_per_T5=segments*cost['base_calls'] if b['truncation']<=Q(str(eps))/3 else '',
                        segments=segments))
                if H in [.01,.02]:
                    # Compare one continuous lift against equal-duration relifting.
                    full=numerical_step(b,u);state=np.array(u,dtype=float)
                    for _ in range(round(H/.001)):
                        sub=affine_bound(model,N,.001,state)
                        state=numerical_step(sub,state)
                    from scipy.integrate import solve_ivp
                    truth=solve_ivp(Lorenz().rhs,(0,H),u,method='DOP853',rtol=2.3e-14,atol=2e-14).y[:,-1]
                    observations.append(dict(initial=name,N=N,H=H,center='energy',
                        continuous_truncation_upper=outward_decimal(b['truncation']),
                        continuous_lift_observed_error=float(np.linalg.norm(full-truth)),
                        relifted_observed_error=float(np.linalg.norm(state-truth)),
                        discrepancy=float(np.linalg.norm(full-state)),
                        classification='binary64 numerical observation, not certificate'))
        for H in [.01,.02]:
            local=AffineLorenz(scale=16.,pivot=u,scales=(16.,)*3)
            try:b=affine_bound(local,3,H,u)
            except ValueError:continue
            full=numerical_step(b,u);state=np.array(u,dtype=float)
            for _ in range(round(H/.001)):
                current=AffineLorenz(scale=16.,pivot=tuple(state),scales=(16.,)*3)
                sub=affine_bound(current,3,.001,state);state=numerical_step(sub,state)
            from scipy.integrate import solve_ivp
            truth=solve_ivp(Lorenz().rhs,(0,H),u,method='DOP853',rtol=2.3e-14,atol=2e-14).y[:,-1]
            observations.append(dict(initial=name,N=3,H=H,center='local',
                continuous_truncation_upper=outward_decimal(b['truncation']),
                continuous_lift_observed_error=float(np.linalg.norm(full-truth)),
                relifted_observed_error=float(np.linalg.norm(state-truth)),
                discrepancy=float(np.linalg.norm(full-state)),
                classification='binary64 numerical observation; local center updated at each relift'))
        print('stage3',name,flush=True)
    save('03_segment_scan.csv',rows);save('03_segment_observations.csv',observations)
    optimized=[]
    for r in read('02_joint_selected.csv'):
        h=float(r['h']);m=math.ceil(5/h);F=float(r['F']);eps=float(r['epsilon'])
        cost=qae_budget([F]*3,eps/3,steps=m,K=int(r['K']))
        optimized.append(dict(initial=r['initial'],H=h,epsilon=eps,N=r['N'],p=r['p'],
            center=r['center'],S=r['S0'],segments=m,
            first_segment_truncation_upper=r['truncation_upper'],
            base_calls_first_segment=cost['base_calls'],stationary_T5_base_calls=m*cost['base_calls'],
            classification='optimistic stationary first-segment resource proxy; no T5 accuracy or changing-input resource guarantee'))
    save('03_optimized_segments.csv',optimized)
    best=[]
    for name in INITIAL:
        for eps in TARGETS:
            candidates=[r for r in optimized if r['initial']==name and r['epsilon']==eps]
            if candidates:best.append(min(candidates,key=lambda r:r['stationary_T5_base_calls']))
    save('03_best_stationary_proxy.csv',best)
    json_save('03_summary.json',dict(
        condition='each row certifies only a segment from its stated initial point',
        T5_counts='optimistic screening: final tolerance reused locally; not a T5 accuracy guarantee',
        invalid=sum(not r['valid'] for r in rows),
        warning='continuous truncated propagation is not equivalent to relifting at internal steps'))
    segment_certificates()


def segment_certificates():
    certificates=[]
    unique={(r['initial'],int(r['N']),float(r['H'])) for r in read('03_segment_scan.csv')
            if r['valid']=='True'}
    for name,N,H in sorted(unique):
        b=affine_bound(AffineLorenz(),N,H,INITIAL[name])
        certificates.append(dict(stage=3,initial=name,center='energy',**b['certificate']))
    for r in read('03_segment_observations.csv'):
        if r['center']!='local':continue
        name=r['initial'];model=AffineLorenz(scale=16.,pivot=INITIAL[name],scales=(16.,)*3)
        b=affine_bound(model,int(r['N']),float(r['H']),INITIAL[name])
        certificates.append(dict(stage=3,initial=name,center='local',**b['certificate']))
    json_save('03_certificates.json',certificates)


def stage4():
    rows=[];resources=[];certificates=[];orthogonal=[]
    for name,u in INITIAL.items():
        for N in [2,3,4]:
            for scales in [(32.,32.,32.),(16.,16.,64.),(32.,16.,64.),(64.,32.,32.)]:
                model=AffineLorenz(scales=scales)
                for label,weights in [('plain',[Q(1)]*(N+2)),
                    ('inverse_factorial',[Q(1,math.factorial(k)) for k in range(N+2)]),
                    ('inverse_degree',[Q(1,k+1) for k in range(N+2)]),
                    ('factorial',[Q(math.factorial(k)) for k in range(N+2)])]:
                    b=affine_bound(model,N,.005,u,weights=weights)
                    # The implemented diagonal sandwich fixes the normalization.
                    kappa=max(weights[:N+1])/min(weights[:N+1])
                    alpha=alpha_exact(model,N)*kappa
                    eps=1e-8;K,w,tail=select_taylor(b,alpha,Q(str(eps))/6)
                    gamma=sum(w[1:]);Fs=[float(f*b['B']*gamma) for f in b['factors']]
                    cost=qae_budget(Fs,eps/3,K=K)
                    rows.append(dict(initial=name,N=N,h=.005,basis=label,
                        Sx=scales[0],Sy=scales[1],Sz=scales[2],B=float(b['B']),
                        alpha_implemented=float(alpha),weight_condition=float(kappa),K=K,
                        truncation_upper=outward_decimal(b['truncation']),
                        deterministic_feasible=bool(b['truncation']<=Q('1e-8')/3),
                        Fmax=max(Fs),**cost,
                        status='degree-diagonal sandwich implementation; exact rational truncation certificate'))
                    if name=='standard' and N==2 and scales==(32.,32.,32.):
                        encoding=compile_weighted(model,N,list(map(float,weights[:N+1])))
                        res=gate_resources(encoding)
                        resources.append(dict(basis=label,N=N,alpha=encoding.alpha,
                            qubits=res['qubits'],cx=res['elementary_operations'].get('cx',0),
                            u=res['elementary_operations'].get('u',0)))
                        certificates.append(dict(stage=4,initial=name,basis=label,**b['certificate']))
        for N in [2,3,4]:
            b=chebyshev_bound(AffineLorenz(),N,.005,u);C=b['data']['C']
            column=[sum(abs(row.get(j,Q(0))) for row in C) for j in range(len(C))]
            majorant=sqrt_upper(max(column)*norm_infinity(C))
            K,_,tail=select_taylor(b,majorant,Q('1e-8')/6)
            term=[{j:Q(1)} for j in b['data']['linear']];total=[{} for _ in range(3)]
            from residual_bounds import row_times
            for k in range(1,K+1):
                term=[{j:v*b['h']/k for j,v in row_times(row,C).items()} for row in term]
                for axis,row in enumerate(term):
                    for j,v in row.items():total[axis][j]=total[axis].get(j,Q(0))+v
            rhos=[sqrt_upper(sum(x*x for x in row.values())) for row in total]
            Fs=[float(s*b['B']*rho) for s,rho in zip(b['factors'],rhos)]
            cost=qae_budget(Fs,1e-8/3,K=1)
            orthogonal.append(dict(initial=name,N=N,h=.005,B=float(b['B']),
                truncation_upper=outward_decimal(b['truncation']),Fmax=max(Fs),
                deterministic_feasible=bool(b['truncation']<=Q('1e-8')/3),
                coefficient_series_majorant=float(majorant),K=K,
                classical_precompute_sparse_products=3*K*sum(map(len,C)),**cost,
                encoding='explicit coefficient-row state preparation; no sparse alpha claimed'))
            certificates.append(dict(stage=4,initial=name,**b['certificate']))
            if N==2 and name=='standard':
                from qiskit import transpile
                row=[total[1].get(j,Q(0)) for j in range(len(C))]
                qc=coefficient_row_hadamard(b['phi'],row)
                gates=transpile(qc,basis_gates=['u','cx'],optimization_level=1)
                resources.append(dict(basis='Chebyshev_row_Hadamard',N=N,alpha='',
                    qubits=qc.num_qubits,cx=gates.count_ops().get('cx',0),u=gates.count_ops().get('u',0)))
        print('stage4',name,flush=True)
    save('04_basis_scan.csv',rows);save('04_weighted_circuit_resources.csv',resources)
    save('04_chebyshev_scan.csv',orthogonal)
    json_save('04_certificates.json',certificates)
    # Orthogonal basis feasibility: an exactly invertible degree-triangular
    # Chebyshev transform changes the residual and target row; no free alpha.
    json_save('04_orthogonal_basis_assessment.json',dict(
        investigated='tensor Chebyshev polynomials, total degree <= N',
        implementation_status='exact separate residual bound + explicit coefficient-row Hadamard; no sparse arithmetic compiler',
        recurrence='v*T_k(v)=(T_(k+1)(v)+T_(k-1)(v))/2 for k>=1',
        derivative='T_k prime=k*U_(k-1); conversion to T basis has O(k) terms',
        sparse_risk='basis conversion can increase row sparsity and preparation cost',
        certificate_risk='retaining total degree after triangular basis change does not justify reusing the monomial truncation certificate',
        conclusion='weighted, anisotropic and orthogonal residuals evaluated; coefficient preparation and classical preprocessing are charged'))


def exp_upper(x):
    x=Q(x);term=Q(1);total=term
    for k in range(1,1000):
        term*=x/k;total+=term
        ratio=x/(k+2)
        if ratio<1:
            tail=term*x/(k+1)/(1-ratio)
            if tail<Q(1,10**70):return total+tail
    raise ValueError('exp enclosure did not converge')


def stage5():
    # Projection to the invariant convex ball is part of this proposed full
    # algorithm. It prevents inaccurate readouts from invalidating the domain.
    R=Q(44);S=Q(64);T=Q(5);L=Q(21);growth=exp_upper(L*T)
    rows=[];allocation=[];certificates=[]
    model=AffineLorenz()
    for N in [3,5,7,9]:
        weights=tuple([Q(1)]*(N+2));_,C=generator(affine_field(model),N)
        a=norm_infinity(C);alpha=alpha_exact(model,N)
        for h in [.001,.005,.01,.02]:
            dt=Q(str(h));m=int(T/dt);J=max(N+8,int(dt*a)+2)
            data=propagation_data(model,N,weights,J)
            components,_=path_bound(data,dt,[(R/S)**(N+1)]*len(data['shell']))
            d=S*sqrt_upper(sum(x*x for x in components))
            # Rigorous, inexpensive upper bound on sum of propagated errors.
            W=m*growth
            B=sqrt_upper(sum((R/S)**(2*k) for k in range(N+1)))
            for eps in TARGETS:
                epsilon=Q(str(eps));local=epsilon/(4*W)
                stub=dict(h=dt,B=B,factors=[S]*3)
                K,w,taylor=select_taylor(stub,alpha,local)
                Fs=[float(S*B*sum(w[1:]))]*3
                cost=qae_budget(Fs,float(local),steps=m,K=K)
                rows.append(dict(N=N,h=h,T=5,epsilon=eps,steps=m,
                    uniform_truncation_upper=outward_decimal(d),
                    propagation_weight_sum_upper=outward_decimal(W),
                    propagated_truncation_upper=outward_decimal(W*d),
                    truncation_budget_certified=bool(W*d<=epsilon/4),
                    admissible_initial_error=outward_decimal(epsilon/(4*growth)),
                    local_vector_readout_budget=outward_decimal(local),
                    taylor_K=K,**cost,total_base_calls=m*cost['base_calls'],
                    guarantee='conditional on QAE success and exact logical gates; invariant-ball projection included',
                    floating_implementation_certified=False))
                if eps==1e-8:
                    certificates.append(dict(N=N,h=str(dt),global_single_step_upper=str(d),
                        weight_sum_upper=str(W),local_budget=str(local),
                        taylor_K=K,taylor_error=str(taylor),
                        initial_error_budget=str(epsilon/(4*growth)),
                        truncation_budget_pass=bool(W*d<=epsilon/4)))
            # Optimizing sum_i c_i/delta_i subject to sum_i w_i delta_i<=eps.
            # Here c_i are equal and w_i=exp(L*(T-t_(i+1))). Float screening.
            wnum=np.exp(21*(5-np.arange(1,m+1)*h))
            uniform=m*float(wnum.sum())
            optimized=float(np.sqrt(wnum).sum())**2
            allocation.append(dict(N=N,h=h,steps=m,
                uniform_continuous_query_objective=uniform,
                optimized_continuous_query_objective=optimized,
                improvement_factor=uniform/optimized,
                classification='numerical continuous allocation model; excludes QAE rounding/repeats and variable F'))
    save('05_full_budget.csv',rows);save('05_precision_allocation.csv',allocation)
    json_save('05_certificates.json',certificates)
    json_save('05_summary.json',dict(T=5,domain_radius=44,one_sided_Lipschitz=21,
        exp_LT_upper=outward_decimal(growth),
        certified_full_trajectory_candidates=sum(r['truncation_budget_certified'] for r in rows),
        certificate='uniform truncation, exact exponential upper bound, Taylor budget, conditional readout union bound',
        projection='project readout to the radius44 ball centered at (0,0,38); nonexpansive relative to true state',
        missing='actual finite-gate and finite-arithmetic error budget; current full-trajectory candidates fail truncation budget',
        conclusion='no certified end-to-end speedup established; failure of this conservative certificate is not failure of numerical trajectories'))
    stability_diagnostics()


def stability_diagnostics():
    """Numerical terminal variational weights; deliberately NOT certificates."""
    from scipy.integrate import solve_ivp
    rows=[];model=Lorenz()
    for name,u in INITIAL.items():
        reference=solve_ivp(model.rhs,(0,5),u,method='DOP853',rtol=2.3e-14,
                            atol=2e-14,dense_output=True)
        def rhs(t,flat):
            x,y,z=reference.sol(t)
            J=np.array([[-model.sigma,model.sigma,0.],
                [model.rho-z,-1.,-x],[y,x,-model.beta]])
            return (-flat.reshape(3,3)@J).ravel()
        coarse=solve_ivp(rhs,(5,0),np.eye(3).ravel(),method='DOP853',
            rtol=1e-10,atol=1e-12,dense_output=True)
        fine=solve_ivp(rhs,(5,0),np.eye(3).ravel(),method='DOP853',
            rtol=2.3e-14,atol=2e-14,dense_output=True)
        for h in [.001,.005,.01,.02]:
            m=round(5/h);times=np.arange(1,m+1)*h
            matrices=fine.sol(times).T.reshape(-1,3,3)
            weights=np.linalg.norm(matrices,ord=2,axis=(1,2))
            other=coarse.sol(times).T.reshape(-1,3,3)
            discrepancy=np.linalg.norm(matrices-other,axis=(1,2)).max()
            uniform=m*float(weights.sum());optimized=float(np.sqrt(weights).sum())**2
            rows.append(dict(initial=name,h=h,T=5,steps=m,
                terminal_initial_amplification=float(np.linalg.norm(fine.sol(0).reshape(3,3),2)),
                terminal_weight_sum=float(weights.sum()),maximum_terminal_weight=float(weights.max()),
                allocation_improvement=uniform/optimized,
                variational_refinement_indicator=float(discrepancy),
                classification='numerical linearized terminal sensitivity only; no nonlinear or all-times guarantee'))
    save('05_variational_diagnostic.csv',rows)


def high_precision_examples():
    """Independent direct-ODE versus lifted-series observations at 120 digits."""
    import mpmath as mp
    from long_validation import taylor_coefficients
    from residual_bounds import row_times
    examples=[]
    selected=read('01_selected.csv');joint=read('02_joint_selected.csv')
    cases=[('original',AffineLorenz(),3,.001,INITIAL['standard'],False)]
    for source,label in [(selected,'selected_increment'),(joint,'selected_remainder')]:
        for name in INITIAL:
            row=next(r for r in source if r['initial']==name and float(r['h'])==.001
                     and float(r['epsilon'])==1e-8)
            cases.append((label+'_'+name,model_for(row),int(row['N']),.001,INITIAL[name],False))
    cases.append(('Chebyshev',AffineLorenz(),2,.005,INITIAL['standard'],True))
    with mp.workdps(120):
        to_mp=lambda x:mp.mpf(Q(x).numerator)/Q(x).denominator
        for label,model,N,h,u,cheb in cases:
            b=chebyshev_bound(model,N,h,u) if cheb else affine_bound(model,N,h,u)
            C=[{j:to_mp(c) for j,c in row.items()} for row in b['data']['C']]
            phi=list(map(to_mp,b['phi']));term=[{i:mp.mpf(1)} for i in b['data']['linear']]
            result=[phi[i] for i in b['data']['linear']];dt=to_mp(Q(str(h)))
            a=to_mp(b['data']['infinity_norm']);weight=mp.mpf(1)
            for k in range(1,400):
                term=[{j:c*dt/k for j,c in row_times(row,C).items()} for row in term]
                result=[r+sum(c*phi[j] for j,c in row.items()) for r,row in zip(result,term)]
                weight*=a*dt/k
                if weight<mp.mpf('1e-110') and k>a*dt:break
            candidate=[to_mp(exact(c))+to_mp(f)*r for c,f,r in zip(model.pivot,b['factors'],result)]
            parameters=[to_mp(exact(x)) for x in (model.sigma,model.rho,model.beta)]
            coarse=taylor_coefficients(list(map(to_mp,u)),80,parameters)
            fine=taylor_coefficients(list(map(to_mp,u)),100,parameters)
            truth=[mp.polyval(a,dt) for a in fine]
            error=mp.sqrt(sum((a-c)**2 for a,c in zip(candidate,truth)))
            check=mp.sqrt(sum((mp.polyval(a,dt)-t)**2 for a,t in zip(coarse,truth)))
            examples.append(dict(case=label,N=N,h=h,pivot=model.pivot,
                certified_truncation=outward_decimal(b['truncation']),
                observed_ideal_error=mp.nstr(error,40),
                ratio=mp.nstr(to_mp(b['truncation'])/error,12),
                reference_refinement_indicator=mp.nstr(check,8),matrix_series_degree=k,
                classification='high precision numerical observation; certificate computed separately'))
            assert error<=to_mp(b['truncation'])
    json_save('01_high_precision_examples.json',examples)


if __name__=='__main__':
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--stage',type=int,choices=range(1,6))
    group.add_argument('--resources-only',action='store_true')
    group.add_argument('--examples-only',action='store_true')
    args=parser.parse_args()
    if args.resources_only:selected_resources()
    elif args.examples_only:high_precision_examples()
    else:globals()[f'stage{args.stage}']()
