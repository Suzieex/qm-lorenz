"""Verify both gates: certified accuracy and honest end-to-end accounting."""
import argparse,csv,hashlib,json,math,time
from pathlib import Path
from fractions import Fraction as Q
from decimal import Decimal,localcontext,ROUND_FLOOR
import numpy as np
from scipy.integrate import solve_ivp
from carleman import Lorenz
from residual_bounds import sqrt_upper,outward_decimal
from global_correction import validate,evaluate,hermite,coefficients,ceildiv,classical_completion

OUT=Path('results/global_correction')
CASES=[('standard',1e-6),('standard',1e-8),('attractor',1e-8),('attractor',1e-10)]
INITIAL={'standard':(1.,1.,1.),'attractor':(-8.,8.,27.)}


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def digest(result):
    h=hashlib.sha256()
    for item in result['locals']:
        for key in ['h','q','E','delta_E','delta_V','endpoint','g_polynomial_components','residual_F_scaled2','residual_V_scaled2']:
            h.update((key+':'+str(item[key])+'\n').encode())
    return h.hexdigest()


def encode_bounds(bounds):
    return {k:(None if v is None else str(v)) for k,v in bounds.items()}


def guide_endpoint_error_lower(certificate):
    """A rigorous lower bound: the guide alone can be proved insufficient."""
    bounds=certificate['bounds']
    if bounds['total_error'] is None:return Q(0)
    d2=sum((Q(x)-Q(y))**2 for x,y in zip(certificate['endpoint'],certificate['nodes'][-1]))
    S=1<<256;lower=Q(math.isqrt(d2.numerator*S*S//d2.denominator),S)
    return max(Q(0),lower-Q(bounds['total_error'])-Q(certificate['endpoint_rounding']))


def lower_decimal(value):
    with localcontext() as ctx:
        ctx.prec=24;ctx.rounding=ROUND_FLOOR
        return str(Decimal(value.numerator)/Decimal(value.denominator))


def qae_budget(scale,budget):
    """Sufficient canonical-QAE count for 3 coherences plus p0, no free prep."""
    scale=Q(scale);budget=Q(budget)
    if scale<=0 or budget<=0:return {}
    delta=min(Q(1,8),budget/(24*sqrt_upper(Q(3))*scale))
    pi=Q(22,7)
    required=max(4*pi/delta,2*pi*sqrt_upper(1/delta),Q(1))
    integer=ceildiv(required.numerator,required.denominator)
    M=1<<max(0,(integer-1).bit_length());q=1-8/(pi*pi);r=1
    while sum(math.comb(r,k)*q**k*(1-q)**(r-k) for k in range(r//2+1,r+1))>Q(1,80):r+=2
    state_accuracy=min(Q(1,64),budget/(32*sqrt_upper(Q(3))*scale))
    return dict(observable_tolerance=outward_decimal(delta),phase_bits=M.bit_length()-1,
        median_repeats=r,state_preparation_calls=4*r*(2*M-1),
        state_preparation_accuracy=outward_decimal(state_accuracy))


def observation(result,truth,spacing=.001):
    """Ordinary floating evaluation only; NEVER used in a certificate."""
    times=np.arange(len(truth))*spacing;ts=np.array(list(map(float,result['subtimes'])))
    values=np.zeros_like(truth);S=1<<result['bits']
    for i,item in enumerate(result['locals']):
        mask=(times>=ts[i])&(times<=ts[i+1])
        tau=(times[mask]-ts[i])/(ts[i+1]-ts[i])
        for j in range(3):
            coef=[float(Q(row[j],S)) for row in item['V']]
            values[mask,j]=np.polynomial.polynomial.polyval(tau,coef)
    error=np.linalg.norm(values-truth,axis=1)
    return values,error


def coefficient_accounting(times,nodes,scale):
    """Input size/LCU normalization for the normalized four-component ODE.

    This is a constructive coefficient LCU (matrix units and tau^k), NOT a
    claim that a time/coefficients oracle or a QLSA is already implemented.
    """
    terms=0;alpha=Q(0);largest_bits=0
    for left,right,t0,t1 in zip(nodes,nodes[1:],times,times[1:]):
        h=t1-t0;c=hermite(left,right,h);B,_=coefficients(c,Q(1))
        field=[[Q(0)]*15 for _ in range(3)]
        for k in range(8):
            field[0][k]=10*(c[1][k]-c[0][k])
            field[1][k]=28*c[0][k]-c[1][k]
            field[2][k]=-Q(float(8/3))*c[2][k]
        for j in range(8):
            for k in range(8):
                field[1][j+k]-=c[0][j]*c[2][k]
                field[2][j+k]+=c[0][j]*c[1][k]
        for axis in range(3):
            for k in range(7):field[axis][k]-=(k+1)*c[axis][k+1]/h
        entries=[v for row in B for v in row if v]
        entries+=[v/scale for row in field for v in row if v]
        terms+=len(entries);alpha=max(alpha,sum(abs(v) for v in entries))
        largest_bits=max(largest_bits,max(max(abs(v.numerator).bit_length(),v.denominator.bit_length()) for v in entries))
    return dict(coefficient_terms=terms,coefficient_LCU_alpha_upper=outward_decimal(alpha),
                largest_exact_coefficient_bits=largest_bits,
                oracle_status='coefficient expansion accounted; no efficient QLSA or QRAM assumed')


def study(out=OUT,quick=False,bits=512,degree=80):
    out.mkdir(parents=True,exist_ok=True);rows=[];certificates=[];costs=[];classical=[]
    cases=[('standard',1e-8)] if quick else CASES
    T=.1 if quick else 5.;target=Q('1e-8');model=Lorenz()
    for name,tolerance in cases:
        u0=INITIAL[name];start=time.perf_counter()
        guide=solve_ivp(model.rhs,(0,T),u0,method='DOP853',rtol=tolerance,
                        atol=tolerance*.01,dense_output=False)
        guide_seconds=time.perf_counter()-start
        times=[Q(float(t)) for t in guide.t];nodes=[[Q(float(x)) for x in row] for row in guide.y.T]
        start=time.perf_counter()
        result=validate(times,nodes,bits=bits,degree=degree,
            progress=lambda a,b:print(name,tolerance,'validated guide intervals',a,'/',b,flush=True))
        validation_seconds=time.perf_counter()-start;b=result['bounds']
        endpoint,rounding=evaluate(result,Q(str(T)))
        error=None if b['total_error'] is None else b['total_error']+b['uniform_binary64_output_rounding']
        passes=error is not None and error<=target
        row=dict(initial=name,guide_rtol=tolerance,T=T,guide_intervals=len(times)-1,
            validated_subintervals=len(result['locals']),guide_rhs_evaluations=guide.nfev,
            K_upper=outward_decimal(b['K']),a_upper=outward_decimal(b['a']),
            two_K_a_upper=outward_decimal(b['rho']),
            contraction_certified=b['rho']<1,
            nonlinear_remainder_upper='' if b['nonlinear_remainder'] is None else outward_decimal(b['nonlinear_remainder']),
            linear_solver_error_upper=outward_decimal(b['linear_error']),
            endpoint_rounding_upper=outward_decimal(rounding),
            uniform_output_rounding_upper=outward_decimal(b['uniform_binary64_output_rounding']),
            full_target_error_upper='' if error is None else outward_decimal(error),
            accuracy_gate_pass=passes,
            guide_seconds=guide_seconds,certificate_seconds=validation_seconds,
            certificate_includes_classical_solution=True,quantum_advantage_gate_pass=False)
        ref=np.load(f'results/longtime/reference_{name}.npz')['state'][:round(T/.001)+1]
        values,observed=observation(result,ref)
        row['observed_max_grid_error']=float(observed.max());row['observed_final_error']=float(observed[-1])
        row['observation_class']='binary64 polynomial evaluation vs prior MP reference; not certification'
        write_csv(out/f'observed_{name}_{tolerance:g}.csv',[
            dict(t=i*.001,x=u[0],y=u[1],z=u[2],observed_error=e)
            for i,(u,e) in enumerate(zip(values,observed))])
        rows.append(row)
        certificates.append(dict(initial=name,guide_rtol=tolerance,T=str(Q(str(T))),
            exact_parameters=list(map(str,(Q(10),Q(28),Q(float(8/3))))),
            times=list(map(str,times)),nodes=[list(map(str,node)) for node in nodes],
            bits=bits,degree=degree,max_step=str(result['max_step']),
            bounds=encode_bounds(b),local_proofs_sha256=digest(result),
            endpoint=list(map(str,endpoint)),endpoint_rounding=str(rounding),
            target=str(target),accuracy_gate_pass=passes))
        row['guide_endpoint_error_lower']=lower_decimal(guide_endpoint_error_lower(certificates[-1]))
        remaining=target-b['nonlinear_remainder']-b['uniform_binary64_output_rounding'] if b['nonlinear_remainder'] is not None else Q(-1)
        budget=qae_budget(b['a'],remaining) if remaining>0 else {}
        accounting=coefficient_accounting(times,nodes,b['a'])
        costs.append(dict(initial=name,guide_rtol=tolerance,accuracy_gate_pass=passes,
            **accounting,**{k:budget.get(k,'') for k in ['phase_bits','median_repeats','state_preparation_calls','state_preparation_accuracy','observable_tolerance']},
            certificate_scalar_integer_multiplications=sum(x['fixed_point_products'] for x in result['locals'])+b['global_fixed_point_products'],
            classic_corrected_endpoint=endpoint,
            conclusion='classical certification already provides target solution when accuracy gate passes; adding quantum work is redundant for this concrete pipeline'))
        print(row,flush=True)
    for name in sorted({n for n,t in cases}):
        ref=np.load(f'results/longtime/reference_{name}.npz')['state'][:round(T/.001)+1]
        times=np.arange(len(ref))*.001
        for tolerance in [1e-6,1e-8,1e-10,1e-12,1e-13]:
            start=time.perf_counter()
            sol=solve_ivp(model.rhs,(0,T),INITIAL[name],method='DOP853',rtol=tolerance,
                atol=tolerance*.01,dense_output=True)
            elapsed=time.perf_counter()-start;error=np.linalg.norm(sol.sol(times).T-ref,axis=1)
            classical.append(dict(initial=name,T=T,rtol=tolerance,rhs_evaluations=sol.nfev,
                elapsed_seconds=elapsed,maximum_observed_grid_error=float(error.max()),
                empirical_target_pass=bool(error.max()<=float(target)),rigorously_certified=False))
    write_csv(out/'accuracy.csv',rows);write_csv(out/'costs.csv',costs);write_csv(out/'classical_direct.csv',classical)
    (out/'certificates.json').write_text(json.dumps(certificates,indent=2))
    summary=dict(target=str(target),T=T,cases=len(rows),accuracy_passes=sum(r['accuracy_gate_pass'] for r in rows),
        advantage_passes=0,proof_scope='All times: exact polynomial linear approximant plus certified nonlinear remainder. Uniform exact-evaluation-to-binary64 rounding included; ordinary floating Horner observations are separate.',
        certificate_method='dyadic polynomial residual validation, signed fundamental products, exact outward norms; no reference in certification',
        decisive_cost_issue='certificate constructs full classical corrected trajectory; it already solves each passing task',
        qae_scope='sufficient readout budgets assuming a normalized endpoint-state oracle; no quantum linear-solver implementation or speedup claimed',
        classical_direct_scope='empirical DOP853 comparison only; certified classical alternative is the validated corrected polynomial itself')
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2),flush=True)


def precision_study(out=OUT):
    """Check whether solving the certificate vector to excess precision is decisive.

    Retain the validated propagators, reduce ONLY the particular-vector degree.
    This distinguishes a usable linearization certificate from a classical
    corrected polynomial that already meets the final target.
    """
    baseline=json.loads((out/'certificates.json').read_text());rows=[];certificates=[]
    for source in baseline:
        if not source['accuracy_gate_pass']:continue
        for p in [8,12,16]:
            result=validate(list(map(Q,source['times'])),[list(map(Q,row)) for row in source['nodes']],
                max_step=Q(source['max_step']),bits=source['bits'],degree=source['degree'],vector_degree=p)
            b=result['bounds'];endpoint,rounding=evaluate(result,Q(source['T']))
            target=Q(source['target']);nl=b['nonlinear_remainder']
            classical=b['total_error']
            classical_pass=classical is not None and classical+b['uniform_binary64_output_rounding']<=target
            linearization_pass=nl is not None and nl+b['uniform_binary64_output_rounding']<=target/2
            rows.append(dict(initial=source['initial'],guide_rtol=source['guide_rtol'],vector_degree=p,
                K_upper=outward_decimal(b['K']),a_upper=outward_decimal(b['a']),
                linear_solver_error_upper=outward_decimal(b['linear_error']),
                nonlinear_remainder_upper='' if nl is None else outward_decimal(nl),
                exact_linearization_gate_pass=linearization_pass,
                classical_certificate_already_sufficient=classical_pass,
                scalar_integer_multiplications=sum(x['fixed_point_products'] for x in result['locals'])+b['global_fixed_point_products']))
            certificate=dict(source);certificate.update(vector_degree=p,bounds=encode_bounds(b),
                local_proofs_sha256=digest(result),endpoint=list(map(str,endpoint)),
                endpoint_rounding=str(rounding),accuracy_gate_pass=classical_pass)
            certificates.append(certificate)
            print('coarse certificate',rows[-1],flush=True)
    write_csv(out/'certificate_precision.csv',rows)
    (out/'precision_certificates.json').write_text(json.dumps(certificates,indent=2))


def completion_study(out=OUT):
    sources=json.loads((out/'precision_certificates.json').read_text());rows=[];certificates=[]
    for index,source in enumerate(sources):
        if source['vector_degree']!=12:continue
        K=Q(source['bounds']['K']);scale=Q(source['bounds']['a'])
        remaining=Q(source['target'])-Q(source['bounds']['nonlinear_remainder'])-Q(source['bounds']['uniform_binary64_output_rounding'])
        budget=qae_budget(scale,remaining)
        for degree in [16,20]:
            start=time.perf_counter()
            result=classical_completion(source['times'],source['nodes'],K,
                max_step=Q(source['max_step']),bits=source['bits'],degree=degree)
            elapsed=time.perf_counter()-start;b=result['bounds'];target=Q(source['target'])
            passed=b['total_error'] is not None and b['total_error']<=target
            endpoint,rounding=evaluate(result,Q(source['T']))
            rows.append(dict(initial=source['initial'],guide_rtol=source['guide_rtol'],
                source_vector_degree=12,completion_degree=degree,
                certified_linear_error=outward_decimal(b['linear_error']),
                certified_total_error='' if b['total_error'] is None else outward_decimal(b['total_error']),
                classical_completion_pass=passed,
                extra_scalar_integer_multiplications=b['extra_scalar_integer_multiplications'],
                extra_classical_seconds=elapsed,
                hypothetical_QAE_state_preparation_calls=budget.get('state_preparation_calls',''),
                comparison='same certified K reused; integer multiplications and quantum oracle calls are different units'))
            certificates.append(dict(source_precision_certificate_index=index,degree=degree,
                source_sha256=hashlib.sha256(json.dumps(source,sort_keys=True).encode()).hexdigest(),
                bounds=encode_bounds(b),local_proofs_sha256=digest(result),
                endpoint=list(map(str,endpoint)),rounding=str(rounding),target_pass=passed))
            print('classical completion',rows[-1],flush=True)
    write_csv(out/'classical_completion.csv',rows)
    (out/'completion_certificates.json').write_text(json.dumps(certificates,indent=2))
    candidate_resources(out)


def candidate_resources(out=OUT):
    rows=[]
    for c in json.loads((out/'precision_certificates.json').read_text()):
        if c['vector_degree']!=12:continue
        b=c['bounds'];remaining=Q(c['target'])-Q(b['nonlinear_remainder'])-Q(b['uniform_binary64_output_rounding'])
        scale=Q(b['a']);budget=qae_budget(scale,remaining)
        rows.append(dict(initial=c['initial'],guide_rtol=c['guide_rtol'],vector_certificate_degree=12,
            **coefficient_accounting(list(map(Q,c['times'])),[list(map(Q,row)) for row in c['nodes']],scale),
            **budget,source_K=b['K'],source_scale=str(scale),remaining_budget=str(remaining),
            scope='conditional normalized endpoint-state oracle; full QLSA and oracle gates not implemented'))
    write_csv(out/'candidate_resources.csv',rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT)
    p.add_argument('--quick',action='store_true');p.add_argument('--bits',type=int,default=512)
    p.add_argument('--degree',type=int,default=80);p.add_argument('--precision-study',action='store_true')
    p.add_argument('--completion-study',action='store_true');args=p.parse_args()
    if args.completion_study:completion_study(args.out)
    elif args.precision_study:precision_study(args.out)
    else:study(args.out,args.quick,args.bits,args.degree)
