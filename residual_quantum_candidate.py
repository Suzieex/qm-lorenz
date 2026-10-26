"""Quantum residual integration candidate versus exact Lorenz polynomial work.

Local mathematical certificates concern ideal exact rational polynomials only.
QAE counts are sufficient oracle budgets, not a simulated global quantum ODE.
"""
import csv,json,math
from pathlib import Path
from fractions import Fraction as Q
import numpy as np
import mpmath as mp
from global_correction import time_coefficients,PARAMETERS,exp_upper_int
from residual_bounds import local_enclosure,sqrt_upper,outward_decimal
from research_bounds import predictor_bounds
from long_validation import taylor_coefficients

OUT=Path('results/quantum_candidates')


def pad(poly,n):return list(poly)+[Q(0)]*max(0,n-len(poly))
def conv(a,b):
    out=[Q(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):out[i+j]+=x*y
    return out
def add(a,b,scale=1):return [x+scale*y for x,y in zip(pad(a,max(len(a),len(b))),pad(b,max(len(a),len(b))))]
def polyval(a,t):
    value=Q(0)
    for c in reversed(a):value=value*t+c
    return value
def derivative(a):return [k*a[k] for k in range(1,len(a))] or [Q(0)]
def bernstein_range(a):
    n=len(a)-1
    values=[sum(a[j]*Q(math.comb(k,j),math.comb(n,j)) for j in range(k+1)) for k in range(n+1)]
    return min(values),max(values)
def absmax(interval):return max(abs(x) for x in interval)


def residual(u,h,p):
    h=Q(str(h));coeff=time_coefficients(list(map(Q,u)),p)
    P=[[a*h**k for k,a in enumerate(row)] for row in coeff]
    sigma,rho,beta=PARAMETERS
    f=[[sigma*x for x in add(P[1],P[0],-1)],
       add(add([rho*x for x in P[0]],P[1],-1),conv(P[0],P[2]),-1),
       add(conv(P[0],P[1]),[beta*x for x in P[2]],-1)]
    R=[add([h*x for x in row],derivative(poly),-1) for row,poly in zip(f,P)]
    delta=[[Q(0)]+[a/(k+1) for k,a in enumerate(row)] for row in R]
    U=[add(a,b) for a,b in zip(P,delta)]
    return dict(P=P,R=R,delta=delta,U=U,h=h,
        integrals=[sum(row) for row in delta],range=list(map(bernstein_range,R)))


def picard_certificate(u,data):
    h=data['h'];tube,_=local_enclosure(type('Model',(),dict(sigma=10.,rho=28.,beta=8/3))(),u,h)
    for polys in [data['P'],data['U']]:
        tube=[(min(lo,bernstein_range(p)[0]),max(hi,bernstein_range(p)[1])) for (lo,hi),p in zip(tube,polys)]
    X,Y,Z=list(map(absmax,tube));sigma,rho,beta=PARAMETERS
    RZ=absmax((rho-tube[2][1],rho-tube[2][0]))
    row=max(2*sigma,RZ+1+X,Y+X+beta)
    column=max(sigma+RZ+Y,sigma+1+X,X+beta)
    jac=sqrt_upper(row*column)
    CZ=absmax((sigma+rho-tube[2][1],sigma+rho-tube[2][0]))
    L=max(Q(0),sqrt_upper(CZ*CZ+Y*Y)/2-1)
    q=Q(exp_upper_int(L*h,1<<256),1<<256)
    d=sqrt_upper(sum(absmax(bernstein_range(poly))**2 for poly in data['delta']))
    return h*q*jac*d


def qae_count(scale,tolerance):
    scale=Q(scale);tolerance=Q(tolerance)
    if scale==0:return 0
    pi=Q(22,7);required=max(4*pi*scale/tolerance,2*pi*sqrt_upper(scale/tolerance),Q(1))
    n=(required.numerator+required.denominator-1)//required.denominator
    M=1<<max(0,(n-1).bit_length());q=1-8/(pi*pi);r=1
    while sum(math.comb(r,k)*q**k*(1-q)**(r-k) for k in range(r//2+1,r+1))>Q(1,60):r+=2
    return r*(2*M-1)


def reference(u,h):
    with mp.workdps(100):
        pars=[mp.mpf(a.numerator)/a.denominator for a in PARAMETERS]
        state=[mp.mpf(float(x)) for x in u];dt=mp.mpf(str(h))/10
        for _ in range(10):state=[mp.polyval(a,dt) for a in taylor_coefficients(state,80,pars)]
        return state


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)


def finite_oracle():
    """An actually built lookup reference oracle; its whole table cost is paid."""
    from qiskit import QuantumCircuit,QuantumRegister,transpile
    from qiskit.quantum_info import Statevector
    from sparse_compiler import conditional
    data=residual([1,1,1],.01,4);poly=data['R'][1];lo,hi=bernstein_range(poly)
    n=5;size=1<<n
    values=[(polyval(poly,Q(2*i+1,2*size))-lo)/(hi-lo) for i in range(size)]
    index=QuantumRegister(n,'index');flag=QuantumRegister(1,'value')
    work=QuantumRegister(n+2,'and');enable=QuantumRegister(1,'enable')
    qc=QuantumCircuit(index,flag,work,enable)
    qc.h(index)
    for i,value in enumerate(values):
        angle=2*math.asin(math.sqrt(float(value)))
        conditional(qc,list(index),[(i>>j)&1 for j in range(n)],
                    lambda en,a=angle:qc.cry(a,en,flag[0]),list(work),enable[0])
    state=Statevector.from_instruction(qc);prob=float(state.probabilities([n])[1])
    exact=sum(values)/size;compiled=transpile(qc,basis_gates=['u','cx'],optimization_level=1)
    return dict(grid=size,probability_observed=prob,probability_exact=str(exact),
        error=abs(prob-float(exact)),qubits=qc.num_qubits,gates=dict(compiled.count_ops()),
        classical_table_evaluations=size,
        caveat='explicit lookup reference, not a cheap arithmetic oracle; its generated table already determines the mean',
        continuous_integral=str(data['integrals'][1]),
        midpoint_integral=str(lo+(hi-lo)*exact),
        midpoint_bias=str(abs(lo+(hi-lo)*exact-data['integrals'][1])))


def mean_estimation_control(oracle):
    """Exact-law QAE simulation versus MC, AND the deterministic alternatives."""
    from carleman import qae_distribution
    from scipy.stats import binom
    data=residual([1,1,1],.01,4);poly=data['R'][1];lo,hi=bernstein_range(poly)
    N=oracle['grid'];values=[(polyval(poly,Q(2*i+1,2*N))-lo)/(hi-lo) for i in range(N)]
    p=sum(values)/N;variance=sum((x-p)**2 for x in values)/N;rows=[];repeats=9
    for bits in [4,6,8,10,12]:
        distribution=qae_distribution(float(p),bits);M=1<<bits
        estimates=np.sin(np.pi*np.arange(M)/M)**2
        order=np.argsort(estimates);estimates=estimates[order];single=distribution[order]
        cdf=np.cumsum(single);cdf=np.clip(cdf,0,1)
        median_cdf=binom.sf(repeats//2,repeats,cdf)
        mass=np.maximum(0,np.diff(np.r_[0.,median_cdf]));mass/=mass.sum()
        calls=repeats*(2*M-1)
        rmse=float(np.sqrt(np.sum(mass*(estimates-float(p))**2)))
        continuous=float(np.sqrt(np.sum(mass*(float(lo)+float(hi-lo)*estimates-float(data['integrals'][1]))**2)))
        rows.append(dict(phase_bits=bits,QAE_A_calls=calls,median_repeats=repeats,
            QAE_normalized_RMSE_exact_law=rmse,
            MC_normalized_RMSE_same_function_calls=float(sqrt_upper(variance/Q(calls))),
            continuous_integral_QAE_RMSE_including_midpoint_bias=continuous,
            classical_exact_finite_sum_function_calls=N,
            classical_exact_continuous_coefficient_terms=len(poly),
            scope='conditional query comparison; QAE distribution simulated classically; no end-to-end quantum speedup'))
    return rows


def study():
    OUT.mkdir(parents=True,exist_ok=True);rows=[];details=[]
    for name,u in [('standard',(1,1,1)),('attractor',(-8,8,27))]:
        for h in [.001,.005,.01]:
            truth=reference(u,h);predictors=predictor_bounds(u,h,12)
            for p in [2,3,4,6,8,12]:
                data=residual(u,h,p);bound=picard_certificate(u,data)
                with mp.workdps(100):
                    approximation=[sum(row) for row in data['U']]
                    error=mp.sqrt(sum((mp.mpf(a.numerator)/a.denominator-b)**2 for a,b in zip(approximation,truth)))
                    assert error<=mp.mpf(bound.numerator)/bound.denominator
                details.append(dict(initial=name,h=h,p=p,bound=str(bound),
                    residual_coefficients=[[str(a) for a in row] for row in data['R']],
                    integrals=list(map(str,data['integrals'])),observed_local_error=mp.nstr(error,24)))
                for epsilon in [1e-6,1e-8,1e-10]:
                    eps=Q(str(epsilon));pred=predictors[p-1]['error']
                    stop=pred<=eps;feasible=bound<=eps/3
                    calls=0;grid_bits=0
                    for poly,(lo,hi) in zip(data['R'],data['range']):
                        delta=eps/(6*sqrt_upper(Q(3)))
                        calls+=qae_count(hi-lo,delta)
                        second=absmax(bernstein_range(derivative(derivative(poly))))
                        needed=sqrt_upper(second/(24*delta))
                        n=(needed.numerator+needed.denominator-1)//needed.denominator
                        grid_bits=max(grid_bits,max(0,(max(n,1)-1).bit_length()))
                    rows.append(dict(initial=name,h=h,p=p,epsilon=epsilon,
                        Taylor_bound=outward_decimal(pred),Picard_bound=outward_decimal(bound),
                        observed_Picard_error=mp.nstr(error,24),classical_predictor_stop=stop,
                        residual_correction_feasible=feasible,
                        proposed_QAE_calls=calls,midpoint_index_bits=grid_bits,
                        gauss_nodes_exact_for_residual=p+1,
                        classical_residual_convolution_products=2*(p+1)**2,
                        classical_integral_coefficient_terms=sum(len(a) for a in data['R']),
                        integral_already_exact_from_known_coefficients=True,
                        scope='local residual-integration experiment; no global quantum trajectory certificate'))
    write_csv(OUT/'residual_integration.csv',rows)
    (OUT/'residual_exact.json').write_text(json.dumps(details,indent=2))
    oracle=finite_oracle();(OUT/'residual_oracle.json').write_text(json.dumps(oracle,indent=2))
    write_csv(OUT/'mean_estimation_control.csv',mean_estimation_control(oracle))
    summary=dict(local_cases=len(details),resource_cases=len(rows),
        feasible_needed_cases=sum(r['residual_correction_feasible'] and not r['classical_predictor_stop'] for r in rows),
        advantage_status='no Lorenz-specific advantage found: residual polynomials integrate exactly classically',
        formal_advantage_scope='Hölder black-box query class, not explicit analytic polynomial Lorenz',oracle=oracle)
    (OUT/'residual_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))


if __name__=='__main__':study()
