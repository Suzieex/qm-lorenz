"""Priority 1--4 implementation/resource study; no claimed single-step speedup.

Optimizes one-step errors on benchmark inputs. Cost/time-density is only a
screening proxy: it does not certify T=5 stability or multi-step readout budgets.
"""
from dataclasses import dataclass
from pathlib import Path
from math import ceil,log2,comb
import csv,json,gzip
import numpy as np
from qiskit import qpy,transpile,__version__ as qiskit_version
from carleman import Lorenz,Carleman
from sparse_compiler import compile_block_encoding,encoded_column,gate_resources,rules_from_model
from sparse_workflow import series_weights,row_series,build_sparse_hadamard,build_row_hadamard


@dataclass(frozen=True)
class ShiftedLorenz(Lorenz):
    pivot:tuple=(0.,0.,38.)

    @property
    def center(self):return np.array(self.pivot,dtype=float)

    def polynomials(self):
        x,y,z=self.center;S=self.scale
        constant=self.rhs(0,self.center)/S
        J=np.array([[-self.sigma,self.sigma,0.],
                    [self.rho-z,-1.,-x],[y,x,-self.beta]])
        result=[]
        for axis in range(3):
            terms={(0,0,0):float(constant[axis])}
            for j in range(3):
                p=[0,0,0];p[j]=1;terms[tuple(p)]=float(J[axis,j])
            result.append(terms)
        result[1][(1,0,1)]=-S;result[2][(1,1,0)]=S
        return result


def write_csv(path,rows):
    with open(path,'w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def taylor_predictor(model,u,h,p):
    x=[float(u[0])];y=[float(u[1])];z=[float(u[2])]
    for k in range(p):
        xz=sum(x[j]*z[k-j] for j in range(k+1));xy=sum(x[j]*y[k-j] for j in range(k+1))
        x.append(model.sigma*(y[k]-x[k])/(k+1))
        y.append((model.rho*x[k]-y[k]-xz)/(k+1))
        z.append((xy-model.beta*z[k])/(k+1))
    return np.array([np.polynomial.polynomial.polyval(h,a) for a in [x,y,z]])


def run(out=Path('results/optimization')):
    out.mkdir(parents=True,exist_ok=True)
    model=Lorenz();compiler=[];normalizations=[];search=[];predictors=[]
    verification=[]
    for N in [1,2,3,5,9,13]:
        encoding=compile_block_encoding(model,N)
        resources=gate_resources(encoding)
        compiler.append(dict(N=N,D_N=comb(N+3,3),
            grid_data_qubits=3*encoding.width,alpha=encoding.alpha,
            logical_qubits=resources['qubits'],cx=resources['elementary_operations'].get('cx',0),
            single_qubit_u=resources['elementary_operations'].get('u',0),
            depth=resources['depth']))
        if N<=3:
            c=Carleman(model,N);err=0.
            for j,power in enumerate(c.exponents):
                column=encoded_column(encoding,power)
                for i,row in enumerate(c.exponents):err=max(err,abs(column.get(row,0)-c.C[i,j]/encoding.alpha))
            invalid=encoded_column(encoding,(N+1,0,0))
            invalid_error=max([abs(v) for v in invalid.values()]+[0.])
            verification.append(dict(N=N,maximum_valid_block_error=err,
                                     invalid_column_error=invalid_error))
        if N==3:
            with gzip.open(out/'sparse_C_N3.qpy.gz','wb') as f:qpy.dump(encoding.circuit,f)
        for grouped,tight in [(False,False),(True,False),(False,True),(True,True)]:
            alpha=sum(r.weight for r in rules_from_model(model,N,grouped,tight))
            w,tail=series_weights(alpha,.001)
            normalizations.append(dict(N=N,group_diagonal=grouped,tight_capacity=tight,
                alpha=alpha,K=len(w)-1,gamma_increment=float(w[1:].sum())))
    for name in ['standard','attractor']:
        reference=np.load(f'results/longtime/reference_{name}.npz')['state']
        u=reference[0]
        pivots={'origin':(0.,0.,0.),'rho':(0.,0.,28.),'energy':(0.,0.,38.),'local':tuple(u)}
        for pivot_name,pivot in pivots.items():
            for S in [24.,32.,48.,64.]:
                shifted=ShiftedLorenz(scale=S,pivot=pivot)
                for N in [1,2,3,5,7]:
                    rules=rules_from_model(shifted,N)
                    alpha=sum(r.weight for r in rules)
                    for h in [.001,.002,.005,.01]:
                        weights,tail=series_weights(alpha,h)
                        K=len(weights)-1;c,rows=row_series(shifted,N,h,K)
                        phi=c.lift(shifted.normalize(u));B=np.linalg.norm(phi)
                        truth=reference[round(h/.001)]
                        increment=sum(rows[1:]);next_u=u+S*(increment@phi)
                        deterministic=np.linalg.norm(next_u-truth)
                        gamma=float(weights[1:].sum())
                        for p in range(min(3,N-1,K-1)+1):
                            predicted=u+S*(sum(rows[1:p+1])@phi) if p else u.copy()
                            residual=sum(rows[p+1:])
                            gamma_res=float(weights[p+1:].sum())
                            row_norms=np.linalg.norm(residual,axis=1)
                            prediction_error=np.linalg.norm(predicted-truth)
                            cutoff=1e-8 # fixed benchmark local target, not a trajectory guarantee
                            search.append(dict(initial_id=name,pivot=pivot_name,
                                pivot_x=pivot[0],pivot_y=pivot[1],pivot_z=pivot[2],
                                S=S,N=N,h=h,p=p,D_N=len(c.exponents),K=K,B=float(B),alpha=alpha,
                                gamma_increment=gamma,gamma_residual=gamma_res,
                                F_sparse=float(S*B*gamma_res),
                                F_row_sum=float(S*B*row_norms.sum()),
                                deterministic_error=float(deterministic),
                                predictor_error=float(prediction_error),
                                local_tolerance=cutoff,
                                deterministic_pass=bool(deterministic<=cutoff/3),
                                predictor_pass_empirical=bool(prediction_error<=cutoff),
                                exact_basis_input=bool(np.count_nonzero(phi)==1),
                                sparse_query_scale_per_time=float(3*K*max(1.,S*B*gamma_res/cutoff)/h),
                                row_query_scale_per_time=float(max(3.,S*B*row_norms.sum()/cutoff)/h),
                                classical_row_precompute_ops=K*c.C.nnz,
                                predictor_support=int(np.count_nonzero(sum(rows[1:p+1]))) if p else 0,
                                trajectory_certified=False))
        for h in [.001,.002,.005,.01]:
            for p in range(1,9):
                predicted=taylor_predictor(model,u,h,p)
                predictors.append(dict(initial_id=name,h=h,p=p,
                    vector_error=float(np.linalg.norm(predicted-reference[round(h/.001)])),
                    convolution_products=2*sum(k+1 for k in range(p)),
                    linear_work_order=p,reference_used_for_benchmark_only=True))
    write_csv(out/'sparse_resources.csv',compiler)
    write_csv(out/'normalizations.csv',normalizations)
    write_csv(out/'joint_search.csv',search)
    write_csv(out/'classical_predictors.csv',predictors)
    # Compare like-sized steps; do not manufacture a winner by shrinking h.
    # An already sufficient classical predictor is reported as a classical
    # stop case. Basis-state inputs remain legitimate sparse-LCU candidates;
    # only the row-precomputed backend then has its answer already available.
    selected=[]
    for name in ['standard','attractor']:
        for h in [.001,.002,.005,.01]:
            candidates=[r for r in search if r['initial_id']==name and r['h']==h
                        and r['deterministic_pass'] and not r['predictor_pass_empirical']]
            if not candidates:continue
            best=min(candidates,key=lambda r:r['sparse_query_scale_per_time'])
            selected.append(best)
    write_csv(out/'selected_per_step.csv',selected)
    (out/'verification.json').write_text(json.dumps(verification,indent=2))
    # End-to-end circuit artifacts: generated sparse increment and explicit
    # coefficient-state norm implementation. State-preparation costs stay visible.
    circuit,meta=build_sparse_hadamard(model,1,.001,np.ones(3),1,K=2)
    with gzip.open(out/'sparse_hadamard_N1_K2.qpy.gz','wb') as f:qpy.dump(circuit,f)
    row_circuit,row_meta=build_row_hadamard(model,3,.001,np.ones(3),1,predictor_order=1,K=12)
    with gzip.open(out/'row_residual_hadamard.qpy.gz','wb') as f:qpy.dump(row_circuit,f)
    row_compiled=transpile(row_circuit,basis_gates=['u','cx'],optimization_level=1)
    summary=dict(qiskit_version=qiskit_version,transpile_basis=['u','cx'],
        transpile_optimization_level=1,block_verification=verification,
        sparse_hadamard_metadata=meta,sparse_hadamard_qubits=circuit.num_qubits,
        row_residual_metadata=row_meta,row_residual_qubits=row_circuit.num_qubits,
        row_residual_gates=dict(row_compiled.count_ops()),
        classical_predictor_smallest_orders={f'{name},h={h}':min(r['p'] for r in predictors
            if r['initial_id']==name and r['h']==h and r['vector_error']<=1e-8)
            for name in ['standard','attractor'] for h in [.001,.002,.005,.01]},
        joint_search_configurations=len(search),
        deterministic_feasible=sum(r['deterministic_pass'] for r in search),
        classical_predictor_pass=sum(r['deterministic_pass'] and r['predictor_pass_empirical'] for r in search),
        caution=['No single-step quantum advantage demonstrated.',
            'Rows are classically precomputed and state preparation is not free.',
            'Joint search is single-step, not a T=5 trajectory certificate.',
            'Reference-based predictor acceptance is an empirical benchmark, not a runtime oracle.',
            'Conditional priority 5 is not promoted before a single-step advantage is established.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':run()
