"""Direct sparse Gauss-stage systems for the known Lorenz linear correction.

The build never precomputes propagators E_i or solves a stage system.  All
reported singular values and accuracy comparisons are floating diagnostics,
not interval certificates, quantum circuits, or gate complexity claims.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from pathlib import Path
import argparse,csv,json,time
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu,eigsh,LinearOperator
from scipy.special import eval_legendre
from spacetime_candidate import Guide,aligned_mesh


@dataclass
class FullStageSystem:
    matrix: object
    rhs: np.ndarray
    mesh: np.ndarray
    stage_count: int
    endpoint_indices: np.ndarray
    local_blocks: list
    previous_blocks: list
    local_rhs: list
    metadata: dict


def tableau(s):
    """Gauss tableau via orthogonal kernels, avoiding power Vandermonde fits."""
    nodes,weights=np.polynomial.legendre.leggauss(s)
    c=(nodes+1)/2;b=weights/2
    P=np.array([eval_legendre(k,nodes) for k in range(s+1)])
    a=np.array([[b[j]*(c[i]+sum(P[k,j]*(P[k+1,i]-P[k-1,i])/2
                    for k in range(1,s))) for j in range(s)] for i in range(s)])
    assert np.max(np.abs(a.sum(axis=1)-c))<5e-13
    return a,b,c


def build_fullstage(guide,M,s=8,formulation='increment',stage_weight='quadrature'):
    """Return CSR B and d with s stages and one endpoint per interval.

    state: stages are w_j*g(t_i+c_j h).
    increment: stages are w_j*(g(t_i+c_j h)-g(t_i)).
    Endpoints are always the unweighted normalized correction g.  Quadrature
    weights w=sqrt(b) are a diagonal change of both unknowns and stage rows.
    These operations change conditioning/history normalization but no oracle
    has been silently assigned their matrix spectral norm as its alpha.
    """
    if formulation not in ['state','increment']:raise ValueError(formulation)
    if stage_weight not in ['quadrature','unit']:raise ValueError(stage_weight)
    mesh=aligned_mesh(guide,M);a,b,c=tableau(s)
    w=np.sqrt(b) if stage_weight=='quadrature' else np.ones(s)
    d=3*(s+1);I=np.eye(3);diags=[];previous=[];rhs=[]
    start=time.perf_counter()
    for left,right in zip(mesh,mesh[1:]):
        h=right-left;pairs=[guide.evaluate(left+ci*h) for ci in c]
        W=np.zeros((d,d));C=np.zeros((d,3));r=np.zeros(d)
        for j in range(s):
            for k,(Ak,rk) in enumerate(pairs):
                W[3*j:3*j+3,3*k:3*k+3]=w[j]/w[k]*((I if j==k else 0)-h*a[j,k]*Ak)
                r[3*j:3*j+3]+=w[j]*h*a[j,k]*rk
            C[3*j:3*j+3]=(-w[j]*I if formulation=='state' else
                -w[j]*h*sum(a[j,k]*pairs[k][0] for k in range(s)))
        W[-3:,-3:]=I
        for k,(Ak,rk) in enumerate(pairs):
            W[-3:,3*k:3*k+3]=-h*b[k]*Ak/w[k]
            r[-3:]+=h*b[k]*rk
        C[-3:]=(-I if formulation=='state' else -I-h*sum(b[k]*pairs[k][0] for k in range(s)))
        diags.append(W);previous.append(C);rhs.append(r)
    # Sparse assembly leaves all known-zero entries out; no global dense B.
    blocks={};rows=[];cols=[];vals=[]
    for i,(W,C) in enumerate(zip(diags,previous)):
        rr,cc=np.nonzero(W);rows.extend((i*d+rr).tolist());cols.extend((i*d+cc).tolist());vals.extend(W[rr,cc].tolist())
        if i:
            rr,cc=np.nonzero(C);rows.extend((i*d+rr).tolist());cols.extend(((i-1)*d+d-3+cc).tolist());vals.extend(C[rr,cc].tolist())
    matrix=sparse.coo_matrix((vals,(rows,cols)),shape=(M*d,M*d)).tocsr()
    matrix.sum_duplicates();matrix.eliminate_zeros()
    meta=dict(formulation=formulation,stage_weight=stage_weight,M=M,s=s,
        dimension=M*d,coefficient_evaluations=M*s,
        build_stage_linear_solves=0,build_propagator_compilations=0,
        build_seconds=time.perf_counter()-start,
        unknown_order='per interval: s three-component weighted stages, then three-component endpoint',
        mathematical_status='floating discretization; not an interval accuracy certificate')
    return FullStageSystem(matrix,np.concatenate(rhs),mesh,s,np.array([i*d+d-3+j for i in range(M) for j in range(3)]),
        diags,previous,rhs,meta)


def structured_solve(system):
    """O(M) direct classical baseline: one 3s-square single-RHS solve per step."""
    s=system.stage_count;previous=np.zeros(3);history=[];start=time.perf_counter()
    for W,C,r in zip(system.local_blocks,system.previous_blocks,system.local_rhs):
        q=r-C@previous
        stages=np.linalg.solve(W[:3*s,:3*s],q[:3*s])
        previous=q[-3:]-W[-3:,:3*s]@stages
        history.append(np.r_[stages,previous])
    return np.concatenate(history),dict(classical_stage_solves=len(system.local_blocks),
        classical_stage_dimension=3*s,classical_rhs_columns=1,
        classical_structured_seconds=time.perf_counter()-start,
        classical_precomputed_propagators=0)


def singular_diagnostics(matrix):
    """Estimate extremal singular values using sparse LU only as a diagnostic.

    lambda_max(B.T B), lambda_max(B^-T B^-1) are obtained by Lanczos.
    Sparse LU is NOT part of the claimed quantum algorithm or a free oracle.
    Residuals measure numerical consistency, not validated eigenvalue bounds.
    """
    matrix=sparse.csr_matrix(matrix);n=matrix.shape[0];start=time.perf_counter()
    if n<=96:
        values=np.linalg.svd(matrix.toarray(),compute_uv=False)
        return dict(sigma_max=float(values[0]),sigma_min=float(values[-1]),
            numerical_condition_number=float(values[0]/values[-1]),
            singular_method='dense SVD on tiny test',singular_max_eigen_residual=0.,
            singular_inverse_eigen_residual=0.,diagnostic_factorization_seconds=0.,
            singular_diagnostic_seconds=time.perf_counter()-start)
    lu=splu(matrix.tocsc());factor_seconds=time.perf_counter()-start
    normal=LinearOperator((n,n),matvec=lambda x:matrix.T@(matrix@x),dtype=float)
    inverse=LinearOperator((n,n),matvec=lambda x:lu.solve(lu.solve(x),trans='T'),dtype=float)
    v0=np.random.default_rng(1729).normal(size=n)
    maximum,vec=eigsh(normal,k=1,which='LA',v0=v0,tol=2e-9,maxiter=800)
    inverted,ivec=eigsh(inverse,k=1,which='LA',v0=v0,tol=2e-9,maxiter=800)
    high=float(maximum[0]);ihigh=float(inverted[0])
    rmax=float(np.linalg.norm(normal@vec[:,0]-high*vec[:,0])/high)
    rinv=float(np.linalg.norm(inverse@ivec[:,0]-ihigh*ivec[:,0])/ihigh)
    return dict(sigma_max=high**.5,sigma_min=ihigh**-.5,
        numerical_condition_number=(high*ihigh)**.5,
        singular_method='Lanczos on B.T B and sparse-LU inverse normal operator; numerical only',
        singular_max_eigen_residual=rmax,singular_inverse_eigen_residual=rinv,
        diagnostic_factorization_seconds=factor_seconds,
        singular_diagnostic_seconds=time.perf_counter()-start)


def local_block_precondition(system):
    """Explicit left block-Jacobi diagnostic, with four-RHS solves charged.

    This intentionally exposes the cost: each inverse action on previous-state
    three columns and RHS needs four RHS columns of a 3s stage solve. No cheap
    quantum application of these inverses is asserted.
    """
    s=system.stage_count;d=3*(s+1);M=len(system.local_blocks)
    blocks=[];rhs=[];rows=[];cols=[];vals=[];start=time.perf_counter()
    for i,(W,C,r) in enumerate(zip(system.local_blocks,system.previous_blocks,system.local_rhs)):
        v=np.column_stack([C,r]);top=np.linalg.solve(W[:3*s,:3*s],v[:3*s])
        bottom=v[-3:]-W[-3:,:3*s]@top;solved=np.vstack([top,bottom])
        Cp=solved[:,:3];rp=solved[:,3];blocks.append(Cp);rhs.append(rp)
        for j in range(d):rows.append(i*d+j);cols.append(i*d+j);vals.append(1.)
        if i:
            rr,cc=np.nonzero(Cp);rows.extend((i*d+rr).tolist());cols.extend(((i-1)*d+d-3+cc).tolist());vals.extend(Cp[rr,cc].tolist())
    B=sparse.coo_matrix((vals,(rows,cols)),shape=(M*d,M*d)).tocsr()
    return B,np.concatenate(rhs),dict(preconditioner='explicit local block-Jacobi left inverse',
        preconditioner_classical_stage_solves=M,preconditioner_stage_dimension=3*s,
        preconditioner_rhs_columns=4,preconditioner_compile_seconds=time.perf_counter()-start,
        preconditioner_quantum_application='not implemented; inverse coefficients/normalization and loading must be charged')


def load_cases():
    coarse=json.loads(Path('results/global_correction/precision_certificates.json').read_text())
    baseline=json.loads(Path('results/global_correction/certificates.json').read_text())
    cases=[]
    for c in coarse:
        if c['vector_degree']!=12:continue
        source=next(b for b in baseline if b['initial']==c['initial'] and b['guide_rtol']==c['guide_rtol'])
        assert source['accuracy_gate_pass'] and source['times']==c['times'] and source['nodes']==c['nodes']
        exact=[Q(x)-Q(y) for x,y in zip(source['endpoint'],c['nodes'][-1])]
        reference=np.array(list(map(float,exact)))
        conversion=float(sum((Q(float(x))-y)**2 for x,y in zip(reference,exact)))**.5
        reference_error=float(Q(source['bounds']['terminal_linear_error'])+Q(source['endpoint_rounding']))+conversion
        cases.append((c,Guide(c),reference,reference_error,float(Q(source['bounds']['nonlinear_remainder']))))
    return cases


def diagnose_system(system,physical_scale,reference,reference_error,nonlinear_error):
    B=system.matrix;q=system.rhs;x,accounting=structured_solve(system)
    start=time.perf_counter();direct=splu(B.tocsc()).solve(q);global_seconds=time.perf_counter()-start
    relative=float(np.linalg.norm(direct-x)/max(np.linalg.norm(x),1e-300))
    residual=float(np.linalg.norm(B@x-q)/(np.linalg.norm(B.data)*np.linalg.norm(x)+np.linalg.norm(q)))
    endpoint=x[-3:];error=float(np.linalg.norm(physical_scale*endpoint-reference))
    endpoint_probability=float(endpoint@endpoint/(x@x))
    rownorm=np.asarray(abs(B).sum(axis=1)).ravel();colnorm=np.asarray(abs(B).sum(axis=0)).ravel()
    row_nnz=np.diff(B.indptr);col_nnz=np.diff(B.tocsc().indptr)
    result=dict(system.metadata,**accounting,**singular_diagnostics(B),
        nonzero_entries=B.nnz,max_row_nonzeros=int(max(row_nnz)),max_column_nonzeros=int(max(col_nnz)),
        maximum_absolute_entry=float(max(abs(B.data))),row_sum_max=float(max(rownorm)),column_sum_max=float(max(colnorm)),
        numerical_sqrt_one_inf_norm=float(np.sqrt(max(rownorm)*max(colnorm))),
        numerical_norm_is_not_block_encoding_alpha=True,
        physical_scale=physical_scale,endpoint_correction_norm=float(physical_scale*np.linalg.norm(endpoint)),
        normalized_full_history_endpoint_probability=endpoint_probability,
        endpoint_amplification_proxy=endpoint_probability**-.5,
        endpoint_probability_scope='unfilled full stage history, changed by stage weighting/padding',
        observed_discretization_endpoint_error=error,
        reference_linear_error_upper=reference_error,certified_baseline_nonlinear_remainder=nonlinear_error,
        combined_endpoint_error_numerical_screen=error+reference_error+nonlinear_error,
        numerical_target_1e8_pass=bool(error+reference_error+nonlinear_error<=1e-8),
        direct_global_vs_structured_relative_difference=relative,
        structured_normalized_residual=residual,global_sparse_solve_diagnostic_seconds=global_seconds,
        new_discretization_interval_certificate=False,
        resource_scope='matrix build and classical diagnostic, not a QLSA execution')
    assert relative<1e-8 and residual<1e-10
    return result,x


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)


def run(out=Path('results/fullstage_qlsa'),quick=False):
    out.mkdir(parents=True,exist_ok=True);rows=[];preconditioners=[]
    for certificate,guide,ref,ref_error,nonlinear in load_cases():
        name=certificate['initial'];base=128 if name=='standard' else 256
        for M in ([base] if quick else [base,2*base]):
            for s in ([8] if quick else [4,8,12]):
                for formulation in ['state','increment']:
                    system=build_fullstage(guide,M,s,formulation)
                    row,x=diagnose_system(system,guide.scale,ref,ref_error,nonlinear)
                    row['initial']=name;rows.append(row)
                    print(name,M,s,formulation,'condition',row['numerical_condition_number'],'p',row['normalized_full_history_endpoint_probability'],'error',row['combined_endpoint_error_numerical_screen'],flush=True)
                    if M==base and s==8:
                        P,pr,meta=local_block_precondition(system)
                        answer=splu(P.tocsc()).solve(pr)
                        rel=float(np.linalg.norm(answer-x)/max(np.linalg.norm(x),1e-300))
                        assert rel<1e-8
                        preconditioners.append(dict(initial=name,M=M,s=s,formulation=formulation,
                            **meta,**singular_diagnostics(P),solution_relative_difference=rel,
                            maximum_absolute_entry=float(max(abs(P.data))),nonzero_entries=P.nnz,
                            warning='Inverse compile is explicit and charged; no efficient quantum preconditioner is claimed'))
        print('completed full-stage case',name,flush=True)
    write_csv(out/'stage_systems.csv',rows)
    (out/'stage_preconditioners.json').write_text(json.dumps(preconditioners,indent=2))
    summary=dict(cases=len(rows),numerical_target_passes=sum(r['numerical_target_1e8_pass'] for r in rows),
        maximum_global_structured_relative_difference=max(r['direct_global_vs_structured_relative_difference'] for r in rows),
        maximum_normalized_equation_residual=max(r['structured_normalized_residual'] for r in rows),
        maximum_singular_eigen_residual=max(max(r['singular_max_eigen_residual'],r['singular_inverse_eigen_residual']) for r in rows),
        quantum_status='No QLSA run or quantum advantage established; stage system does avoid E,d preprocessing',
        mathematical_status='Numerical accuracy screen using previously certified linear reference; new full-stage discretization is not interval certified',
        preconditioner_status='Four-RHS classical inverse compilation charged, not a free quantum primitive')
    (out/'stage_summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('results/fullstage_qlsa'))
    parser.add_argument('--quick',action='store_true');args=parser.parse_args()
    run(args.out,args.quick)
