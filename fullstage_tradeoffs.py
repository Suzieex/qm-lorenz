"""Numerical resource tradeoffs for complete-stage linear correction systems.

Padding and direct inverse-element readout are tested on identical equations.
Singular values are numerical diagnostics, not certified inverse-norm bounds.
No quantity called a proxy below is a gate count or a complexity lower bound.
"""
from pathlib import Path
import csv,json,math
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu
from fullstage_system import load_cases,build_fullstage,structured_solve,singular_diagnostics,tableau
from fullstage_encoding import diagonal_lcu
from residual_quantum_candidate import qae_count
from fractions import Fraction as Q


def append_endpoint_copies(matrix,rhs,x,copies):
    """Add exactly copies equations y_0=x_end, y_k=y_(k-1)."""
    matrix=sparse.csr_matrix(matrix);n=matrix.shape[0]
    if copies==0:return matrix,np.asarray(rhs),np.asarray(x)
    extra=sparse.kron(sparse.diags([np.ones(copies),-np.ones(copies-1)],
                                 [0,-1],shape=(copies,copies)),sparse.eye(3),format='csr')
    link=sparse.csr_matrix((-np.ones(3),(np.arange(3),np.arange(n-3,n))),shape=(3*copies,n))
    padded=sparse.bmat([[matrix,None],[link,extra]],format='csr')
    vector=np.r_[rhs,np.zeros(3*copies)]
    answer=np.r_[x,np.tile(x[-3:],copies)]
    assert np.linalg.norm(padded@answer-vector)<1e-9*max(1.,np.linalg.norm(vector))
    return padded,vector,answer


def uniform_sparse_alpha(matrix):
    """Explicit uniform row/column-slot walk normalization for stored matrix.

    This is achievable using entry oracles plus uniform padded-slot states,
    not the numeric spectral norm. Root's compiled oracle cost remains extra.
    Padding counts are rounded to powers of two for an actual Hadamard slot
    preparation; alpha=sqrt(Sr Sc)*maxabs(B).
    """
    row=int(max(np.diff(matrix.indptr)));col=int(max(np.diff(matrix.tocsc().indptr)))
    sr=1<<max(0,(row-1).bit_length());sc=1<<max(0,(col-1).bit_length())
    return float(np.sqrt(sr*sc)*max(abs(matrix.data))),sr,sc


def shift_alpha_fast(matrix,pad_identity=True):
    """Exactly the compiler's shift-LCU alpha, without retaining every lookup.

    Choosing the largest finite binary64 absolute value is also exact for
    their rational interpretations. Rational summation is performed AFTER
    taking that maximum. Cross-checked against diagonal_lcu on base systems.
    """
    coo=sparse.coo_matrix(matrix);n=coo.shape[0];p=1<<max(0,(n-1).bit_length())
    maxima=np.zeros(p);np.maximum.at(maxima,(coo.row-coo.col)%p,abs(coo.data))
    if pad_identity and p>n:maxima[0]=max(maxima[0],1.)
    return float(sum((Q(float(v)) for v in maxima if v),Q(0)))


def zero_field_inverse(system):
    """Exact algebraic zero-field B0 and its analytic prefix inverse P.

    Stage weights are the same stored floats as the original builder.
    No propagation matrix, local inverse or ODE solve is compiled.
    """
    s=system.stage_count;d=3*(s+1);M=len(system.mesh)-1;n=M*d
    w=np.sqrt(tableau(s)[1]) if system.metadata['stage_weight']=='quadrature' else np.ones(s)
    state=system.metadata['formulation']=='state';I=sparse.eye(n,format='lil')
    B0=I.copy();P=I.copy()
    for k in range(M):
        if k:
            for j in range(3):B0[k*d+d-3+j,(k-1)*d+d-3+j]=-1.
            if state:
                for st in range(s):
                    for j in range(3):B0[k*d+3*st+j,(k-1)*d+d-3+j]=-w[st]
        for previous in range(k):
            for j in range(3):P[k*d+d-3+j,previous*d+d-3+j]=1.
            if state:
                for st in range(s):
                    for j in range(3):P[k*d+3*st+j,previous*d+d-3+j]=w[st]
    return B0.tocsr(),P.tocsr()


def prefix_audits(cases):
    rows=[]
    for certificate,guide,reference,reference_error,nonlinear in cases:
        name=certificate['initial'];M=128 if name=='standard' else 256
        for formulation in ['state','increment']:
            sys=build_fullstage(guide,M,8,formulation,'quadrature');x,_=structured_solve(sys)
            B0,P=zero_field_inverse(sys);I=sparse.eye(sys.matrix.shape[0],format='csr')
            identity_error=P@B0-I
            assert not identity_error.nnz or max(abs(identity_error.data))<2e-14
            K=B0-sys.matrix
            # This is the cancellation-aware I-PK, not a bound on P times B.
            pre=I-P@K;pb=P@sys.rhs
            difference=P@sys.matrix-pre
            assert not difference.nnz or max(abs(difference.data))<2e-13
            residual=float(np.linalg.norm(pre@x-pb))
            assert residual<1e-9*max(1.,np.linalg.norm(pb))
            alphaB=shift_alpha_fast(sys.matrix);alphaP=shift_alpha_fast(P)
            alphaK=shift_alpha_fast(K,pad_identity=False)
            alpha_pre=shift_alpha_fast(pre)
            assert abs(alphaB-diagonal_lcu(sys.matrix).alpha)<1e-10
            sd=singular_diagnostics(pre);beta=max(1.,1/sd['sigma_min'])
            rows.append(dict(initial=name,M=M,s=8,formulation=formulation,
                dimension=pre.shape[0],original_diagonal_lcu_alpha=alphaB,
                prefix_diagonal_lcu_alpha=alphaP,perturbation_diagonal_lcu_alpha=alphaK,
                naive_composed_alpha=alphaP*alphaB,
                cancelled_composed_alpha=1+alphaP*alphaK,
                directly_reencoded_shift_alpha=alpha_pre,
                numerical_preconditioned_beta=beta,
                numerical_preconditioned_condition=sd['numerical_condition_number'],
                reencoded_alpha_beta=alpha_pre*beta,
                direct_inverse_physical_scale=2*guide.scale*beta*np.linalg.norm(pb),
                maximum_step=float(max(np.diff(sys.mesh))),
                alphaP_times_maximum_step=alphaP*max(np.diff(sys.mesh)),
                prefix_rhs_norm=float(np.linalg.norm(pb)),
                prefix_identity_max_error=float(max(abs(identity_error.data))) if identity_error.nnz else 0.,
                cancelled_matrix_max_error=float(max(abs(difference.data))) if difference.nnz else 0.,
                original_solution_preconditioned_residual=residual,
                preprocessing='analytic zero-field prefix; no local inverse or transition compilation',
                normalization='exact shift/diagonal LCU on stored finite matrix; inverse singular values numerical',
                not_free='prefix/coupling lookup, RHS prefix state loading and primitive application remain charged'))
            print('prefix audit',name,formulation,'naive',alphaP*alphaB,'cancelled',1+alphaP*alphaK,'direct',alpha_pre,'beta',beta,flush=True)
    return rows


def prefix_scaling_control():
    """Check the cancellation-aware normalization versus M at fixed T."""
    class Constant:
        times=np.array([0.,5.])
        def evaluate(self,t):
            return np.array([[-10.,10.,0.],[27.,-1.,-1.],[1.,1.,-8/3]]),np.array([1.,-2.,1.])
    rows=[]
    for M in [16,32,64,128]:
        sys=build_fullstage(Constant(),M,4,'increment','quadrature')
        B0,P=zero_field_inverse(sys);K=B0-sys.matrix
        pre=sparse.eye(sys.matrix.shape[0],format='csr')-P@K
        alphaP=shift_alpha_fast(P);alphaK=shift_alpha_fast(K,pad_identity=False)
        rows.append(dict(M=M,T=5,s=4,alphaP=alphaP,alphaK=alphaK,
                         alphaP_times_h=alphaP*5/M,
                         cancelled_composed_alpha=1+alphaP*alphaK,
                         directly_reencoded_shift_alpha=shift_alpha_fast(pre),
                         naive_composed_alpha=alphaP*shift_alpha_fast(sys.matrix),
                         scope='constant Lorenz Jacobian linear control; not nonlinear trajectory accuracy'))
    assert max(r['cancelled_composed_alpha'] for r in rows)<1.01*min(r['cancelled_composed_alpha'] for r in rows)
    return rows


def scalar_padding_control():
    """Analytic singular spectrum of D_L verifies what padding can change."""
    rows=[];M=64
    for p in [0,8,16,32,64,128]:
        L=M+p
        # Singular values of lower bidiagonal D_L with diagonal1/subdiag-1.
        high=2*math.cos(math.pi/(2*L+1));low=2*math.sin(math.pi/(4*L+2))
        D=sparse.diags([np.ones(L),-np.ones(L-1)],[0,-1]).toarray()
        singular=np.linalg.svd(D,compute_uv=False)
        assert abs(singular[0]-high)<1e-12 and abs(singular[-1]-low)<1e-12
        probability=(p+1)/L
        rows.append(dict(original_steps=M,padding_copies=p,dimension=L,
                         exact_spectrum_sigma_max_formula=high,
                         exact_spectrum_sigma_min_formula=low,
                         numerical_condition=high/low,endpoint_union_probability=probability,
                         condition_times_amplification=(high/low)/math.sqrt(probability),
                         scope='analytic scalar x_prime=0 control; not a Lorenz lower bound'))
    return rows


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader();w.writerows(rows)


def run(out=Path('results/fullstage_qlsa')):
    out.mkdir(parents=True,exist_ok=True);rows=[]
    cases=load_cases()
    for certificate,guide,reference,reference_error,nonlinear in cases:
        name=certificate['initial'];M=128 if name=='standard' else 256
        system=build_fullstage(guide,M,8,'increment','quadrature')
        x,accounting=structured_solve(system)
        eps=1e-8-nonlinear-reference_error
        assert eps>0
        for padding in [0,M//2,M,2*M]:
            B,b,padded=append_endpoint_copies(system.matrix,system.rhs,x,padding)
            sd=singular_diagnostics(B);beta=max(1.,1/sd['sigma_min'])
            slot_alpha,sr,sc=uniform_sparse_alpha(B)
            alpha=diagonal_lcu(B).alpha
            endpoint_norm=np.linalg.norm(x[-3:]);history_norm=np.linalg.norm(padded)
            probability=(padding+1)*endpoint_norm**2/history_norm**2
            rhs_norm=np.linalg.norm(b)
            # Implementing A^-1/(2 beta) leaves reciprocal-polynomial margin.
            # Physical g_j = C * <j|A^-1/(2 beta)|b/||b||>, C=2*s*beta*||b||.
            C=2*guide.scale*beta*rhs_norm
            selectors=np.zeros((B.shape[0],3));selectors[-3:]=np.eye(3)
            # Diagnostic only: three transpose solves are NOT a free row oracle.
            adjoint=splu(B.tocsc()).solve(selectors,trans='T')
            row_norms=np.linalg.norm(adjoint,axis=0)
            row_scales=guide.scale*rhs_norm*row_norms
            row_prepare_amplification=2*beta/row_norms
            assert np.allclose(row_scales*row_prepare_amplification,C,rtol=2e-14)
            each_tolerance=Q(str(eps))/3
            # Together with eta, sqrt(3)*(epsilon/3+epsilon/12)<epsilon.
            qae_calls=3*qae_count(Q(str(2*C)),each_tolerance)
            eta=eps/(12*max(C,1e-300))
            inv_degree_proxy=alpha*beta*math.log(max(3.,alpha*beta/eta))
            rows.append(dict(initial=name,M=M,s=8,padding_copies=padding,
                dimension=B.shape[0],implemented_diagonal_lcu_alpha=alpha,
                alternative_uniform_slot_alpha=slot_alpha,
                row_slots_power_two=sr,column_slots_power_two=sc,
                numerical_sigma_min=sd['sigma_min'],numerical_beta_inverse_norm=beta,
                numerical_condition=sd['numerical_condition_number'],
                unscaled_rhs_norm=rhs_norm,endpoint_union_probability=probability,
                endpoint_amplification=probability**-.5,
                alpha_beta=alpha*beta,
                alpha_beta_times_endpoint_amplification=alpha*beta/math.sqrt(probability),
                direct_inverse_physical_amplitude_scale=C,
                adjoint_row_norm_min=float(min(row_norms)),adjoint_row_norm_max=float(max(row_norms)),
                hypothetical_prepared_adjoint_readout_scale_max=float(max(row_scales)),
                reciprocal_adjoint_preparation_amplification_min=float(min(row_prepare_amplification)),
                reciprocal_adjoint_preparation_amplification_max=float(max(row_prepare_amplification)),
                row_scale_times_prep_amplification_max=float(max(row_scales*row_prepare_amplification)),
                direct_inverse_qae_calls_numerical_budget=qae_calls,
                reciprocal_degree_proxy=inv_degree_proxy,
                direct_inverse_total_query_proxy=qae_calls*inv_degree_proxy,
                physical_error_budget=eps,
                original_endpoint_error_numerical=float(np.linalg.norm(guide.scale*x[-3:]-reference)),
                padding_solution_equivalence_residual=float(np.linalg.norm(B@padded-b)),
                classification='numerical resource screen; beta not interval-certified; not quantum gate timing',
                inverse_oracle='A_inverse/(2 beta); includes reference-assisted signed matrix element readout',
                degree_proxy_status='O(alpha beta log(alpha beta/eta)) with hidden theorem constants; NOT a sufficient exact query count',
                qae_count_status='conditional sufficient A/A_dagger budget if displayed beta were a valid upper bound; not a complexity lower bound'))
            print('padding audit',name,padding,'p',probability,'alpha_beta',alpha*beta,'inverse_scale',C,flush=True)
    scalar=scalar_padding_control()
    prefix=prefix_audits(cases)
    scaling=prefix_scaling_control()
    write_csv(out/'tradeoff_padding.csv',rows)
    write_csv(out/'tradeoff_scalar_control.csv',scalar)
    write_csv(out/'tradeoff_prefix.csv',prefix)
    write_csv(out/'tradeoff_prefix_scaling_control.csv',scaling)
    (out/'tradeoff_summary.json').write_text(json.dumps(dict(
        Lorenz_padding_cases=len(rows),scalar_controls=len(scalar),Lorenz_prefix_cases=len(prefix),prefix_scaling_controls=len(scaling),
        endpoint_padding_exact_algebra=True,
        query_status='all displayed inverse-norm dependent resource values are numerical feasibility diagnostics',
        conclusion='padding helps endpoint probability but shifts inverse norm; direct inverse-element readout replaces history probability by inverse normalization',
        not_claimed=['No universal quantum lower bound','No full QSVT/QLSA circuit','No rigorous singular-value certificate']),indent=2))
    return rows


if __name__=='__main__':run()
