"""Numerical audit of global space-time linear solving for Lorenz correction.

No matrix norm below is claimed to be an implemented block-encoding alpha.
No quantum oracle, QLSA circuit or hardware speedup is implemented here.
Exact prefix preconditioning is tested separately from changing discretization.
"""
from pathlib import Path
from fractions import Fraction as Q
from math import ceil,log2,sqrt
import argparse,csv,json,time
import numpy as np
from scipy.linalg import svdvals
from scipy.special import eval_legendre
from global_correction import hermite,PARAMETERS

OUT=Path('results/quantum_candidates')


class Guide:
    def __init__(self,certificate):
        self.times=np.array([float(Q(x)) for x in certificate['times']])
        self.exact_times=list(map(Q,certificate['times']))
        self.nodes=[list(map(Q,row)) for row in certificate['nodes']]
        self.scale=float(Q(certificate['bounds']['a']))
        self.curves=[];self.residuals=[]
        sigma,rho,beta=PARAMETERS
        for i in range(len(self.nodes)-1):
            h=self.exact_times[i+1]-self.exact_times[i]
            c=hermite(self.nodes[i],self.nodes[i+1],h)
            r=[[Q(0)]*15 for _ in range(3)]
            for k in range(8):
                r[0][k]=sigma*(c[1][k]-c[0][k])
                r[1][k]=rho*c[0][k]-c[1][k]
                r[2][k]=-beta*c[2][k]
            for j in range(8):
                for k in range(8):
                    r[1][j+k]-=c[0][j]*c[2][k]
                    r[2][j+k]+=c[0][j]*c[1][k]
            for axis in range(3):
                for k in range(7):r[axis][k]-=(k+1)*c[axis][k+1]/h
            self.curves.append(np.array([[float(x) for x in a] for a in c]))
            # First cancel polynomial coefficients EXACTLY, then evaluate the
            # small residual. Direct f(c)-c' loses unnecessary precision.
            self.residuals.append(np.array([[float(x) for x in a] for a in r]))

    def evaluate(self,t):
        i=int(np.clip(np.searchsorted(self.times,t,side='right')-1,0,len(self.curves)-1))
        tau=(t-self.times[i])/(self.times[i+1]-self.times[i])
        tau=min(1.,max(0.,tau))
        c=np.polynomial.polynomial.polyval(tau,self.curves[i].T)
        r=np.polynomial.polynomial.polyval(tau,self.residuals[i].T)/self.scale
        x,y,z=c
        A=np.array([[-10.,10.,0.],[28.-z,-1.,-x],[y,x,-float(8/3)]])
        return A,r


def midpoint_system(guide,M,T=5.):
    h=T/M;As=[];rs=[]
    for t in (np.arange(M)+.5)*h:
        A,r=guide.evaluate(t);As.append(A);rs.append(r)
    matrix=np.zeros((3*M,3*M));rhs=np.zeros(3*M);I=np.eye(3)
    for k,(A,r) in enumerate(zip(As,rs)):
        matrix[3*k:3*k+3,3*k:3*k+3]=I-h*A/2
        if k:matrix[3*k:3*k+3,3*k-3:3*k]=-(I+h*A/2)
        rhs[3*k:3*k+3]=h*r
    # Classical forward solver uses the ORIGINAL block structure, not dense
    # inversion of this artificially enlarged matrix.
    state=np.zeros(3);history=[];start=time.perf_counter()
    for A,r in zip(As,rs):
        state=np.linalg.solve(I-h*A/2,(I+h*A/2)@state+h*r)
        history.append(state.copy())
    elapsed=time.perf_counter()-start
    return matrix,rhs,np.array(history),elapsed


def volterra_trapezoidal(guide,M,T=5.):
    h=T/M;data=[guide.evaluate(t) for t in np.arange(M+1)*h]
    matrix=np.zeros((3*M,3*M));rhs=np.zeros(3*M);I=np.eye(3)
    prefix_r=h*data[0][1]/2
    for k in range(1,M+1):
        A,r=data[k];row=slice(3*k-3,3*k)
        for j in range(1,k):matrix[row,3*j-3:3*j]=-h*data[j][0]
        matrix[row,row]=I-h*A/2
        rhs[row]=prefix_r+h*r/2
        prefix_r+=h*r
    # Prefix-cached forward recurrence is O(M), despite the dense lower
    # triangular Volterra matrix. A quadratic classical solve is unfair.
    prefix=h*data[0][1]/2;history=[];start=time.perf_counter()
    for A,r in data[1:]:
        state=np.linalg.solve(I-h*A/2,prefix+h*r/2)
        history.append(state.copy());prefix+=h*(A@state+r)
    elapsed=time.perf_counter()-start
    return matrix,rhs,np.array(history),elapsed


def gauss_legendre_two_stage(guide,M,T=5.):
    """Order4 collocation; charge every 6x6 transition construction solve.

    The reference global matrix requires E,d: four RHS columns per local solve.
    A classical endpoint calculation instead solves only the one RHS it needs.
    """
    h=T/M;I=np.eye(3);r3=np.sqrt(3.)
    a=np.array([[.25,.25-r3/6],[.25+r3/6,.25]])
    c=np.array([.5-r3/6,.5+r3/6]);transitions=[];stages=[]
    start=time.perf_counter()
    for k in range(M):
        pairs=[guide.evaluate((k+ci)*h) for ci in c]
        A1,r1=pairs[0];A2,r2=pairs[1]
        block=np.block([[I-h*a[0,0]*A1,-h*a[0,1]*A1],
                        [-h*a[1,0]*A2,I-h*a[1,1]*A2]])
        all_rhs=np.column_stack([np.vstack([A1,A2]),np.r_[r1,r2]])
        K=np.linalg.solve(block,all_rhs)
        E=I+h*(K[:3,:3]+K[3:,:3])/2
        d=h*(K[:3,3]+K[3:,3])/2
        transitions.append((E,d));stages.append((block,A1,r1,A2,r2))
    preprocessing_seconds=time.perf_counter()-start
    matrix=np.zeros((3*M,3*M));rhs=np.zeros(3*M)
    for k,(E,d) in enumerate(transitions):
        matrix[3*k:3*k+3,3*k:3*k+3]=I
        if k:matrix[3*k:3*k+3,3*k-3:3*k]=-E
        rhs[3*k:3*k+3]=d
    state=np.zeros(3);history=[];start=time.perf_counter()
    for E,d in transitions:
        state=E@state+d;history.append(state.copy())
    forward_seconds=time.perf_counter()-start
    # Independent classical realization, one RHS per step, no E,d preparation.
    direct=np.zeros(3);direct_history=[];start=time.perf_counter()
    for block,A1,r1,A2,r2 in stages:
        K=np.linalg.solve(block,np.r_[A1@direct+r1,A2@direct+r2])
        direct=direct+h*(K[:3]+K[3:])/2;direct_history.append(direct.copy())
    direct_seconds=time.perf_counter()-start
    return matrix,rhs,np.array(history),forward_seconds,dict(
        transition_precompute_6_by_6_solves=M,transition_precompute_rhs_columns=4,
        transition_precompute_seconds=preprocessing_seconds,
        classical_direct_6_by_6_solves=M,classical_direct_rhs_columns=1,
        classical_direct_seconds=direct_seconds,
        direct_stage_vs_transition_relative_difference=float(np.linalg.norm(np.array(direct_history)-history)/max(np.linalg.norm(history),1e-300)))


def aligned_mesh(guide,M):
    """Keep every guide polynomial boundary; bisect largest intervals."""
    mesh=list(guide.times)
    if M<len(mesh)-1:raise ValueError('Insufficient nodes to retain guide boundaries')
    while len(mesh)-1<M:
        i=int(np.argmax(np.diff(mesh)));mesh.insert(i+1,(mesh[i]+mesh[i+1])/2)
    return np.array(mesh)


def gauss_collocation_aligned(guide,M,stage_count=8):
    """Higher-order control to avoid unresolved polynomial-residual aliasing."""
    mesh=aligned_mesh(guide,M);nodes,weights=np.polynomial.legendre.leggauss(stage_count)
    c=(nodes+1)/2;b=weights/2
    # Stable orthogonal-kernel integration of the Lagrange basis. Avoid an
    # ill-conditioned power-basis Vandermonde inversion.
    P=np.array([eval_legendre(k,nodes) for k in range(stage_count+1)])
    a=np.empty((stage_count,stage_count))
    for i in range(stage_count):
        for j in range(stage_count):
            a[i,j]=b[j]*(c[i]+sum(P[k,j]*(P[k+1,i]-P[k-1,i])/2 for k in range(1,stage_count)))
    if np.max(np.abs(a.sum(axis=1)-c))>1e-12:raise AssertionError('Collocation row consistency')
    I=np.eye(3);transitions=[];stages=[];start=time.perf_counter()
    for left,right in zip(mesh,mesh[1:]):
        h=right-left;pairs=[guide.evaluate(left+ci*h) for ci in c]
        block=np.zeros((3*stage_count,3*stage_count))
        for i,(Ai,ri) in enumerate(pairs):
            for j in range(stage_count):block[3*i:3*i+3,3*j:3*j+3]=-h*a[i,j]*Ai+(I if i==j else 0)
        all_rhs=np.column_stack([np.vstack([Ai for Ai,ri in pairs]),np.concatenate([ri for Ai,ri in pairs])])
        K=np.linalg.solve(block,all_rhs).reshape(stage_count,3,4)
        E=I+h*np.einsum('s,sij->ij',b,K[:,:,:3]);d=h*np.einsum('s,si->i',b,K[:,:,3])
        transitions.append((E,d));stages.append((block,pairs,h))
    preprocessing_seconds=time.perf_counter()-start
    matrix=np.zeros((3*M,3*M));rhs=np.zeros(3*M)
    for k,(E,d) in enumerate(transitions):
        matrix[3*k:3*k+3,3*k:3*k+3]=I
        if k:matrix[3*k:3*k+3,3*k-3:3*k]=-E
        rhs[3*k:3*k+3]=d
    state=np.zeros(3);history=[];start=time.perf_counter()
    for E,d in transitions:state=E@state+d;history.append(state.copy())
    forward_seconds=time.perf_counter()-start
    direct=np.zeros(3);direct_history=[];start=time.perf_counter()
    for block,pairs,h in stages:
        K=np.linalg.solve(block,np.concatenate([Ai@direct+ri for Ai,ri in pairs])).reshape(stage_count,3)
        direct=direct+h*np.einsum('s,si->i',b,K);direct_history.append(direct.copy())
    direct_seconds=time.perf_counter()-start
    return matrix,rhs,np.array(history),forward_seconds,dict(
        transition_precompute_stage_system_dimension=3*stage_count,
        transition_precompute_stage_solves=M,transition_precompute_rhs_columns=4,
        transition_precompute_seconds=preprocessing_seconds,
        classical_direct_stage_solves=M,classical_direct_rhs_columns=1,
        classical_direct_seconds=direct_seconds,
        collocation_stage_count=stage_count,mesh_kind='guide boundaries retained; largest intervals bisected',
        direct_stage_vs_transition_relative_difference=float(np.linalg.norm(np.array(direct_history)-history)/max(np.linalg.norm(history),1e-300)))


def diagnostics(matrix,rhs,history,reference,scale,name,M,elapsed,reference_error=0.,nonlinear_error=0.):
    values=svdvals(matrix);sigma_max=float(values[0]);sigma_min=float(values[-1])
    norm_majorant=float(np.sqrt(np.linalg.norm(matrix,1)*np.linalg.norm(matrix,np.inf)))
    dense=np.linalg.solve(matrix,rhs).reshape(M,3)
    norm2=float(np.sum(history*history));endpoint2=float(history[-1]@history[-1])
    probability=endpoint2/norm2 if norm2 else 1.
    endpoint_error=float(np.linalg.norm(scale*history[-1]-reference))
    condition=sigma_max/sigma_min
    return dict(method=name,M=M,history_dimension=3*M,
        history_data_qubits=ceil(log2(3*M)),matrix_nonzero_numeric=int(np.count_nonzero(matrix)),
        numerical_norm_majorant=norm_majorant,sigma_max=sigma_max,sigma_min=sigma_min,
        numerical_condition_number=condition,
        normalized_history_endpoint_probability=probability,
        endpoint_postselection_amplification_factor=1/sqrt(probability),
        illustrative_condition_times_postselection=condition/sqrt(probability),
        illustrative_majorant_inverse_times_postselection=norm_majorant/sigma_min/sqrt(probability),
        observed_discretization_endpoint_error=endpoint_error,
        empirical_endpoint_target_pass=bool(endpoint_error<=1e-8),
        empirical_combined_endpoint_error_screen=endpoint_error+reference_error+nonlinear_error,
        empirical_combined_target_screen_pass=bool(endpoint_error+reference_error+nonlinear_error<=1e-8),
        reference_linear_endpoint_error_upper=reference_error,
        certified_nonlinear_remainder_upper=nonlinear_error,
        dense_vs_structured_solution_difference=float(np.linalg.norm(dense-history)),
        dense_vs_structured_relative_difference=float(np.linalg.norm(dense-history)/max(np.linalg.norm(history),1e-300)),
        structured_relative_linear_system_residual=float(np.linalg.norm(matrix@history.ravel()-rhs)/(np.linalg.norm(matrix)*np.linalg.norm(history)+np.linalg.norm(rhs)+1e-300)),
        structured_classical_seconds=elapsed,structured_classical_3_by_3_solves=M,
        structured_classical_work_order='O(M) with prefix reuse',
        transition_precompute_6_by_6_solves=0,transition_precompute_rhs_columns=0,
        transition_precompute_seconds=0.,classical_direct_6_by_6_solves=0,
        classical_direct_rhs_columns=0,classical_direct_seconds=0.,
        direct_stage_vs_transition_relative_difference=0.,
        guide_polynomial_preprocessing_seconds=0.,
        transition_precompute_stage_system_dimension=0,transition_precompute_stage_solves=0,
        classical_direct_stage_solves=0,collocation_stage_count=0,mesh_kind='uniform',
        matrix_norm_status='numerical diagnostic only; NOT an implemented block-encoding alpha',
        proxy_status='illustrative QLSA factors, NOT lower bounds or complete query/gate costs',
        reference_status='baseline validated polynomial endpoint for same guide; this discretization is not certified',
        omitted_quantum_costs='coefficient/time oracle, RHS preparation, QLSA precision, reflections, endpoint amplification and physical readout')


def run(out=OUT):
    out.mkdir(parents=True,exist_ok=True)
    certificates=json.loads(Path('results/global_correction/precision_certificates.json').read_text())
    baseline=json.loads(Path('results/global_correction/certificates.json').read_text())
    rows=[];checks=[]
    for c in certificates:
        if c['vector_degree']!=12:continue
        name=c['initial'];group_start=len(rows);start=time.perf_counter();guide=Guide(c)
        guide_preprocessing_seconds=time.perf_counter()-start
        source=next(b for b in baseline if b['initial']==name and b['guide_rtol']==c['guide_rtol'])
        assert source['accuracy_gate_pass']
        assert source['times']==c['times'] and source['nodes']==c['nodes']
        # Baseline endpoint is the corrected v; subtract its guide endpoint
        # to obtain physical g, whose history is encoded in this audit.
        reference_exact=[Q(x)-Q(y) for x,y in zip(source['endpoint'],c['nodes'][-1])]
        reference=np.array(list(map(float,reference_exact)))
        subtraction_error=float(sum((Q(float(x))-y)**2 for x,y in zip(reference,reference_exact)))**.5
        reference_error=float(Q(source['bounds']['terminal_linear_error'])+Q(source['endpoint_rounding']))+subtraction_error
        nonlinear_error=float(Q(source['bounds']['nonlinear_remainder']))
        for M in [16,32,64,128]:
            A,b,history,elapsed=midpoint_system(guide,M)
            raw=diagnostics(A,b,history,reference,guide.scale,'differential_midpoint',M,elapsed,reference_error,nonlinear_error)
            raw.update(initial=name,guide_rtol=c['guide_rtol']);rows.append(raw)
            L=np.kron(np.tril(np.ones((M,M))),np.eye(3))
            P=L@A;q=L@b
            exact=diagnostics(P,q,history,reference,guide.scale,'prefix_preconditioned_midpoint',M,elapsed,reference_error,nonlinear_error)
            exact.update(initial=name,guide_rtol=c['guide_rtol']);rows.append(exact)
            checks.append(dict(initial=name,M=M,
                numerical_exact_preconditioning_solution_difference=float(np.linalg.norm(np.linalg.solve(P,q).reshape(M,3)-history)),
                numerical_exact_preconditioning_relative_difference=float(np.linalg.norm(np.linalg.solve(P,q).reshape(M,3)-history)/max(np.linalg.norm(history),1e-300)),
                prefix_preconditioner_norm_majorant=float(M),
                prefix_norm_not_free=True))
            C,d,nystrom,nystrom_elapsed=volterra_trapezoidal(guide,M)
            trap=diagnostics(C,d,nystrom,reference,guide.scale,'volterra_trapezoidal_nodes',M,nystrom_elapsed,reference_error,nonlinear_error)
            trap.update(initial=name,guide_rtol=c['guide_rtol']);rows.append(trap)
        for M in [32,64,128,256]:
            C,d,history,elapsed,accounting=gauss_legendre_two_stage(guide,M)
            highorder=diagnostics(C,d,history,reference,guide.scale,'gauss_legendre_order4',M,elapsed,reference_error,nonlinear_error)
            highorder.update(initial=name,guide_rtol=c['guide_rtol'],structured_classical_3_by_3_solves=0,
                structured_classical_work_order='O(M) direct 6x6 stages; E,d precomputation charged separately',
                collocation_stage_count=2,transition_precompute_stage_system_dimension=6,
                transition_precompute_stage_solves=M,classical_direct_stage_solves=M,**accounting)
            rows.append(highorder)
        for M in [128,256]:
            if M<len(guide.times)-1:continue
            C,d,history,elapsed,accounting=gauss_collocation_aligned(guide,M)
            highorder=diagnostics(C,d,history,reference,guide.scale,'aligned_gauss_legendre_order16',M,elapsed,reference_error,nonlinear_error)
            highorder.update(initial=name,guide_rtol=c['guide_rtol'],structured_classical_3_by_3_solves=0,
                structured_classical_work_order='O(M) direct 24x24 stages; E,d four-RHS preprocessing charged',**accounting)
            rows.append(highorder)
        for row in rows[group_start:]:row['guide_polynomial_preprocessing_seconds']=guide_preprocessing_seconds
        print('space-time audit complete:',name,flush=True)
    consistency=dict(cases=len(rows),
        max_dense_vs_structured_relative_difference=max(r['dense_vs_structured_relative_difference'] for r in rows),
        max_structured_relative_linear_system_residual=max(r['structured_relative_linear_system_residual'] for r in rows),
        max_direct_stage_transition_relative_difference=max(r['direct_stage_vs_transition_relative_difference'] for r in rows),
        max_exact_prefix_relative_difference=max(c['numerical_exact_preconditioning_relative_difference'] for c in checks),
        classification='numerical consistency, not interval accuracy certification')
    assert consistency['max_dense_vs_structured_relative_difference']<1e-8
    assert consistency['max_direct_stage_transition_relative_difference']<1e-8
    assert consistency['max_exact_prefix_relative_difference']<1e-8
    assert consistency['max_structured_relative_linear_system_residual']<1e-10
    with (out/'spacetime.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    (out/'spacetime_preconditioner_checks.json').write_text(json.dumps(checks,indent=2))
    (out/'spacetime_consistency_checks.json').write_text(json.dumps(consistency,indent=2))
    summary=dict(cases=len(rows),T=5,Ms_low_order=[16,32,64,128],Ms_order4=[32,64,128,256],Ms_aligned_order16=[128,256],
        empirical_target_passes=sum(r['empirical_endpoint_target_pass'] for r in rows),
        empirical_combined_target_screen_passes=sum(r['empirical_combined_target_screen_pass'] for r in rows),
        norm_warning='sqrt(norm1 norminf) and SVD are numerical diagnostics, no corresponding cheap block encoding is supplied',
        conditioning_warning='Conditioning alone is insufficient to claim QLSA advantage; prefix L has a growing norm and changing RHS preparation cost',
        endpoint_probability_warning='Unpadded history only; endpoint padding can change probability and other resource costs',
        transition_preprocessing_scope='E,d compilation is charged for this formulation; a full-stage QLSA could avoid it using more unknowns and would require a separate audit',
        classical_warning='Both differential and integral forms have O(M) structured forward solutions',
        precision_warning='Only the three aligned high-order cases meet the empirical target; none is interval-certified here',
        final_output='Three physical endpoint corrections; a full classical history would also incur its output-size cost')
    (out/'spacetime_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=OUT)
    run(parser.parse_args().out)
