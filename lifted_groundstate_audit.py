"""Lorenz-specific audit of unconstrained quadratic ground-state lifting.

An exact rational PSD, symmetric, normalized moment matrix can satisfy every
lifted implicit-Euler equation while failing rank one and the original ODE
step. This does not refute algorithms that separately impose the tensor-product
constraint; it tests the sufficiency of an unconstrained ground-state solve.
"""
import argparse,json,itertools
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
from scipy.optimize import root


def det3(a):
    return (a[0][0]*(a[1][1]*a[2][2]-a[1][2]*a[2][1])
        -a[0][1]*(a[1][0]*a[2][2]-a[1][2]*a[2][0])
        +a[0][2]*(a[1][0]*a[2][1]-a[1][1]*a[2][0]))


def lifted_rows(h,sigma,rho,beta,u0):
    return [{(0,1):1+h*sigma,(0,2):-h*sigma,(0,0):-u0[0]},
            {(0,2):1+h,(0,1):-h*rho,(1,3):h,(0,0):-u0[1]},
            {(0,3):1+h*beta,(1,2):-h,(0,0):-u0[2]}]


def original_residual(mu,u0,h,sigma,rho,beta):
    x,y,z=mu
    return [x-u0[0]-h*sigma*(y-x),
            y-u0[1]-h*(rho*x-y-x*z),
            z-u0[2]-h*(x*y-beta*z)]


def solve_exact(A,b):
    rows=[list(row)+[value] for row,value in zip(A,b)];n=len(b)
    for k in range(n):
        pivot=next(i for i in range(k,n) if rows[i][k])
        rows[k],rows[pivot]=rows[pivot],rows[k];v=rows[k][k]
        rows[k]=[x/v for x in rows[k]]
        for i in range(n):
            if i==k:continue
            v=rows[i][k];rows[i]=[x-v*y for x,y in zip(rows[i],rows[k])]
    return [row[-1] for row in rows]


def audit(out=Path('results/quantum_candidates')):
    out.mkdir(parents=True,exist_ok=True)
    h=Q(1,100);sigma=Q(10);rho=Q(28);beta=Q(float(8/3))
    u0=[Q(1)]*3;mu=u0.copy()
    C=[[Q(30),beta-1,rho-2],[beta-1,Q(30),Q(0)],
       [rho-2,Q(0),Q(30)]]
    Y=[[Q(1)]+mu]
    for i in range(3):Y.append([mu[i]]+[mu[i]*mu[j]+C[i][j] for j in range(3)])
    rows=lifted_rows(h,sigma,rho,beta,u0)
    residual=[sum(c*Y[i][j] for (i,j),c in row.items()) for row in rows]
    assert residual==[Q(0)]*3
    principal=[Q(1),C[0][0],C[0][0]*C[1][1]-C[0][1]**2,det3(C)]
    dominance=min(C[i][i]-sum(abs(C[i][j]) for j in range(3) if j!=i) for i in range(3))
    assert all(x>0 for x in principal) and dominance>0
    assert all(Y[i][j]==Y[j][i] for i in range(4) for j in range(4))
    # The spurious moments even admit a distribution supported wholly INSIDE
    # the known radius44 Lorenz ball. Boundedness alone does not fix rank one.
    signs=list(itertools.product([-1,1],repeat=3))
    a=(beta-1)/30;b=(rho-2)/30
    probabilities=[(1+a*s[0]*s[1]+b*s[0]*s[2])/8 for s in signs]
    assert min(probabilities)>0 and sum(probabilities)==1
    for i in range(3):
        assert sum(p*s[i] for p,s in zip(probabilities,signs))==0
        for j in range(3):
            assert 30*sum(p*s[i]*s[j] for p,s in zip(probabilities,signs))==C[i][j]
    # max ||mu+sqrt30*s-(0,0,38)||² =1461+78 sqrt30 <44².
    assert 78**2*30<(44**2-1461)**2
    physical_residual=original_residual(mu,u0,h,sigma,rho,beta)
    assert any(physical_residual)
    rank_minor=((1+h*sigma)*(1+h)-h*h*sigma*rho)*(1+h*beta)
    assert rank_minor!=0
    # Symmetric orthonormal Frobenius basis: E_ii, (E_ij+E_ji)/sqrt2.
    pairs=[(i,j) for i in range(4) for j in range(i,4)]
    L=np.zeros((3,10))
    for axis,row in enumerate(rows):
        for pair,c in row.items():
            L[axis,pairs.index(pair)]=float(c)/(np.sqrt(2) if pair[0]!=pair[1] else 1)
    Yf=np.array(Y,dtype=float)
    vector=np.array([Yf[i,j]*(np.sqrt(2) if i!=j else 1) for i,j in pairs])
    normalized=vector/np.linalg.norm(vector);H=L.T@L
    eigenvalues=np.linalg.eigvalsh(H)
    ground_residual=float(np.linalg.norm(L@normalized))
    schmidt=np.linalg.svd(Yf/np.linalg.norm(Yf),compute_uv=False)
    normalized_energy=float(np.linalg.norm(L@normalized)**2)
    # Exact orthogonal projection of the NATURAL physical initial product
    # q0 tensor q0 onto ker(L): the infinite imaginary-time limit when overlap
    # is nonzero. Frobenius weights are 1(diagonal),2(off-diagonal).
    Crows=[[row.get(pair,Q(0)) for pair in pairs] for row in rows]
    weights=[Q(1 if i==j else 2) for i,j in pairs]
    initial_product=[Q(1)]*10
    gram=[[sum(Crows[i][k]*Crows[j][k]/weights[k] for k in range(10)) for j in range(3)] for i in range(3)]
    forcing=[sum(row) for row in Crows];multipliers=solve_exact(gram,forcing)
    projected=[initial_product[k]-sum(Crows[j][k]*multipliers[j] for j in range(3))/weights[k] for k in range(10)]
    assert all(sum(a*b for a,b in zip(row,projected))==0 for row in Crows)
    projection_matrix=[[Q(0)]*4 for _ in range(4)]
    for (i,j),value in zip(pairs,projected):projection_matrix[i][j]=projection_matrix[j][i]=value
    minor=projection_matrix[0][0]*projection_matrix[1][1]-projection_matrix[0][1]**2
    assert minor!=0 and projection_matrix[0][0]!=0
    recovered=[projection_matrix[0][j]/projection_matrix[0][0] for j in range(1,4)]
    recovered_residual=original_residual(recovered,u0,h,sigma,rho,beta)
    assert any(recovered_residual)
    # Physical nonlinear implicit-Euler root near the initial state, numerical
    # comparison only; it is not needed for the exact counterexample.
    parameters=list(map(float,[sigma,rho,beta]));dt=float(h)
    def implicit(v):
        return np.array(original_residual(v,np.ones(3),dt,*parameters))
    solution=root(implicit,np.array([1.,1.26,1.-dt*(float(beta)-1)]),tol=1e-12)
    if not solution.success and np.linalg.norm(implicit(solution.x))>1e-12:
        raise RuntimeError(solution.message)
    q=np.concatenate(([1.],solution.x));Yphysical=np.outer(q,q)
    physical_vector=np.array([Yphysical[i,j]*(np.sqrt(2) if i!=j else 1) for i,j in pairs])
    physical_vector/=np.linalg.norm(physical_vector)
    exact=dict(h=str(h),sigma=str(sigma),rho=str(rho),beta=str(beta),
        initial_state=list(map(str,u0)),spurious_mean=list(map(str,mu)),
        covariance=[[str(x) for x in row] for row in C],
        moment_matrix=[[str(x) for x in row] for row in Y],
        exact_lifted_residual=list(map(str,residual)),
        exact_original_residual_at_mean=list(map(str,physical_residual)),
        leading_principal_minors_of_Y=list(map(str,principal)),
        covariance_diagonal_dominance_margin=str(dominance),
        exact_nonzero_constraint_rank_minor=str(rank_minor),
        symmetry=True,PSD=True,positive_definite=True,Y00='1',
        exact_matrix_rank=4,exact_Schmidt_rank_of_normalized_vectorization=4,
        symmetric_basis_dimension=10,exact_constraint_rank=3,
        exact_zero_energy_subspace_dimension=7,
        rank_one_condition_satisfied=False)
    exact.update(bounded_support_construction='mu+sqrt(30)*(sx,sy,sz), sx,sy,sz in {-1,1}',
        bounded_support_probabilities=list(map(str,probabilities)),
        support_ball_center=[0,0,38],support_ball_radius=44,
        support_max_radius_squared='1461+78*sqrt(30) < 1936',
        support_containment_exact_integer_check=[78**2*30,(44**2-1461)**2])
    exact.update(initial_product_kernel_projection_matrix=[[str(x) for x in row] for row in projection_matrix],
        initial_product_projection_nonzero_rank2_minor=str(minor),
        initial_product_projection_recovered_mean=list(map(str,recovered)),
        initial_product_projection_original_residual=list(map(str,recovered_residual)),
        initial_product_projection_original_residual_norm=float(np.linalg.norm(list(map(float,recovered_residual)))),
        initial_product_projection_interpretation='exact infinite imaginary-time limit for this unconstrained H; product consistency not preserved')
    numerical=dict(original_implicit_root=solution.x.tolist(),
        root_original_residual_norm=float(np.linalg.norm(implicit(solution.x))),
        physical_root_lifted_residual_norm=float(np.linalg.norm(L@physical_vector)),
        spurious_mean_original_residual_norm=float(np.linalg.norm(list(map(float,physical_residual)))),
        spurious_normalized_lifted_residual_norm=ground_residual,
        spurious_normalized_energy=normalized_energy,
        H_eigenvalues=eigenvalues.tolist(),
        smallest_positive_eigenvalue=float(eigenvalues[-3]),
        spurious_Schmidt_coefficients=schmidt.tolist(),
        physical_root_overlap_with_spurious=float(abs(physical_vector@normalized)),
        note='Floating diagnostics only; rank, PSD and zero residual follow from exact data above.')
    summary=dict(method='implicit-Euler polynomial lifting followed by unconstrained H=L_dagger L ground-state preparation',
        result='generic ground state is insufficient: symmetry, PSD and Y00=1 do not enforce identical tensor-product rank1',
        mathematical_status='exact rational counterexample with the repository binary64 beta interpreted exactly',
        caveat='Not a refutation of a method that separately enforces rank-one/product consistency or proves its prepared ground state has that property.',
        paper='https://arxiv.org/html/2504.13174',
        comparison_scope='A Lorenz-specific feasibility test inspired by generic nonlinear lifting, not a reproduction of that paper entire Chebyshev construction.',
        **exact,**numerical)
    (out/'groundstate_audit.json').write_text(json.dumps(summary,indent=2))
    np.savez(out/'groundstate_matrices.npz',constraint_matrix=L,H=H,
             spurious_moment_matrix=Yf,spurious_symmetric_state=normalized,
             physical_symmetric_state=physical_vector)
    print(json.dumps(dict(exact_lifted_residual=exact['exact_lifted_residual'],
        exact_original_residual=exact['exact_original_residual_at_mean'],
        principal_minors=exact['leading_principal_minors_of_Y'],
        rank=4,nullity=7,physical_root=numerical['original_implicit_root'],
        spurious_mean_residual_norm=numerical['spurious_mean_original_residual_norm'],
        caveat=summary['caveat']),indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,
        default=Path('results/quantum_candidates'))
    audit(parser.parse_args().out)
