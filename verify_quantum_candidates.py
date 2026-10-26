"""Replay exact local bounds/counterexamples; label floating diagnostics separately."""
import csv,json,itertools
from fractions import Fraction as Q
from pathlib import Path
from residual_quantum_candidate import residual,picard_certificate,polyval,bernstein_range
from lifted_groundstate_audit import lifted_rows,original_residual,det3,solve_exact


def verify(root=Path('results/quantum_candidates')):
    states={'standard':[1,1,1],'attractor':[-8,8,27]};count=0
    for c in json.loads((root/'residual_exact.json').read_text()):
        u=states[c['initial']];d=residual(u,c['h'],c['p'])
        assert [[str(x) for x in row] for row in d['R']]==c['residual_coefficients']
        assert list(map(str,d['integrals']))==c['integrals']
        assert picard_certificate(u,d)==Q(c['bound'])
        count+=1
    c=json.loads((root/'groundstate_audit.json').read_text())
    Y=[list(map(Q,row)) for row in c['moment_matrix']];C=[list(map(Q,row)) for row in c['covariance']]
    h,sigma,rho,beta=map(Q,(c['h'],c['sigma'],c['rho'],c['beta']));u=list(map(Q,c['initial_state']))
    assert all(Y[i][j]==Y[j][i] for i in range(4) for j in range(4)) and Y[0][0]==1
    for row in lifted_rows(h,sigma,rho,beta,u):assert sum(v*Y[i][j] for (i,j),v in row.items())==0
    assert original_residual(u,u,h,sigma,rho,beta)==list(map(Q,c['exact_original_residual_at_mean']))
    assert det3(C)>0 and C[0][0]*C[1][1]-C[0][1]**2>0 and C[0][0]>0
    signs=list(itertools.product([-1,1],repeat=3));p=list(map(Q,c['bounded_support_probabilities']))
    assert sum(p)==1 and min(p)>0
    for i in range(3):
        assert sum(v*s[i] for v,s in zip(p,signs))==0
        for j in range(3):assert 30*sum(v*s[i]*s[j] for v,s in zip(p,signs))==C[i][j]
    assert 78**2*30<(44**2-1461)**2
    assert ((1+h*sigma)*(1+h)-h*h*sigma*rho)*(1+h*beta)!=0
    pairs=[(i,j) for i in range(4) for j in range(i,4)]
    rows=[[row.get(pair,Q(0)) for pair in pairs] for row in lifted_rows(h,sigma,rho,beta,u)]
    weights=[Q(1 if i==j else 2) for i,j in pairs]
    gram=[[sum(rows[i][k]*rows[j][k]/weights[k] for k in range(10)) for j in range(3)] for i in range(3)]
    multipliers=solve_exact(gram,list(map(sum,rows)))
    projected=[1-sum(rows[j][k]*multipliers[j] for j in range(3))/weights[k] for k in range(10)]
    matrix=[list(map(Q,row)) for row in c['initial_product_kernel_projection_matrix']]
    assert [matrix[i][j] for i,j in pairs]==projected
    assert matrix[0][0]*matrix[1][1]-matrix[0][1]**2==Q(c['initial_product_projection_nonzero_rank2_minor'])!=0
    a1=11+beta;a2=38*beta;a3=540*beta
    lit=json.loads((root/'literature_eligibility.json').read_text())
    assert Q(lit['exact_Routh_margin'])==a1*a2-a3<0
    assert Q(lit['exact_energy_R_squared_lower_for_displayed_tensors'])==76/beta>1
    oracle=json.loads((root/'residual_oracle.json').read_text())
    d=residual([1,1,1],.01,4);poly=d['R'][1];lo,hi=bernstein_range(poly);n=oracle['grid']
    mean=sum((polyval(poly,Q(2*i+1,2*n))-lo)/(hi-lo) for i in range(n))/n
    assert mean==Q(oracle['probability_exact'])
    assert Q(oracle['midpoint_bias'])==abs(lo+(hi-lo)*mean-d['integrals'][1])
    with (root/'spacetime.csv').open() as f:rows=list(csv.DictReader(f))
    for r in rows:
        assert float(r['dense_vs_structured_relative_difference'])<1e-9
        assert 0<float(r['normalized_history_endpoint_probability'])<=1
    print(f'Replayed {count} exact local certificates, bounded-support and kernel-projection counterexamples, and theorem prerequisites.')
    print(f'Checked {len(rows)} numerical space-time consistency records; these are NOT accuracy certificates.')
    return count,len(rows)


if __name__=='__main__':verify()
