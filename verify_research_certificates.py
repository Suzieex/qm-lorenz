"""Replay research bounds with no numerical reference trajectories.

Also verifies exact sufficient QAE inequalities for the recorded global budget.
"""
import json,math
from fractions import Fraction as Q
from research_bounds import (AffineLorenz,affine_bound,chebyshev_bound,
    scalar_tail,sqrt_upper,propagation_data,generator,affine_field)
from research_workflow import OUT,alpha_exact,exp_upper
from residual_bounds import norm_infinity,path_bound


def verify():
    count=0
    for filename in ['01_certificates.json','03_certificates.json','04_certificates.json']:
        for c in json.loads((OUT/filename).read_text()):
            model=AffineLorenz(pivot=tuple(map(Q,c['pivot'])),scales=tuple(map(Q,c['scales'])))
            u=[Q(str(x)) for x in c['u']]
            if c.get('basis')=='tensor_Chebyshev_total_degree':
                b=chebyshev_bound(model,c['N'],Q(c['h']),u)
            else:b=affine_bound(model,c['N'],Q(c['h']),u,weights=list(map(Q,c['weights'])))
            assert b['truncation']==Q(c['truncation'])
            assert b['certificate']['tube']==c['tube']
            count+=1
    growth=exp_upper(Q(105));model=AffineLorenz();R=Q(44);S=Q(64)
    for c in json.loads((OUT/'05_certificates.json').read_text()):
        N=c['N'];h=Q(c['h']);m=int(5/h)
        _,C=generator(affine_field(model),N);a=norm_infinity(C)
        J=max(N+8,int(h*a)+2);data=propagation_data(model,N,tuple([Q(1)]*(N+2)),J)
        components,_=path_bound(data,h,[(R/S)**(N+1)]*len(data['shell']))
        bound=S*sqrt_upper(sum(x*x for x in components))
        W=m*growth;local=Q('1e-8')/(4*W)
        assert bound==Q(c['global_single_step_upper'])
        assert W==Q(c['weight_sum_upper']) and local==Q(c['local_budget'])
        B=sqrt_upper(sum((R/S)**(2*k) for k in range(N+1)))
        tail=scalar_tail(alpha_exact(model,N),h,c['taylor_K'])*S*B
        assert tail==Q(c['taylor_error']) and tail<=local
        assert c['truncation_budget_pass']==(W*bound<=Q('1e-8')/4)
        count+=1
    # Check the recorded sufficient-QAE counts independently, using pi<22/7
    # and an exact binomial tail rather than the study's floating evaluation.
    from research_workflow import read
    for row in read('05_full_budget.csv'):
        N=int(row['N']);h=Q(row['h']);m=int(row['steps']);eps=Q(row['epsilon'])
        local=eps/(4*m*growth);K=int(row['taylor_K']);alpha=alpha_exact(model,N)
        gamma=sum((alpha*h)**k/math.factorial(k) for k in range(1,K+1))
        B=sqrt_upper(sum((R/S)**(2*k) for k in range(N+1)));F=S*B*gamma
        delta=local/sqrt_upper(Q(3));pi=Q(22,7);M=1<<int(row['max_phase_bits'])
        assert M>=4*pi*F/delta and M>=2*pi*sqrt_upper(F/delta)
        r=int(row['repeats']);q=1-8/(pi*pi)
        failure=sum(math.comb(r,k)*q**k*(1-q)**(r-k) for k in range(r//2+1,r+1))
        assert 3*m*failure<=Q(1,20)
        assert int(row['qae_A_calls'])==3*r*(2*M-1)
    print(f'Verified {count} rational bounds and 48 sufficient-QAE budgets; no reference trajectory used.')
    return count


if __name__=='__main__':verify()
