"""Exact a posteriori residual bounds for computed Gauss-stage trajectories.

The mathematical object certified here is a continuous piecewise rational
polynomial.  The input stage data may come from an inaccurate floating solver:
its accuracy is never assumed.  Existing validated K and nonlinear remainders
are reused for exactly the same guide.  This is CLASSICAL certification and
already constructs a classical answer; it is not a quantum algorithm result.
"""
from pathlib import Path
from fractions import Fraction as Q
from math import comb
from bisect import bisect_right
import argparse,csv,hashlib,json,time
import numpy as np
from global_correction import PARAMETERS,hermite,restrict_polynomial
from residual_bounds import sqrt_upper,outward_decimal
from fullstage_system import load_cases,build_fullstage,structured_solve,tableau

OUT=Path('results/fullstage_qlsa')


def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()
def add(a,b,scale=1):
    n=max(len(a),len(b))
    return [(a[k] if k<len(a) else Q(0))+scale*(b[k] if k<len(b) else Q(0)) for k in range(n)]
def conv(a,b):
    p=[Q(0)]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        for j,y in enumerate(b):p[i+j]+=x*y
    return p
def evaluate(a,t):
    value=Q(0)
    for c in reversed(a):value=value*t+c
    return value
def derivative(a,h=Q(1)):return [k*a[k]/h for k in range(1,len(a))] or [Q(0)]


def lagrange_basis(nodes):
    basis=[]
    for j,t in enumerate(nodes):
        poly=[Q(1)];denom=Q(1)
        for k,u in enumerate(nodes):
            if j==k:continue
            poly=conv(poly,[-u,Q(1)]);denom*=t-u
        basis.append([c/denom for c in poly])
    return basis


def field(c):
    sigma,rho,beta=PARAMETERS;x,y,z=c
    return [[sigma*t for t in add(y,x,-1)],
            add(add([rho*t for t in x],y,-1),conv(x,z),-1),
            add(conv(x,y),[beta*t for t in z],-1)]


def jacobian_product(c,g):
    sigma,rho,beta=PARAMETERS;x,y,z=c;gx,gy,gz=g
    return [[sigma*t for t in add(gy,gx,-1)],
            add(add(add([rho*t for t in gx],conv(z,gx),-1),gy,-1),conv(x,gz),-1),
            add(add(conv(y,gx),conv(x,gy)),[beta*t for t in gz],-1)]


def residual(c,g,h):
    # D=G'-J(c)G-(f(c)-c'), in physical time and physical coordinates.
    forcing=[add(f,derivative(p,h),-1) for f,p in zip(field(c),c)]
    return [add(add(derivative(p,h),a,-1),r,-1)
            for p,a,r in zip(g,jacobian_product(c,g),forcing)]


def bernstein_abs(poly,subdivisions=1):
    """Exact convex-hull enclosure using dyadic de Casteljau subdivision."""
    if subdivisions<1 or subdivisions&(subdivisions-1):raise ValueError('Use a power of two')
    n=len(poly)-1
    bernstein=[sum(poly[j]*Q(comb(k,j),comb(n,j)) for j in range(k+1)) for k in range(n+1)]
    parts=[bernstein]
    for _ in range(subdivisions.bit_length()-1):
        split=[]
        for values in parts:
            levels=[values]
            for k in range(n):levels.append([(a+b)/2 for a,b in zip(levels[-1],levels[-1][1:])])
            split.append([level[0] for level in levels]);split.append([level[-1] for level in reversed(levels)])
        parts=split
    return max(abs(x) for values in parts for x in values)


def guide_piece(source,left,right):
    times=list(map(Q,source['times']))
    i=min(len(times)-2,max(0,bisect_right(times,left)-1))
    assert times[i]<=left<right<=times[i+1],'Guide boundary omitted by stage mesh'
    h=times[i+1]-times[i]
    curve=hermite(list(map(Q,source['nodes'][i])),list(map(Q,source['nodes'][i+1])),h)
    return [restrict_polynomial(p,(left-times[i])/h,(right-left)/h) for p in curve]


def construct(source,guide,M,s,coefficient_bits=256):
    """Construct exact integrated collocation curves, corrected to be continuous."""
    system=build_fullstage(guide,M,s,'increment','quadrature')
    history,_=structured_solve(system);history=history.reshape(M,3*(s+1))
    _,b,nodes_float=tableau(s);nodes=list(map(lambda x:Q(float(x)),nodes_float))
    weights=list(map(lambda x:Q(float(x)),np.sqrt(b)));basis=lagrange_basis(nodes)
    scale=Q(float(guide.scale));previous=[Q(0)]*3;pieces=[]
    for step,(left_float,right_float) in enumerate(zip(system.mesh,system.mesh[1:])):
        left,right=Q(float(left_float)),Q(float(right_float));h=right-left
        c=guide_piece(source,left,right)
        stage=history[step,:3*s].reshape(s,3);target=[scale*Q(float(v)) for v in history[step,-3:]]
        slopes=[]
        for j,tau in enumerate(nodes):
            cj=[[evaluate(p,tau)] for p in c]
            gj=[[previous[k]+scale*Q(float(stage[j,k]))/weights[j]] for k in range(3)]
            force=[f[0]-evaluate(derivative(p,h),tau) for f,p in zip(field(cj),c)]
            slopes.append([a[0]+r for a,r in zip(jacobian_product(cj,gj),force)])
        g=[]
        for k in range(3):
            derivative_poly=[sum(slopes[j][k]*basis[j][degree] for j in range(s)) for degree in range(s)]
            poly=[previous[k]]+[h*v/(degree+1) for degree,v in enumerate(derivative_poly)]
            # Store an exactly specified compact dyadic polynomial. Rounding
            # is included in the subsequent exact residual, not assumed away.
            grid=1<<coefficient_bits
            poly=[Q((v.numerator*grid)//v.denominator,grid) for v in poly]
            poly[0]=previous[k]
            # Arbitrary float solver endpoint mismatch is included, not ignored.
            poly[1]+=target[k]-sum(poly);g.append(poly)
        assert all(g[k][0]==previous[k] and sum(g[k])==target[k] for k in range(3))
        pieces.append(dict(left=str(left),right=str(right),g=[[str(x) for x in p] for p in g]))
        previous=target
    return system,pieces


def bound(source,pieces,subdivisions=1):
    assert list(map(str,PARAMETERS))==source['exact_parameters']
    previous=[Q(0)]*3;start=Q(0);maximum=Q(0);worst=-1;local=[]
    for i,piece in enumerate(pieces):
        left,right=Q(piece['left']),Q(piece['right']);assert left==start
        g=[list(map(Q,p)) for p in piece['g']]
        assert [p[0] for p in g]==previous,'Discontinuous correction polynomial'
        c=guide_piece(source,left,right);D=residual(c,g,right-left)
        norm=sqrt_upper(sum(bernstein_abs(p,subdivisions)**2 for p in D),bits=200)
        local.append(str(norm))
        if norm>maximum:maximum=norm;worst=i
        previous=list(map(sum,g));start=right
    assert start==Q(source['T'])==Q(source['times'][-1])
    K=Q(source['bounds']['K']);nonlinear=Q(source['bounds']['nonlinear_remainder'])
    linear=K*maximum;total=linear+nonlinear
    endpoint=[Q(x)+y for x,y in zip(source['nodes'][-1],previous)]
    rounded=list(map(float,endpoint));rounding=sqrt_upper(sum((Q(x)-y)**2 for x,y in zip(rounded,endpoint)),bits=200)
    return dict(K=str(K),nonlinear_remainder=str(nonlinear),residual_sup=str(maximum),
        linear_error=str(linear),total_exact_polynomial_error=str(total),endpoint_rounding=str(rounding),
        endpoint_total_error=str(total+rounding),all_times_target_pass=total<=Q('1e-8'),
        endpoint_target_pass=total+rounding<=Q('1e-8'),worst_piece=worst,
        local_residual_bounds=local,endpoint=list(map(str,endpoint)),rounded_endpoint=rounded)


def source_for(case):
    sources=json.loads(Path('results/global_correction/certificates.json').read_text())
    if 'source_sha256' in case:return next(c for c in sources if digest(c)==case['source_sha256'])
    return next(c for c in sources if c['initial']==case['initial'] and c['guide_rtol']==case['guide_rtol'])


def run(out=OUT):
    out.mkdir(parents=True,exist_ok=True);certificates=[];rows=[]
    for case,guide,*_ in load_cases():
        source=source_for(case)
        assert source['nodes']==case['nodes'] and source['times']==case['times']
        for s in [8,12]:
            M=128 if case['initial']=='standard' else 256;start=time.perf_counter()
            system,pieces=construct(source,guide,M,s)
            for subdivisions in ([1] if s==8 else [1,4]):
                bounds=bound(source,pieces,subdivisions)
                certificate=dict(initial=case['initial'],guide_rtol=case['guide_rtol'],M=M,s=s,
                    exact_parameters=list(map(str,PARAMETERS)),coefficient_bits=256,
                    source_path='results/global_correction/certificates.json',source_sha256=digest(source),
                    source_local_proofs_sha256=source['local_proofs_sha256'],subdivisions=subdivisions,
                    pieces=pieces,bounds=bounds,
                    scope='strict all-times bound for continuous exact rational polynomial; source K/nonlinear certificate reused',
                    quantum_status='classically reconstructed certificate/answer, not a quantum result or free quantum preprocessing')
                certificates.append(certificate)
                rows.append(dict(initial=case['initial'],M=M,s=s,subdivisions=subdivisions,
                    residual_sup=outward_decimal(Q(bounds['residual_sup'])),
                    linear_error=outward_decimal(Q(bounds['linear_error'])),
                    nonlinear_remainder=outward_decimal(Q(bounds['nonlinear_remainder'])),
                    all_times_error=outward_decimal(Q(bounds['total_exact_polynomial_error'])),
                    all_times_target_pass=bounds['all_times_target_pass'],elapsed_seconds=time.perf_counter()-start,
                    status='strict conditional on the existing replayable K/nonlinear source certificate'))
                print(rows[-1],flush=True)
                if bounds['all_times_target_pass']:break
            if bounds['all_times_target_pass']:break
    passing=[c for c in certificates if c['bounds']['all_times_target_pass']]
    failed=[dict({k:v for k,v in c.items() if k!='pieces'},curve_sha256=digest(c['pieces']))
            for c in certificates if not c['bounds']['all_times_target_pass']]
    (out/'certificates.json').write_text(json.dumps(passing,indent=2))
    (out/'certificate_failures.json').write_text(json.dumps(failed,indent=2))
    with (out/'certificate_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)
    return passing


def verify(path=OUT/'certificates.json',verify_source=False):
    sources=json.loads(Path('results/global_correction/certificates.json').read_text());checked=set()
    certificates=json.loads(Path(path).read_text())
    for c in certificates:
        source=next(s for s in sources if digest(s)==c['source_sha256'])
        assert c['initial']==source['initial'] and c['guide_rtol']==source['guide_rtol']
        assert len(c['pieces'])==c['M']
        assert all(len(poly)==c['s']+1 for piece in c['pieces'] for poly in piece['g'])
        assert source['local_proofs_sha256']==c['source_local_proofs_sha256']
        assert c['exact_parameters']==source['exact_parameters']==list(map(str,PARAMETERS))
        if verify_source and c['source_sha256'] not in checked:
            from verify_global_correction import verify_one
            verify_one(source);checked.add(c['source_sha256'])
        result=bound(source,c['pieces'],c['subdivisions'])
        assert result==c['bounds']
        print(c['initial'],c['s'],'exact residual replayed; strict all-times pass:',result['all_times_target_pass'],flush=True)
    replay=dict(certificates=len(certificates),all_times_passes=sum(c['bounds']['all_times_target_pass'] for c in certificates),
        source_certificates_replayed=verify_source,source_sha256=sorted(checked),
        stage_residual_replay='exact rational algebra, no reference ODE or numerical solve',
        scope='only already certified K and nonlinear remainder reused; full source proof replay requires --verify-source')
    Path(path).with_name('certificate_replay.json').write_text(json.dumps(replay,indent=2))
    return len(certificates)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true')
    parser.add_argument('--verify-source',action='store_true');args=parser.parse_args()
    verify(verify_source=args.verify_source) if args.verify else run()
