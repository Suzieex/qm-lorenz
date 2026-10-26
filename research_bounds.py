"""Exact affine/weighted Carleman bounds and validated classical predictors.

Only Fraction/interval operations enter certificates. Resource counts and
floating observations are deliberately kept in research_workflow.py.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from functools import lru_cache
from math import factorial
import numpy as np
from carleman import Lorenz,basis
from residual_bounds import (local_enclosure,absmax,point,add,mul,
    row_times,norm_infinity,geometric_integral_tail,sqrt_upper,outward_decimal)


def exact(x):
    return x if isinstance(x,Q) else Q(float(x))


@dataclass(frozen=True)
class AffineLorenz(Lorenz):
    pivot:tuple=(0.,0.,38.)
    scales:tuple=(64.,64.,64.)

    @property
    def center(self):return np.array(self.pivot,dtype=float)

    def normalize(self,u):return (np.asarray(u)-self.center)/np.array(self.scales)

    def restore(self,v):return self.center+np.array(self.scales)*np.asarray(v)

    def polynomials(self):
        return [{p:float(c) for p,c in row.items()} for row in affine_field(self)]


def affine_field(model):
    x,y,z=map(exact,model.pivot);a,b,c=map(exact,model.scales)
    sigma,rho,beta=map(exact,(model.sigma,model.rho,model.beta))
    if min(a,b,c)<=0:raise ValueError('Positive scales required')
    result=[{(0,0,0):sigma*(y-x)/a,(1,0,0):-sigma,(0,1,0):sigma*b/a},
        {(0,0,0):(rho*x-y-x*z)/b,(1,0,0):(rho-z)*a/b,
         (0,1,0):-Q(1),(0,0,1):-x*c/b,(1,0,1):-a*c/b},
        {(0,0,0):(x*y-beta*z)/c,(1,0,0):y*a/c,
         (0,1,0):x*b/c,(0,0,1):-beta,(1,1,0):a*b/c}]
    return [{p:c for p,c in row.items() if c} for row in result]


def generator(field,N,weights=None):
    exponents=basis(N);index={a:i for i,a in enumerate(exponents)}
    weights=[Q(1)]*(N+1) if weights is None else list(map(exact,weights))
    if len(weights)!=N+1 or min(weights)<=0:raise ValueError('Positive layer weights required')
    rows=[]
    for a in exponents:
        row={}
        for j,poly in enumerate(field):
            if not a[j]:continue
            for p,c in poly.items():
                target=tuple(a[k]+p[k]-(j==k) for k in range(3))
                if target in index:
                    value=a[j]*c*weights[sum(a)]/weights[sum(target)]
                    k=index[target];row[k]=row.get(k,Q(0))+value
        rows.append({j:c for j,c in row.items() if c})
    return exponents,rows


@lru_cache(maxsize=128)
def propagation_data(model,N,weights,J):
    field=affine_field(model);exponents,C=generator(field,N,weights[:N+1])
    extended,full=generator(field,N+1,weights)
    shell=extended[len(exponents):]
    G=[{j-len(exponents):v for j,v in row.items() if j>=len(exponents)}
       for row in full[:len(exponents)]]
    index={a:i for i,a in enumerate(exponents)}
    linear=[index[(1,0,0)],index[(0,1,0)],index[(0,0,1)]]
    rows=[{i:Q(1)} for i in linear];H=[]
    for _ in range(J+1):
        H.append([row_times(row,G) for row in rows])
        rows=[row_times(row,C) for row in rows]
    M=[{j:abs(v) for j,v in row.items()} for row in C]
    positive=[{i:Q(1)} for i in linear]
    for _ in range(J+1):positive=[row_times(row,M) for row in positive]
    return dict(exponents=exponents,C=C,G=G,shell=shell,H=H,
        positive_tail_rows=positive,linear=linear,infinity_norm=norm_infinity(C))


def product(values):
    result=Q(1)
    for v in values:result*=v
    return result


def affine_bound(model,N,h,u,weights=None,tube=None,extra_paths=8):
    """Certified ideal truncation, Taylor remainder and input-lift norm.

    No global invariant-ball assumption is needed: a validated local tube is
    sufficient. weights[k] rescales every monomial of degree k.
    """
    h=Q(str(h));u=list(map(exact,u));N=int(N)
    if h<=0 or N<1:raise ValueError('Require h>0,N>=1')
    weights=tuple([Q(1)]*(N+2) if weights is None else map(exact,weights))
    if len(weights)!=N+2 or min(weights)<=0:raise ValueError('Need N+2 positive layer weights')
    field=affine_field(model);_,C=generator(field,N,weights[:N+1]);a=norm_infinity(C)
    J=max(N+extra_paths,int(h*a)+2)
    data=propagation_data(model,N,weights,J)
    # Caller may supply only a previously validated tube for this same u,h.
    if tube is None:tube,proof=local_enclosure(model,u,h)
    else:proof={'caller_supplied_validated_tube':True}
    center=list(map(exact,model.pivot));scales=list(map(exact,model.scales))
    vbox=[((lo-c)/s,(hi-c)/s) for (lo,hi),c,s in zip(tube,center,scales)]
    maxima=list(map(absmax,vbox))
    monomials=[weights[N+1]*product(maxima[j]**p for j,p in enumerate(a)) for a in data['shell']]
    from residual_bounds import path_bound
    components,tails=path_bound(data,h,monomials)
    factors=[s/weights[1] for s in scales]
    truncation=sqrt_upper(sum((s*x)**2 for s,x in zip(factors,components)))
    v0=[(x-c)/s for x,c,s in zip(u,center,scales)]
    phi=[weights[sum(a)]*product(v0[j]**p for j,p in enumerate(a)) for a in data['exponents']]
    B=sqrt_upper(sum(x*x for x in phi))
    return dict(truncation=truncation,B=B,phi=phi,factors=factors,data=data,
        tube=tube,tube_proof=proof,weights=weights,h=h,
        certificate=dict(truncation=str(truncation),outward=outward_decimal(truncation),
            tube=[[str(x) for x in interval] for interval in tube],proof=proof,
            J=J,parameters=[str(exact(x)) for x in (model.sigma,model.rho,model.beta)],
            pivot=[str(x) for x in center],scales=[str(x) for x in scales],
            weights=[str(x) for x in weights],N=N,h=str(h),u=[str(x) for x in u]))


def scalar_tail(a,h,K):
    ratio=h*a/(K+2)
    if ratio>=1:return None
    return (h*a)**(K+1)/factorial(K+1)/(1-ratio)


def select_taylor(bound,alpha,budget):
    """Implementable LCU alpha >= ||C||_2; physical output norm budget."""
    a=exact(alpha);h=bound['h'];factor=max(bound['factors'])*bound['B']
    for K in range(1,300):
        tail=scalar_tail(a,h,K)
        if tail is not None and factor*tail<=budget:
            weights=[(a*h)**k/factorial(k) for k in range(K+1)]
            return K,weights,factor*tail
    raise ValueError('Taylor order exceeded')


def lie_derivative(poly,field):
    result={}
    for a,c in poly.items():
        for j in range(3):
            if not a[j]:continue
            for b,d in field[j].items():
                target=tuple(a[k]+b[k]-(j==k) for k in range(3))
                result[target]=result.get(target,Q(0))+a[j]*c*d
    return {a:c for a,c in result.items() if c}


def interval_polynomial(poly,box):
    powers=[]
    for j,interval in enumerate(box):
        maximum=max([a[j] for a in poly]+[0]);row=[point(1)]
        for _ in range(maximum):row.append(mul(row[-1],interval))
        powers.append(row)
    total=point(0)
    for a,c in poly.items():
        term=point(c)
        for j,p in enumerate(a):term=mul(term,powers[j][p])
        total=add(total,term)
    return total


@lru_cache(maxsize=64)
def derivative_polynomials(u,p):
    local=AffineLorenz(pivot=tuple(u),scales=(1.,1.,1.))
    field=affine_field(local);polys=[{a:Q(1)} for a in [(1,0,0),(0,1,0),(0,0,1)]]
    derivatives=[]
    for _ in range(p+1):
        polys=[lie_derivative(poly,field) for poly in polys];derivatives.append(polys)
    return derivatives


def predictor_bounds(u,h,max_order=10,tube=None):
    """Exact classical time-Taylor values and rigorous Lagrange remainders."""
    h=Q(str(h));u=tuple(map(exact,u));model=Lorenz()
    if tube is None:tube,_=local_enclosure(model,u,h)
    box=[(lo-x,hi-x) for (lo,hi),x in zip(tube,u)]
    derivatives=derivative_polynomials(u,max_order)
    field=affine_field(AffineLorenz(pivot=u,scales=(1.,1.,1.)))
    previous=[{a:Q(1)} for a in [(1,0,0),(0,1,0),(0,0,1)]]
    cumulative=0;precompute=[]
    for current in derivatives:
        cumulative+=sum(len(field[j]) for poly in previous for a in poly for j in range(3) if a[j])
        precompute.append(cumulative);previous=current
    value=list(u);results=[];zero=(0,0,0)
    for p in range(1,max_order+1):
        value=[v+h**p/factorial(p)*poly.get(zero,Q(0))
               for v,poly in zip(value,derivatives[p-1])]
        components=[h**(p+1)/factorial(p+1)*absmax(interval_polynomial(poly,box))
                    for poly in derivatives[p]]
        results.append(dict(p=p,value=value.copy(),error=sqrt_upper(sum(x*x for x in components)),
            convolution_products=p*(p+1),certificate_terms=sum(map(len,derivatives[p])),
            certificate_precompute_products=precompute[p],
            certificate_coefficient_max_bits=max(max(abs(c.numerator).bit_length(),c.denominator.bit_length())
                for poly in derivatives[p] for c in poly.values())))
    return results


@lru_cache(maxsize=12)
def chebyshev_polynomials(N):
    """Exact tensor Chebyshev basis, total degree <= N, in monomials."""
    univariate=[{0:Q(1)},{1:Q(1)}]
    for k in range(1,N):
        following={j+1:2*c for j,c in univariate[k].items()}
        for j,c in univariate[k-1].items():following[j]=following.get(j,Q(0))-c
        univariate.append({j:c for j,c in following.items() if c})
    result=[]
    for a in basis(N):
        poly={(0,0,0):Q(1)}
        for axis,degree in enumerate(a):
            new={}
            for power,c in poly.items():
                for k,d in univariate[degree].items():
                    target=list(power);target[axis]+=k;target=tuple(target)
                    new[target]=new.get(target,Q(0))+c*d
            poly=new
        result.append(poly)
    return result


def chebyshev_expansion(poly,exponents,polynomials):
    work=poly.copy();result={};index={a:i for i,a in enumerate(exponents)}
    while work:
        a=max(work,key=lambda a:(sum(a),a));i=index[a]
        coefficient=work[a]/polynomials[i][a];result[i]=coefficient
        for p,c in polynomials[i].items():
            work[p]=work.get(p,Q(0))-coefficient*c
            if not work[p]:del work[p]
    return result


def chebyshev_bound(model,N,h,u):
    """Separate Chebyshev residual derivation; no monomial-bound substitution.

    Its general coefficient-row preparation is implemented, but no cheap
    arithmetic sparse block encoding is claimed for this basis.
    """
    h=Q(str(h));field=affine_field(model);exponents=basis(N)
    extended=basis(N+1);polys=chebyshev_polynomials(N+1)
    full=[chebyshev_expansion(lie_derivative(poly,field),extended,polys)
          for poly in polys[:len(exponents)]]
    C=[{j:c for j,c in row.items() if j<len(exponents)} for row in full]
    G=[{j-len(exponents):c for j,c in row.items() if j>=len(exponents)} for row in full]
    linear=[exponents.index(a) for a in [(1,0,0),(0,1,0),(0,0,1)]]
    a=norm_infinity(C);J=max(N+8,int(h*a)+2)
    rows=[{i:Q(1)} for i in linear];H=[]
    for _ in range(J+1):
        H.append([row_times(row,G) for row in rows]);rows=[row_times(row,C) for row in rows]
    M=[{j:abs(c) for j,c in row.items()} for row in C]
    rows=[{i:Q(1)} for i in linear]
    for _ in range(J+1):rows=[row_times(row,M) for row in rows]
    tube,proof=local_enclosure(model,u,h)
    scales=list(map(exact,model.scales));center=list(map(exact,model.pivot))
    box=[((lo-c)/s,(hi-c)/s) for (lo,hi),c,s in zip(tube,center,scales)]
    monomials=[absmax(interval_polynomial(poly,box)) for poly in polys[len(exponents):]]
    data=dict(C=C,G=G,linear=linear,H=H,positive_tail_rows=rows,
              infinity_norm=a,exponents=exponents,shell=extended[len(exponents):])
    from residual_bounds import path_bound
    components,_=path_bound(data,h,monomials)
    v0=[(exact(x)-c)/s for x,c,s in zip(u,center,scales)]
    phi=[sum(c*product(v0[j]**p for j,p in enumerate(a)) for a,c in poly.items())
         for poly in polys[:len(exponents)]]
    B=sqrt_upper(sum(x*x for x in phi));bound=sqrt_upper(sum((s*b)**2 for s,b in zip(scales,components)))
    return dict(truncation=bound,B=B,phi=phi,factors=scales,data=data,h=h,
        certificate=dict(N=N,h=str(h),pivot=list(map(str,center)),scales=list(map(str,scales)),
            u=list(map(str,u)),tube=[[str(x) for x in interval] for interval in tube],
            proof=proof,truncation=str(bound),basis='tensor_Chebyshev_total_degree',J=J))
