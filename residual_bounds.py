"""Rigorous rational single-step bounds for ideal Lorenz-Carleman truncation.

All certified operations use Fraction, dyadic outward enclosure and a rational
geometric remainder. No DOP853 trajectory, floating matrix exponential or
numerical quadrature enters these bounds. The ideal generator is assembled
algebraically from exact representations of the stored binary64 parameters.
This does not enclose floating implementation/Taylor/gate/measurement errors.
"""
from fractions import Fraction as F
from functools import lru_cache
from math import factorial,isqrt
from decimal import Decimal,localcontext,ROUND_CEILING
from carleman import Lorenz,basis


def rational(x):
    return x if isinstance(x,F) else F(x)


def sqrt_upper(x,bits=100):
    x=F(x)
    if x<0:raise ValueError('Negative radicand')
    scale=1<<(2*bits)
    n=(x.numerator*scale+x.denominator-1)//x.denominator
    root=isqrt(n)
    if root*root<n:root+=1
    return F(root,1<<bits)


def outward_decimal(x,digits=24):
    with localcontext() as ctx:
        ctx.prec=digits;ctx.rounding=ROUND_CEILING
        return str(Decimal(x.numerator)/Decimal(x.denominator))


def outward_interval(interval,bits=80):
    scale=1<<bits;lo,hi=interval
    return (F((lo*scale).numerator//(lo*scale).denominator,scale),
            F(-((-hi*scale).numerator//(-hi*scale).denominator),scale))


def add(a,b):return a[0]+b[0],a[1]+b[1]
def neg(a):return -a[1],-a[0]
def sub(a,b):return add(a,neg(b))
def mul(a,b):
    products=[x*y for x in a for y in b]
    return min(products),max(products)
def point(x):return F(x),F(x)
def absmax(a):return max(abs(a[0]),abs(a[1]))


def rhs_box(box,parameters):
    sigma,rho,beta=parameters;x,y,z=box
    return [mul(point(sigma),sub(y,x)),
            sub(sub(mul(point(rho),x),y),mul(x,z)),
            sub(mul(x,y),mul(point(beta),z))]


def local_enclosure(model,u0,h,refinements=8,bits=80):
    """Exit-time validated tube, then Picard image intersections.

    A strict self-inclusion gives existence of the local enclosure. Every
    refinement is an integral-equation image of an already valid enclosure.
    Outward dyadic rounding prevents uncontrolled denominator growth.
    """
    h=F(h);u=[F(float(x)) for x in u0]
    if h<=0:raise ValueError('h must be positive')
    parameters=tuple(F(float(x)) for x in [model.sigma,model.rho,model.beta])
    f0=rhs_box([point(x) for x in u],parameters)
    radii=[2*h*max(F(1),absmax(x)) for x in f0]
    for attempt in range(40):
        box=[outward_interval((x-r,x+r),bits) for x,r in zip(u,radii)]
        image=[add(point(x),mul((F(0),h),f)) for x,f in zip(u,rhs_box(box,parameters))]
        if all(a[0]<b[0] and b[1]<a[1] for a,b in zip(box,image)):break
        # Enlarge offending coordinates independently. Uniform inflation can
        # preserve a bad aspect ratio forever (e.g. x'=0 initially, y' large).
        radii=[max(r,2*max(abs(a-x),abs(b-x)))
               for r,x,(a,b) in zip(radii,u,image)]
    else:raise ValueError('Failed to validate a local tube; reduce h')
    initial_box=[[str(x) for x in interval] for interval in box]
    initial_image=[[str(x) for x in interval] for interval in image]
    for _ in range(refinements):
        image=[outward_interval(add(point(x),mul((F(0),h),f)),bits)
               for x,f in zip(u,rhs_box(box,parameters))]
        box=[(max(a[0],b[0]),min(a[1],b[1])) for a,b in zip(box,image)]
    return box,dict(strict_self_inclusion=True,inflations=attempt,
                    picard_refinements=refinements,rounding_bits=bits,
                    initial_box=initial_box,initial_integral_image=initial_image)


def exact_polynomials(model):
    sigma,rho,beta=map(lambda x:F(float(x)),[model.sigma,model.rho,model.beta])
    S=F(float(model.scale));center=sigma+rho
    return [{(1,0,0):-sigma,(0,1,0):sigma},
            {(1,0,0):-sigma,(0,1,0):-F(1),(1,0,1):-S},
            {(1,1,0):S,(0,0,1):-beta,(0,0,0):-beta*center/S}]


def rational_generator(model,N):
    exponents=basis(N);index={a:i for i,a in enumerate(exponents)}
    rows=[{} for _ in exponents]
    for i,a in enumerate(exponents):
        for j,poly in enumerate(exact_polynomials(model)):
            if not a[j]:continue
            lowered=list(a);lowered[j]-=1
            for power,coefficient in poly.items():
                target=tuple(lowered[k]+power[k] for k in range(3))
                if target in index:
                    k=index[target];rows[i][k]=rows[i].get(k,F(0))+a[j]*coefficient
        rows[i]={k:v for k,v in rows[i].items() if v}
    return exponents,rows


def row_times(row,matrix):
    result={}
    for i,a in row.items():
        for j,b in matrix[i].items():result[j]=result.get(j,F(0))+a*b
    return {j:a for j,a in result.items() if a}


def norm_infinity(matrix):return max([sum(abs(x) for x in row.values()) for row in matrix]+[F(0)])


@lru_cache(maxsize=24)
def path_data(N,scale,sigma=10.,rho=28.,beta=8/3,J=12):
    model=Lorenz(sigma=sigma,rho=rho,beta=beta,scale=scale)
    exponents,C=rational_generator(model,N)
    extended,Gfull=rational_generator(model,N+1)
    shell=[i for i,a in enumerate(extended) if sum(a)==N+1]
    shell_index={j:k for k,j in enumerate(shell)}
    G=[{shell_index[j]:v for j,v in row.items() if j in shell_index}
       for row in Gfull[:len(exponents)]]
    index={a:i for i,a in enumerate(exponents)}
    linear=[index[(1,0,0)],index[(0,1,0)],index[(0,0,1)]]
    powers=[{j:F(1)} for j in linear]
    H=[]
    for k in range(J+1):
        H.append([row_times(row,G) for row in powers])
        powers=[row_times(row,C) for row in powers]
    absolute=[{j:abs(v) for j,v in row.items()} for row in C]
    positive=[{j:F(1)} for j in linear]
    for _ in range(J+1):positive=[row_times(row,absolute) for row in positive]
    # A nonnegative degree-block comparison system. Diagonal blocks retain
    # their infinity logarithmic norm; zero is used if it is negative.
    layers=[[i for i,a in enumerate(exponents) if sum(a)==k] for k in range(1,N+1)]
    A=[]
    for i,indices in enumerate(layers):
        row={}
        for j,columns in enumerate(layers):
            if i==j:
                entries=[C[r].get(r,F(0))+sum(abs(C[r].get(c,F(0))) for c in columns if c!=r) for r in indices]
                value=max(F(0),max(entries))
            else:value=max(sum(abs(C[r].get(c,F(0))) for c in columns) for r in indices)
            if value:row[j]=value
        A.append(row)
    return dict(exponents=exponents,C=C,G=G,shell=[extended[i] for i in shell],
                linear=linear,H=H,positive_tail_rows=positive,A=A,
                infinity_norm=norm_infinity(C))


def geometric_integral_tail(h,a,J):
    ratio=h*a/(J+3)
    if ratio>=1:raise ValueError('Increase J to bound the exponential series tail')
    return h**(J+2)/factorial(J+2)/(1-ratio)


def path_bound(data,h,monomials):
    J=len(data['H'])-1;components=[F(0)]*3
    for k,H in enumerate(data['H']):
        weight=h**(k+1)/factorial(k+1)
        for j,row in enumerate(H):components[j]+=weight*sum(abs(v)*monomials[a] for a,v in row.items())
    forcing=[sum(abs(v)*monomials[a] for a,v in row.items()) for row in data['G']]
    gmax=max(forcing)
    tail_factor=geometric_integral_tail(h,data['infinity_norm'],J)
    tails=[sum(row.values())*gmax*tail_factor for row in data['positive_tail_rows']]
    return [x+y for x,y in zip(components,tails)],tails


def block_bound(data,h,monomials):
    A=data['A'];N=len(A);J=len(data['H'])-1
    forcing=max(sum(abs(v)*monomials[a] for a,v in row.items()) for row in data['G'])
    row={0:F(1)};integral=F(0)
    for k in range(J+1):
        integral+=h**(k+1)/factorial(k+1)*row.get(N-1,F(0))
        row=row_times(row,A)
    tail=sum(row.values())*geometric_integral_tail(h,norm_infinity(A),J)
    return sqrt_upper(F(3))*(integral+tail)*forcing


def certified_bounds(model,N,h,u0,radius=44,extra_paths=8):
    h=F(str(h));S=F(float(model.scale));R=F(radius)
    if not isinstance(N,int) or N<1:raise ValueError('Require integer N>=1')
    if h<=0 or S<=0:raise ValueError('Require h>0 and S>0')
    center=[F(0),F(0),F(float(model.sigma))+F(float(model.rho))]
    u=[F(float(x)) for x in u0]
    if sum((x-c)**2 for x,c in zip(u,center))>R*R:
        raise ValueError('Initial point outside the stated invariant ball')
    if model.sigma!=10 or model.rho!=28 or model.beta!=8/3 or R<44:
        raise ValueError('Global radius guarantee currently implemented for standard Lorenz, R>=44')
    beta=F(float(model.beta));c=center[2]
    # On the sphere, (||u-center||^2)'/2 <= -R^2+beta^2*c^2/(4*(beta-1)).
    # Check this using the exact stored parameters, not a nearby rational beta.
    if beta<=1 or beta*beta*c*c>=4*(beta-1)*R*R:
        raise ValueError('Invariant sphere inequality not validated')
    # J may be raised to make the rational geometric tail valid.
    _,C=rational_generator(model,N);a=norm_infinity(C)
    J=max(N+extra_paths,int(h*a)+2)
    data=path_data(N,model.scale,model.sigma,model.rho,model.beta,J)
    tube,proof=local_enclosure(model,u0,h)
    vbox=[((lo-c)/S,(hi-c)/S) for (lo,hi),c in zip(tube,center)]
    magnitudes=[min(absmax(interval),R/S) for interval in vbox]
    local=[np_product([magnitudes[j]**p for j,p in enumerate(power)]) for power in data['shell']]
    global_monomials=[(R/S)**(N+1)]*len(local)
    path_local,tails=path_bound(data,h,local)
    path_global,_=path_bound(data,h,global_monomials)
    result=dict(block_global=S*block_bound(data,h,global_monomials),
                block_local=S*block_bound(data,h,local),
                path_global=S*sqrt_upper(sum(x*x for x in path_global)),
                path_local=S*sqrt_upper(sum(x*x for x in path_local)),
                path_local_tail=S*sqrt_upper(sum(x*x for x in tails)))
    certificate=dict(N=N,h=str(h),S=str(S),initial_state=[str(x) for x in u],
        exact_parameters=[str(F(float(x))) for x in [model.sigma,model.rho,model.beta]],
        center=[str(x) for x in center],radius=str(R),series_last_power=J,
        local_tube=[[str(x) for x in interval] for interval in tube],tube_proof=proof,
        exact_upper_bounds={k:str(v) for k,v in result.items()},
        outward_decimal_bounds={k:outward_decimal(v) for k,v in result.items()},
        scope='Ideal algebraic rational-generator Carleman truncation only; no floating implementation errors.')
    return result,certificate,data


def np_product(values):
    value=F(1)
    for x in values:value*=x
    return value
