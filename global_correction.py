"""Validated global linear defect correction for the binary64 Lorenz parameters.

The guide is arbitrary exact rational C3 piecewise Hermite interpolation.
All certificates use integer fixed point and exact Fraction arithmetic. No
reference trajectory or floating matrix exponential enters certification.
"""
from fractions import Fraction as Q
from math import ceil,comb,factorial,isqrt
from bisect import bisect_left
from residual_bounds import sqrt_upper,outward_decimal

PARAMETERS=(Q(10),Q(28),Q(float(8/3)))
HERMITE_INVERSE=((35,-15,Q(5,2),Q(-1,6)),
                 (-84,39,-7,Q(1,2)),
                 (70,-34,Q(13,2),Q(-1,2)),
                 (-20,10,-2,Q(1,6)))


def ceildiv(a,b):return -(-a//b)
def norm(v):return max(abs(x) for x in v)
def matrix_norm(A):return max(sum(abs(x) for x in A[i:i+3]) for i in (0,3,6))
def mm(A,B):
    return [A[i]*B[j]+A[i+1]*B[j+3]+A[i+2]*B[j+6] for i in (0,3,6) for j in range(3)]
def mv(A,v):return [sum(A[3*i+j]*v[j] for j in range(3)) for i in range(3)]


def polynomial_range_upper_int(coefficients,bernstein_degree=24):
    """All-tau bound via Bernstein convex hull, plus an absolute power tail.

    Every rational coefficient conversion is enclosed by integer floor/ceil;
    point sampling and floating cancellation are not used.
    """
    d=min(bernstein_degree,len(coefficients)-1);maximum=0
    for k in range(d+1):
        lower=sum(coefficients[j]*comb(k,j)//comb(d,j) for j in range(k+1))
        maximum=max(maximum,abs(lower),abs(lower+k+1))
    return maximum+sum(abs(x) for x in coefficients[d+1:])


def nonlinear_enclosure(K,a):
    rho=2*K*a
    if rho>=1:return rho,None,None
    radicand=1-rho;scale=1<<512
    lower=Q(isqrt((radicand.numerator*scale*scale)//radicand.denominator),scale)
    R=2*a/(1+lower)
    if K*R>=1:raise ValueError('Outward root too coarse')
    if a+K*R*R/2>R:raise ValueError('Invariant radius inequality failed')
    return rho,R,K*R*R/2


def time_coefficients(u,degree=3):
    sigma,rho,beta=PARAMETERS;x,y,z=([Q(a)] for a in u)
    for k in range(degree):
        x.append(sigma*(y[k]-x[k])/(k+1))
        y.append((rho*x[k]-y[k]-sum(x[j]*z[k-j] for j in range(k+1)))/(k+1))
        z.append((sum(x[j]*y[k-j] for j in range(k+1))-beta*z[k])/(k+1))
    return x,y,z


def hermite(left,right,h):
    """Exact degree7 interpolation matching derivatives 0..3 at both ends."""
    left=time_coefficients(left);right=time_coefficients(right);result=[]
    for a,b in zip(left,right):
        first=[a[k]*h**k for k in range(4)]
        residual=[factorial(k)*b[k]*h**k-sum(factorial(j)//factorial(j-k)*first[j]
                   for j in range(k,4)) for k in range(4)]
        result.append(first+[sum(c*r for c,r in zip(row,residual)) for row in HERMITE_INVERSE])
    return result


def restrict_polynomial(coefficients,start,length):
    return [sum(c*comb(j,k)*start**(j-k)*length**k
                for j,c in enumerate(coefficients) if j>=k) for k in range(len(coefficients))]


def coefficients(curve,h):
    """Dimensionless B(tau)=h J(c(tau)), d(tau)=-h Q(c(tau))."""
    sigma,rho,beta=PARAMETERS
    A=[[Q(0)]*9 for _ in range(8)]
    for k in range(8):
        A[k][3]=-curve[2][k];A[k][5]=-curve[0][k]
        A[k][6]=curve[1][k];A[k][7]=curve[0][k]
    A[0][0]=-sigma;A[0][1]=sigma;A[0][3]+=rho;A[0][4]=-1;A[0][8]=-beta
    A=[[h*x for x in row] for row in A]
    b=[[Q(0)]*3 for _ in range(15)]
    for j in range(8):
        for k in range(8):
            b[j+k][1]+=h*curve[0][j]*curve[2][k]
            b[j+k][2]-=h*curve[0][j]*curve[1][k]
    return A,b


def exp_upper_int(a,S):
    """Integer upper bound on S exp(a), a>=0, with a rigorous series tail."""
    if a<0:raise ValueError('Nonnegative exponential bound required')
    ai=ceildiv(a.numerator*S,a.denominator);term=S;total=S
    for k in range(1,2000):
        term=ceildiv(term*ai,S*k);total+=term
        if ai<S*(k+2):
            following=ceildiv(term*ai,S*(k+1))
            if following<=2:
                return total+ceildiv(following*S*(k+2),S*(k+2)-ai)
    raise ValueError('Exponential enclosure did not converge')


def local_proof(curve,h,initial,bits=512,degree=80,vector_degree=None):
    """Polynomial approximants and rigorous local residual integrals.

    Coefficients of A,b are rounded DOWN to dyadic points (entry error<1/S).
    Residuals are checked afterwards with exact integer products at scale S^2.
    Therefore rounding in recurrence, coefficient loading and tail are covered.
    """
    vector_degree=degree if vector_degree is None else vector_degree
    if vector_degree<8:raise ValueError('Vector polynomial degree must be >=8')
    S=1<<bits;A,b=coefficients(curve,h)
    bound=max(sum(abs(A[k][i+j]) for k in range(8) for j in range(3)) for i in (0,3,6))
    q=exp_upper_int(bound,S)
    Af=[[int(x*S//1) for x in row] for row in A]
    bf=[[int(x*S//1) for x in row] for row in b]
    identity=[S,0,0,0,S,0,0,0,S];F=[identity];V=[list(initial)]
    products=0
    for n in range(max(degree,vector_degree)):
        ft=[0]*9;vt=[x*S for x in bf[n]] if n<len(bf) else [0]*3
        for k in range(min(n,7)+1):
            if n<degree:
                fm=mm(Af[k],F[n-k]);products+=27;ft=[x+y for x,y in zip(ft,fm)]
            if n<vector_degree:
                vm=mv(Af[k],V[n-k]);products+=9;vt=[x+y for x,y in zip(vt,vm)]
        if n<degree:F.append([x//(S*(n+1)) for x in ft])
        if n<vector_degree:V.append([x//(S*(n+1)) for x in vt])
    residual_F=0;residual_V=0
    for n in range(max(degree,vector_degree)+8):
        rf=[(n+1)*x*S for x in F[n+1]] if n+1<len(F) else [0]*9
        rv=[(n+1)*x*S for x in V[n+1]] if n+1<len(V) else [0]*3
        rf_error=0;rv_error=0
        for k in range(max(0,n-max(degree,vector_degree)),min(n,7)+1):
            if n-k<len(F):
                fm=mm(Af[k],F[n-k]);products+=27
                rf=[x-y for x,y in zip(rf,fm)];rf_error+=3*matrix_norm(F[n-k])
            if n-k<len(V):
                vm=mv(Af[k],V[n-k]);products+=9
                rv=[x-y for x,y in zip(rv,vm)];rv_error+=3*norm(V[n-k])
        if n<len(bf):rv=[x-y*S for x,y in zip(rv,bf[n])];rv_error+=S
        residual_F+=ceildiv(matrix_norm(rf)+rf_error,n+1)
        residual_V+=ceildiv(norm(rv)+rv_error,n+1)
    delta_F=ceildiv(q*residual_F,S*S);delta_V=ceildiv(q*residual_V,S*S)
    endpoint_F=[sum(row[j] for row in F) for j in range(9)]
    endpoint_V=[sum(row[j] for row in V) for j in range(3)]
    difference=[]
    for axis in range(3):
        powers=[]
        for k in range(vector_degree+1):
            if k<8:
                cf=int(curve[axis][k]*S//1)
                powers.append(V[k][axis]-cf)
            else:powers.append(V[k][axis])
        difference.append(polynomial_range_upper_int(powers)+8)
    return dict(h=h,q=q,E=endpoint_F,delta_E=delta_F,delta_V=delta_V,
        endpoint=endpoint_V,g_polynomial_components=difference,
        V=V,residual_F_scaled2=residual_F,residual_V_scaled2=residual_V,
        fixed_point_products=products)


def global_proof(locals,bits=512):
    """Signed matrix products BEFORE taking norms; no global exp(LT)."""
    S=1<<bits;m=len(locals);eta=[0];K_nodes=[0];max_product_radius=0;products=0
    for j in range(1,m+1):
        P=[S,0,0,0,S,0,0,0,S];radius=0;Kj=0;error=0
        for i in reversed(range(j)):
            item=locals[i];pn=matrix_norm(P);upper=pn+radius
            h=item['h']
            Kj+=ceildiv(h.numerator*item['q']*upper,h.denominator*S)
            error+=ceildiv(upper*item['delta_V'],S)
            # Right multiplication: Psi(tj,ti)=Psi(tj,t_(i+1))*E_i.
            en=matrix_norm(item['E']);de=item['delta_E']
            radius=ceildiv(pn*de+radius*en+radius*de,S)+3
            P=[x//S for x in mm(P,item['E'])];products+=27
            max_product_radius=max(max_product_radius,radius)
        K_nodes.append(Kj);eta.append(error)
    Kinf=0;linear_error=0;a_squared=0
    for i,item in enumerate(locals):
        h=item['h'];q=item['q']
        Kinf=max(Kinf,ceildiv(q*(K_nodes[i]*h.denominator+h.numerator*S),S*h.denominator))
        local_error=ceildiv(q*eta[i],S)+item['delta_V']
        linear_error=max(linear_error,local_error)
        a_squared=max(a_squared,sum((x+local_error)**2 for x in item['g_polynomial_components']))
    K=sqrt_upper(Q(3))*Q(Kinf,S)
    a=sqrt_upper(Q(a_squared,S*S));eta2=sqrt_upper(Q(3))*Q(linear_error,S)
    rho,R,remainder=nonlinear_enclosure(K,a)
    max_value=max(sum(abs(row[j]) for row in item['V']) for item in locals for j in range(3))
    uniform_rounding=sqrt_upper(Q(3))*(Q(max_value,S*(1<<52))+Q(1,1<<1074))
    return dict(K=K,a=a,linear_error=eta2,rho=rho,R=R,
        nonlinear_remainder=remainder,total_error=None if remainder is None else remainder+eta2,
        uniform_binary64_output_rounding=uniform_rounding,
        terminal_linear_error=sqrt_upper(Q(3))*Q(eta[-1],S),
        max_product_radius=Q(max_product_radius,S),
        global_fixed_point_products=products)


def validate(times,nodes,max_step=Q('0.02'),bits=512,degree=80,progress=None,vector_degree=None):
    times=list(map(Q,times));nodes=[list(map(Q,row)) for row in nodes]
    if len(times)!=len(nodes) or len(times)<2:raise ValueError('Bad guide layout')
    if times[0]!=0 or any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Increasing times required')
    if bits<32 or degree<8 or max_step<=0:raise ValueError('Bad certificate parameters')
    S=1<<bits
    if any((x*S).denominator!=1 for x in nodes[0]):raise ValueError('Initial state must fit dyadic grid exactly')
    initial=[int(x*S) for x in nodes[0]];locals=[];subtimes=[times[0]]
    for index in range(len(times)-1):
        h=times[index+1]-times[index];curve=hermite(nodes[index],nodes[index+1],h)
        steps=ceil(h/max_step)
        for k in range(steps):
            local_curve=[restrict_polynomial(row,Q(k,steps),Q(1,steps)) for row in curve]
            proof=local_proof(local_curve,h/steps,initial,bits,degree,vector_degree)
            initial=proof['endpoint'];locals.append(proof)
            subtimes.append(times[index]+h*Q(k+1,steps))
        if progress and (index+1)%20==0:progress(index+1,len(times)-1)
    global_bounds=global_proof(locals,bits)
    return dict(bounds=global_bounds,locals=locals,subtimes=subtimes,bits=bits,degree=degree,
        vector_degree=degree if vector_degree is None else vector_degree,
        max_step=max_step,times=times,nodes=nodes)


def classical_completion(times,nodes,K,max_step=Q('0.02'),bits=512,degree=20):
    """Reuse a certified K, recompute ONLY the particular 3-vector.

    A pointwise global residual bound controls the new linear solution error.
    No fundamental matrix hierarchy or signed all-pairs products are repeated.
    degree=0 matrix work inside local_proof is a small, counted dummy residual
    check; its returned E is not used as a transition in this routine.
    """
    times=list(map(Q,times));nodes=[list(map(Q,row)) for row in nodes];K=Q(K)
    if K<0 or len(times)!=len(nodes) or len(times)<2 or times[0]!=0:
        raise ValueError('Bad completion inputs')
    if any(b<=a for a,b in zip(times,times[1:])):raise ValueError('Increasing times required')
    S=1<<bits;initial=[int(x*S) for x in nodes[0]];locals=[];subtimes=[times[0]]
    if any((x*S).denominator!=1 for x in nodes[0]):raise ValueError('Initial state must fit dyadic grid')
    residual_sup=Q(0);approx_g=Q(0)
    for index in range(len(times)-1):
        h=times[index+1]-times[index];curve=hermite(nodes[index],nodes[index+1],h)
        steps=ceil(h/max_step)
        for k in range(steps):
            local_curve=[restrict_polynomial(row,Q(k,steps),Q(1,steps)) for row in curve]
            proof=local_proof(local_curve,h/steps,initial,bits,0,degree)
            initial=proof['endpoint'];locals.append(proof);subtimes.append(times[index]+h*Q(k+1,steps))
            # A degree<=degree+7 residual satisfies sup norm <=(degree+8)*
            # sum_k norm(R_k)/(k+1). Convert d/dtau to d/dt with 1/h_i.
            residual_sup=max(residual_sup,Q((degree+8)*proof['residual_V_scaled2'],S*S)/(h/steps))
            approx_g=max(approx_g,sqrt_upper(sum(Q(x,S)**2 for x in proof['g_polynomial_components'])))
    eta=K*sqrt_upper(Q(3))*residual_sup;a=approx_g+eta
    rho,R,remainder=nonlinear_enclosure(K,a)
    max_value=max(sum(abs(row[j]) for row in item['V']) for item in locals for j in range(3))
    rounding=sqrt_upper(Q(3))*(Q(max_value,S*(1<<52))+Q(1,1<<1074))
    bounds=dict(K=K,a=a,linear_error=eta,rho=rho,R=R,nonlinear_remainder=remainder,
        uniform_binary64_output_rounding=rounding,
        total_error=None if remainder is None else remainder+eta+rounding,
        extra_scalar_integer_multiplications=sum(x['fixed_point_products'] for x in locals),
        residual_sup_infinity=residual_sup)
    return dict(bounds=bounds,locals=locals,subtimes=subtimes,bits=bits,degree=degree,
        times=times,nodes=nodes,max_step=max_step)


def evaluate(result,time):
    """Return binary64 output with an exact, separately bounded rounding error."""
    t=Q(time);ts=result['subtimes']
    if t<ts[0] or t>ts[-1]:raise ValueError('Time outside certificate')
    i=max(0,bisect_left(ts,t)-1);i=min(i,len(result['locals'])-1)
    tau=(t-ts[i])/(ts[i+1]-ts[i]);S=1<<result['bits'];coeff=result['locals'][i]['V']
    values=[]
    for axis in range(3):
        value=Q(0)
        for row in reversed(coeff):value=value*tau+Q(row[axis],S)
        values.append(value)
    rounded=list(map(float,values));rounding=sqrt_upper(sum((Q(x)-y)**2 for x,y in zip(rounded,values)))
    return rounded,rounding
