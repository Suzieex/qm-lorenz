"""Independent identities, analytic flows and numerical consistency checks."""
import unittest,math,copy
from fractions import Fraction as Q
import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp
from carleman import Lorenz
from global_correction import (hermite,time_coefficients,restrict_polynomial,
    polynomial_range_upper_int,exp_upper_int,local_proof,validate,evaluate,
    PARAMETERS,matrix_norm,classical_completion)


class GlobalCorrection(unittest.TestCase):
    def test_exact_C3_guide_and_restriction(self):
        left=[Q(1),Q(2),Q(3)];right=[Q(11,10),Q(13,7),Q(20,9)];h=Q(1,20)
        curve=hermite(left,right,h);l=time_coefficients(left);r=time_coefficients(right)
        for axis,row in enumerate(curve):
            for k in range(4):
                derivative=sum(math.factorial(j)//math.factorial(j-k)*row[j] for j in range(k,8))
                self.assertEqual(derivative,math.factorial(k)*r[axis][k]*h**k)
                self.assertEqual(row[k],l[axis][k]*h**k)
            restricted=restrict_polynomial(row,Q(1,3),Q(1,5))
            for t in [Q(0),Q(2,7),Q(1)]:
                a=sum(c*t**j for j,c in enumerate(restricted))
                b=sum(c*(Q(1,3)+t/5)**j for j,c in enumerate(row))
                self.assertEqual(a,b)

    def test_Bernstein_range_preserves_cancellation(self):
        S=1<<100;coeff=[0,S,-S]+[0]*30
        bound=polynomial_range_upper_int(coeff)
        self.assertGreaterEqual(Q(bound,S),Q(1,4))
        self.assertLess(Q(bound,S),Q(3,10))
        for k in range(101):
            t=Q(k,100);value=sum(a*t**j for j,a in enumerate(coeff))
            self.assertLessEqual(abs(value),bound)

    def test_exponential_enclosure(self):
        with mp.workdps(100):
            for a in [Q(0),Q(1,7),Q(2)]:
                S=1<<128;q=exp_upper_int(a,S)
                self.assertGreaterEqual(mp.mpf(q)/S,mp.exp(mp.mpf(a.numerator)/a.denominator))

    def test_local_fundamental_against_analytic_exponential(self):
        S=1<<128;h=Q('0.01');curve=[[Q(0)]*8 for _ in range(3)]
        result=local_proof(curve,h,[0,0,0],128,24)
        with mp.workdps(120):
            dt=mp.mpf(h.numerator)/h.denominator
            A=mp.matrix([[-10,10,0],[28,-1,0],[0,0,-mp.mpf(PARAMETERS[2].numerator)/PARAMETERS[2].denominator]])
            exact=mp.expm(dt*A);error=max(sum(abs(exact[i,j]-mp.mpf(result['E'][3*i+j])/S) for j in range(3)) for i in range(3))
            self.assertLessEqual(error,mp.mpf(result['delta_E'])/S)

    def test_global_certificate_and_output(self):
        guide=solve_ivp(Lorenz().rhs,(0,.1),[1,1,1],method='DOP853',rtol=1e-6,atol=1e-8)
        result=validate([Q(float(t)) for t in guide.t],
            [[Q(float(x)) for x in row] for row in guide.y.T],bits=256,degree=48)
        b=result['bounds'];self.assertLess(b['rho'],1)
        self.assertLess(b['total_error'],Q('1e-8'))
        from long_validation import taylor_coefficients
        for t in [Q(0),Q('0.037'),Q('0.1')]:
            value,rounding=evaluate(result,t)
            self.assertLessEqual(rounding,b['uniform_binary64_output_rounding'])
            # Binary64 DOP853 reference error can exceed this very small bound.
            # Use an independent high-precision direct nonlinear integration.
            with mp.workdps(100):
                state=[mp.mpf(1)]*3;remaining=mp.mpf(t.numerator)/t.denominator
                parameters=[mp.mpf(x.numerator)/x.denominator for x in PARAMETERS]
                while remaining>0:
                    step=min(remaining,mp.mpf('.01'))
                    coeff=taylor_coefficients(state,60,parameters)
                    state=[mp.polyval(a,step) for a in coeff];remaining-=step
                error=mp.sqrt(sum((mp.mpf(a)-x)**2 for a,x in zip(value,state)))
                upper=b['total_error']+rounding
                self.assertLessEqual(error,mp.mpf(upper.numerator)/upper.denominator)
        from global_correction_study import digest,encode_bounds
        from verify_global_correction import verify_one
        endpoint,rounding=evaluate(result,Q('0.1'))
        cert=dict(exact_parameters=list(map(str,PARAMETERS)),times=list(map(str,result['times'])),
            nodes=[list(map(str,row)) for row in result['nodes']],max_step=str(result['max_step']),
            bits=256,degree=48,local_proofs_sha256=digest(result),bounds=encode_bounds(b),
            endpoint=list(map(str,endpoint)),endpoint_rounding=str(rounding),T='1/10',
            target='1/100000000',accuracy_gate_pass=True)
        self.assertTrue(verify_one(cert))
        altered=copy.deepcopy(cert);altered['bounds']['K']='0'
        with self.assertRaises(AssertionError):verify_one(altered)

    def test_classical_completion_reuses_K(self):
        guide=solve_ivp(Lorenz().rhs,(0,.1),[1,1,1],method='DOP853',rtol=1e-6,atol=1e-8)
        times=[Q(float(t)) for t in guide.t];nodes=[[Q(float(x)) for x in row] for row in guide.y.T]
        source=validate(times,nodes,bits=256,degree=48,vector_degree=12)
        result=classical_completion(times,nodes,source['bounds']['K'],bits=256,degree=20)
        self.assertLess(result['bounds']['total_error'],Q('1e-8'))
        self.assertLess(result['bounds']['extra_scalar_integer_multiplications'],
            sum(x['fixed_point_products'] for x in source['locals']))
        one,_=evaluate(source,Q('0.1'));two,_=evaluate(result,Q('0.1'))
        self.assertLess(np.linalg.norm(np.array(one)-two),1e-10)

    def test_conditional_readout_budget(self):
        from global_correction_study import qae_budget
        from residual_bounds import sqrt_upper
        for s in [Q('1e-6'),Q('5e-7')]:
            budget=Q('1e-8');sqrt3=sqrt_upper(Q(3))
            result=qae_budget(s,budget);delta=min(Q(1,8),budget/(24*sqrt3*s))
            zeta=min(Q(1,64),budget/(32*sqrt3*s));pi=Q(22,7)
            self.assertLess(delta+2*zeta,Q(1,4))
            self.assertLessEqual(6*sqrt3*s*(delta+2*zeta),budget)
            M=1<<result['phase_bits']
            self.assertGreaterEqual(M,4*pi/delta)
            self.assertGreaterEqual(M,2*pi*sqrt_upper(1/delta))
            r=result['median_repeats'];q=1-8/(pi*pi)
            failure=sum(math.comb(r,k)*q**k*(1-q)**(r-k) for k in range(r//2+1,r+1))
            self.assertLessEqual(4*failure,Q(1,20))


if __name__=='__main__':unittest.main()
