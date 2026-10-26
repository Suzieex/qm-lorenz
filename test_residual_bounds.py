"""Exact identities and independent numerical checks of the new bounds."""
import unittest
from fractions import Fraction as F
import mpmath as mp
from carleman import Lorenz
from residual_bounds import (sqrt_upper,outward_interval,local_enclosure,
    exact_polynomials,rational_generator,path_data,certified_bounds,outward_decimal)
from residual_bound_study import numerical_observations,to_mp


class ResidualBounds(unittest.TestCase):
    def test_outward_arithmetic(self):
        for x in [F(1,3),F(-1,3),F(2),F(0)]:
            lo,hi=outward_interval((x,x),bits=12)
            self.assertLessEqual(lo,x);self.assertGreaterEqual(hi,x)
        for x in [F(2),F(1,1000000),F(0)]:
            upper=sqrt_upper(x)
            self.assertGreaterEqual(upper*upper,x)
            self.assertGreaterEqual(F(outward_decimal(upper)),upper)

    def test_exact_residual_identity(self):
        model=Lorenz();N=3;v=[F(1,17),F(-2,19),F(1,7)]
        data=path_data(N,64.,J=11)
        phi=lambda a:mp_product([v[j]**p for j,p in enumerate(a)])
        field=[sum(c*phi(a) for a,c in p.items()) for p in exact_polynomials(model)]
        for i,a in enumerate(data['exponents']):
            derivative=F(0)
            for j,p in enumerate(a):
                if p:
                    reduced=list(a);reduced[j]-=1
                    derivative+=p*phi(reduced)*field[j]
            retained=sum(c*phi(data['exponents'][j]) for j,c in data['C'][i].items())
            residual=sum(c*phi(data['shell'][j]) for j,c in data['G'][i].items())
            self.assertEqual(derivative,retained+residual)

    def test_leading_coupling_paths_and_cancellations(self):
        data=path_data(3,64.,J=11)
        self.assertTrue(all(not row for H in data['H'][:2] for row in H))
        H=data['H'][2]
        y={data['shell'][j]:v for j,v in H[1].items()}
        z={data['shell'][j]:v for j,v in H[2].items()}
        self.assertEqual(H[0],{})
        self.assertEqual(y,{(3,0,1):F(64**3)})
        self.assertEqual(z,{(3,1,0):F(-64**3)})

    def test_local_tube_handles_zero_initial_derivative(self):
        model=Lorenz()
        for h in ['0.001','0.005','0.01']:
            tube,proof=local_enclosure(model,[1,1,1],F(h))
            self.assertTrue(proof['strict_self_inclusion'])
            self.assertTrue(all(lo<=F(1)<=hi for lo,hi in tube))

    def test_default_bound_is_tight_and_reference_independent(self):
        bounds,certificate,data=certified_bounds(Lorenz(),3,.001,[1,1,1])
        self.assertLess(bounds['path_local'],F('7e-9'))
        self.assertGreater(bounds['path_local'],F('6e-9'))
        self.assertLess(bounds['path_local'],bounds['path_global'])
        self.assertLess(bounds['block_local'],bounds['block_global'])
        observed=numerical_observations(Lorenz(),3,.001,[1,1,1],data)
        with mp.workdps(100):
            self.assertLess(mp.mpf(observed['observed_ideal_mp_error_decimal']),to_mp(bounds['path_local']))
        self.assertLess(observed['voc_reconstruction_error'],1e-25)
        for name,value in bounds.items():
            self.assertGreaterEqual(F(certificate['outward_decimal_bounds'][name]),value)

    def test_other_initial_state_and_order(self):
        for N,u,h in [(1,[-8,8,27],.01),(5,[1,1,1],.0005),(3,[20,20,30],.005)]:
            bounds,_,data=certified_bounds(Lorenz(),N,h,u)
            observed=numerical_observations(Lorenz(),N,h,u,data)
            self.assertLessEqual(observed['observed_ideal_mp_error'],float(bounds['path_local']))
            self.assertLessEqual(bounds['path_local'],bounds['path_global'])

    def test_domain_conditions_are_enforced(self):
        with self.assertRaises(ValueError):certified_bounds(Lorenz(),3,.001,[100,100,100])
        with self.assertRaises(ValueError):certified_bounds(Lorenz(rho=30),3,.001,[1,1,1])


def mp_product(values):
    result=F(1)
    for v in values:result*=v
    return result


if __name__=='__main__':unittest.main()
