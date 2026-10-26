import unittest
from fractions import Fraction as Q
from fullstage_certificate import lagrange_basis,evaluate,bernstein_abs,residual,add,bound


class StageCertificate(unittest.TestCase):
    def test_exact_lagrange_and_integral(self):
        nodes=[Q(1,7),Q(3,7),Q(6,7)];basis=lagrange_basis(nodes)
        for i,p in enumerate(basis):
            for j,t in enumerate(nodes):self.assertEqual(evaluate(p,t),int(i==j))
        # Reconstruct the exact polynomial 2-3t+5t².
        original=[Q(2),Q(-3),Q(5)];recovered=[Q(0)]*3
        for t,p in zip(nodes,basis):recovered=add(recovered,[evaluate(original,t)*v for v in p])
        self.assertEqual(original,recovered)

    def test_bernstein_bound_contains_polynomial_extrema(self):
        # 4t(1-t) reaches 1; subdivision reduces its unsplit Bernstein bound 2.
        p=[Q(0),Q(4),Q(-4)]
        self.assertEqual(bernstein_abs(p),2)
        self.assertEqual(bernstein_abs(p,2),1)

    def test_zero_correction_residual_is_negative_guide_defect(self):
        # For a constant guide, -r=-f(c) exactly, with stored exact parameters.
        from fullstage_certificate import field
        c=[[Q(1)],[Q(2)],[Q(3)]];zero=[[Q(0)]]*3
        self.assertEqual(residual(c,zero,Q(1,10)),[[-v for v in p] for p in field(c)])

    def test_bad_trial_curve_is_not_assumed_accurate(self):
        from global_correction import PARAMETERS
        # At the origin the true guide/correction is identically zero. K=1
        # is conservative on [.0,.01], since .01 exp(.39)<1 in infinity norm.
        source=dict(times=['0','1/100'],T='1/100',nodes=[['0']*3,['0']*3],
            exact_parameters=list(map(str,PARAMETERS)),bounds={'K':'1','nonlinear_remainder':'0'})
        pieces=[dict(left='0',right='1/100',g=[['0','1/10000'],['0'],['0']])]
        result=bound(source,pieces)
        self.assertFalse(result['all_times_target_pass'])
        self.assertGreater(Q(result['total_exact_polynomial_error']),Q(1,10000))
        pieces[0]['g'][0][0]='1/10000'
        with self.assertRaises(AssertionError):bound(source,pieces)


if __name__=='__main__':unittest.main()
