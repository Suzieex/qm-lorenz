"""Independent exact identities and native-oracle/structured-solver checks."""
import unittest,tempfile,contextlib,io,json
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
import mpmath as mp
from residual_quantum_candidate import (residual,picard_certificate,reference,
    polyval,bernstein_range,finite_oracle)


class QuantumCandidates(unittest.TestCase):
    def test_known_residual_is_exactly_integrable(self):
        for p in [2,3,4,8]:
            d=residual([1,1,1],.005,p)
            for axis,row in enumerate(d['R']):
                self.assertTrue(all(c==0 for c in row[:p]))
                self.assertEqual(d['integrals'][axis],sum(c/Q(k+1) for k,c in enumerate(row)))
                self.assertEqual(sum(d['U'][axis]),sum(d['P'][axis])+d['integrals'][axis])
            nodes,weights=np.polynomial.legendre.leggauss(p+1)
            for row,truth in zip(d['R'],d['integrals']):
                value=np.dot(weights,np.polynomial.polynomial.polyval((nodes+1)/2,list(map(float,row))))/2
                self.assertLess(abs(value-float(truth)),1e-12*max(abs(float(truth)),1e-20))

    def test_local_certificate_against_independent_nonlinear_solution(self):
        for u in [[1,1,1],[-8,8,27]]:
            d=residual(u,.005,4);upper=picard_certificate(u,d)
            for t in [Q(1,2),Q(1)]:
                truth=reference(u,.005*float(t))
                values=[polyval(row,t) for row in d['U']]
                with mp.workdps(100):
                    error=mp.sqrt(sum((mp.mpf(v.numerator)/v.denominator-r)**2 for v,r in zip(values,truth)))
                    self.assertLessEqual(error,mp.mpf(upper.numerator)/upper.denominator)

    def test_native_finite_mean_oracle(self):
        try:import qiskit
        except ImportError:self.skipTest('Install requirements-quantum.txt')
        result=finite_oracle()
        self.assertLess(result['error'],1e-12)
        self.assertEqual(result['classical_table_evaluations'],result['grid'])
        self.assertGreater(result['gates']['cx'],0)

    def test_lorenz_groundstate_rank_constraint(self):
        from lifted_groundstate_audit import audit
        with tempfile.TemporaryDirectory() as folder,contextlib.redirect_stdout(io.StringIO()):
            r=audit(Path(folder))
        self.assertEqual(r['exact_lifted_residual'],['0']*3)
        self.assertEqual(r['exact_matrix_rank'],4)
        self.assertEqual(r['exact_zero_energy_subspace_dimension'],7)
        self.assertTrue(all(Q(x)>0 for x in r['leading_principal_minors_of_Y']))
        self.assertGreater(r['spurious_mean_original_residual_norm'],.25)
        self.assertNotEqual(Q(r['initial_product_projection_nonzero_rank2_minor']),0)
        self.assertGreater(r['initial_product_projection_original_residual_norm'],1e-3)
        self.assertEqual(sum(map(Q,r['bounded_support_probabilities'])),1)
        left,right=r['support_containment_exact_integer_check'];self.assertLess(left,right)

    def test_spacetime_implicit_methods_against_known_linear_flow(self):
        from spacetime_candidate import midpoint_system,gauss_legendre_two_stage,gauss_collocation_aligned
        class Constant:
            times=np.linspace(0,5,6)
            def evaluate(self,t):return -np.diag([1.,2.,3.]),np.array([1.,2.,3.])
        guide=Constant();M=8;h=5/M;lam=np.array([1.,2.,3.])
        A,b,history,_=midpoint_system(guide,M)
        stability=(1-lam*h/2)/(1+lam*h/2)
        np.testing.assert_allclose(history[-1],1-stability**M,atol=1e-13)
        np.testing.assert_allclose(np.linalg.solve(A,b).reshape(M,3),history,atol=1e-13)
        A,b,history,_,meta=gauss_legendre_two_stage(guide,M)
        z=-lam*h;stability=(1+z/2+z*z/12)/(1-z/2+z*z/12)
        np.testing.assert_allclose(history[-1],1-stability**M,atol=2e-13)
        self.assertEqual(meta['transition_precompute_rhs_columns'],4)
        self.assertEqual(meta['classical_direct_rhs_columns'],1)
        A,b,history,_,meta=gauss_collocation_aligned(guide,M)
        np.testing.assert_allclose(history[-1],1-np.exp(-5*lam),atol=2e-11)
        self.assertLess(meta['direct_stage_vs_transition_relative_difference'],1e-12)


if __name__=='__main__':unittest.main()
