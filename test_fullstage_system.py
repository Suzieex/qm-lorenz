"""Analytic-flow, representation and sparse-norm tests for full stage systems."""
import unittest
from unittest.mock import patch
import numpy as np
from scipy.sparse.linalg import spsolve
from fullstage_system import (build_fullstage,structured_solve,tableau,
    local_block_precondition,singular_diagnostics)


class ConstantGuide:
    times=np.linspace(0,2,5)
    def evaluate(self,t):return -np.diag([1.,2.,3.]),np.array([1.,2.,3.])


class FullStageChecks(unittest.TestCase):
    def test_analytic_diagonal_flow(self):
        for mode in ['state','increment']:
            system=build_fullstage(ConstantGuide(),16,4,mode)
            x,accounting=structured_solve(system)
            expected=1-np.exp(-2*np.arange(1,4))
            np.testing.assert_allclose(x[-3:],expected,atol=3e-12,rtol=0)
            np.testing.assert_allclose(spsolve(system.matrix,system.rhs),x,atol=2e-13)
            self.assertEqual(accounting['classical_rhs_columns'],1)
            self.assertEqual(accounting['classical_stage_solves'],16)
            self.assertEqual(system.metadata['build_stage_linear_solves'],0)

    def test_implicit_midpoint_stability_function(self):
        M=8;system=build_fullstage(ConstantGuide(),M,1)
        x,_=structured_solve(system);z=-2/M*np.arange(1,4)
        expected=1-((1+z/2)/(1-z/2))**M
        np.testing.assert_allclose(x[-3:],expected,atol=2e-14,rtol=0)

    def test_polynomial_forcing_stage_values(self):
        class Polynomial:
            times=np.linspace(0,2,5)
            def evaluate(self,t):return np.zeros((3,3)),np.array([1,t,t**3])
        system=build_fullstage(Polynomial(),8,4,'state')
        x,_=structured_solve(system);a,b,c=tableau(4)
        for i,local in enumerate(x.reshape(8,-1)):
            left,right=system.mesh[i:i+2];times=left+c*(right-left)
            expected=np.array([times,times**2/2,times**4/4]).T
            observed=local[:-3].reshape(4,3)/np.sqrt(b)[:,None]
            np.testing.assert_allclose(observed,expected,atol=5e-14,rtol=0)
        np.testing.assert_allclose(x[-3:],[2,2,4],atol=5e-14,rtol=0)

    def test_stage_changes_preserve_physical_solution(self):
        a,b,c=tableau(4);answers={}
        for mode in ['state','increment']:
            for weight in ['quadrature','unit']:
                system=build_fullstage(ConstantGuide(),8,4,mode,weight)
                x,_=structured_solve(system);local=x.reshape(8,-1)
                stage=local[:,:-3].reshape(8,4,3)
                if weight=='quadrature':stage=stage/np.sqrt(b)[None,:,None]
                if mode=='increment':stage=stage+np.vstack([np.zeros(3),local[:-1,-3:]])[:,None,:]
                answers[mode,weight]=(stage,local[:,-3:])
        reference=answers['state','unit']
        for stage,endpoints in answers.values():
            np.testing.assert_allclose(stage,reference[0],atol=3e-14,rtol=0)
            np.testing.assert_allclose(endpoints,reference[1],atol=3e-14,rtol=0)

    def test_build_has_no_stage_solve(self):
        with patch('numpy.linalg.solve',side_effect=AssertionError('build must not solve stages')):
            system=build_fullstage(ConstantGuide(),8,8)
        self.assertEqual(system.metadata['build_propagator_compilations'],0)

    def test_preconditioning_invariance_and_cost(self):
        system=build_fullstage(ConstantGuide(),8,4)
        P,q,metadata=local_block_precondition(system)
        x,_=structured_solve(system)
        np.testing.assert_allclose(spsolve(P,q),x,atol=3e-14,rtol=0)
        self.assertEqual(metadata['preconditioner_rhs_columns'],4)
        self.assertEqual(metadata['preconditioner_classical_stage_solves'],8)

    def test_sparse_singular_diagnostic_against_dense(self):
        system=build_fullstage(ConstantGuide(),8,4)
        result=singular_diagnostics(system.matrix)
        values=np.linalg.svd(system.matrix.toarray(),compute_uv=False)
        self.assertAlmostEqual(result['sigma_max']/values[0],1,places=8)
        self.assertAlmostEqual(result['sigma_min']/values[-1],1,places=8)


if __name__=='__main__':unittest.main()
