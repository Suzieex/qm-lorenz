"""Nontrivial checks of affine certificates, predictors and weighted circuits."""
import unittest
from fractions import Fraction as Q
import numpy as np
from carleman import Lorenz
from residual_bounds import certified_bounds
from research_bounds import (AffineLorenz,affine_field,generator,affine_bound,
    predictor_bounds,product,select_taylor)
from research_workflow import alpha_exact,qae_budget,numerical_step,exp_upper


class ResearchWorkflow(unittest.TestCase):
    def test_default_agrees_with_existing_certificate(self):
        a=certified_bounds(Lorenz(),3,.001,[1,1,1])[0]['path_local']
        b=affine_bound(AffineLorenz(),3,.001,[1,1,1])['truncation']
        self.assertLess(abs(a-b),Q('1e-25'))

    def test_affine_chain_rule_and_truncation_observation(self):
        model=AffineLorenz(pivot=(1.125,2.25,29.125),scales=(16.,32.,64.))
        v=[Q(1,10),Q(-1,5),Q(3,20)];field=affine_field(model)
        value=[sum(c*product(v[j]**p for j,p in enumerate(a)) for a,c in poly.items()) for poly in field]
        physical=model.rhs(0,model.restore(list(map(float,v))))/np.array(model.scales)
        np.testing.assert_allclose(list(map(float,value)),physical,atol=1e-13)
        from scipy.integrate import solve_ivp
        u=[-8.,8.,27.];h=.005;b=affine_bound(model,3,h,u)
        truth=solve_ivp(Lorenz().rhs,(0,h),u,method='DOP853',rtol=2.3e-14,atol=2e-14).y[:,-1]
        self.assertLess(np.linalg.norm(truth-numerical_step(b,u)),float(b['truncation']))

    def test_weighting_preserves_physical_map_and_certificate(self):
        model=AffineLorenz();u=[1,1,1]
        plain=affine_bound(model,3,.005,u)
        weighted=affine_bound(model,3,.005,u,weights=[1,1,Q(1,2),Q(1,6),Q(1,24)])
        np.testing.assert_allclose(numerical_step(plain,u),numerical_step(weighted,u),atol=1e-13)
        # Finite signed-path contributions are invariant, but absolute tails
        # may differ because the similarity changes the infinity norm.
        self.assertLess(abs(float(plain['truncation']-weighted['truncation'])),1e-15)

    def test_predictor_has_independent_enclosed_error(self):
        import mpmath as mp
        from long_validation import taylor_coefficients
        for u in [(1.,1.,1.),(-8.,8.,27.)]:
            results=predictor_bounds(u,.005,8)
            with mp.workdps(80):
                parameters=[mp.mpf(float(x)) for x in (10,28,8/3)]
                coefficients=taylor_coefficients([mp.mpf(x) for x in u],80,parameters)
                truth=[mp.polyval(a,mp.mpf('0.005')) for a in coefficients]
                for r in results:
                    values=[mp.mpf(x.numerator)/x.denominator for x in r['value']]
                    error=mp.sqrt(sum((a-b)**2 for a,b in zip(values,truth)))
                    self.assertLessEqual(error,mp.mpf(r['error'].numerator)/r['error'].denominator)

    def test_chebyshev_residual_identity_and_observation(self):
        from research_bounds import chebyshev_bound,chebyshev_polynomials,lie_derivative
        b=chebyshev_bound(AffineLorenz(),2,.005,[1,1,1]);data=b['data']
        polys=chebyshev_polynomials(3);field=affine_field(AffineLorenz())
        v=[Q(1,7),Q(-1,11),Q(2,9)]
        evaluate=lambda poly:sum(c*product(v[j]**p for j,p in enumerate(a)) for a,c in poly.items())
        values=list(map(evaluate,polys));D=len(data['C'])
        for i,poly in enumerate(polys[:D]):
            exact_value=evaluate(lie_derivative(poly,field))
            retained=sum(c*values[j] for j,c in data['C'][i].items())
            residual=sum(c*values[D+j] for j,c in data['G'][i].items())
            self.assertEqual(exact_value,retained+residual)
        from scipy.integrate import solve_ivp
        truth=solve_ivp(Lorenz().rhs,(0,.005),[1,1,1],method='DOP853',rtol=2.3e-14,atol=2e-14).y[:,-1]
        b['certificate']['pivot']=['0','0','38']
        self.assertLess(np.linalg.norm(numerical_step(b,[1,1,1])-truth),float(b['truncation']))

    def test_predictor_degree_boundary(self):
        from residual_bounds import row_times
        u=[-8,8,27];h=Q('0.005');b=affine_bound(AffineLorenz(),3,h,u)
        term=[{i:Q(1)} for i in b['data']['linear']];value=list(map(Q,u))
        predictions=predictor_bounds(u,h,3)
        for p in range(1,4):
            term=[{j:c*h/p for j,c in row_times(row,b['data']['C']).items()} for row in term]
            value=[v+s*sum(c*b['phi'][j] for j,c in row.items())
                   for v,s,row in zip(value,b['factors'],term)]
            if p<3:self.assertEqual(value,predictions[p-1]['value'])
            else:self.assertNotEqual(value,predictions[p-1]['value'])

    def test_alpha_is_implemented_and_bounds_spectral_norm(self):
        from sparse_compiler import compile_block_encoding
        for model in [AffineLorenz(),AffineLorenz(pivot=(1.,1.,1.),scales=(16.,32.,64.))]:
            alpha=alpha_exact(model,3)
            self.assertAlmostEqual(float(alpha),compile_block_encoding(model,3).alpha,places=11)
            _,C=generator(affine_field(model),3)
            matrix=np.array([[float(row.get(j,0)) for j in range(len(C))] for row in C])
            self.assertGreaterEqual(float(alpha),np.linalg.norm(matrix,2))

    def test_weighted_native_circuit(self):
        from weighted_compiler import compile_weighted
        from sparse_compiler import encoded_column
        model=AffineLorenz(scales=(16.,32.,64.));weights=[1.,1.,.5]
        encoding=compile_weighted(model,2,weights)
        exponents,C=generator(affine_field(model),2,weights)
        for j in [0,1,4,8]:
            column=encoded_column(encoding,exponents[j])
            np.testing.assert_allclose([column.get(a,0) for a in exponents],
                [float(row.get(j,0))/encoding.alpha for row in C],atol=5e-13)
        self.assertNotIn('unitary',encoding.circuit.count_ops())

    def test_general_coefficient_state_reads_signed_overlap(self):
        from weighted_compiler import coefficient_row_hadamard
        from qiskit.quantum_info import Statevector
        phi=np.array([1.,-.3,.2,0.,-.7]);row=np.array([-.2,.4,0.,.1,.5])
        qc=coefficient_row_hadamard(phi,row)
        probability=Statevector.from_instruction(qc).probabilities([0])[0]
        expected=(1+row@phi/(np.linalg.norm(phi)*np.linalg.norm(row)))/2
        self.assertAlmostEqual(probability,expected,places=11)

    def test_qae_and_propagation_budget(self):
        a=qae_budget([1.,2.,3.],1e-5,K=2)
        b=qae_budget([1.,2.,3.],1e-6,K=2)
        self.assertGreaterEqual(b['base_calls'],a['base_calls'])
        self.assertEqual(a['base_calls'],2*a['qae_A_calls'])
        e=exp_upper(Q(105))
        import mpmath as mp
        with mp.workdps(180):self.assertGreater(mp.mpf(e.numerator)/e.denominator,mp.exp(105))


if __name__=='__main__':unittest.main()
