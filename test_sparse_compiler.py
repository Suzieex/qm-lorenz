"""Gate-level tests; install requirements-quantum.txt to run this suite."""
import unittest
import numpy as np
try:
    import qiskit
    from qiskit.quantum_info import Statevector
    from carleman import Lorenz,Carleman
    from sparse_compiler import compile_block_encoding,encoded_column,simulate_sparse
    from sparse_workflow import build_sparse_lcu,build_sparse_hadamard,build_row_hadamard,row_series
    from optimization_study import ShiftedLorenz,taylor_predictor
    HAS_QISKIT=True
except ImportError:
    HAS_QISKIT=False


@unittest.skipUnless(HAS_QISKIT,'Install requirements-quantum.txt')
class SparseCompiler(unittest.TestCase):
    def test_generated_block_for_all_valid_columns(self):
        model=Lorenz()
        for N in [1,2,3]:
            encoding=compile_block_encoding(model,N);c=Carleman(model,N)
            for j,power in enumerate(c.exponents):
                column=encoded_column(encoding,power)
                actual=np.array([column.get(row,0) for row in c.exponents])
                np.testing.assert_allclose(actual,c.C[:,j]/encoding.alpha,atol=4e-13)
            for invalid in [(N+1,0,0),(0,N+1,0)]:
                self.assertLess(max([abs(a) for a in encoded_column(encoding,invalid).values()]+[0.]),1e-13)

    def test_loose_and_grouped_normalizations_encode_same_operator(self):
        model=Lorenz();c=Carleman(model,2)
        for grouped,tight in [(False,False),(True,False),(False,True)]:
            encoding=compile_block_encoding(model,2,grouped,tight)
            for j in [0,1,3,7]:
                column=encoded_column(encoding,c.exponents[j])
                actual=np.array([column.get(row,0) for row in c.exponents])
                np.testing.assert_allclose(actual,c.C[:,j]/encoding.alpha,atol=3e-13)
            self.assertLess(max([abs(a) for a in encoded_column(encoding,(3,0,0)).values()]+[0.]),1e-13)

    def test_shifted_model_chain_rule_and_generated_encoding(self):
        model=ShiftedLorenz(scale=48.,pivot=(1.,2.,29.));c=Carleman(model,2)
        encoding=compile_block_encoding(model,2)
        for j in [0,2,5]:
            column=encoded_column(encoding,c.exponents[j])
            np.testing.assert_allclose([column.get(row,0) for row in c.exponents],
                                       c.C[:,j]/encoding.alpha,atol=3e-13)
        v=np.array([.1,-.2,.15])
        np.testing.assert_allclose((c.C@c.lift(v))[c.linear],
                                  model.rhs(0,model.restore(v))/model.scale,atol=1e-13)

    def test_base_inverse_and_no_dense_unitary_gate(self):
        encoding=compile_block_encoding(Lorenz(),2)
        state=simulate_sparse(encoding.circuit,encoding.data_basis((1,0,1)))
        self.assertAlmostEqual(sum(abs(a)**2 for a in state.values()),1.,places=12)
        inverse=simulate_sparse(encoding.circuit.inverse(),state)
        self.assertAlmostEqual(abs(inverse.get(encoding.data_basis((1,0,1)),0)),1.,places=12)
        self.assertNotIn('unitary',encoding.circuit.count_ops())
        self.assertGreater(encoding.alpha,np.linalg.norm(Carleman(Lorenz(),2).C,2))

    def test_two_factor_generated_lcu_block(self):
        model=Lorenz();q,meta=build_sparse_lcu(model,1,.001,K=2)
        c=Carleman(model,1);D=.001*c.C+.001**2*c.C@c.C/2
        for j,power in enumerate(c.exponents):
            key=sum(a<<(2*i) for i,a in enumerate(power))
            state=simulate_sparse(q,key)
            for i,row in enumerate(c.exponents):
                output=sum(a<<(2*k) for k,a in enumerate(row))
                self.assertAlmostEqual(abs(state.get(output,0)-D[i,j]/meta['gamma']),0.,places=12)

    def test_generated_end_to_end_hadamard(self):
        model=Lorenz();u=np.array([1.,1.,1.]);h=.001
        qc,meta=build_sparse_hadamard(model,1,h,u,1,K=2)
        state=simulate_sparse(qc)
        probability=sum(abs(a)**2 for k,a in state.items() if not k&1)
        c=Carleman(model,1);phi=c.lift(model.normalize(u))
        D=h*c.C+h*h*c.C@c.C/2
        expected=(1+(D@phi)[c.linear[1]]/(meta['B']*meta['gamma']))/2
        self.assertAlmostEqual(probability,expected,places=11)

    def test_row_norm_has_an_actual_state_preparation_implementation(self):
        qc,meta=build_row_hadamard(Lorenz(),1,.001,np.ones(3),1,K=3)
        probability=Statevector.from_instruction(qc).probabilities([0])[0]
        self.assertAlmostEqual(probability,meta['expected_test_probability'],places=11)
        self.assertGreater(sum(qc.count_ops().values()),0)

    def test_predictor_matches_original_time_taylor_and_residual(self):
        model=Lorenz();u=np.array([-8.,8.,27.]);h=.002
        c,rows=row_series(model,5,h,12);phi=c.lift(model.normalize(u))
        for p in [1,2,3]:
            low=u+model.scale*(sum(rows[1:p+1])@phi)
            np.testing.assert_allclose(low,taylor_predictor(model,u,h,p),atol=2e-12)
            residual=sum(rows[p+1:])
            full=u+model.scale*(sum(rows[1:])@phi)
            np.testing.assert_allclose(low+model.scale*(residual@phi),full,atol=2e-12)


if __name__=='__main__':unittest.main()
