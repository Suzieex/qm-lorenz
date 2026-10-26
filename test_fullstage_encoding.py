import unittest
import numpy as np
from scipy.sparse import csr_matrix
from fullstage_encoding import (diagonal_lcu,reconstruct,compile_shift_encoding,
    encoded_column,native_rhs_and_signed_overlap)


class StageEncoding(unittest.TestCase):
    def test_decomposition_and_exact_alpha(self):
        from fractions import Fraction as Q
        a=np.array([[2.,0.,-3.],[.5,-1.,0.],[0.,4.,2.]])
        d=diagonal_lcu(csr_matrix(a));wanted=np.eye(4);wanted[:3,:3]=a
        np.testing.assert_array_equal(reconstruct(d),wanted)
        self.assertEqual(d.alpha_exact,sum(d.weights.values(),Q(0)))
        self.assertGreaterEqual(d.alpha,np.linalg.norm(wanted,2))

    def test_native_block_handles_signs_zero_and_padding(self):
        try:import qiskit
        except ImportError:self.skipTest('Install requirements-quantum.txt')
        a=np.array([[2.,0.,-3.],[.5,-1.,0.],[0.,4.,2.]])
        encoding=compile_shift_encoding(csr_matrix(a))
        expected=reconstruct(encoding.decomposition)/encoding.decomposition.alpha
        for j in range(4):np.testing.assert_allclose(encoded_column(encoding,j),expected[:,j],atol=5e-13)
        self.assertNotIn('unitary',encoding.circuit.count_ops())

    def test_rhs_and_signed_interference(self):
        try:import qiskit
        except ImportError:self.skipTest('Install requirements-quantum.txt')
        a=np.array([[1.,-.25],[.5,1.]])
        encoding=compile_shift_encoding(csr_matrix(a))
        result=native_rhs_and_signed_overlap(encoding,np.array([1.,-2.]),1)
        self.assertLess(result['signed_test_probability'],.5)
        self.assertLess(result['signed_test_error'],1e-12)


if __name__=='__main__':unittest.main()
