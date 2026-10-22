"""Structural and circuit tests; standard library unittest, no extra dependency."""
import unittest
import numpy as np
from scipy.linalg import expm
from carleman import (Lorenz, Carleman, DenseLCUCircuit, ProductLCUCircuit, unitary_dilation,
                      taylor_data, qae_distribution)


class Validation(unittest.TestCase):
    def test_generator_chain_rule_for_interior_degrees(self):
        model, v = Lorenz(), np.array([.1, -.2, .25])
        c = Carleman(model, 4)
        f = model.rhs(0, model.restore(v))/model.scale
        phi = c.lift(v)
        result = c.C @ phi
        for i, a in enumerate(c.exponents):
            if sum(a) == 4:
                continue
            expected = 0.
            for j in range(3):
                if a[j]:
                    reduced = np.array(a)
                    reduced[j] -= 1
                    expected += a[j]*np.prod(v**reduced)*f[j]
            self.assertAlmostEqual(result[i], expected, places=12)

    def test_julia_unitarity_and_nonhermitian_block(self):
        A = np.array([[.1, .7], [-.2, .05]])
        U = unitary_dilation(A)
        np.testing.assert_allclose(U.T @ U, np.eye(4), atol=2e-14)
        np.testing.assert_allclose(U[:2, :2], A)

    def test_taylor_error_bound(self):
        C = Carleman(Lorenz(), 3).C
        E, _, _, tail = taylor_data(C, .004, tol=1e-11)
        self.assertLessEqual(np.linalg.norm(E-expm(.004*C), 2), tail+2e-14)

    def test_lcu_and_signed_hadamard_recovery(self):
        c = Carleman(Lorenz(), 2)
        for increment in [False, True]:
            circuit = DenseLCUCircuit(c, .001, increment=increment)
            for u in [np.array([1., 1., 1.]), np.array([-8., 8., 27.])]:
                p, B, error = circuit.hadamard_probabilities(u)
                expected = c.model.restore((circuit.E @ c.lift(c.model.normalize(u)))[c.linear])
                np.testing.assert_allclose(circuit.recover(u, p, B), expected, atol=2e-12)
                self.assertLess(error, 2e-14)
                state, _ = circuit.output(u)
                D = circuit.E-np.eye(len(c.C)) if increment else circuit.E
                target = D @ c.lift(c.model.normalize(u))/(B*circuit.gamma)
                np.testing.assert_allclose(state[:len(c.C)], target, atol=2e-13)

    def test_qae_is_exact_when_phase_is_on_grid(self):
        for j in [0, 1, 5, 16, 25, 32]:
            size = 64
            p = np.sin(np.pi*j/size)**2
            distribution = qae_distribution(p, 6)
            values = np.sin(np.pi*np.arange(size)/size)**2
            self.assertAlmostEqual(distribution.sum(), 1., places=13)
            self.assertLess(np.sum(distribution*(values-p)**2), 1e-24)

    def test_product_flags_match_dense_power_circuit(self):
        c = Carleman(Lorenz(), 2)
        u = np.array([-8., 8., 27.])
        for increment in [False, True]:
            dense = DenseLCUCircuit(c, .001, increment)
            product = ProductLCUCircuit(c, .001, increment)
            pd, _, _ = dense.hadamard_probabilities(u)
            pp, _, _ = product.hadamard_probabilities(u)
            np.testing.assert_allclose(pp, pd, atol=2e-13)
            state, _ = product.output(u)
            np.testing.assert_allclose(product.apply(product.apply(state), True), state,
                                       atol=2e-13)


if __name__ == '__main__':
    unittest.main()
