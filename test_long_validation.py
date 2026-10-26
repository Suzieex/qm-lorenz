"""Independent checks for the long-horizon reference and QAE sampler."""
import unittest
import numpy as np
import mpmath as mp
from scipy.linalg import expm
from carleman import Lorenz,Carleman,qae_distribution,qae_rejection_samples
from long_validation import mp_reference,mp_increment_rows,first_exit


class LongValidation(unittest.TestCase):
    def test_qae_sampler_matches_folded_fft_distribution(self):
        p=.2;bits=6;M=2**bits
        rng=np.random.default_rng(20261026)
        values=qae_rejection_samples(np.full(50000,p),bits,1,rng)
        indices=np.rint(np.arcsin(np.sqrt(values))*M/np.pi).astype(int)
        observed=np.bincount(indices,minlength=M//2+1)/len(values)
        expected=np.zeros(M//2+1)
        for k,prob in enumerate(qae_distribution(p,bits)):
            expected[min(k,M-k)]+=prob
        self.assertLess(.5*np.abs(observed-expected).sum(),.015)

    def test_qae_high_resolution_and_endpoints(self):
        rng=np.random.default_rng(12)
        for p in [0.,.5,1.]:
            samples=qae_rejection_samples(np.full(50,p),40,7,rng)
            np.testing.assert_allclose(samples,p,atol=2e-15)
        samples=qae_rejection_samples(np.full(1000,.500123),40,7,rng)
        self.assertLess(np.median(np.abs(samples-.500123)),2e-12)

    def test_reference_refinement_and_binary_parameter(self):
        model=Lorenz();u=np.ones(3)
        coarse=mp_reference(model,u,.02,.001,10,45,60)
        fine=mp_reference(model,u,.02,.001,5,55,80)
        with mp.workdps(85):
            error=max(abs(a-b) for x,y in zip(coarse,fine) for a,b in zip(x,y))
            self.assertLess(error,mp.mpf('1e-40'))
            self.assertNotEqual(mp.mpf(model.beta),mp.mpf(8)/3)

    def test_mp_row_recurrence_matches_matrix_exponential(self):
        model=Lorenz();c=Carleman(model,3)
        rows,exponents,K,tail=mp_increment_rows(model,3,.001,digits=60,tolerance='1e-50')
        actual=np.array([[float(v) for v in row] for row in rows])
        expected=(expm(.001*c.C)-np.eye(len(c.C)))[c.linear]
        np.testing.assert_allclose(actual,expected,atol=3e-15,rtol=1e-12)
        self.assertEqual(exponents,c.exponents)
        self.assertLess(float(tail),1e-50)
        self.assertGreater(K,0)

    def test_first_exit_is_not_endpoint_error(self):
        times=np.arange(4)*.1
        self.assertAlmostEqual(first_exit(times,[0.,.01,.2,.01],.1),.2)
        self.assertIsNone(first_exit(times,[0.,.01,.02,.01],.1))


if __name__=='__main__':
    unittest.main()
