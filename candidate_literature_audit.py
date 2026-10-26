"""Check concrete theorem prerequisites; do not infer a universal no-go."""
import json
from fractions import Fraction as Q
from pathlib import Path
import numpy as np
from carleman import Lorenz


SOURCES=[
 dict(method='quantum residual integration',source='https://arxiv.org/abs/quant-ph/0510045',
      requirement='finite-smoothness Holder black-box class with specified query access',
      Lorenz_status='test against analytic deterministic integration, not Monte Carlo only'),
 dict(method='corrected general Carleman / nonresonance',source='https://quantum-journal.org/papers/q-2026-10-08-2233/',
      requirement='specific stable/Lyapunov or conserved/nonresonant hypotheses; nonresonant branch excludes positive real eigenvalues and needs quantitative gap plus convergence ratio',
      Lorenz_status='origin/equilibria fail spectral premise; energy recentering introduces forcing and needs a separate convergence criterion'),
 dict(method='no dissipative conditions Carleman',source='https://arxiv.org/abs/2405.12714',
      requirement='remaining spectral/nonresonance and convergence conditions; revised analysis must be checked against 2026 corrections',
      Lorenz_status='title does not establish a chaotic Lorenz guarantee'),
 dict(method='Fourier-Koopman',source='https://arxiv.org/abs/2512.06488',
      requirement='specified Fourier nonlinearity G0+G1 exp(iu); restricted regimes and classical readout',
      Lorenz_status='polynomial Lorenz is not this input class; a valid recast and all new modes must be provided'),
 dict(method='coherent Lorenz time marching',source='https://arxiv.org/abs/2506.21354',
      requirement='copy count, recursive gate count and success probability must all be charged',
      Lorenz_status='published gate cost includes exponential dependence on step count; linear copies alone insufficient'),
 dict(method='physics-informed effective Hamiltonian',source='https://arxiv.org/abs/2504.13174',
      requirement='special identical tensor-product state within degenerate ground space, spectral gap and output normalization',
      Lorenz_status='explicit lifted Lorenz counterexample tests rank-one consistency; not a refutation of an algorithm that enforces that constraint'),
 dict(method='bounded quantum advantage for nonlinear fluid dynamics',source='https://doi.org/10.1103/xysy-q3fp',
      requirement='large spatial lattice, selected observables, specified accuracy and full resource model',
      Lorenz_status='different problem scaling; cannot transfer advantage to fixed three-dimensional precise trajectory')]


def audit(out=Path('results/quantum_candidates')):
    out.mkdir(parents=True,exist_ok=True);model=Lorenz();beta=model.beta
    equilibria=[np.zeros(3),np.array([np.sqrt(beta*27),np.sqrt(beta*27),27.]),
                np.array([-np.sqrt(beta*27),-np.sqrt(beta*27),27.])]
    rows=[]
    for state in equilibria:
        x,y,z=state;A=np.array([[-10.,10.,0.],[28-z,-1.,-x],[y,x,-beta]])
        eigen=np.linalg.eigvals(A)
        rows.append(dict(equilibrium=state.tolist(),numerical_eigenvalues=[[float(a.real),float(a.imag)] for a in eigen],
                         maximum_real_part=float(eigen.real.max()),nonpositive_real_spectrum=bool(np.all(eigen.real<=0))))
    forcing=beta*38;decay=beta
    # For the displayed Euclidean coefficient criteria, min_r (a*r+b/r)/decay.
    # This is NOT an optimization over all metrics, centers, or encodings.
    criterion=[dict(tensor_representation=name,F2_norm=norm,
        minimum_R_over_scalar_radius=float(2*np.sqrt(norm*forcing)/decay))
        for name,norm in [('ordered_product',1.),('symmetrized',1/np.sqrt(2))]]
    bq=Q(float(beta));a1=11+bq;a2=38*bq;a3=540*bq
    assert a1*a2-a3<0
    summary=dict(sources=SOURCES,equilibria=rows,
        analytic_origin_positive_eigenvalue='(-11+sqrt(1201))/2 > 0',
        nonzero_equilibrium_Routh_coefficients=list(map(str,(a1,a2,a3))),
        exact_Routh_margin=str(a1*a2-a3),
        exact_energy_R_squared_lower_for_displayed_tensors=str(Q(76)/bq),
        energy_center=[0,0,38],energy_center_forcing_norm=forcing,
        energy_center_euclidean_criteria=criterion,
        limitation='numerical eigenvalues plus analytic origin sign; convergence criteria only for displayed representations, not a theorem against all transformations')
    (out/'literature_eligibility.json').write_text(json.dumps(summary,indent=2));return summary


if __name__=='__main__':print(json.dumps(audit(),indent=2))
