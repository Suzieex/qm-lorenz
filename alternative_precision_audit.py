"""Audit precision loopholes without changing the Lorenz endpoint task.

The Taylor scans are observed multiprecision errors, NOT interval certificates.
The native circuit is an arithmetic component, NOT a complete quantum ODE solver.
Digital basis encoding removes an amplitude-estimation readout requirement but
does not by itself accelerate the arithmetic being made reversible.
"""
from pathlib import Path
import argparse
import csv
import json
import math
import time
import warnings

import mpmath as mp
from global_correction import PARAMETERS
from long_validation import taylor_coefficients


SOURCES = [
    dict(method='Chebyshev quantum spectral ODE',
         source='https://arxiv.org/abs/1901.00961',
         scope='linear time-dependent ODE; prepares a state proportional to the solution',
         unresolved_here='must first supply a valid nonlinear-to-linear reduction and charge classical coordinate extraction'),
    dict(method='quantum Newton', source='https://arxiv.org/abs/2109.08470',
         scope='QLSA plus l_infinity tomography between iterations; paper states log^4(N)/epsilon_s^2 per iteration',
         unresolved_here='the three physical coordinates are not the full time-coefficient vector needed to rebuild a generic Newton oracle'),
    dict(method='quantum multigrid',source='https://arxiv.org/abs/2404.07466',
         scope='finite-element linear systems; output encodes a sequence of iterates in a quantum state',
         unresolved_here='a Lorenz causal time-system needs its own contraction, transfer operators, implementation and readout analysis'),
    dict(method='compact iterative quantum PDE algorithms',
         source='https://doi.org/10.1103/PhysRevResearch.7.023262',
         scope='LCU matrix-vector/inversion iterations for discretized fluid equations',
         unresolved_here='the PDE grid-size advantage and best-case access model do not establish advantage for fixed three-dimensional Lorenz')]


def integrate(initial, degree, steps, digits=100):
    """Fixed-mesh high-order direct Taylor; report all scalar work explicitly."""
    with mp.workdps(digits):
        parameters=[mp.mpf(a.numerator)/a.denominator for a in PARAMETERS]
        dt=mp.mpf(5)/steps
        u=list(map(mp.mpf,initial));start=time.perf_counter()
        for _ in range(steps):
            u=[mp.polyval(row,dt) for row in taylor_coefficients(u,degree,parameters)]
        return tuple(u),dict(seconds=time.perf_counter()-start,
            quadratic_convolution_products=steps*degree*(degree+1),
            other_scalar_products=steps*6*degree,
            divisions_by_small_integer=steps*3*degree)


def arithmetic_components():
    """Compile actual unsigned multiplication on computational-basis inputs."""
    from qiskit import QuantumCircuit,transpile
    from qiskit.circuit.library import RGQFTMultiplier
    from qiskit.quantum_info import Statevector
    rows=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',DeprecationWarning)
        for bits in [2,3,4,8,16]:
            multiplier=RGQFTMultiplier(bits)
            compiled=transpile(multiplier,basis_gates=['u','cx'],optimization_level=1)
            row=dict(bits_per_operand=bits,output_bits=2*bits,qubits=multiplier.num_qubits,
                     u_gates=int(compiled.count_ops().get('u',0)),
                     cx_gates=int(compiled.count_ops().get('cx',0)),
                     classical_schoolbook_partial_AND_terms=bits**2,
                     statevector_checked=bits<=4,
                     status='native arithmetic component only; elementary quantum and classical gates are not equal-time units',
                     synthesis_scope='ideal continuous u rotations; fault-tolerant synthesis and the complete solver are not counted')
            if bits<=4:
                a=(1<<bits)-3;b=(1<<bits)-1
                circuit=QuantumCircuit(multiplier.num_qubits)
                for j in range(bits):
                    if (a>>j)&1:circuit.x(j)
                    if (b>>j)&1:circuit.x(bits+j)
                circuit.compose(multiplier,inplace=True)
                state=Statevector.from_instruction(circuit)
                index=a+(b<<bits)+((a*b)<<(2*bits))
                probability=float(abs(state.data[index])**2)
                assert abs(probability-1)<1e-11
                row.update(a=a,b=b,expected_product=a*b,
                           correct_computational_basis_probability=probability)
            rows.append(row)
    return rows


def run(out=Path('results/fullstage_qlsa')):
    out.mkdir(parents=True,exist_ok=True);rows=[];references={}
    for name,initial in [('standard',(1,1,1)),('attractor',(-8,8,27))]:
        reference,work=integrate(initial,80,250,100)
        cross,crosswork=integrate(initial,88,200,110)
        with mp.workdps(110):
            gap=mp.sqrt(sum((a-b)**2 for a,b in zip(reference,cross)))
            assert gap<mp.mpf('1e-60')
        references[name]=dict(primary_degree=80,primary_steps=250,primary_digits=100,
                              cross_degree=88,cross_steps=200,cross_digits=110,
                              observed_reference_difference=mp.nstr(gap,20),
                              status='independent step/order/precision comparison, NOT a rigorous reference enclosure')
        for degree in [8,12,16,24,32,48,64,96]:
            answer,work=integrate(initial,degree,100,100)
            with mp.workdps(110):
                error=mp.sqrt(sum((a-b)**2 for a,b in zip(answer,reference)))
            rows.append(dict(initial=name,T=5,step=.05,steps=100,degree=degree,
                arithmetic_digits=100,observed_endpoint_error=mp.nstr(error,24),
                observed_achieved_bits=float(-mp.log(error,2)),
                target_1e8_observed_pass=bool(error<mp.mpf('1e-8')),
                target_1e20_observed_pass=bool(error<mp.mpf('1e-20')),
                target_1e40_observed_pass=bool(error<mp.mpf('1e-40')),
                **work,status='observed multiprecision error; no interval certification'))
        print('alternative precision scan:',name,flush=True)
    with (out/'alternative_precision.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader();writer.writerows(rows)
    components=arithmetic_components()
    (out/'alternative_arithmetic.json').write_text(json.dumps(components,indent=2))
    result=dict(task='Lorenz63, known classical initial data, T=5, precise classical endpoint',
        exact_parameters=list(map(str,PARAMETERS)),sources=SOURCES,references=references,
        precision_scan_cases=len(rows),
        digital_encoding=dict(mapping='|u>|0> -> |u>|F_b(u)>; compute-copy-uncompute can clean work registers',
            readout='basis-encoded deterministic bits can be measured once in the ideal circuit',
            consequence='there is no universal 1/epsilon readout lower bound for the original known-input classical problem',
            cost='a reversible implementation of a classical Taylor circuit inherits that circuit arithmetic and uncomputation work',
            caveat='compiled multipliers are arithmetic components; the complete reversible Lorenz integrator is not implemented'),
        iterative_refinement=dict(
            conditional_recurrence='e_(k+1) <= theta*e_k + C*e_k^2 with fixed relative solve error theta<1',
            iterations='after entering the local contraction region, O(log(1/epsilon)) iterations are possible',
            caveat='not a Lorenz algorithm or certification by itself; rebuilding an entire time-dependent guide generally requires time coefficients, not only three endpoint numbers',
            classical_comparison='classical iterative refinement or direct high-order integration must receive the same warm start and preprocessing'),
        same_task_advantage_established=False,
        conclusion='digital encoding and iterative refinement are valid precision loopholes, but no lower-work Lorenz quantum algorithm was obtained; state-output spectral/Newton/multigrid claims do not supply one automatically',
        validation=f'{len(rows)} observed Taylor cases, two order/step/precision reference cross-checks, 3 native multiplier statevector checks, 5 compiled multiplier resource records')
    (out/'alternative_summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(cases=len(rows),same_task_advantage_established=False,
                          arithmetic_components=len(components)),indent=2))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,default=Path('results/fullstage_qlsa'))
    run(parser.parse_args().out)
