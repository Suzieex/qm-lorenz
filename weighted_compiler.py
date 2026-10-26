"""Implemented layer-weight similarity using two degree-diagonal filters.

Normalization is alpha_base * max(w)/min(w), NOT a numerical matrix norm.
Both filters have fresh failure flags. This conservative implementation may
lose more in alpha than it gains in lifted-input norm.
"""
from math import acos
from qiskit import QuantumCircuit,QuantumRegister
from sparse_compiler import (compile_block_encoding,CompiledEncoding,add_word,
                              conditional,prepare_real)


def compile_weighted(model,N,weights):
    if len(weights)!=N+1 or min(weights)<=0:raise ValueError('Positive layer weights required')
    base=compile_block_encoding(model,N);b=base.width
    flags=QuantumRegister(2,'weight_failure')
    qc=QuantumCircuit(*base.circuit.qregs,flags,name='Weighted_Carleman')
    registers={r.name:r for r in base.circuit.qregs}
    data=registers['exp'];total=registers['sum'];work=registers['and']
    enable=registers['rotation'][0]
    def diagonal(factors,flag):
        for j in range(3):add_word(qc,list(data[j*b:(j+1)*b]),list(total),list(work))
        # Invalid exponents need no special treatment: C's encoded block is
        # zero there and diagonal filters preserve the exponent register.
        for degree,factor in enumerate(factors):
            angle=2*acos(min(1.,max(0.,float(factor))))
            if angle:
                conditional(qc,list(total),[(degree>>j)&1 for j in range(len(total))],
                    lambda en:qc.cry(angle,en,flag),list(work),enable)
        for j in reversed(range(3)):
            add_word(qc,list(data[j*b:(j+1)*b]),list(total),list(work),inverse=True)
    lo=min(weights);hi=max(weights)
    diagonal([lo/w for w in weights],flags[0])
    qc.compose(base.circuit,inplace=True)
    diagonal([w/hi for w in weights],flags[1])
    return CompiledEncoding(qc,base.alpha*hi/lo,N,b,base.data_indices,base.rules)


def coefficient_row_hadamard(phi,row):
    """Explicit compact-basis preparation for any real lift and target row.

    Classical coefficient computation is required; no efficient oracle assumed.
    """
    import numpy as np
    from math import ceil,log2
    bits=max(1,ceil(log2(len(phi))));size=2**bits
    vectors=[]
    for source in [phi,row]:
        target=np.zeros(size);target[:len(source)]=list(map(float,source));vectors.append(target)
    data=QuantumRegister(bits,'basis');work=QuantumRegister(bits+2,'prep_and')
    enable=QuantumRegister(1,'prep_enable');test=QuantumRegister(1,'test')
    input_prep=QuantumCircuit(data,work,enable);row_prep=input_prep.copy()
    prepare_real(input_prep,list(data),vectors[0],list(work),enable[0])
    prepare_real(row_prep,list(data),vectors[1],list(work),enable[0])
    V=input_prep.compose(row_prep.inverse())
    qc=QuantumCircuit(test,data,work,enable)
    qc.h(test[0]);qc.append(V.to_gate().control(1),[test[0]]+list(V.qubits));qc.h(test[0])
    return qc
