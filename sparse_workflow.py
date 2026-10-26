"""Native sparse product-LCU and row-overlap readout circuits.

The row-overlap alternative pays for classical coefficient precomputation and
O(D_N)-size state preparation. Its smaller normalization is not a free change
to the normalization of the sparse product-LCU circuit.
"""
from math import ceil,log2
import numpy as np
from qiskit import QuantumCircuit,QuantumRegister
from carleman import basis
from sparse_compiler import (compile_block_encoding,prepare_real,mcx,
                             less_than_constant)


def series_weights(alpha,h,tolerance=1e-13,K=None):
    a=alpha*h;weights=[1.]
    for k in range(1,300):
        weights.append(weights[-1]*a/k)
        ratio=a/(k+2)
        tail=weights[-1]*a/(k+1)/(1-ratio) if ratio<1 else np.inf
        if (K is not None and k==K) or (K is None and tail<=tolerance):
            return np.array(weights),float(tail)
    raise ValueError('Taylor degree too large')


class PolynomialLift:
    """Monomial layout without constructing a dense Carleman matrix."""
    def __init__(self,model,N):
        self.model=model;self.order=N;self.exponents=basis(N)
        self.index={a:i for i,a in enumerate(self.exponents)}
        self.linear=np.array([self.index[(1,0,0)],self.index[(0,1,0)],self.index[(0,0,1)]])

    def lift(self,v):
        return np.prod(np.asarray(v)[None,:]**np.array(self.exponents),axis=1)

    def sparse_generator(self):
        from scipy.sparse import coo_matrix
        r=[];c=[];values=[]
        for i,a in enumerate(self.exponents):
            for j,poly in enumerate(self.model.polynomials()):
                if not a[j]:continue
                reduced=list(a);reduced[j]-=1
                for power,coefficient in poly.items():
                    if not coefficient:continue
                    target=tuple(reduced[k]+power[k] for k in range(3))
                    if target in self.index:
                        r.append(i);c.append(self.index[target]);values.append(a[j]*coefficient)
        result=coo_matrix((values,(r,c)),shape=(len(self.exponents),)*2).tocsr()
        result.eliminate_zeros()
        return result


def row_series(model,N,h,K):
    """All P1 C^l/l! rows via sparse adjoint recurrence, O(K*nnz(C))."""
    c=PolynomialLift(model,N);c.C=c.sparse_generator();C=c.C
    term=np.zeros((3,len(c.exponents)));term[np.arange(3),c.linear]=1.
    rows=[term.copy()]
    for k in range(1,K+1):
        term=(C.T@term.T).T*(h/k)
        rows.append(term.copy())
    return c,rows


def build_sparse_lcu(model,N,h,predictor_order=0,K=None):
    """Unitary block encoding of sum_{l=p+1}^K (h C)^l/l!.

    Fresh type/threshold/failure registers per factor; clean arithmetic scratch
    is reused. No dense matrix UnitaryGate is used. The lifted input is separate.
    Full statevector simulation of high K is intentionally not attempted.
    """
    encoding=compile_block_encoding(model,N)
    weights,tail=series_weights(encoding.alpha,h,K=K)
    K=len(weights)-1
    if not 0<=predictor_order<K:raise ValueError('Need 0 <= p < K')
    b=encoding.width;tbits=encoding.circuit.qregs[1].size
    obits=max(1,ceil(log2(K+1)))
    data=QuantumRegister(3*b,'exp');order=QuantumRegister(obits,'order')
    enable=QuantumRegister(1,'order_enable')
    control_work=QuantumRegister(max(obits+2,3*b+2),'outer_and')
    rotation=QuantumRegister(1,'outer_rotation')
    # Nonzero failure branches retain type/threshold/failure. Other arithmetic
    # registers are uncomputed by the base circuit and may be reused.
    blocks=[];shared={}
    fresh_names={'type','threshold','failure'}
    registers=[data,order,enable,control_work,rotation]
    for k in range(K):
        current={}
        for reg in encoding.circuit.qregs[1:]:
            if reg.name in fresh_names:
                current[reg.name]=QuantumRegister(reg.size,f'{reg.name}_{k}')
                registers.append(current[reg.name])
            else:
                if reg.name not in shared:
                    shared[reg.name]=QuantumRegister(reg.size,reg.name)
                    registers.append(shared[reg.name])
                current[reg.name]=shared[reg.name]
        blocks.append(current)
    qc=QuantumCircuit(*registers,name='Sparse_residual_LCU')
    amplitudes=np.zeros(2**obits)
    gamma=float(weights[predictor_order+1:].sum())
    amplitudes[predictor_order+1:K+1]=np.sqrt(weights[predictor_order+1:]/gamma)
    prep=QuantumCircuit(*registers)
    prepare_real(prep,list(order),amplitudes,list(control_work),rotation[0])
    qc.compose(prep,inplace=True)
    controlled=encoding.circuit.to_gate().control(1)
    for k in range(K):
        less_than_constant(qc,list(order),k+1,enable[0],list(control_work))
        qc.x(enable[0])
        mapping=list(data)
        for reg in encoding.circuit.qregs[1:]:mapping.extend(blocks[k][reg.name])
        qc.append(controlled,[enable[0]]+mapping)
        qc.x(enable[0])
        less_than_constant(qc,list(order),k+1,enable[0],list(control_work))
    qc.compose(prep.inverse(),inplace=True)
    return qc,dict(alpha=encoding.alpha,gamma=gamma,K=K,predictor_order=predictor_order,
                   taylor_tail_bound=tail,base_calls_per_application=K,
                   data_qubits=list(range(3*b)),width=b)


def grid_vector(c,coefficients):
    b=ceil(log2(c.order+2));size=2**(3*b)
    vector=np.zeros(size)
    for exponent,value in zip(c.exponents,coefficients):
        index=sum(int(a)<<(j*b) for j,a in enumerate(exponent))
        vector[index]=value
    return vector,b


def build_sparse_hadamard(model,N,h,u,coordinate,predictor_order=0,K=None):
    """Full generated sparse LCU, input preparation and coordinate interference."""
    block,meta=build_sparse_lcu(model,N,h,predictor_order,K)
    c=PolynomialLift(model,N);phi=c.lift(model.normalize(u));B=np.linalg.norm(phi)
    vector,b=grid_vector(c,phi)
    V=QuantumCircuit(*block.qregs)
    registers={r.name:r for r in block.qregs}
    prepare_real(V,list(registers['exp']),vector,list(registers['outer_and']),
                 registers['outer_rotation'][0])
    V.compose(block,inplace=True)
    V.x(registers['exp'][coordinate*b])
    test=QuantumRegister(1,'test')
    circuit=QuantumCircuit(test,*block.qregs,name='Sparse_residual_Hadamard')
    circuit.h(test[0]);circuit.append(V.to_gate().control(1),[test[0]]+list(V.qubits));circuit.h(test[0])
    meta.update(B=float(B),F=float(model.scale*B*meta['gamma']))
    return circuit,meta


def build_row_hadamard(model,N,h,u,coordinate,predictor_order=0,K=12):
    """Explicit coefficient-state implementation of a smaller row norm.

    <row/||row|| | phi/B> = residual_coordinate/(B ||row||).
    Classical rows and input coefficients are supplied explicitly; this costs
    O(K*nnz(C)) preprocessing and O(D_N)-scale state preparation, not polylog D.
    """
    c,series=row_series(model,N,h,K)
    row=sum(series[predictor_order+1:])[coordinate]
    phi=c.lift(model.normalize(u));B=float(np.linalg.norm(phi))
    row_norm=float(np.linalg.norm(row))
    if row_norm==0:return None,dict(classical_zero_residual=True,F=0.)
    phi_grid,b=grid_vector(c,phi);row_grid,_=grid_vector(c,row)
    data=QuantumRegister(3*b,'data');work=QuantumRegister(3*b+2,'and')
    enable=QuantumRegister(1,'enable')
    phi_circuit=QuantumCircuit(data,work,enable)
    row_circuit=QuantumCircuit(data,work,enable)
    prepare_real(phi_circuit,list(data),phi_grid,list(work),enable[0])
    prepare_real(row_circuit,list(data),row_grid,list(work),enable[0])
    V=phi_circuit.compose(row_circuit.inverse())
    test=QuantumRegister(1,'test')
    qc=QuantumCircuit(test,data,work,enable,name='Residual_row_Hadamard')
    qc.h(test[0]);qc.append(V.to_gate().control(1),[test[0]]+list(V.qubits));qc.h(test[0])
    return qc,dict(B=B,row_norm=row_norm,F=model.scale*B*row_norm,
                   classical_residual=float(model.scale*(row@phi)),
                   expected_test_probability=float((1+(row@phi)/(B*row_norm))/2),
                   classical_precompute_terms=K*c.C.nnz,
                   row_support=int(np.count_nonzero(row)),
                   lift_support=int(np.count_nonzero(phi)),K=K,predictor_order=predictor_order)
