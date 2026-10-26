"""Explicit diagonal/shift LCU for the directly assembled stage system.

The reference compiler uses a charged sparse classical lookup, not QRAM or
an efficient arithmetic oracle. Its normalization follows its decomposition,
not an SVD. All circuit angles are ideal real rotations represented in float.
"""
from dataclasses import dataclass
from fractions import Fraction as Q
from math import ceil, log2, acos
from pathlib import Path
import json
import numpy as np
from scipy.sparse import csr_matrix


@dataclass
class ShiftDecomposition:
    dimension: int
    padded_dimension: int
    diagonals: dict
    weights: dict
    alpha_exact: Q

    @property
    def alpha(self): return float(self.alpha_exact)


def diagonal_lcu(matrix, pad_identity=True):
    """A=sum_d D_d S_d, (S_d)_{i,j}=1 iff i=j+d mod P.

    Padded identity makes the augmented linear solve nonsingular. Every sparse
    float input entry is interpreted as an exact rational in alpha accounting.
    """
    a=csr_matrix(matrix,dtype=float).copy();a.sum_duplicates();a.eliminate_zeros()
    if a.shape[0]!=a.shape[1]:raise ValueError('A square matrix is required')
    n=a.shape[0];p=1<<max(0,(n-1).bit_length());diagonals={}
    coo=a.tocoo()
    for i,j,v in zip(coo.row,coo.col,coo.data):
        diagonals.setdefault(int((i-j)%p),{})[int(i)]=float(v)
    if pad_identity:
        for i in range(n,p):diagonals.setdefault(0,{})[i]=1.
    weights={d:max(abs(Q(v)) for v in values.values()) for d,values in diagonals.items()}
    if not weights:raise ValueError('The zero matrix has no normalized LCU')
    return ShiftDecomposition(n,p,diagonals,weights,sum(weights.values(),Q(0)))


def reconstruct(decomposition):
    d=decomposition;a=np.zeros((d.padded_dimension,d.padded_dimension))
    for shift,values in d.diagonals.items():
        for i,value in values.items():a[i,(i-shift)%d.padded_dimension]=value
    return a


def resources(decomposition):
    d=decomposition;n=max(1,ceil(log2(d.padded_dimension)))
    types=max(1,ceil(log2(len(d.weights))));controls=n+types
    entries=sum(map(len,d.diagonals.values()))
    # Each entry lookup uses 2 equality-controlled AND computations plus a CRY.
    # The AND ladder has 2c-3 Toffolis for c>=2, each <=6 CX. CRY <=2 CX.
    lookup_cx_bound=entries*(12*max(0,2*controls-3)+2)
    return dict(dimension=d.dimension,padded_dimension=d.padded_dimension,
        shifts=len(d.weights),sparse_lookup_entries=entries,alpha=d.alpha,
        alpha_exact=str(d.alpha_exact),data_qubits=n,type_qubits=types,
        lookup_CX_upper_bound=lookup_cx_bound,
        classical_lookup_entries_prepared=entries,
        normalization_status='exact sum of diagonal maxima for the displayed shift LCU',
        resource_status='lookup CX upper bound only; PREPARE, shifts and synthesis are additional',
        oracle_status='sparse compiled lookup paid explicitly; no free QRAM or fast arithmetic claimed')


@dataclass
class CompiledShiftEncoding:
    circuit: object
    decomposition: ShiftDecomposition
    data_qubits: tuple


def compile_shift_encoding(matrix):
    """PREPARE-dagger SELECT PREPARE, with native Boolean/rotation gates.

    Default SELECT flips a flag (zero success amplitude); each nonzero lookup
    rotates it to give signed success amplitude a_ij / w_shift.
    """
    from qiskit import QuantumCircuit,QuantumRegister
    from sparse_compiler import prepare_real,conditional,add_one
    d=diagonal_lcu(matrix);info=resources(d)
    n=info['data_qubits'];t=info['type_qubits']
    data=QuantumRegister(n,'row');choice=QuantumRegister(t,'shift')
    flag=QuantumRegister(1,'failure');enable=QuantumRegister(1,'enable')
    work=QuantumRegister(n+t+2,'and')
    qc=QuantumCircuit(data,choice,flag,enable,work,name='stage_shift_LCU')
    amplitudes=np.zeros(1<<t);shifts=sorted(d.weights)
    amplitudes[:len(shifts)]=[np.sqrt(float(d.weights[s]/d.alpha_exact)) for s in shifts]
    prep=QuantumCircuit(*qc.qregs)
    prepare_real(prep,list(choice),amplitudes,list(work),enable[0])
    qc.compose(prep,inplace=True);qc.x(flag[0])
    for index,shift in enumerate(shifts):
        pattern=[(index>>j)&1 for j in range(t)]
        signed=shift if shift<=d.padded_dimension//2 else shift-d.padded_dimension
        # Adding a classical constant uses its binary expansion, not |d| adds.
        for k in range(n):
            if not ((abs(signed)>>k)&1):continue
            conditional(qc,list(choice),pattern,
                lambda en,k=k: add_one(qc,list(data[k:]),en,list(work),inverse=signed<0),
                list(work),enable[0])
        weight=float(d.weights[shift])
        for row,value in sorted(d.diagonals[shift].items()):
            angle=2*acos(max(-1.,min(1.,value/weight)))-np.pi
            if abs(angle)<1e-16:continue
            controls=list(choice)+list(data)
            values=pattern+[(row>>j)&1 for j in range(n)]
            conditional(qc,controls,values,
                lambda en,a=angle:qc.cry(a,en,flag[0]),list(work),enable[0])
    qc.compose(prep.inverse(),inplace=True)
    return CompiledShiftEncoding(qc,d,tuple(qc.find_bit(q).index for q in data))


def encoded_column(encoding,column):
    from sparse_compiler import simulate_sparse
    state=simulate_sparse(encoding.circuit,column,prune=1e-16)
    return np.array([state.get(i,0) for i in range(encoding.decomposition.padded_dimension)])


def native_rhs_and_signed_overlap(encoding,rhs,coordinate):
    """Execute RHS preparation and signed Hadamard readout for B/alpha.

    Replacing this block by an inverse block would require a real QLSA/QSVT;
    this diagnostic does NOT claim that replacement has been implemented.
    """
    from qiskit import QuantumCircuit,transpile
    from sparse_compiler import prepare_real,simulate_sparse
    registers=encoding.circuit.qregs;data,choice,flag,enable,work=registers
    vector=np.zeros(encoding.decomposition.padded_dimension)
    vector[:len(rhs)]=rhs
    prep=QuantumCircuit(*registers)
    prepare_real(prep,list(data),vector,list(work),enable[0])
    rhs_state=simulate_sparse(prep)
    probability=sum(abs(a)**2 for a in rhs_state.values())
    expected=vector/np.linalg.norm(vector)
    prep_error=max(abs(rhs_state.get(i,0)-x) for i,x in enumerate(expected))
    qc=QuantumCircuit(encoding.circuit.num_qubits+1)
    qc.h(0);qc.x(0)
    for j in range(len(data)):
        if (coordinate>>j)&1:qc.cx(0,j+1)
    qc.x(0)
    qc.append(prep.to_gate().control(),list(range(qc.num_qubits)))
    qc.append(encoding.circuit.to_gate().control(),list(range(qc.num_qubits)))
    qc.h(0)
    observed=sum(abs(a)**2 for key,a in simulate_sparse(qc).items() if not (key&1))
    matrix=reconstruct(encoding.decomposition)
    target=(1+float(matrix[coordinate]@expected)/encoding.decomposition.alpha)/2
    compiled=transpile(prep,basis_gates=['u','cx'],optimization_level=1)
    assert abs(probability-1)<1e-12 and prep_error<1e-12 and abs(observed-target)<1e-12
    return dict(rhs_preparation_maximum_amplitude_error=float(prep_error),
        rhs_preparation_gates=dict(compiled.count_ops()),
        signed_test_probability=float(observed),signed_test_expected=float(target),
        signed_test_error=float(abs(observed-target)),coordinate=int(coordinate),
        scope='RHS loading and signed B/alpha overlap executed; inverse block not implemented')


def run(out=Path('results/fullstage_qlsa')):
    from qiskit import transpile
    from fullstage_system import build_fullstage,load_cases
    out.mkdir(parents=True,exist_ok=True);rows=[]
    for certificate,guide,*_ in load_cases():
        name=certificate['initial']
        base=128 if name=='standard' else 256
        for M in [base,2*base]:
            for s in [4,8,12]:
                for formulation in ['state','increment']:
                    system=build_fullstage(guide,M,s,formulation=formulation)
                    info=resources(diagonal_lcu(system.matrix))
                    info.update(initial=name,M=M,stage_count=s,formulation=formulation,
                        matrix_nonzeros=int(system.matrix.nnz))
                    rows.append(info)
    # Small Lorenz-coefficient system executes the SAME generated shift compiler.
    class ToyGuide:
        times=np.array([0.,.005,.01])
        def evaluate(self,t):
            return np.array([[-10.,10.,0.],[27.,-1.,-1.],[1.,1.,-8/3]]),np.array([1.,-2.,.5])
    system=build_fullstage(ToyGuide(),2,1,formulation='increment')
    encoding=compile_shift_encoding(system.matrix)
    desired=reconstruct(encoding.decomposition)/encoding.decomposition.alpha
    errors=[float(np.max(np.abs(encoded_column(encoding,j)-desired[:,j])))
            for j in range(encoding.decomposition.padded_dimension)]
    compiled=transpile(encoding.circuit,basis_gates=['u','cx'],optimization_level=1)
    toy=dict(**resources(encoding.decomposition),qubits=compiled.num_qubits,
        logical_gates=dict(encoding.circuit.count_ops()),elementary_gates=dict(compiled.count_ops()),
        depth=compiled.depth(),all_padded_columns_tested=len(errors),
        maximum_block_entry_error=max(errors),
        caveat='native circuit verification of a small stage matrix; not a QLSA solve or hardware timing')
    toy['rhs_and_signed_readout']=native_rhs_and_signed_overlap(encoding,system.rhs,len(system.rhs)-2)
    assert max(errors)<1e-12
    (out/'encoding_resources.json').write_text(json.dumps(rows,indent=2))
    (out/'encoding_native_check.json').write_text(json.dumps(toy,indent=2))
    print(json.dumps(toy,indent=2));return rows,toy


if __name__=='__main__':run()
