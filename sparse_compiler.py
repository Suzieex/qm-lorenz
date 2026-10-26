"""Arithmetic Lorenz-Carleman block encoding, without dense UnitaryGate/SVD.

Exponent-grid data |a,b,c>, total degree <= N. All other grid entries are zero
in the encoded block. Derivatives use coherent threshold counting, not table
lookups or data-dependent arcsin arithmetic. Exact real gate angles are logical
idealizations; elementary gate synthesis and coefficient precision remain costs.
"""
from dataclasses import dataclass
from math import ceil,log2,asin,sqrt
import numpy as np
from qiskit import QuantumCircuit,QuantumRegister,transpile


@dataclass(frozen=True)
class Rule:
    shift: tuple
    axis: int
    coefficient: float
    cap: int
    kind: str='derivative'
    diagonal: tuple=()

    @property
    def weight(self):
        return self.cap*(max(abs(x) for x in self.diagonal)
                         if self.kind=='diagonal' else abs(self.coefficient))


def rules_from_model(model,N,group_diagonal=True,tight_capacity=True):
    rules=[];diagonal=np.zeros(3)
    for axis,polynomial in enumerate(model.polynomials()):
        for power,coefficient in polynomial.items():
            if coefficient==0:continue
            shift=list(power);shift[axis]-=1
            if sum(abs(x) for x in shift)==0 and group_diagonal:
                diagonal[axis]+=coefficient
                continue
            cap=N-max(sum(shift),0) if tight_capacity else N
            if cap>0:rules.append(Rule(tuple(shift),axis,float(coefficient),cap))
    nonzero=diagonal[diagonal!=0]
    if nonzero.size:
        if np.all(nonzero<0) or np.all(nonzero>0):
            rules.append(Rule((0,0,0),-1,float(np.sign(nonzero[0])),N,
                              'diagonal',tuple(diagonal)))
        else:
            for j,c in enumerate(diagonal):
                if c:rules.append(Rule((0,0,0),j,float(c),N))
    return rules


def mcx(qc,controls,target,work):
    controls=list(controls);n=len(controls)
    if n==0:qc.x(target)
    elif n==1:qc.cx(controls[0],target)
    elif n==2:qc.ccx(*controls,target)
    else:
        if len(work)<n-2:raise ValueError('Insufficient clean AND workspace')
        qc.ccx(controls[0],controls[1],work[0])
        for j in range(2,n-1):qc.ccx(work[j-2],controls[j],work[j-1])
        qc.ccx(work[n-3],controls[-1],target)
        for j in range(n-2,1,-1):qc.ccx(work[j-2],controls[j],work[j-1])
        qc.ccx(controls[0],controls[1],work[0])


def conditional(qc,controls,values,operation,work,enable):
    negative=[q for q,v in zip(controls,values) if not v]
    for q in negative:qc.x(q)
    if controls:
        mcx(qc,controls,enable,work)
        operation(enable)
        mcx(qc,controls,enable,work)
    else:operation(None)
    for q in reversed(negative):qc.x(q)


def prepare_real(qc,qubits,amplitudes,work,enable,extra_controls=(),extra_values=()):
    """Binary-tree state preparation: rotations/Boolean controls, no unitary table."""
    amplitudes=np.asarray(amplitudes,dtype=float)
    if len(amplitudes)!=2**len(qubits):raise ValueError('State size mismatch')
    norm=np.linalg.norm(amplitudes)
    if norm==0:raise ValueError('Cannot prepare a zero state')
    amplitudes=amplitudes/norm
    def recurse(bits,lo,hi,controls,values):
        if not bits:return
        middle=(lo+hi)//2
        left=float(np.sum(amplitudes[lo:middle]**2))
        right=float(np.sum(amplitudes[middle:hi]**2))
        if left+right==0:return
        angle=2*asin(sqrt(min(1.,right/(left+right))))
        target=bits[-1]
        if angle:
            conditional(qc,controls,values,
                lambda en:qc.ry(angle,target) if en is None else qc.cry(angle,en,target),work,enable)
        if left:recurse(bits[:-1],lo,middle,controls+[target],values+[0])
        if right:recurse(bits[:-1],middle,hi,controls+[target],values+[1])
    recurse(list(qubits),0,len(amplitudes),list(extra_controls),list(extra_values))
    for index,amplitude in enumerate(amplitudes):
        if amplitude<0:
            values=list(extra_values)+[(index>>j)&1 for j in range(len(qubits))]
            controls=list(extra_controls)+list(qubits)
            conditional(qc,controls,values,
                lambda en:qc.z(en) if en is not None else setattr(qc,'global_phase',qc.global_phase+np.pi),work,enable)


def prepare_uniform(qc,qubits,M,work,enable,controls=(),values=()):
    """Uniform [0,M): O(b^2) controlled gates, not an O(2^b) state table."""
    def rec(bits,count,cs,vs):
        if not bits:return
        size=2**len(bits)
        if count==size:
            for target in bits:
                conditional(qc,cs,vs,lambda en,t=target:qc.h(t) if en is None else qc.ch(en,t),work,enable)
            return
        half=size//2;left=min(count,half);right=max(0,count-half);target=bits[-1]
        if right:
            angle=2*asin(sqrt(right/count))
            conditional(qc,cs,vs,lambda en:qc.ry(angle,target) if en is None else qc.cry(angle,en,target),work,enable)
        if left:rec(bits[:-1],left,cs+[target],vs+[0])
        if right:rec(bits[:-1],right,cs+[target],vs+[1])
    if not 1<=M<=2**len(qubits):raise ValueError('Invalid interval')
    rec(list(qubits),M,list(controls),list(values))


def add_one(qc,bits,control,work,inverse=False):
    """Controlled +/-1 modulo 2^len(bits), Boolean reversible implementation."""
    if not inverse:
        for j in range(len(bits)-1,0,-1):mcx(qc,[control]+list(bits[:j]),bits[j],work)
        qc.cx(control,bits[0])
    else:
        qc.cx(control,bits[0])
        for j in range(1,len(bits)):mcx(qc,[control]+list(bits[:j]),bits[j],work)


def add_word(qc,word,total,work,inverse=False):
    indices=list(range(len(word)))
    if inverse:indices.reverse()
    for j in indices:add_one(qc,total[j:],word[j],work,inverse)


def less_than_constant(qc,bits,bound,target,work):
    if bound>=2**len(bits):qc.x(target);return
    if bound<=0:return
    for j in range(len(bits)-1,-1,-1):
        if not (bound>>j)&1:continue
        controls=list(bits[j:]);values=[0]+[(bound>>k)&1 for k in range(j+1,len(bits))]
        negative=[q for q,v in zip(controls,values) if not v]
        for q in negative:qc.x(q)
        mcx(qc,controls,target,work)
        for q in reversed(negative):qc.x(q)


def greater_than_word(qc,a,b,target,equality,work):
    for x,y,e in zip(a,b,equality):qc.x(e);qc.cx(x,e);qc.cx(y,e)
    for j in range(len(a)-1,-1,-1):
        qc.x(b[j]);mcx(qc,list(equality[j+1:])+[a[j],b[j]],target,work);qc.x(b[j])
    for x,y,e in reversed(list(zip(a,b,equality))):qc.cx(y,e);qc.cx(x,e);qc.x(e)


@dataclass
class CompiledEncoding:
    circuit: object
    alpha: float
    N: int
    width: int
    data_indices: tuple
    rules: tuple

    def data_basis(self,exponent):
        return sum(int(a)<<self.data_indices[j][0] for j,a in enumerate(exponent))


def compile_block_encoding(model,N,group_diagonal=True,tight_capacity=True):
    rules=rules_from_model(model,N,group_diagonal,tight_capacity)
    alpha=sum(rule.weight for rule in rules)
    b=ceil(log2(N+2));tbits=max(1,ceil(log2(len(rules))));w=b+2
    data=QuantumRegister(3*b,'exp');types=QuantumRegister(tbits,'type')
    threshold=QuantumRegister(b,'threshold');total=QuantumRegister(w,'sum')
    zeros=QuantumRegister(2,'pad');eq=QuantumRegister(w,'eq')
    comparisons=QuantumRegister(3,'cmp');valid=QuantumRegister(1,'valid')
    flag=QuantumRegister(1,'failure');select=QuantumRegister(1,'select')
    rotation=QuantumRegister(1,'rotation');work=QuantumRegister(b+tbits+7,'and')
    qc=QuantumCircuit(data,types,threshold,total,zeros,eq,comparisons,valid,flag,select,rotation,work,name='Lorenz_sparse_C')
    axes=[list(data[j*b:(j+1)*b]) for j in range(3)]
    probabilities=np.zeros(2**tbits)
    probabilities[:len(rules)]=[r.weight/alpha for r in rules]
    prep=QuantumCircuit(*qc.qregs)
    prepare_real(prep,list(types),np.sqrt(probabilities),list(work),rotation[0])
    for index,rule in enumerate(rules):
        prepare_uniform(prep,list(threshold),rule.cap,list(work),rotation[0],list(types),[(index>>j)&1 for j in range(tbits)])
    qc.compose(prep,inplace=True)
    for index,rule in enumerate(rules):
        pattern=[(index>>j)&1 for j in range(tbits)]
        negative=[q for q,v in zip(types,pattern) if not v]
        for q in negative:qc.x(q)
        mcx(qc,list(types),select[0],list(work))
        for q in reversed(negative):qc.x(q)
        for j,delta in enumerate(rule.shift):
            for _ in range(abs(delta)):add_one(qc,axes[j],select[0],list(work),inverse=delta>0)
        if rule.kind=='derivative':
            for axis in axes:add_word(qc,axis,list(total),list(work))
            cutoff=N-max(sum(rule.shift),0)
            less_than_constant(qc,list(total),cutoff+1,valid[0],list(work))
            greater_than_word(qc,axes[rule.axis],list(threshold),comparisons[0],list(eq[:b]),list(work))
            qc.cx(select[0],flag[0])
            mcx(qc,[select[0],valid[0],comparisons[0]],flag[0],list(work))
            greater_than_word(qc,axes[rule.axis],list(threshold),comparisons[0],list(eq[:b]),list(work))
            less_than_constant(qc,list(total),cutoff+1,valid[0],list(work))
            for axis in reversed(axes):add_word(qc,axis,list(total),list(work),inverse=True)
        else:
            padded_threshold=list(threshold)+list(zeros)
            for j,axis in enumerate(axes):
                add_word(qc,axis,list(total),list(work))
                greater_than_word(qc,list(total),padded_threshold,comparisons[j],list(eq),list(work))
            less_than_constant(qc,list(total),N+1,valid[0],list(work))
            qc.cx(select[0],flag[0])
            maximum=max(abs(c) for c in rule.diagonal)
            for j,coefficient in enumerate(rule.diagonal):
                if not coefficient:continue
                controls=[select[0],valid[0],comparisons[j]]
                values=[1,1,1]
                if j:controls.append(comparisons[j-1]);values.append(0)
                angle=-2*asin(abs(coefficient)/maximum)
                conditional(qc,controls,values,
                    lambda en:qc.cry(angle,en,flag[0]),list(work),rotation[0])
            less_than_constant(qc,list(total),N+1,valid[0],list(work))
            for j in range(2,-1,-1):
                greater_than_word(qc,list(total),padded_threshold,comparisons[j],list(eq),list(work))
                add_word(qc,axes[j],list(total),list(work),inverse=True)
        if rule.coefficient<0:qc.z(select[0])
        for q in negative:qc.x(q)
        mcx(qc,list(types),select[0],list(work))
        for q in reversed(negative):qc.x(q)
    qc.compose(prep.inverse(),inplace=True)
    data_indices=tuple(tuple(qc.find_bit(q).index for q in axis) for axis in axes)
    return CompiledEncoding(qc,float(alpha),N,b,data_indices,tuple(rules))


def simulate_sparse(circuit,initial=0,prune=1e-15):
    """Exact logical-gate sparse-state simulator for these arithmetic circuits.

    Does not allocate 2^qubits. Pruning is only a numerical verification option;
    it is not a quantum algorithm or a hardware cost reduction.
    """
    from qiskit.circuit import ControlledGate
    state=dict(initial) if isinstance(initial,dict) else {int(initial):1.+0j}
    def apply(op,bits,guards=()):
        nonlocal state
        name=op.name
        native={'x','cx','ccx','z','h','ry','ch','cry'}
        if name not in native:
            if isinstance(op,ControlledGate):
                n=op.num_ctrl_qubits
                extra=tuple((bits[j],(op.ctrl_state>>j)&1) for j in range(n))
                apply(op.base_gate,bits[n:],guards+extra)
            elif op.definition is not None:
                definition=op.definition
                for instruction in definition.data:
                    locations=[bits[definition.find_bit(q).index] for q in instruction.qubits]
                    apply(instruction.operation,locations,guards)
                phase=float(definition.global_phase)
                if phase:
                    state={k:v*(np.exp(1j*phase) if all(((k>>q)&1)==value for q,value in guards) else 1) for k,v in state.items()}
            else:raise ValueError(f'Unsupported verification gate {name}')
            return
        if name in ['x','cx','ccx','z']:
            new={}
            for key,amplitude in state.items():
                if not all(((key>>q)&1)==value for q,value in guards):
                    new[key]=new.get(key,0)+amplitude;continue
                if name=='z':target=key;amplitude*=(-1 if (key>>bits[0])&1 else 1)
                else:
                    controls=bits[:-1]
                    target=key^(1<<bits[-1]) if all((key>>q)&1 for q in controls) else key
                new[target]=new.get(target,0)+amplitude
            state=new
        elif name in ['h','ry','ch','cry']:
            controlled=name.startswith('c');target=bits[-1]
            if name in ['h','ch']:matrix=np.array([[1,1],[1,-1]])/sqrt(2)
            else:
                angle=float(op.params[0]);matrix=np.array([[np.cos(angle/2),-np.sin(angle/2)],[np.sin(angle/2),np.cos(angle/2)]])
            new={}
            for key,amplitude in state.items():
                if not all(((key>>q)&1)==value for q,value in guards) or (controlled and not (key>>bits[0])&1):
                    new[key]=new.get(key,0)+amplitude;continue
                old=(key>>target)&1
                for value in [0,1]:
                    result=(key&~(1<<target))|(value<<target)
                    new[result]=new.get(result,0)+amplitude*matrix[value,old]
            state={k:v for k,v in new.items() if abs(v)>prune}
    for instruction in circuit.data:
        bits=[circuit.find_bit(q).index for q in instruction.qubits]
        apply(instruction.operation,bits)
    if float(circuit.global_phase):state={k:v*np.exp(1j*float(circuit.global_phase)) for k,v in state.items()}
    return state


def encoded_column(encoding,exponent):
    state=simulate_sparse(encoding.circuit,encoding.data_basis(exponent))
    mask=sum(1<<q for axis in encoding.data_indices for q in axis)
    output={}
    for key,amplitude in state.items():
        if key&~mask:continue
        indices=tuple((key>>axis[0])&((1<<encoding.width)-1) for axis in encoding.data_indices)
        output[indices]=amplitude
    return output


def gate_resources(encoding):
    native=dict(encoding.circuit.count_ops())
    compiled=transpile(encoding.circuit,basis_gates=['u','cx'],optimization_level=1)
    return dict(qubits=compiled.num_qubits,logical_operations=native,
                elementary_operations=dict(compiled.count_ops()),depth=compiled.depth())
