"""Lorenz63, compressed monomial Carleman lifting, and dense reference circuits.

No quantum speedup is claimed: state preparation and SELECT blocks are dense
reference unitaries. QAE uses the exact two-dimensional Grover invariant subspace.
"""
from dataclasses import dataclass
from math import ceil, log2
import numpy as np
from scipy.linalg import expm
from scipy.integrate import solve_ivp


@dataclass(frozen=True)
class Lorenz:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0
    scale: float = 64.0

    @property
    def center(self):
        return np.array([0.0, 0.0, self.rho + self.sigma])

    def rhs(self, t, u):
        x, y, z = u
        return np.array([self.sigma*(y-x), self.rho*x-y-x*z,
                         x*y-self.beta*z])

    def normalize(self, u):
        return (np.asarray(u)-self.center)/self.scale

    def restore(self, v):
        return self.center+self.scale*np.asarray(v)

    def polynomials(self):
        # f(v)=(sigma(eta-xi), -sigma*xi-eta-S*xi*zeta,
        #       S*xi*eta-beta*zeta-beta*(rho+sigma)/S)
        return [
            {(1, 0, 0): -self.sigma, (0, 1, 0): self.sigma},
            {(1, 0, 0): -self.sigma, (0, 1, 0): -1.0,
             (1, 0, 1): -self.scale},
            {(1, 1, 0): self.scale, (0, 0, 1): -self.beta,
             (0, 0, 0): -self.beta*(self.rho+self.sigma)/self.scale}]


def basis(order):
    if order < 1:
        raise ValueError('Carleman order must be at least one')
    return [(a, b, degree-a-b) for degree in range(order+1)
            for a in range(degree, -1, -1)
            for b in range(degree-a, -1, -1)]


class Carleman:
    def __init__(self, model, order):
        self.model, self.order = model, order
        self.exponents = basis(order)
        index = {a: k for k, a in enumerate(self.exponents)}
        self.linear = np.array([index[(1, 0, 0)], index[(0, 1, 0)],
                                index[(0, 0, 1)]])
        self.C = np.zeros((len(index), len(index)))
        for row, a in enumerate(self.exponents):
            for j, polynomial in enumerate(model.polynomials()):
                if a[j] == 0:
                    continue
                reduced = list(a)
                reduced[j] -= 1
                for power, coefficient in polynomial.items():
                    target = tuple(reduced[k]+power[k] for k in range(3))
                    if target in index:
                        self.C[row, index[target]] += a[j]*coefficient

    def lift(self, v):
        return np.prod(np.asarray(v)[None, :] ** np.asarray(self.exponents), axis=1)

    def exact_map(self, u, h):
        v = self.model.normalize(u)
        y = expm(h*self.C) @ self.lift(v)
        return self.model.restore(y[self.linear])


def reference(model, u, times):
    times = np.asarray(times, dtype=float)
    sol = solve_ivp(model.rhs, (0.0, float(times[-1])), np.asarray(u),
                    method='DOP853', t_eval=times, rtol=2.3e-14, atol=2e-14)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def taylor_data(C, h, tol=1e-13, max_order=256):
    """Return E_K and LCU weights with a rigorous scalar tail majorant.

    When next weight ratio is <1, tail <= next_weight/(1-next_ratio).
    alpha=sqrt(||C||_1 ||C||_inf) bounds its spectral norm.
    """
    alpha = float(np.sqrt(np.linalg.norm(C, 1)*np.linalg.norm(C, np.inf)))
    a = h*alpha
    weights = [1.0]
    E = np.eye(len(C))
    term = E.copy()
    for k in range(1, max_order+1):
        weights.append(weights[-1]*a/k)
        term = term @ (h*C)/k
        E += term
        next_weight = weights[-1]*a/(k+1)
        ratio = a/(k+2)
        tail = next_weight/(1-ratio) if ratio < 1 else np.inf
        if tail <= tol:
            return E, np.array(weights), alpha, float(tail)
    raise RuntimeError('Taylor truncation did not converge')


def householder_prepare(target):
    target = np.asarray(target, dtype=float)
    target = target/np.linalg.norm(target)
    e0 = np.zeros_like(target)
    e0[0] = 1.0
    w = e0-target
    if np.linalg.norm(w) < 1e-14:
        return np.eye(len(target))
    w /= np.linalg.norm(w)
    return np.eye(len(target))-2*np.outer(w, w)


def unitary_dilation(A):
    """Julia unitary [[A,sqrt(I-AA*)],[sqrt(I-A*A),-A*]]."""
    left, singular, right_h = np.linalg.svd(A)
    if singular.max() > 1+1e-10:
        raise ValueError('Dilation requires a contraction')
    root = np.sqrt(np.maximum(0, 1-singular**2))
    top_right = (left*root) @ left.T
    bottom_left = (right_h.T*root) @ right_h
    return np.block([[A, top_right], [bottom_left, -A.T]])


class DenseLCUCircuit:
    """Explicit PREP -> SELECT -> PREP^dagger and Hadamard tests.

    SELECT_l is a *dense reference dilation* of (C/alpha)^l, NOT a cheap
    oracle. A scalable implementation must use product block encodings with
    fresh flags (and their extra resources), or another justified compiler.
    """
    def __init__(self, carleman, h, increment=True, tol=1e-13):
        self.system, self.h, self.increment = carleman, h, increment
        self.E, weights, self.alpha, self.tail = taylor_data(carleman.C, h, tol)
        self.K = len(weights)-1
        self.orders = np.arange(1 if increment else 0, self.K+1)
        self.gamma = float(weights[self.orders].sum())
        self.order_size = 2**ceil(log2(max(1, len(self.orders))))
        self.data_size = 2**ceil(log2(len(carleman.C)))
        probabilities = np.zeros(self.order_size)
        probabilities[:len(self.orders)] = weights[self.orders]/self.gamma
        self.prep = householder_prepare(np.sqrt(probabilities))
        padded = np.zeros((self.data_size, self.data_size))
        padded[:len(carleman.C), :len(carleman.C)] = carleman.C/self.alpha
        powers = [np.eye(self.data_size)]
        for _ in range(self.K):
            powers.append(powers[-1] @ padded)
        self.select = [unitary_dilation(powers[k]) for k in self.orders]
        while len(self.select) < self.order_size:
            self.select.append(np.eye(2*self.data_size))

    @property
    def qubits_with_hadamard(self):
        return int(log2(self.order_size))+1+int(log2(self.data_size))+1

    def apply(self, state, inverse=False):
        shape = (self.order_size, 2*self.data_size)
        state = np.asarray(state).reshape(shape)
        state = self.prep @ state
        state = np.stack([(U.T if inverse else U) @ row
                          for U, row in zip(self.select, state)])
        return (self.prep.T @ state).ravel()

    def output(self, u):
        phi = self.system.lift(self.system.model.normalize(u))
        B = np.linalg.norm(phi)
        lifted = np.zeros(self.data_size)
        lifted[:len(phi)] = phi/B
        lift_gate = householder_prepare(lifted)
        state = np.zeros((self.order_size, 2, self.data_size))
        state[0, 0] = lift_gate[:, 0]
        return self.apply(state.ravel()), float(B)

    def hadamard_probabilities(self, u):
        """Build each controlled-branch Hadamard state, not just formula p."""
        output, B = self.output(u)
        probabilities = []
        normalization_error = 0.0
        for j in self.system.linear:
            branch = output.copy()
            # R_j^dagger: XOR the data basis index with j in every ancilla block.
            branch = branch.reshape(-1, self.data_size)[:,
                                    np.arange(self.data_size) ^ int(j)].ravel()
            zero = np.zeros_like(branch)
            zero[0] = 1.0
            final_state = np.concatenate(((zero+branch)/2, (zero-branch)/2))
            normalization_error = max(normalization_error,
                                      abs(np.linalg.norm(final_state)-1))
            probabilities.append(float(np.sum(final_state[:len(branch)]**2)))
        return np.array(probabilities), B, normalization_error

    def recover(self, u, probabilities, B):
        estimates = B*self.gamma*(2*np.asarray(probabilities)-1)
        if self.increment:
            estimates += self.system.model.normalize(u)
        return self.system.model.restore(estimates)


class ProductLCUCircuit(DenseLCUCircuit):
    """LCU with products of a single base block encoding, not dense C^l.

    K fresh flags allow the l-th branch to apply UC l times. In a circuit,
    K calls to UC are controlled by the predicate (l >= t+1). The simulator
    applies these branch by branch. Base UC and lifting remain dense reference
    gates; no sparse compiler or physical elementary gate count is claimed.
    """
    def __init__(self, carleman, h, increment=True, tol=1e-13):
        self.system, self.h, self.increment = carleman, h, increment
        self.E, weights, self.alpha, self.tail = taylor_data(carleman.C, h, tol)
        self.K = len(weights)-1
        self.orders = np.arange(1 if increment else 0, self.K+1)
        self.gamma = float(weights[self.orders].sum())
        self.order_size = 2**ceil(log2(max(1, len(self.orders))))
        self.data_size = 2**ceil(log2(len(carleman.C)))
        self.flags_size = 2**self.K
        if self.order_size*self.flags_size*self.data_size > 2**23:
            raise ValueError('Product statevector too large; use a smaller order/step')
        probabilities = np.zeros(self.order_size)
        probabilities[:len(self.orders)] = weights[self.orders]/self.gamma
        self.prep = householder_prepare(np.sqrt(probabilities))
        padded = np.zeros((self.data_size, self.data_size))
        padded[:len(carleman.C), :len(carleman.C)] = carleman.C/self.alpha
        self.base = unitary_dilation(padded)

    @property
    def qubits_with_hadamard(self):
        return int(log2(self.order_size))+self.K+int(log2(self.data_size))+1

    def apply(self, state, inverse=False):
        state = np.asarray(state).reshape(self.order_size,
                                         self.flags_size*self.data_size)
        state = self.prep @ state
        U = self.base.T if inverse else self.base
        for slot, degree in enumerate(self.orders):
            branch = state[slot].reshape(self.flags_size, self.data_size)
            steps = range(int(degree)-1, -1, -1) if inverse else range(int(degree))
            for flag in steps:
                shape = (2**flag, 2, 2**(self.K-flag-1), self.data_size)
                reordered = branch.reshape(shape).transpose(1, 3, 0, 2)
                transformed = U @ reordered.reshape(2*self.data_size, -1)
                branch = transformed.reshape(2, self.data_size, shape[0], shape[2]).transpose(2, 0, 3, 1).reshape(self.flags_size, self.data_size)
            state[slot] = branch.ravel()
        return (self.prep.T @ state).ravel()

    def output(self, u):
        phi = self.system.lift(self.system.model.normalize(u))
        B = np.linalg.norm(phi)
        lifted = np.zeros(self.data_size)
        lifted[:len(phi)] = phi/B
        lift_gate = householder_prepare(lifted)
        state = np.zeros((self.order_size, self.flags_size, self.data_size))
        state[0, 0] = lift_gate[:, 0]
        return self.apply(state.ravel()), float(B)


def qae_distribution(p, phase_bits):
    """Exact canonical QAE/QPE distribution via the 2D invariant subspace.

    This executes controlled Grover powers and inverse QFT in reduced space;
    it is not a heuristic Gaussian error model or a hardware execution.
    """
    if not 0 <= p <= 1:
        raise ValueError('Probability outside [0,1]')
    size = 2**phase_bits
    theta = np.arcsin(np.sqrt(p))
    k = np.arange(size)
    cos, sin = np.cos(2*k*theta), np.sin(2*k*theta)
    amplitudes = np.stack((cos*np.sqrt(p)+sin*np.sqrt(1-p),
                           -sin*np.sqrt(p)+cos*np.sqrt(1-p)), axis=1)/np.sqrt(size)
    after_iqft = np.fft.fft(amplitudes, axis=0)/np.sqrt(size)
    probabilities = np.sum(np.abs(after_iqft)**2, axis=1)
    return probabilities/probabilities.sum()


def qae_samples(p, phase_bits, repeats, trials, rng):
    distribution = qae_distribution(p, phase_bits)
    size = len(distribution)
    outcomes = rng.choice(size, size=(trials, repeats), p=distribution)
    estimates = np.sin(np.pi*outcomes/size)**2
    return np.median(estimates, axis=1)
