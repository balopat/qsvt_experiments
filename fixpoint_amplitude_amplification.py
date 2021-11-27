from typing import List, Iterable, Callable

import cirq
import numpy as np
import scipy.linalg
from matplotlib import pyplot as plt


def u(n_qubits: int) -> cirq.Gate:
    # we seed it to 1 to keep it even
    return cirq.MatrixGate(name="u",
                           matrix=cirq.testing.random_unitary(2 ** n_qubits,
                                                              random_state=2))


def rot_state(name: str, state_vector: np.ndarray) -> Callable[
    [float], cirq.Gate]:
    """Rotates the state around a given state."""
    return lambda phi: cirq.MatrixGate(name=name,
                                       matrix=scipy.linalg.expm(
                                           1j * phi * np.outer(state_vector,
                                                               state_vector))
                                       )


def qsp(qs: List[cirq.Qid],
        a: Callable[[float], cirq.Gate],
        b: Callable[[float], cirq.Gate],
        u: cirq.Gate,
        u_inv: cirq.Gate,
        phis: List[float]) -> Iterable[cirq.Operation]:
    yield u(*qs)
    for k in range(len(phis) - 1, 0, -2):
        yield a(phis[k])(*qs)
        yield u_inv(*qs)
        yield b(phis[k - 1])(*qs)
        yield u(*qs)


def plot():
    """"""
    n_qubits = 3
    oracle = u(n_qubits)
    oracle_inverse = cirq.inverse(oracle)

    s_a = 3
    s_b = 4

    a = rot_state("Ra", cirq.to_valid_state_vector(s_a, n_qubits))
    b = rot_state("Rb", cirq.to_valid_state_vector(s_b, n_qubits))

    phis = [
       -1.58023603, -1.55147987, -1.6009483, -1.52812171,
       -1.62884337, -1.49242141, -1.67885248, -1.41255145,
       -1.8386054, -0.87463828, -0.87463828, -1.8386054,
       -1.41255145, -1.67885248, -1.49242141, -1.62884337,
       -1.52812171, -1.6009483, -1.55147987, -1.58023603, ]

    # phis = [
    #     0.01558127, -0.01805798, 0.05705643, -0.01661832,
    #     0.16163773, 0.09379074, -2.62342885, 0.49168481,
    #     0.92403822, -0.09696846, -0.09696846, 0.92403822,
    #     0.49168481, -2.62342885, 0.09379074, 0.16163773,
    #     -0.01661832, 0.05705643, -0.01805798, 1.5863776]
    fig, axs = plt.subplots(nrows=len(phis) // 2 //3 + 1, ncols=3)
    i = 0
    print(axs)
    for k in range(0, len(phis) + 1, 2):
        unitary = cirq.unitary(cirq.Circuit(qsp(
            cirq.LineQubit.range(3),
            a, b, oracle, oracle_inverse,
            phis[:k]
        )))
        probs = [[np.abs(a) ** 2 for a in r] for r in unitary]
        print(probs[s_a][s_b])
        axs[i // 3][i % 3].set_title(f"{i}")
        axs[i//3][i%3].imshow(probs)
        i += 1

    fig.show()


if __name__ == '__main__':
    plot()
