#    Copyright 2021 Balint Pato
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

from typing import List, Iterable, Callable

import cirq
import numpy as np
import pyqsp.phases
import scipy.linalg
from matplotlib import pyplot as plt

from plot_qsp import qsp_plot


def u(p: float, s, s_a, s_b, n_qubits: int) -> cirq.Gate:
    u = s * scipy.linalg.expm(1j * p * cirq.unitary(cirq.CX))
    return cirq.MatrixGate(name="u", matrix=s * np.conj(u[s_a][s_b]) / np.abs(
        u[s_a][s_b]) * u)


def rot_state(name: str, state_vector: np.ndarray) -> Callable[
    [float], cirq.Gate]:
    """Rotates the state around a given state."""
    return lambda phi: cirq.MatrixGate(name=f"{name}[{phi:.2f}]",
                                       matrix=scipy.linalg.expm(
                                           1j * phi * (2 * np.outer(
                                               state_vector,
                                               state_vector) - np.identity(4)))
                                       )


def qsp(qs: List[cirq.Qid],
        a: Callable[[float], cirq.Gate],
        b: Callable[[float], cirq.Gate],
        u: cirq.Gate,
        u_inv: cirq.Gate,
        phis: List[float]) -> Iterable[cirq.Operation]:
    # reverse operation order for circuits
    # we mandatorily start with U, as this is the U|B0> in Eq (13)
    if len(phis) == 0:
        return [u(*qs)]

    ops = []
    i = 1

    for phi in phis[::-1]:
        if i % 2 == 1:
            ops += [u(*qs)]
            ops += [a(phi)(*qs)]
        else:
            ops += [u_inv(*qs)]
            ops += [b(phi)(*qs)]

        i += 1

    return ops



def experiment(coeffs, title=None, convention='wx', basis='x', filename=None,
               target_fn=None,
               target_fn_label: str = None):
    """The main function to qsp the two cases presented in the paper."""
    if not title:
        title = f"QSP({coeffs}, conv={convention}, basis={basis}) response"
    if not filename:
        filename = f"qsp_{title}_{convention}_{basis}.png"
    print(f"QSP circuit for {title}")
    print(qsp(0.123, coeffs, convention, basis))

    # we qsp from -1 to 1
    a_s = np.linspace(-1, 1, 60)
    poly_as = [
        qsp_response(a, coeffs, convention, basis) for a in a_s
    ]

    qsp_plot(a_s, filename, poly_as, target_fn, target_fn_label, title)

def experiment(phis, title="", npoints=50):
    """"""
    n_qubits = 2

    s_a = 2
    s_b = 3

    a = rot_state("Ra", cirq.to_valid_state_vector(s_a, n_qubits))
    b = rot_state("Rb", cirq.to_valid_state_vector(s_b, n_qubits))

    i = 0

    a_s = []
    f_as = []

    d = len(phis) - 1

    p = 0.123
    unitary = u(p, -1, s_a, s_b, n_qubits)
    unitary_inverse = cirq.inverse(unitary)
    circuit = cirq.Circuit(
        qsp(cirq.LineQubit.range(2), a, b, unitary, unitary_inverse, phis))
    print(circuit)

    project = lambda uni: uni[s_a][s_b] if len(phis) % 2 == 1 else uni[s_b][s_b]

    qsp_response(a, a_s, b, f_as, n_qubits, npoints, phis, project, -1, s_a,
                 s_b)

    qsp_response(a, a_s, b, f_as, n_qubits, npoints, phis, project, 1, s_a, s_b)

    plt.figure(figsize=[10, 10])
    plt.title(f"Poly(a) {title}")
    plt.plot(np.real(a_s), np.real(f_as), marker="*")
    plt.plot(np.real(a_s), np.imag(f_as), marker="^")
    plt.plot(np.real(a_s), np.abs(f_as) ** 2, marker="^")
    plt.legend(["re", "im", "amp"])

    plt.show()


def qsp_response(a, a_s, b, f_as, n_qubits, npoints, phis, project, s, s_a,
                 s_b):
    for p in np.linspace(1e-8, np.pi, npoints):
        unitary = u(p, s, s_a, s_b, n_qubits)
        unitary_inverse = cirq.inverse(unitary)
        a_s.append(cirq.unitary(unitary)[s_a][s_b])
        circuit = cirq.Circuit(
            qsp(cirq.LineQubit.range(2), a, b, unitary, unitary_inverse, phis))
        transformed = cirq.unitary(circuit)
        f_as.append(project(transformed))


def wx_to_r(wx_phis):
    d = len(wx_phis) - 1
    phis = [wx_phis[0] + wx_phis[-1] + (d - 1) * np.pi / 2]
    phis += [wx_phi - np.pi / 2 for wx_phi in wx_phis[1:-1]]
    # phis += [wx_phis[-1]]
    print(phis)
    return phis


if __name__ == '__main__':
    experiment(title="T1", phis=wx_to_r([0, 0]), npoints=100)
    experiment(title="T2", phis=wx_to_r([0, 0, 0]), npoints=100)
    experiment(title="T3", phis=wx_to_r([0, 0, 0, 0]), npoints=100)
    experiment(title="T4", phis=wx_to_r([0, 0, 0, 0, 0]), npoints=100)
    experiment(title="T5", phis=wx_to_r([0, 0, 0, 0, 0, 0]), npoints=100)
    # these are the same as in the Martyn et al paper
    wx_phis=pyqsp.phases.FPSearch().generate(10, 0.5)


    experiment(title="FPSearch(10,0.5)", phis=wx_to_r(wx_phis), npoints=100)