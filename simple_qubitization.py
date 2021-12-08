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

import dataclasses
from typing import List, Iterable, Callable, Tuple

import cirq
import numpy as np
import pyqsp.phases
import scipy.linalg
from matplotlib import pyplot as plt

from plot_qsp import qsp_plot
from qsp import to_r_z_from_wx


@dataclasses.dataclass
class FixedPointAmplitudeAmplification:
    """Amplitude amplification inputs.

    Based on the inputs for an amplitude amplification problem, it
    creates the fixed point amplitude amplification circuit and after the
    proper projection (depending on whether the number of
    coefficients is even or odd) it returns the amplitude.

    On a real quantum computer, we'll need to provide all of u, u_inv, a0, b0,
    rotA0=e^(i * phi * (2|a0><a0|-I)) and rotB0=e^(i * phi * (2|b0><b0|-I))
    for the algorithm as black boxes, but in this simulation we can just
    calculate them from u, a0, b0. Finally, coeffs determine the polynomial
    we'd like to convert <a0|u|b0> with.

    Attributes:
        u: the unitary to amplify
        a0: the goal state
        b0: the starting state
        coeffs: the coefficients in QSP(R, <0|.|0>) convention
    """

    u: cirq.Gate
    a0: cirq.STATE_VECTOR_LIKE
    b0: cirq.STATE_VECTOR_LIKE
    coeffs: List[float]

    u_inv: cirq.Gate = None
    rot_a0: Callable[[float], cirq.Gate] = None
    rot_b0: Callable[[float], cirq.Gate] = None
    num_qubits: int = 2
    _amplitude_projector: Callable[[np.ndarray], np.complex] = None

    def __post_init__(self):
        self.u_inv = cirq.inverse(self.u)
        self.rot_a0 = self._rot_state("a0", self.a0)
        self.rot_b0 = self._rot_state("b0", self.b0)
        self.num_qubits = cirq.num_qubits(self.u)
        self._amplitude_projector = lambda uni: (
            self.a0 @ uni @ self.b0
            if len(self.coeffs) % 2 == 1
            else self.b0 @ uni @ self.b0
        )

    def _rot_state(
        self, name: str, state_vector: np.ndarray
    ) -> Callable[[float], cirq.Gate]:
        """Rotates the state around a given state."""
        return lambda phi: cirq.MatrixGate(
            name=f"{name}[{phi:.2f}]",
            matrix=scipy.linalg.expm(
                1j * phi * (2 * np.outer(state_vector, state_vector) - np.identity(4))
            ),
        )

    def get_circuit(self) -> cirq.Circuit:
        qs = cirq.LineQubit.range(self.num_qubits)
        # reverse operation order for circuits
        # we mandatorily start with U, as this is the U|B0> in Eq (13)
        if len(self.coeffs) == 0:
            return cirq.Circuit(self.u(*qs))

        ops = []
        i = 1

        for phi in self.coeffs[::-1]:
            if i % 2 == 1:
                ops += [self.u(*qs)]
                ops += [self.rot_a0(phi)(*qs)]
            else:
                ops += [self.u_inv(*qs)]
                ops += [self.rot_b0(phi)(*qs)]

            i += 1

        return cirq.Circuit(ops)

    def run(self) -> float:
        return self._amplitude_projector(cirq.unitary(self.get_circuit()))

    def __str__(self):
        return f"""FixedPointAmplification:
num qubits: {self.num_qubits},
u: {self.u},
a0: {self.a0},
b0: {self.b0},             
{self.get_circuit()}"""


class Experiment:
    def __init__(
        self,
        coeffs: List[float],
        n_points: int,
        basis_a: int = 2,
        basis_b: int = 3,
        n_qubits: int = 2,
    ):
        self.coeffs = coeffs
        self.basis_a = basis_a
        self.basis_b = basis_b
        self.n_points = n_points
        self.a_s = []
        self.fa_s = []
        self.a0 = cirq.to_valid_state_vector(basis_a, n_qubits)
        self.b0 = cirq.to_valid_state_vector(basis_b, n_qubits)

    def _get_u_gate_and_initial_amplitude(
        self, p: float, sign: int
    ) -> Tuple[float, cirq.Gate]:
        """Creates a CNOT-like unitary with a real amplitude."""
        u = sign * scipy.linalg.expm(1j * p * cirq.unitary(cirq.CX))
        a = u[self.basis_a][self.basis_b]
        new_a = a * sign * np.conj(a) / np.abs(a)
        return new_a, cirq.MatrixGate(
            name="u", matrix=sign * np.conj(a) / np.abs(a) * u
        )

    def _run_half(self, sign: int):
        for p in np.linspace(1e-8, np.pi, self.n_points):
            a, u = self._get_u_gate_and_initial_amplitude(p, sign)
            fp_amp = self._get_fpamp(u)
            self.a_s.append(a)
            self.fa_s.append(fp_amp.run())

    def _get_fpamp(self, u):
        return FixedPointAmplitudeAmplification(u, self.a0, self.b0, self.coeffs)

    def run(self) -> Tuple[List[float], List[float]]:
        _, sample_fpamp = self._get_u_gate_and_initial_amplitude(0.123, -1)
        print(self._get_fpamp(sample_fpamp))

        self._run_half(-1)
        self._run_half(1)
        return self.a_s, self.fa_s


def experiment(
    coeffs,
    npoints=50,
    title=None,
    filename="fp_amp.png",
    target_fn=None,
    target_fn_label: str = None,
):
    """The main function to qsp the two cases presented in the paper."""
    title = f"Fixed amplitude amplification for {title}"

    a_s, f_as = Experiment(coeffs, npoints).run()

    qsp_plot(np.real(a_s), f_as, filename, target_fn, target_fn_label, title)


if __name__ == "__main__":
    experiment(
        title="$T_1$",
        coeffs=to_r_z_from_wx([0, 0]),
        npoints=10,
        filename="fp_amp_t1.png",
        target_fn=lambda a_s: a_s,
        target_fn_label="$T_1(a)=a$",
    )
    experiment(
        title="$T_2$",
        coeffs=to_r_z_from_wx([0, 0, 0]),
        npoints=100,
        filename="fp_amp_t2.png",
        target_fn=lambda a_s: 2 * a_s ** 2 - 1,
        target_fn_label="$T_2(a)=2a^2-1$",
    )
    experiment(
        title="$T_3$",
        coeffs=to_r_z_from_wx([0, 0, 0, 0]),
        npoints=100,
        filename="fp_amp_t3.png",
        target_fn=lambda a_s: 4 * a_s ** 3 - 3 * a_s,
        target_fn_label="$T_3(a)=4 a^3-3 a$",
    )
    experiment(
        title="$T_4$",
        coeffs=to_r_z_from_wx([0, 0, 0, 0, 0]),
        npoints=100,
        filename="fp_amp_t4.png",
        target_fn=lambda a_s: 8 * a_s ** 4 - 8 * a_s ** 2 + 1,
        target_fn_label="$T_4(a)=8 a^4-8 a^2 +1$",
    )
    experiment(
        title="$T_5$",
        coeffs=to_r_z_from_wx([0, 0, 0, 0, 0, 0]),
        npoints=100,
        filename="fp_amp_t5.png",
        target_fn=lambda a_s: 16 * a_s ** 5 - 20 * a_s ** 3 + 5 * a_s,
        target_fn_label="$T_5(a)=16 a^5-20 a^3 + 5 a$",
    )
    # these are the same as in the Martyn et al paper
    wx_phis = pyqsp.phases.FPSearch().generate(10, 0.5)

    experiment(
        title="FPSearch(10,0.5)",
        coeffs=to_r_z_from_wx(wx_phis),
        npoints=100,
        filename="fp_amp_fpsearch_10_0.5.png",
    )
