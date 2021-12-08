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

from typing import List

import cirq
import numpy as np
import pyqsp.phases

from plot_qsp import qsp_plot


def wx(a: float) -> cirq.Gate:
    """ "The Wx operator for the signal in the Wx QSP convention.

    Args:
        a: the input signal.
    Returns:
        the gate for the X rotation by theta
    """
    return cirq.Rx(rads=-2 * np.arccos(a))


def r(a: float) -> cirq.Gate:
    """ "The R operator in the reflection QSP convention.

    Args:
        a: the input parameter
    Returns:
        the gate for R(a)
    """
    return cirq.MatrixGate(
        name=f"R({a})",
        matrix=np.array([[a, np.sqrt(1 - a ** 2)], [np.sqrt(1 - a ** 2), -a]]),
    )


def s(phi: float) -> cirq.Gate:
    """The S operator for processing the signal.

    Args:
         phi: the rotation angle.
    Returns:
          the gate for the -2*phi Z rotation for the given angle, phi
    """
    return cirq.Rz(rads=-2 * phi)


def qsp(theta: float, wx_phis: List[float], convention: str, basis="x") -> cirq.Circuit:
    """Returns the complete QSP sequence as a cirq.Circuit.

    For a given qubit, signal and QSP angles, it returns back a list of
    operations corresponding to the full QSP sequence.

    Args:
         q: the qubit to apply to operations to
         theta: the signal angle (in this file it is the X rotation angle)
         wx_phis: the QSP angles in Wx convention
         convention: the convention to convert to, possible values are "wx" for
            the Wx QSP convention and "r" for reflection QSP convention
         basis: the QSP signal basis - either "z" for <0|U|0> or "x" for
            <+|U|+>
    Returns:
          the cirq Circuit representing the QSP sequence
    """
    q = cirq.NamedQubit("q")
    d = len(wx_phis) - 1
    signal = wx if convention == "wx" else r
    processor = s
    if len(wx_phis) == 0:
        return cirq.Circuit()

    if convention == "wx":
        phis = wx_phis
    elif basis == "x":
        phis = to_r_x_from_wx(wx_phis)
    else:
        phis = to_r_z_from_wx(wx_phis)

    if convention == "r" and basis == "z":
        # use operator product order for easier understanding
        ops = [cirq.GlobalPhaseOperation(np.exp(1j * (phis[0]))), signal(theta)(q)]
        for j in range(2, d + 1):
            ops.append(processor(phis[j - 1])(q))
            ops.append(signal(theta)(q))

    else:
        # use operator product order for easier understanding
        ops = [processor(phis[0])(q)]
        for phi in phis[1:]:
            ops.append(signal(theta)(q))
            ops.append(processor(phi)(q))

    # note the reverse order for the circuit notation
    return cirq.Circuit(ops[::-1])


def to_r_z_from_wx(wx_phis: List[float]) -> List[float]:
    """Convert from Wx convention to R convention in Z basis.

    It returns one less phis, as per Theorem II.3
    Args:
        wx_phis: the list of phase angles in the Wx convention
            (typically coming from pyqsp)
    Returns:
         the list of phase angles that can be used in reflection convention in Z
         basis
    """
    d = len(wx_phis) - 1
    phis = [wx_phis[0] + wx_phis[-1] + (d - 1) * np.pi / 2]
    phis += [wx_phi - np.pi / 2 for wx_phi in wx_phis[1:-1]]
    return phis


def to_r_x_from_wx(wx_phis: List[float]) -> List[float]:
    phis = [wx_phis[0] - np.pi / 4]
    phis += [wx_phi - np.pi / 2 for wx_phi in wx_phis[1:-1]]
    phis += [wx_phis[-1] - np.pi / 4]
    return phis


def qsp_response(
    theta: float, wx_phis: List[float], convention: str, basis: str = "x"
) -> float:
    """Returns the QSP response for a given theta.

    The QSP sequence for the given signal and QSP angles defines a unitary, U.
    This function returns <b|U|b> for a basis state |b>.

    Args:
          theta: the X rotation
          wx_phis: the QSP angles in Wx convention
          convention: the convention to convert to, possible values are "wx" for
            the Wx QSP convention and "r" for reflection QSP convention
          basis: the QSP signal basis - either "z" for <0|U|0> or "x" for
            <+|U|+>
    Returns:
          the QSP response, which is the overlap of the signal basis state with
          the basis state evolved by U, <b|U|b>
    """
    meas_state = 1 / np.sqrt(2) * np.array([1, 1]) if basis == "x" else np.array([1, 0])

    d = len(wx_phis) - 1
    if convention == "wx" or basis == "z":
        factor = 1
    elif convention == "r" and basis == "x":
        # see Theorem II.2. to see where this is coming from
        factor = 1j ** d

    circuit = qsp(theta, wx_phis, convention, basis)

    return factor * meas_state.conj().T @ circuit.final_state_vector(meas_state)


def experiment(
    coeffs,
    title=None,
    convention="wx",
    basis="x",
    filename=None,
    target_fn=None,
    target_fn_label: str = None,
):
    if not title:
        title = f"QSP({coeffs}, conv={convention}, basis={basis}) response"
    if not filename:
        filename = f"qsp_{title}_{convention}_{basis}.png"
    print(f"QSP circuit for {title}")
    print(qsp(0.123, coeffs, convention, basis))

    # we qsp from -1 to 1
    a_s = np.linspace(-1, 1, 60)
    poly_as = [qsp_response(a, coeffs, convention, basis) for a in a_s]

    qsp_plot(a_s, poly_as, filename, target_fn, target_fn_label, title)


if __name__ == "__main__":
    # sanity check for different conventions, all Chebyshev polynomials
    # are below for conv=r basis=z
    # TODO: turn these into tests
    experiment(
        coeffs=[0, 0],
        convention="wx",
        basis="x",
        title="$QSP(T_1, conv=wx, basis=x)$",
        filename="t_1_wx_x.png",
        target_fn=lambda a_s: a_s,
        target_fn_label="$T_1(a)=a$",
    )
    experiment(
        coeffs=[0, 0],
        convention="wx",
        basis="z",
        title="$QSP(T_1, conv=wx, basis=z)$",
        filename="t_1_wx_z.png",
        target_fn=lambda a_s: a_s,
        target_fn_label="$T_1(a)=a$",
    )
    experiment(
        coeffs=[0, 0],
        convention="r",
        basis="x",
        title="$QSP(T_1, conv=r, basis=x)$",
        filename="t_1_r_x.png",
        target_fn=lambda a_s: a_s,
        target_fn_label="$T_1(a)=a$",
    )

    # this matches the numbers reported in the paper
    fpsearch_10_05 = pyqsp.phases.FPSearch().generate(10, 0.5)
    experiment(
        coeffs=fpsearch_10_05,
        convention="wx",
        basis="z",
        title="QSP(FPSearch(10, 0.5), conv=wx, basis=z)",
        filename="fpsearch_10_0.5_wx_z.png",
    )

    experiment(
        coeffs=fpsearch_10_05,
        convention="r",
        basis="z",
        title="QSP(FPSearch(10, 0.5), conv=r, basis=z)",
        filename="fpsearch_10_0.5_r_z.png",
    )

    # Sign function approximation QSP phase angles from the paper
    # this does not match (in fact it's non-deterministic):
    # >>> poly = pyqsp.poly.PolySign().generate(degree=19, delta=10)
    # >>> pyqsp.angle_sequence.QuantumSignalProcessingPhases(poly)
    # opened issue: https://github.com/ichuang/pyqsp/issues/1
    poly_sign_from_paper = [
        0.01558127,
        -0.01805798,
        0.05705643,
        -0.01661832,
        0.16163773,
        0.09379074,
        -2.62342885,
        0.49168481,
        0.92403822,
        -0.09696846,
        -0.09696846,
        0.92403822,
        0.49168481,
        -2.62342885,
        0.09379074,
        0.16163773,
        -0.01661832,
        0.05705643,
        -0.01805798,
        1.5863776,
    ]

    experiment(
        coeffs=poly_sign_from_paper,
        convention="wx",
        basis="x",
        title="PolySign(19, 10, conv=wx, basis=x)",
        filename="polysign_19_10_wx_x.png",
    )

    experiment(
        coeffs=poly_sign_from_paper,
        convention="r",
        basis="x",
        title="PolySign(19, 10, conv=r, basis=x)",
        filename="polysign_19_10_r_x.png",
    )

    experiment(
        coeffs=poly_sign_from_paper,
        convention="r",
        basis="z",
        title="PolySign(19, 10, conv=r, basis=z)",
        filename="polysign_19_10_r_z.png",
    )

    experiment(
        coeffs=[0, 0],
        convention="r",
        basis="z",
        title="$QSP(T_1, conv=r, basis=z)$",
        filename="t_1.png",
    )
    experiment(
        coeffs=[0, 0, 0],
        convention="r",
        basis="z",
        title="$QSP(T_2, conv=r, basis=z)$",
        filename="t_2.png",
        target_fn=lambda a_s: 2 * a_s ** 2 - 1,
        target_fn_label="$T_2(a)=2a^2-1$",
    )
    experiment(
        coeffs=[0, 0, 0, 0],
        convention="r",
        basis="z",
        title="$QSP(T_3, conv=r, basis=z)$",
        filename="t_3.png",
        target_fn=lambda a_s: 4 * a_s ** 3 - 3 * a_s,
        target_fn_label="$T_3(a)=4 a^3-3 a$",
    )
    experiment(
        coeffs=[0, 0, 0, 0, 0],
        convention="r",
        basis="z",
        title="$QSP(T_4, conv=r, basis=z)$",
        filename="t_4.png",
        target_fn=lambda a_s: 8 * a_s ** 4 - 8 * a_s ** 2 + 1,
        target_fn_label="$T_4(a)=8 a^4-8 a^2 +1$",
    )
    experiment(
        coeffs=[0, 0, 0, 0, 0, 0],
        convention="r",
        basis="z",
        title="$QSP(T_5, conv=r, basis=z)$",
        filename="t_5.png",
        target_fn=lambda a_s: 16 * a_s ** 5 - 20 * a_s ** 3 + 5 * a_s,
        target_fn_label="$T_5(a)=16 a^5-20 a^3 + 5 a$",
    )
    experiment(
        coeffs=pyqsp.phases.FPSearch().generate(9, 0.5),
        convention="r",
        basis="z",
        title="QSP(FPSearch(9, 0.5), conv=r, basis=z)",
        filename="fpsearch_9_0.5_r_z.png",
    )
