from typing import List

import cirq
import numpy as np
import pyqsp.phases
from matplotlib import pyplot as plt


def wx(a: float) -> cirq.Gate:
    """"The Wx operator for the signal in the Wx QSP convention.

    Args:
        a: the input signal.
    Returns:
        the gate for the X rotation by theta
    """
    return cirq.Rx(rads=-2 * np.arccos(a))


def r(a: float) -> cirq.Gate:
    """"The R operator in the reflection QSP convention.

    Args:
        a: the input parameter
    Returns:
        the gate for R(a)
    """
    return cirq.MatrixGate(name=f"R({a})",
                           matrix=np.array([[a, np.sqrt(1 - a ** 2)],
                                            [np.sqrt(1 - a ** 2), -a]
                                            ]))


def s(phi: float) -> cirq.Gate:
    """The S operator for processing the signal.

    Args:
         phi: the rotation angle.
    Returns:
          the gate for the -2*phi Z rotation for the given angle, phi
    """
    return cirq.Rz(rads=-2 * phi)


def qsp(theta: float, wx_phis: List[float], convention: str) -> cirq.Circuit:
    """Returns the complete QSP sequence as a cirq.Circuit.

    For a given qubit, signal and QSP angles, it returns back a list of
    operations corresponding to the full QSP sequence.

    Args:
         q: the qubit to apply to operations to
         theta: the signal angle (in this file it is the X rotation angle)
         wx_phis: the QSP angles in Wx convention
         convention: the convention to convert to, possible values are "wx" for
            the Wx QSP convention and "r" for reflection QSP convention
    Returns:
          list of operations representing the QSP sequence
    """
    q = cirq.NamedQubit('q')
    signal = wx if convention == 'wx' else r
    processor = s
    if len(wx_phis) == 0:
        return cirq.Circuit()

    if convention == 'wx':
        phis = wx_phis
    else:
        phis = [wx_phis[0] - np.pi / 4]
        phis += [wx_phi - np.pi / 2 for wx_phi in wx_phis[1:-1]]
        phis += [wx_phis[-1] - np.pi / 4]

    # note the reverse order in the circuit notation
    ops = [processor(phis[0])(q)]
    for phi in phis[1:]:
        ops.append(signal(theta)(q))
        ops.append(processor(phi)(q))
    return cirq.Circuit(ops[::-1])


def qsp_response(theta: float, wx_phis: List[float],
                 convention: str, basis: str = 'x') -> float:
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
    meas_state = 1 / np.sqrt(2) * np.array(
        [1, 1]) if basis == 'x' else np.array([1, 0])

    d = len(wx_phis) - 1
    factor = 1 if convention == 'wx' else 1j ** d

    circuit = qsp(theta, wx_phis, convention)

    return factor * meas_state.conj().T @ circuit.final_state_vector(meas_state)


def plot(coeffs, title=None, convention='wx', basis='x', filename=None):
    """The main function to plot the two cases presented in the paper."""
    if not title:
        title = f"QSP({coeffs}, conv={convention}, basis={basis}) response"
    if not filename:
        filename = f"qsp_{coeffs}_{convention}_{basis}.png"
    # we plot from -1 to 1
    a_s = np.linspace(-1, 1, 50)
    poly_as = [
        qsp_response(a, coeffs, convention, basis) for a in a_s
    ]

    print(f"QSP circuit for {title}")
    print(qsp(0.5, coeffs, convention))

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})
    plt.title(title)
    plt.plot(a_s, np.real(poly_as), marker="*")
    plt.plot(a_s, np.imag(poly_as), marker="v")
    plt.plot(a_s, np.abs(poly_as) ** 2, marker="^")
    plt.legend(["Re(Poly(a))",
                "Im(Poly(a))",
                "$|Poly(a)|^2$"])
    plt.ylabel(r"$f(a)$")
    plt.xlabel(r"a")
    plt.savefig(f"plots/{filename}")
    plt.show()


if __name__ == '__main__':
    plot(coeffs=[0, 0], convention="wx", basis="z")
    plot(coeffs=[0, 0], convention="wx", basis="x")
    plot(coeffs=[0, 0], convention="r", basis="x")
    plot(coeffs=[0, 0], convention="r", basis="z")

    # this matches the numbers reported in the paper
    fpsearch_10_05 = pyqsp.phases.FPSearch().generate(10, 0.5)
    plot(coeffs=fpsearch_10_05, convention="wx", basis="z",
         title="QSP(FPSearch(10, 0.5), conv=wx, basis=z)",
         filename="fpsearch_10_0.5_wx_z.png")

    plot(coeffs=fpsearch_10_05, convention="r", basis="z",
         title="QSP(FPSearch(10, 0.5), conv=r, basis=z)",
         filename="fpsearch_10_0.5_r_z.png")

    # Sign function approximation QSP phase angles from the paper
    # this does not match (in fact it's non-deterministic):
    # >>> poly = pyqsp.poly.PolySign().generate(degree=19, delta=10)
    # >>> pyqsp.angle_sequence.QuantumSignalProcessingPhases(poly)
    # opened issue: https://github.com/ichuang/pyqsp/issues/1
    poly_sign_from_paper = [
        0.01558127, -0.01805798, 0.05705643, -0.01661832,
        0.16163773, 0.09379074, -2.62342885, 0.49168481,
        0.92403822, -0.09696846, -0.09696846, 0.92403822,
        0.49168481, -2.62342885, 0.09379074, 0.16163773,
        -0.01661832, 0.05705643, -0.01805798, 1.5863776]

    plot(coeffs=poly_sign_from_paper, convention="wx", basis="x",
         title="PolySign(19, 10, conv=wx, basis=x)",
         filename="polysign_19_10_wx_x.png")

    plot(coeffs=poly_sign_from_paper, convention="r", basis="x",
         title="PolySign(19, 10, conv=r, basis=x)",
         filename="polysign_19_10_r_x.png")
