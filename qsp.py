from typing import List, Iterable

import cirq
import numpy as np
import pyqsp
import pyqsp.phases
import sympy
from pyqsp import angle_sequence
from matplotlib import pyplot as plt


def w(a: float) -> cirq.Gate:
    """"The Wx operator for the signal in the Wx QSP convention.

    Args:
        a: the input signal.
    Returns:
        the gate for the X rotation by theta
    """
    return cirq.Rx(rads=-2 * np.arccos(a))


def r(a: float) -> cirq.Gate:
    """"The R operator in the reflection QSP convention.


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


def qsp(theta: float, wx_phis: List[float],
        convention: str) -> cirq.Circuit:
    """Returns the complete QSP sequence.

    For a given qubit, signal and QSP angles, it returns back a list of
    operations corresponding to the full QSP sequence.

    Args:
         q: the qubit to apply to operations to
         theta: the signal angle (in this file it is the X rotation angle)
         phis: the QSP angles
    Returns:
          list of operations representing the QSP sequence
    """
    q = cirq.NamedQubit('q')
    signal = w if convention == 'wx' else r
    processor = s
    if not wx_phis:
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


def qsp_response(theta, wx_phis, convention, basis='x') -> float:
    """Returns the QSP response for a given theta.

    The QSP sequence for the given signal and QSP angles defines a unitary, U.
    This function returns the probability that U preserves the |0> state, i.e.
    |<0|U|0>|^2.

    Args:
          theta: the X rotation
          wx_phis: the QSP angles
    Returns:
          the |0> -> |0> probability
    """
    meas_state = 1 / np.sqrt(2) * np.array(
        [1, 1]) if basis == 'x' else np.array([1, 0])

    d = len(wx_phis) - 1
    factor = 1 if convention == 'wx' else 1j ** d

    circuit = qsp(theta, wx_phis, convention)

    return factor * meas_state.conj().T @ circuit.final_state_vector(meas_state)


def plot(coeffs, title=None, convention='wx', basis='x'):
    """The main function to plot the two cases presented in the paper."""
    if not title:
        title = f"QSP({coeffs}, conv={convention}, basis={basis}) response"
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
    plt.savefig("qsp.png")
    plt.show()


if __name__ == '__main__':
    plot(coeffs=[0, 0], convention="wx", basis="z")
    plot(coeffs=[0, 0], convention="wx", basis="z")
    plot(coeffs=[0, 0], convention="r", basis="z")

    # bb1 = [0, 0, 0]

    # wxphis = [-1.58023603, -1.55147987, -1.6009483, -1.52812171, -1.62884337, \
    #        -1.49242141, -1.67885248, -1.41255145, -1.8386054, -0.87463828, \
    #        -0.87463828, -1.8386054, -1.41255145, -1.67885248, -1.49242141, \
    #        -1.62884337, -1.52812171, -1.6009483, -1.55147987, -1.58023603]
    #
    # wxphis = [-1.57576949, -1.56061906, -1.58668456, -1.54830449, -1.60140168, \
    #           -1.52944546, -1.6278833, -1.48692359, -1.71436243, -1.15604816, \
    #           -1.15604816, -1.71436243, -1.48692359, -1.6278833, -1.52944546, \
    #           -1.60140168, -1.54830449, -1.58668456, -1.56061906, -1.57576949]
    # bb1 = [wxphis[0] - np.pi / 4] + [phi - np.pi / 2 for phi
    #                                  in wxphis[1:-1]] + [
    #           wxphis[-1] - np.pi / 4]
    # # poly = pyqsp.poly.PolySign().generate(degree=19, delta=10.0)
    # bb1 = pyqsp.angle_sequence.QuantumSignalProcessingPhases(poly)
    # bb1 = pyqsp.phases.FPSearch().generate(20, 0.8)
    #
    # print(repr(bb1))
    # basis = 'x'
    # bb1=[0,0]
    # pyqsp.response.PlotQSPResponse(bb1, measurement=basis, plot_magnitude=True)
    # pyqsp.response.PlotQSPResponse(bb1, measurement=basis)
