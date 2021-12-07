from typing import List, Iterable

import cirq
import numpy as np
from matplotlib import pyplot as plt


def w(theta: float) -> cirq.Gate:
    """"The W operator for the signal.

    Args:
        theta: the input signal.
    Returns:
        the gate for the X rotation by theta
    """
    # note, at this point, we are just directly using the Rx rotation. 
    return cirq.Rx(rads=theta)


def s(phi: float) -> cirq.Gate:
    """The S operator for processing the signal.

    Args:
         phi: the rotation angle.
    Returns:
          the gate for the -2*phi Z rotation for the given angle, phi
    """
    return cirq.Rz(rads=-2 * phi)


def qsp(q: cirq.Qid, theta: float, phis: List[float]) -> Iterable[
    cirq.Operation]:
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
    for phi in list(reversed(phis))[:-1]:
        yield s(phi)(q)
        yield w(theta)(q)
    yield s(phis[0])(q)


def probability_state_stays_zero(theta, phis) -> float:
    """Returns the probability that |0> stays |0> after applying QSP.

    The QSP sequence for the given signal and QSP angles defines a unitary, U.
    This function returns the probability that U preserves the |0> state, i.e.
    |<0|U|0>|^2.

    Args:
          theta: the X rotation
          phis: the QSP angles
    Returns:
          the |0> -> |0> probability
    """
    state = qsp_circuit(phis, theta).final_state_vector(0)
    return np.abs(state[0]) ** 2


def qsp_circuit(phis, theta):
    return cirq.Circuit(qsp(cirq.NamedQubit("q0"), theta, phis))


def plot():
    """The main function to plot the two cases presented in the paper."""

    # we plot from -pi to pi
    signal = np.linspace(-np.pi, np.pi, 50)

    # the |0> -> |0> probability with pure X rotations
    unprocessed_probabilities = list(
        map(lambda th: probability_state_stays_zero(th, (0, 0)), signal))

    # the BB1 angle sequence
    eta = 1 / 2 * np.arccos(-1 / 4)
    bb1 = [np.pi / 2, -eta, 2 * eta, 0, -2 * eta, eta]

    # the |0> -> |0> probability with BB1 processed X rotations
    bb1_probabilities = list(
        map(lambda th: probability_state_stays_zero(th, bb1), signal))

    # print out an example QSP circuit
    print("The QSP circuit for the BB1 transformed Rx rotation at theta -1")
    print(qsp_circuit(bb1, np.pi/2))

    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica"]})
    plt.title("The effect of the BB1 sequence")
    plt.plot(signal / np.pi, unprocessed_probabilities)
    plt.plot(signal / np.pi, bb1_probabilities)
    plt.legend([r"$U=R_x(\theta)$", r"$U=BB1(R_x(\theta))$"])
    plt.ylabel(r"$|\langle 0 | U(\theta)| 0 \rangle|^2$")
    plt.xlabel(r"$\frac{\theta}{\pi}$")
    plt.savefig("bb1_plot.png")
    plt.show()


if __name__ == '__main__':
    plot()
