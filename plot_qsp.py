from matplotlib import pyplot as plt
import numpy as np


def qsp_plot(a_s, poly_as, filename, target_fn, target_fn_label, title):
    plt.rcParams.update(
        {
            "text.usetex": True,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica"],
        }
    )
    plt.title(title)
    plt.plot(
        a_s,
        np.real(poly_as),
        marker="*",
        markersize="5",
        linestyle=("" if target_fn is not None else "--"),
        linewidth=1,
    )
    plt.plot(
        a_s, np.imag(poly_as), marker="v", markersize="5", linestyle="--", linewidth=1
    )
    plt.plot(
        a_s,
        np.abs(poly_as) ** 2,
        marker="^",
        markersize="5",
        linestyle="--",
        linewidth=1,
    )
    if target_fn is not None:
        plt.plot(a_s, target_fn(a_s))
    legend = ["Re(Poly(a))", "Im(Poly(a))", "$|Poly(a)|^2$", target_fn_label]
    plt.legend(legend)
    plt.ylabel(r"$f(a)$")
    plt.xlabel(r"a")
    plt.savefig(f"plots/{filename}")
    plt.show()
