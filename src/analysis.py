import io
import math
import time
from datetime import datetime

import numpy as np
import warnings
import matplotlib.pyplot as plt

from cavity import Cavity
from results import CavityResults


def run_cavity_job(epics_name, n_samples, timeout):
    # We want all warnings to be raised as exceptions
    np.seterr(all='raise')
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        # Create a cavity object for interacting with this cavity
        cavity = Cavity.get_cavity(epics_name=epics_name)

        # Setup for storing multiple samples' results
        tdoff = cavity.get_tdoff()
        cavity_results = CavityResults(epics_name=epics_name, tdoff=tdoff)

        with cavity.scope_mode():
            for i in range(n_samples):
                # Initialize the values in case of exception
                tdoff_error = math.nan
                coef = []
                img_buf = None
                error = ""

                try:
                    # Wait until there is no trip
                    start = datetime.now()
                    while not cavity.is_stable_running():
                        time.sleep(1)
                        if (datetime.now() - start).total_seconds() > timeout:
                            raise RuntimeError(f"{epics_name}: {start.strftime('%Y-%m-%d %H:%M:%S')} "
                                               "sample timed out waiting for stable running")

                    # Get the detune angle and reflected power waveforms.  This waits for the data to be ready.
                    deta, crfp = cavity.get_waveforms()
                    time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]

                    # Check again.  If we're not still in stable operations, then we probably have
                    # a problem in the data just collected.
                    if not cavity.is_stable_running():
                        raise RuntimeError("Trip occurred during data collection")

                    # Calculate the second order fit of these
                    X = np.tan(np.radians(deta))
                    coef = np.polynomial.polynomial.polyfit(X, crfp, deg=2)

                    # Find the argmin of CRFP - i.e., the detune angle producing the lowest
                    # reflected power.
                    try:
                        tdoff_error = np.degrees(np.arctan(get_quadratic_min(coef)))
                    except Exception as exc:
                        error += repr(exc)
                        tdoff_error = math.nan

                    # Save a plot of the data and the fit
                    img_buf = get_plot_img(deta, crfp, coef, tdoff_error, epics_name, time_string)

                except Exception as exc:
                    error += repr(exc)

                # Collect the results
                cavity_results.append_result(tdoff_error, coef, img_buf, error)

    return cavity_results


def get_quadratic_min(coefs):
    """Return the input that produces a zero derivative."""
    if coefs[2] <= 0:
        raise RuntimeError("Fit has a negative second derivative")

    return -coefs[1] / (2 * coefs[2])


def get_plot_img(deta, crfp, coef, tdoff_error, epics_name, time_string):
    """Generates a plot and returns it as a buffer variable."""

    # Create a figure for the plot
    fig = plt.figure(figsize=(6, 4))

    # Plot the data
    plt.scatter(deta, crfp, color='blue', alpha=0.5, s=2)

    # Plot the fit and min point
    plot_x = np.linspace(np.min(deta), np.max(deta), 100)
    x = np.tan(np.radians(plot_x))
    y = coef[0] + coef[1] * x + coef[2] * np.power(x, 2)
    plt.plot(plot_x, y, 'r')

    # Plot the min point
    ymin = np.min(crfp) - (np.max(crfp) - np.min(crfp)) / 2
    x_center = np.min(deta) + (np.max(deta) - np.min(deta)) / 2
    if math.isnan(tdoff_error):
        plt.text(x_center, ymin, f"min CRFP @\ndeta = {tdoff_error}",
                 ha="center")
    else:
        plt.vlines(tdoff_error, ymin=ymin, ymax=np.max(crfp), zorder=3)
        plt.text(x_center, ymin, f"min CRFP @\ndeta = {round(tdoff_error, 2)}",
                 ha="center")

    c = np.round(coef, 5).astype('str')
    coef_str = f"CRFP = {c[0]} + {c[1]}*tan(d) + {c[2]}*tan(d)^2"
    title = f"{epics_name} ({time_string})\n{coef_str}"
    plt.title(title)
    plt.ylabel("CRFP")
    plt.xlabel("DETA2")

    # Write the plot out as an image file in a buffer variable
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf
