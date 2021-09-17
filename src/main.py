import os
import math
import numpy as np
import argparse
import concurrent.futures

from _version import __version__
from analysis import run_cavity_job
from email_sender import EmailSender
from results import CavityResults, ResultSet, ResultTextFormatter


def process_cavities(cavities, n_samples, timeout):
    """Run TDOFF analysis for the specified cavities."""
    rs = ResultSet()

    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        # Use submit to leverage Future interface.  Creates a dictionary
        # of Future objects mapped to cavity epics names.
        futures = {}
        for cavity in cavities:
            futures[executor.submit(run_cavity_job, cavity, n_samples=n_samples,
                                    timeout=timeout)] = cavity

    for future in concurrent.futures.as_completed(futures):
        try:
            results = future.result()
        except Exception as exc:
            results = CavityResults(epics_name=futures[future], tdoff=math.nan)
            results.append_result(tdoff_error=math.nan,
                                  coefs=np.array([math.nan, math.nan, math.nan]),
                                  img_buf=None, error_message=f"ALL FAILED: {repr(exc)}")

        rs.add_cavity_results(results)

    return rs


def main():
    # Setup parser.  You can target either a cavity or a zone.  Secondary check is
    # required to make sure that the user hasn't blocked all output of results.
    parser = argparse.ArgumentParser(prog=f"detuneReport",
                                     description="Calculate the detune offset error of a 12 GeV zone or cavity using "
                                                 "waveforms")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-z", "--zone", type=str, nargs=1,
                       help="EPICS name of a zone to check.  E.g. R1M")
    group.add_argument("-c", "--cavity", type=str, nargs=1,
                       help="EPICS name of cavity to check.  E.g., R1M1")
    parser.add_argument("-e", "--email", type=str, nargs='+',
                        help="Space separated list of email addresses to report")
    parser.add_argument("-q", "--quiet", action='store_true', help="Suppresses text output")
    parser.add_argument("-n", "--n-samples", type=int, default=1,
                        help="Number of samples to collect per cavity")
    parser.add_argument("-t", "--timeout", type=float, default=20,
                        help="How long each sample should wait for stable operations.")
    parser.add_argument("-v", "--version", action='version', version='%(prog)s ' + __version__)

    args = parser.parse_args()

    # Make sure the user has specified some form of output
    if (args.email is None) and args.quiet:
        print("Error: User selection of quiet and no emails will produce no output.")
        exit(1)

    if args.cavity is not None:
        cavities = args.cavity
    elif args.zone is not None:
        cavities = [f"{args.zone[0]}{i}" for i in range(1, 9)]
    else:
        raise ValueError("Cavity or Zone must be supplied to CLI.")

    # Go get the data and analyze it
    result_set = process_cavities(cavities, n_samples=args.n_samples,
                                  timeout=args.timeout)

    if args.email is not None:
        e_mail = EmailSender(subject="Detune Error Report", fromaddr=os.getlogin(),
                             toaddrs=args.email)
        e_mail.send_html_email(rs=result_set)
    if not args.quiet:
        formatter = ResultTextFormatter()
        print(formatter.format(result_set))


if __name__ == "__main__":
    main()
