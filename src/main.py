import os
import math
from typing import List

import numpy as np
import argparse
import concurrent.futures

from _version import __version__
from analysis import run_cavity_job
from ced import CED
from email_sender import EmailSender
from results import CavityResults, ResultSet, ResultTextFormatter, ResultsException


def process_cavities(cavities, n_samples, timeout, force_periodic):
    """Run TDOFF analysis for the specified cavities."""
    rs = ResultSet()

    try:
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            # Use submit to leverage Future interface.  Creates a dictionary
            # of Future objects mapped to cavity epics names.
            futures = {}
            for epics_name in cavities.keys():
                cav_type = cavities[epics_name]
                futures[executor.submit(run_cavity_job, epics_name, cav_type, n_samples=n_samples,
                                        timeout=timeout, force_periodic=force_periodic)] = epics_name

        for future in concurrent.futures.as_completed(futures):
            try:
                results = future.result()
            except Exception as exc:
                results = CavityResults(epics_name=futures[future], tdoff=math.nan)
                results.append_result(tdoff_error=math.nan,
                                      coefs=np.array([math.nan, math.nan, math.nan]),
                                      img_buf=None, error_message=f"ALL FAILED: {repr(exc)}")

            rs.add_cavity_results(results)

    except Exception as exc:
        print(exc)

    return rs


def main():
    try:
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
        parser.add_argument("--cavity-type", type=str,
                            help="Manually specify the cavity type.  By default, query CED.  E.g., 'C75' or 'C100'")
        parser.add_argument("-e", "--email", type=str, nargs='+',
                            help="Space separated list of email addresses to report")
        parser.add_argument("-q", "--quiet", action='store_true', help="Suppresses text output")
        parser.add_argument("-n", "--n-samples", type=int, default=1,
                            help="Number of samples to collect per cavity")
        parser.add_argument("-t", "--timeout", type=float, default=20,
                            help="How long each sample should wait for stable operations.")
        parser.add_argument("-f", "--force-periodic", action='store_true', default=False,
                            help="Allow the program to interrupt any scope mode operation.")
        parser.add_argument("-m", "--min-error", type=float, default=0,
                            help="Minimum detune offset error to produce output.  -1 to always alert.")
        parser.add_argument("-v", "--version", action='version', version='%(prog)s ' + __version__)

        args = parser.parse_args()

        # Make sure the user has specified some form of output
        if (args.email is None) and args.quiet:
            raise ValueError("User selection of quiet and no emails will produce no output.")

        if args.cavity is not None:
            cavity_names = args.cavity
        elif args.zone is not None:
            cavity_names = [f"{args.zone[0]}{i}" for i in range(1, 9)]
        else:
            raise ValueError("Cavity or Zone must be supplied to CLI.")

        # Construct the cavities dictionary.  Matches epics_name to cavity type.
        if args.cavity_type is None:
            ced = CED()
            cavities = get_ced_cavities(cavity_names, ced=ced)
        else:
            cavities = {}
            for e_name in cavity_names:
                cavities[e_name] = args.cavity_type

        # Go get the data and analyze it
        result_set = process_cavities(cavities, n_samples=args.n_samples,timeout=args.timeout,
                                      force_periodic=args.force_periodic)

        try:
            # Short circuit here if no cavity had a significant error.  Otherwise, process the results and output.
            max_error = result_set.get_max_average_tdoff_error()
            if args.min_error > max_error:
                return
        except ResultsException as exc:
            # Let the program continue.  More information will be given in the report.
            print(f"Warning: {exc}")

        if args.email is not None:
            e_mail = EmailSender(subject="Detune Error Report", fromaddr=os.getlogin(),
                                 toaddrs=args.email)
            e_mail.send_html_email(rs=result_set)
        if not args.quiet:
            formatter = ResultTextFormatter()
            print(formatter.format(result_set))

    except Exception as exc:
        print(f"Error: {exc}")
        exit(1)


def get_ced_cavities(epics_names: List[str], ced: CED):
    properties = ['EPICSName', 'CavityType']
    elements = ced.query_inventory(element_type='CryoCavity', properties=properties)
    cavities = {}
    missing = []
    for e_name in epics_names:
        found = False
        for e in elements:
            if e['properties']['EPICSName'] == e_name:
                cavities[e_name] = e['properties']['CavityType']
                found = True
                break

        if not found:
            missing.append(e_name)

    if len(missing) > 0:
        raise ValueError(f"CED lookup failed for {missing}.")

    return cavities


if __name__ == "__main__":
    main()
