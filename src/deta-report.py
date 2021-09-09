#!/usr/csite/pubtools/bin/python3.6

import math
import epics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io
import time
from datetime import datetime
import urllib
import base64
import argparse
import concurrent.futures

from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
import email.mime.application

import smtplib
import imghdr


class CavityResults:
    """A class for holding the results of analyzing a cavity.  Just a light wrapper."""

    def __init__(self, epics_name):
        self.epics_name = epics_name
        self.tdoff = None
        self.tdoff_errors = []
        self.coefs = []
        self.img_bufs = []
        self.error_messages = []

    def append_result(self, tdoff_error, coefs, img_buf, error_message):
        self.tdoff_errors.append(tdoff_error)
        self.coefs.append(coefs)
        self.img_bufs.append(img_buf)
        self.error_messages.append(error_message)

class ResultSet:

    def __init__(self):
        # Data Structures keyed on epic_name
        self.epics_names = []
        self.tdoff_errors = {}
        self.coefs = {}
        self.imgs = {}
        self.errors = {}

    def add_cavity_results(self, cavity_results: CavityResults):
        self.epics_names.append(cavity_results.epics_name)
        self.tdoff_errors[cavity_results.epics_name] = cavity_results.tdoff_errors
        self.coefs[cavity_results.epics_name] = cavity_results.coefs
        self.imgs[cavity_results.epics_name] = cavity_results.img_bufs
        self.errors[cavity_results.epics_name] = cavity_results.error_messages

    def to_table_string(self):
        """Return a table formatted human readable string."""
        t_fmt = "{:<7} {:<13} {:<13} {:<10} {}\n"

        out = t_fmt.format("Cavity", "TDOFF_Err_Avg", "TDOFF_Err_Std", "N_Success",
                           "Error_Msgs")
        for name in sorted(self.epics_names):
            avg = round(np.nanmean(self.tdoff_errors[name]), 3)
            std = round(np.nanstd(self.tdoff_errors[name]), 3)
            n_suc = np.count_nonzero(~np.isnan(self.tdoff_errors[name]))
            out += t_fmt.format(name, avg, std, n_suc, self.errors[name])
            
        return out


class EmailMessage:

    def __init__(self, subject, fromaddr, toaddrs, smtp_server='localhost'):
        self.subject = subject
        self.fromaddr = fromaddr
        self.toaddrs = toaddrs
        self.smtp_server = smtp_server
        self.message = None
         

    def init_message(self):
            self.message = """
<html>
<head>
<style>
p {
  margin: 1px;
}
p .more_space {
  margin-bottom: 10px;
}
h1 {
  padding-left: 10px;
  padding-bottom: 4px;
  background-color: #342E63;
  color: white;
}
h2 {
  padding-left: 10px;
  padding-bottom: 4px;
  background-color: #52489C;
  color: white;
}
table, th, td {
  border: 1px solid black;
  border-collapse: collapse;
}
th, td {
  padding-left: 15px;
  padding-right: 15px;
  text-align: left;
}
th {
  background-color: #EBEBEB;
}
</style>
</head>
<body>
"""

    def finalize_message(self):
        self.message += """</body></html>"""

    def generate_message(self, rs: ResultSet):
        # Setup the header, etc.
        self.init_message()
        
        # TODO: Add column for count of "good" non-nan samples
        # TODO: Add new TDOFF value
        # TODO: Format error messages better
        # Add the summary section/table
        self.message += "<h1>Result Summary</h1>\n"
        self.message += "<table><tr><th>Cavity</th><th>TDOFF Error Mean (degs)</th>"
        self.message += "<th>TDOFF Error Std Dev</th>"
        self.message += "<th>Error Message</th></tr>\n"
        for name in sorted(rs.epics_names):
            tdoff_error = rs.tdoff_errors[name]
            coefs = rs.coefs[name]
            err = rs.errors[name]
            self.message += f"<tr><td>{name}</td><td>{round(np.nanmean(tdoff_error), 2)}</td>"
            self.message += f"<td>{round(np.nanstd(tdoff_error), 2)}</td>"
            self.message += f"<td>{err}</td></tr>\n"
        self.message += f"</table>"

        # Include the plots in the message.
        self.message += "<h1>Plots</h1>"
        for name in sorted(rs.epics_names):
            self.message += f"<h2>{name}</h2>"
            cav_imgs = rs.imgs[name]
            for img_buf in cav_imgs:
                if img_buf is None:
                    self.message += f"<p>{name} experienced an error.</p>"
                    self.message += f"<p>{rs.errors[name]}</p>\n"
                else:
                    string = base64.b64encode(img_buf.read()).decode()
                    uri = 'data:image/png;base64,' + string
                    self.message += f'<img width="400" heigh="200" src="{uri}"/>\n'
    
                    # Put the cursor back at the start so they can be attachments.
                    img_buf.seek(0)
        
        self.finalize_message()

    def send_html_email(self, rs: ResultSet):
        self.generate_message(rs=rs)

        msg = MIMEMultipart('mixed')
        msg['Subject'] = self.subject
        msg['From'] = self.fromaddr
        msg['To'] = ",".join(self.toaddrs)

        part1 = MIMEText(self.message, 'html')
        msg.attach(part1)
        if rs.imgs is not None:
            for epics_name in sorted(rs.imgs.keys()):
                if rs.imgs[epics_name] is not None:
                    for img_buf in rs.imgs[epics_name]:
                        if img_buf is not None:
                            att = email.mime.image.MIMEImage(img_buf.read(),
                                                             _subtype='png')
                            att.add_header('Content-Disposition', 'attachment',
                                           filename=f"{epics_name}.png")
                            msg.attach(att)

        with smtplib.SMTP(self.smtp_server) as server:
            server.sendmail(msg['From'], self.toaddrs, msg.as_string())

    

def is_gradient_ramping(epics_name):
    # If the cavity is ramping is saved as the 11th bit in the
    # R...STAT1 PV
    stat1 = epics.caget(f"{epics_name}STAT1")

    # We're ramping if the bit is not 0
    is_ramping = int(stat1) & 0x0800 > 0

    return(is_ramping)


def is_rf_on(epics_name):
    rf_on = epics.caget(f"{epics_name}RFONr")
    is_on = rf_on == 1
    return is_on


def is_stable_running(epics_name):
    is_stable = (not is_gradient_ramping(epics_name)) and is_rf_on(epics_name)
    return is_stable


def get_quadratic_min(coefs):
    """Return the input that produces a zero derivative."""
    if coefs[2] <= 0:
        raise RuntimeError("Fit has a negative second derivative")

    return -coefs[1] / (2 * coefs[2])


def get_plot_img(deta, crfp, coef, tdoff_error, epics_name, time_string):
    """Generates a plot and returns it as a buffer variable."""

    # Create a figure for the plot
    fig = plt.figure(figsize=(6,4))

    # Plot the data
    plt.scatter(deta, crfp, color='blue', alpha=0.5, s=2)

    # Plot the fit and min point
    plot_x = np.linspace(np.min(deta), np.max(deta), 100)
    x = np.tan(np.radians(plot_x))
    y = coef[0] + coef[1] * x + coef[2] * np.power(x,2)
    plt.plot(plot_x, y, 'r')

    # Plot the min point
    ymin = np.min(crfp) - (np.max(crfp) - np.min(crfp))/2
    x_center = np.min(deta) + (np.max(deta) - np.min(deta))/2
    if math.isnan(tdoff_error):
        plt.text(x_center, ymin, f"min CRFP @\ndeta = {tdoff_error}",
             ha="center")
    else:
        plt.vlines(tdoff_error, ymin=ymin, ymax=np.max(crfp), zorder=3)
        plt.text(x_center, ymin, f"min CRFP @\ndeta = {round(tdoff_error,2)}",
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


def run_cavity_job(epics_name, n_samples, timeout):

    # Setup for storing multiple samples' results
    cavity_results = CavityResults(epics_name=epics_name)

    # TODO: Update procedure based on Rama's guidance
    for i in range(n_samples):
        try:
            # Intialize the values in case of exception
            tdoff_error = math.nan
            coef = []
            img_buf = None
            error = ""

            # Wait until there is no trip
            start = datetime.now()
            while not is_stable_running(epics_name):
                time.sleep(1)
                if (datetime.now() - start).total_seconds() > timeout:
                    raise RuntimeError(f"{epics_name}: {start.strftime('%Y-%m-%d %H:%M:%S')} "
                                       "sample timed out waiting for stable running")

            # Get the detune angle and reflected power waveforms
            deta = epics.caget(f"{epics_name}WFSDETA2")
            crfp = epics.caget(f"{epics_name}WFSCRFP")
            time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]

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

        # TODO: Update based on scope delay or sample rate or something.
        # Sleep so new data has time to accrue
        if i < n_samples - 1:
            time.sleep(2)

    return cavity_results


def process_cavities(cavities, n_samples, timeout):
    """Run tdoffset analysis for the specified cavities."""
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
            print(repr(exc))
            results = CavityResults(epics_name=futures[future])
            results.append_result(tdoff_error=math.nan,
                                  coefs=np.array([math.nan, math.nan, math.nan]),
                                  img_buf=None, error_message=f"{repr(exc)}")
            #print(f"Cavity: {epics_name}")
            #print(f"Min CRFP DETA2 value: {tdoff_error}")
            #print(f"Coeficients: {coefs}")
            #print(f"Img: {im}")
            # plt.imshow(mpimg.imread(img))
            # plt.show()

        rs.add_cavity_results(results)

    return rs


def main():

    # Setup parser.  You can target either a cavity or a zone.  Secondary check is
    # required to make sure that the user hasn't blocked all output of results.
    parser = argparse.ArgumentParser(description="Calculate the detune offset error of a"
                                                 " 12 GeV zone or cavity using waveforms")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-z", "--zone", type=str, nargs=1,
                       help="EPICS name of a zone to check.  E.g. R1M")
    group.add_argument("-c", "--cavity", type=str, nargs=1,
                       help="EPICS name of cavity to check.  E.g., R1M1")
    parser.add_argument("-e", "--email", type=str, nargs='+',
                        help="Space separated list of email addresses to report")
    parser.add_argument("-q", "--quiet", action='store_true', help="Suppresses text output")
    parser.add_argument("-n", "--n-samples", type=int, nargs=1, default=1,
                        help="Number of samples to collect per cavity")
    parser.add_argument("-t", "--timeout", type=float, nargs=1, default=20,
                        help="How long each sample should wait for stable operations.")

    args = parser.parse_args()
   
    # Make sure the user has specified some form of output
    if (args.email is None) and (args.quiet):
        print("Error: User selection of quiet and no emails will produce no output.")
        exit(1)

    if args.cavity is not None:
        cavities = args.cavity
    elif args.zone is not None:
        cavities = [f"{args.zone[0]}{i}" for i in range(1, 9)]

    # Go get the data and analyze it
    result_set = process_cavities(cavities, n_samples=args.n_samples[0], timeout=args.timeout)

    if args.email is not None:
        email = EmailMessage(subject="Detune Error Report", fromaddr=os.getlogin(),
                             toaddrs=args.email)
        email.send_html_email(rs=result_set)
    if not args.quiet:
       print(result_set.to_table_string()) 

if __name__ == "__main__":
    main()

