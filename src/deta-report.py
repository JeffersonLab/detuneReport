#!/usr/csite/pubtools/bin/python3.7

import epics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import io
import time
import datetime
import urllib
import base64
import concurrent.futures

from email.message import EmailMessage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from email.mime.image import MIMEImage
import email.mime.application

import smtplib
import imghdr


class EmailMessage:

    def __init__(self, subject, fromaddr, toaddrs, smtp_server='localhost'):
        self.subject = subject
        self.fromaddr = fromaddr
        self.toaddrs = toaddrs
        self.smtp_server = smtp_server
        self.message = None

        # Data Structures keyed on epic_name
        self.epics_names = []
        self.d_mins = {}
        self.coefs = {}
        self.imgs = {}
        self.errors = {}
         

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

    def add_cavity_results(self, epics_name, d_min, coefs, img, error=""):
        self.epics_names.append(epics_name)
        self.d_mins[epics_name] = d_min
        self.coefs[epics_name] = coefs
        self.imgs[epics_name] = img
        self.errors[epics_name] = error

    def generate_message(self):
        # Setup the header, etc.
        self.init_message()

        # Add the summary section/table
        self.message += "<h1>Result Summary</h1>\n"
        self.message += "<table><tr><th>Cavity</th><th>Min Deta2</th>"
        self.message += "<th>Fit Coefs</th>"
        self.message += "<th>Error Message</th></tr>\n"
        for name in sorted(self.epics_names):
            d_min = self.d_mins[name]
            coefs = self.coefs[name]
            err = self.errors[name]
            self.message += f"<tr><td>{name}</td><td>{d_min}</td>"
            self.message += f"<td>{list(coefs)}</td>"
            self.message += f"<td>{err}</td></tr>\n"
        self.message += f"</table>"

        # Include the plots in the message.
        self.message += "<h1>Plots</h1>"
        for name in sorted(self.epics_names):
            self.message += f"<h2>{name}</h2>"
            img = self.imgs[name]
            if img is None:
                self.message += f"<p>{name} experienced an error.</p>"
                self.message += f"<p>{self.errors[name]}</p>\n"
            else:
                string = base64.b64encode(img.read()).decode()
                uri = 'data:image/png;base64,' + string
                #uri = 'data:image/png;base64,' + urllib.parse.quote(string)
                self.message += f'<img src="{uri}"/>\n'

                # Put the cursor back at the start so they can be attachments.
                img.seek(0)
        
        self.finalize_message()

    def send_html_email(self):
        self.generate_message()

        msg = MIMEMultipart('mixed')
        msg['Subject'] = self.subject
        msg['From'] = self.fromaddr
        msg['To'] = ",".join(self.toaddrs)

        part1 = MIMEText(self.message, 'html')
        msg.attach(part1)
        if self.imgs is not None:
            for filename in self.imgs.keys():
                if self.imgs[filename] is not None:
                    att = email.mime.image.MIMEImage(self.imgs[filename].read(),
                                                     _subtype='png')
                    att.add_header('Content-Disposition', 'attachment',
                                   filename=f"{filename}.png")
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



def get_plot_img(deta, crrp, coef, epics_name, time_string):
    """Generates a plot and returns it as a buffer variable."""

    # Create a figure for the plot
    fig = plt.figure(figsize=(6,4))

    # Plot the data
    plt.scatter(deta, crrp, color='blue', alpha=0.5, s=2)

    # Plot the fit and min point
    x = np.linspace(np.min(deta), np.max(deta), 100)
    y = coef[0] + coef[1] * x + coef[2] * np.power(x,2)
    plt.plot(x, y, 'r')

    # Plot the min point
    d_min = get_quadratic_min(coef)
    ymin = np.min(crrp) - (np.max(crrp) - np.min(crrp))/2
    x_center = np.min(deta) + (np.max(deta) - np.min(deta))/2
    plt.vlines(d_min, ymin=ymin, ymax=np.max(crrp), zorder=3)
    plt.text(x_center, ymin, f"min CRRP @\ndeta = {round(d_min,2)}",
             ha="center")

    c = np.round(coef, 5).astype('str')
    coef_str = f"CRRP = {c[0]} + {c[1]}*d + {c[2]}*d^2"
    title = f"{epics_name} ({time_string})\n{coef_str}"
    plt.title(title)
    plt.ylabel("CRRP")
    plt.xlabel("DETA2")

    # Write the plot out as an image file in a buffer variable
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    plt.close()
    return buf


def run_cavity_job(epics_name, timeout=30.0):

    # Wait until there is no trip
    while not is_stable_running(epics_name):
        time.sleep(1)

    # Get the detune angle and reflected power waveforms
    deta = epics.caget(f"{epics_name}WFSDETA2")
    crrp = epics.caget(f"{epics_name}WFSCRRP")
    time_string = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-5]

    # Calculate the second order fit of these
    coef = np.polynomial.polynomial.polyfit(deta, crrp, deg=2)
    
    # Find the argmin of CRRP - i.e., the detune angle producing the lowest
    # reflected power.  Take the derivative, set it to zero, solve for deta.
    # Easy for crrp = c0 + c1*d + c2*^2.
    #d_min = -coef[1] / (2 * coef[2])
    d_min = get_quadratic_min(coef)

    # Print out the parameter estimates, and detune angle producing the minimum
    # reflected power.

    # Save a plot of the data and the fit
    im = get_plot_img(deta, crrp, coef, epics_name, time_string)

    return (d_min, coef, im, epics_name)


def main():

    cavities = [f"R15{c}" for c in range(1,9)]

    # TODO: Convert this to using executor.submit to leverage the futures
    # interface
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = executor.map(run_cavity_job, cavities, timeout=120)

    fromaddr = 'adamc@jlab.org'
    toaddrs = ['adamc@jlab.org']

    email = EmailMessage(subject="Detune Error Report", fromaddr=fromaddr,
                         toaddrs=toaddrs)

    for result in results:
        try:
            d_min = result[0]
            coefs = result[1]
            img = result[2]
            epics_name = result[3]
            
            email.add_cavity_results(epics_name=epics_name, d_min=d_min,
                                     coefs=coefs, img=img)

            #print(f"Cavity: {epics_name}")
            #print(f"Min CRRP DETA2 value: {d_min}")
            #print(f"Coeficients: {coefs}")
            #print(f"Img: {im}")
            # plt.imshow(mpimg.imread(img))
            # plt.show()
        except Exception as exc:
            email.add_cavity_results(epics_name="", d_min="", coefs="",
                                     img=None, error=f"{type(exc)}: {exc}")
            print("{exc.args}")

    email.send_html_email()

if __name__ == "__main__":
    main()

