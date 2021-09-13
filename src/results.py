import base64

import numpy as np
import math


class CavityResults:
    """A class for holding the results of analyzing a cavity.  Just a light wrapper."""

    def __init__(self, epics_name, tdoff):
        self.epics_name = epics_name
        self.tdoff = tdoff
        self.tdoff_errors = []
        self.coefs = []
        self.img_buffers = []
        self.error_messages = []

    def append_result(self, tdoff_error, coefs, img_buf, error_message):
        self.tdoff_errors.append(tdoff_error)
        self.coefs.append(coefs)
        self.img_buffers.append(img_buf)
        self.error_messages.append(error_message)


class ResultSet:
    """A class for working with multiple CavityResults."""

    def __init__(self):
        # Data Structures keyed on epic_name
        self.epics_names = []
        self.tdoffs = {}
        self.tdoff_errors = {}
        self.coefs = {}
        self.img_buffers = {}
        self.errors = {}

    def add_cavity_results(self, cavity_results: CavityResults):
        name = cavity_results.epics_name
        self.epics_names.append(name)
        self.tdoffs[name] = cavity_results.tdoff
        self.tdoff_errors[name] = cavity_results.tdoff_errors
        self.coefs[name] = cavity_results.coefs
        self.img_buffers[name] = cavity_results.img_buffers
        self.errors[name] = cavity_results.error_messages

    def get_tdoff_stats(self, epics_name):
        """ Calculate some basic statistics about the sample estimates."""
        n_success = np.count_nonzero(~np.isnan(self.tdoff_errors[epics_name]))
        if n_success > 0:
            avg = np.round(np.nanmean(self.tdoff_errors[epics_name]), 3)
            std = np.round(np.nanstd(self.tdoff_errors[epics_name]), 3)
        else:
            avg = math.nan
            std = math.nan

        return avg, std, n_success


class ResultFormatter:
    """Base class for formatting ResultSet data."""

    def format(self, rs: ResultSet):
        pass

    @staticmethod
    def _generate_summary_table_rows(fmt: str, rs: ResultSet) -> str:
        """Generates a the rows of a summary table according to specified format."""
        out = ""

        for name in sorted(rs.epics_names):
            avg, std, n_suc = rs.get_tdoff_stats(name)
            tdoff = round(rs.tdoffs[name], 3)
            tdoff_new = round(rs.tdoffs[name] + avg, 3)
            out += fmt.format(name, tdoff_new, tdoff, avg, std, n_suc)

        return out


class ResultTextFormatter(ResultFormatter):
    def format(self, rs: ResultSet):
        out = self.generate_summary_table(rs)
        return out

    def generate_summary_table(self, rs: ResultSet):
        """Return a table formatted human readable string."""
        t_fmt = "{:<7} {:>10} {:>10} {:>13} {:>13} {:>10}\n"

        out = t_fmt.format("Cavity", "TDOFF_New", "TDOFF_Cur", "TDOFF_Err_Avg",
                           "TDOFF_Err_Std", "N_Good")
        out += self._generate_summary_table_rows(t_fmt, rs)

        t_fmt = "{:<7} {:<5}  {}\n"
        e_out = "\n" + t_fmt.format("Cavity", "Run#", "Error_Message")
        no_errors = True
        for name in sorted(rs.epics_names):
            for rn, error in enumerate(rs.errors[name]):
                if error != "":
                    e_out += t_fmt.format(name, rn + 1, error)
                    no_errors = False

        if no_errors:
            out += "\nNo Errors"
        else:
            out += e_out

        return out


class ResultHTMLFormatter(ResultFormatter):
    """Generate HTML formatted output for a ResultSet"""

    def format(self, rs: ResultSet):
        html = self.opening_html()
        html += self.generate_summary_table(rs)
        html += self.generate_plot_html(rs)
        html += self.closing_html()
        return html

    def generate_summary_table(self, rs):
        """Return a html table of the data."""

        out = "<h1>Result Summary</h1>\n"
        out += "<table>"
        h_fmt = "<tr><th>{:<7}</th><th>{:>10}</th><th>{:>10}</th>"
        h_fmt += "<th>{:>13}</th><th>{:>13}</th><th>{:>10}</th></tr>\n"
        r_fmt = "<tr><td>{:<7}</td><td>{:>10}</td><td>{:>10}</td>"
        r_fmt += "<td>{:>13}</td><td>{:>13}</td><td>{:>10}</td></tr>\n"

        # Set up the first summary table
        out += "<thead>"
        out += h_fmt.format("Cavity", "TDOFF_New", "TDOFF_Cur", "TDOFF_Err_Avg",
                            "TDOFF_Err_Std", "N_Good")
        out += "</thead><tbody>"
        out += self._generate_summary_table_rows(r_fmt, rs)
        out += "</tbody></table>"

        # Setup the second table for error messages
        e_out = "<br><table>"
        h_fmt = "<tr><th>{:<7}</th><th>{:<5}</th><th>{}</th></tr>"
        r_fmt = "<tr><td>{:<7}</td><td>{:<5}</td><td>{}</td></tr>"
        e_out += "\n" + h_fmt.format("Cavity", "Run#", "Error_Message")
        no_errors = True
        for name in sorted(rs.epics_names):
            for rn, error in enumerate(rs.errors[name]):
                if error != "":
                    e_out += r_fmt.format(name, rn + 1, error)
                    no_errors = False

        e_out += "</table>\n"

        if no_errors:
            out += "<p><strong>No Errors</strong></p>\n"
        else:
            out += e_out

        return out

    @staticmethod
    def opening_html():
        out = """
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
        return out

    @staticmethod
    def closing_html():
        out = """</body></html>"""
        return out

    @staticmethod
    def generate_plot_html(rs: ResultSet):
        # Include the plots in the message.
        out = "<h1>Plots</h1>"
        for name in sorted(rs.epics_names):
            out += f"<h2>{name}</h2>"
            cav_images = rs.img_buffers[name]
            for run_num, img_buf in enumerate(cav_images):
                if img_buf is None:
                    out += f"<br><p>{name} run {run_num + 1} experienced"
                    out += f" an error.</p>"
                    out += f"<p>{rs.errors[name][run_num]}</p>\n"
                else:
                    # Make sure that we start at the front of the buffer
                    img_buf.seek(0)
                    data_string = base64.b64encode(img_buf.read()).decode()
                    uri = f'data:image/png;charset=UTF-8;base64,{data_string}'
                    out += f'<img width="400" height="200" '
                    out += f'src="{uri}"/>\n'

        return out
