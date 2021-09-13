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

    def to_table_string(self):
        """Return a table formatted human readable string."""
        t_fmt = "{:<7} {:>10} {:>10} {:>13} {:>13} {:>10}\n"

        out = t_fmt.format("Cavity", "TDOFF_New", "TDOFF_Cur", "TDOFF_Err_Avg",
                           "TDOFF_Err_Std", "N_Good")
        for name in sorted(self.epics_names):
            avg, std, n_suc = self.get_tdoff_stats(name)

            tdoff = round(self.tdoffs[name], 3)
            tdoff_new = round(self.tdoffs[name] + avg, 3)
            out += t_fmt.format(name, tdoff_new, tdoff, avg, std, n_suc)

        t_fmt = "{:<7} {:<5}  {}\n"
        e_out = "\n" + t_fmt.format("Cavity", "Run#", "Error_Message")
        no_errors = True
        for name in sorted(self.epics_names):
            for rn, error in enumerate(self.errors[name]):
                e_out += t_fmt.format(name, rn + 1, error)
                no_errors = False

        if no_errors:
            out += "\nNo Errors"
        else:
            out += e_out

        return out

    def to_html_table_string(self):
        """Return a html table of the data."""

        out = "<table>"
        h_fmt = "<tr><th>{:<7}</th><th>{:>10}</th><th>{:>10}</th>"
        h_fmt += "<th>{:>13}</th><th>{:>13}</th><th>{:>10}</th></tr>\n"
        r_fmt = "<tr><td>{:<7}</td><td>{:>10}</td><td>{:>10}</td>"
        r_fmt += "<td>{:>13}</td><td>{:>13}</td><td>{:>10}</td></tr>\n"

        # Set up the first summary table
        out += "<thead>"
        out += h_fmt.format("Cavity", "TDOFF_New", "TDOFF_Cur", "TDOFF_Err_Avg",
                            "TDOFF_Err_Std", "N_Good")
        out += "</thead><tbody>"
        for name in sorted(self.epics_names):
            avg, std, n_suc = self.get_tdoff_stats(name)

            tdoff = round(self.tdoffs[name], 3)
            tdoff_new = round(self.tdoffs[name] + avg, 3)
            out += r_fmt.format(name, tdoff_new, tdoff, avg, std, n_suc)

        # Setup the second table for error messages
        e_out = "</tbody></table><br><table>"
        h_fmt = "<tr><th>{:<7}</th><th>{:<5}</th><th>{}</th></tr>"
        r_fmt = "<tr><td>{:<7}</td><th>{:<5}</th><td>{}</td></tr>"
        e_out += "\n" + h_fmt.format("Cavity", "Run#", "Error_Message")
        no_errors = True
        for name in sorted(self.epics_names):
            for rn, error in enumerate(self.errors[name]):
                e_out += r_fmt.format(name, rn + 1, error)
                no_errors = False

        e_out += "</table>\n"

        if no_errors:
            out += "<p><strong>No Errors</strong></p>\n"
        else:
            out += e_out

        return out
