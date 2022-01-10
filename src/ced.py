"""This module contains any network specific code.  Currently defines single class for accessing system trust store."""
from typing import List, Dict

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.ssl_ import create_urllib3_context


class SSLContextAdapter(HTTPAdapter):
    """An HTTPAdapter that loads the default system SSL trust store

    This is needed since the requests module ships with its own CA cert store that does not include the JLab PKI"""

    def init_poolmanager(self, *args, **kwargs):
        """Overrides the parent method to include call to load_default_certs()"""
        context = create_urllib3_context()
        kwargs['ssl_context'] = context
        context.load_default_certs()  # this loads the OS default trusted CA certs
        return super(SSLContextAdapter, self).init_poolmanager(*args, **kwargs)


class CED:
    """A class for managing queries with a given CED instance"""

    def __init__(self, ced_server="ced.acc.jlab.org", ced_instance='ced', ced_workspace="ops"):
        self.ced_server = ced_server
        self.ced_instance = ced_instance
        self.ced_workspace = ced_workspace

    def query_inventory(self, element_type: str = None, properties: List[str] = None):
        """Query the CED inventory with an option element type filter.

        Args:
            element_type:  A specific element type to request
            properties:  Which properties to include in response
        """

        if element_type is None:
            inventory_url = f"http://{self.ced_server}/inventory"
        else:
            inventory_url = f"http://{self.ced_server}/inventory/{element_type}"

        request_params = {'ced': self.ced_instance, 'workspace': self.ced_workspace, 'out': 'json'}
        request_params['p']: properties

        return CED._get_ced_elements(inventory_url, request_params)

    @staticmethod
    def _get_ced_elements(url: str, parameters) -> List[Dict]:
        """Queries the CED with the supplied URL and parameter dictionary.  parameters MUST include out=json."""

        with requests.Session() as s:
            adapter = SSLContextAdapter()
            s.mount(url, adapter)
            r = s.get(url, params=parameters)

        if r.status_code != 200:
            raise ValueError(
                "Received error response from {}.  status_code={}.  response={}".format(ced_url, r.status_code,
                                                                                        r.text))

        # The built-in JSON decoder will raise a ValueError if parsing non-JSON content
        out = r.json()
        if out['stat'] != 'ok':
            raise ValueError("Received non-ok status response")

        return out['Inventory']['elements']
