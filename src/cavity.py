import epics


class Cavity:

    def __init__(self, epics_name):
        self.epics_name = epics_name
        self.rf_on = epics.PV(f"{epics_name}RFONr")
        self.stat1 = epics.PV(f"{epics_name}STAT1")
        self.tdoff = epics.PV(f"{epics_name}TDOFF")
        self.deta2 = epics.PV(f"{epics_name}WFSDETA2", auto_monitor=False)
        self.crfp = epics.PV(f"{epics_name}WFSCRFP", auto_monitor=False)

        self.pvs = []
        self.pvs.append(self.rf_on)
        self.pvs.append(self.stat1)
        self.pvs.append(self.tdoff)
        self.pvs.append(self.deta2)
        self.pvs.append(self.crfp)

        for pv in self.pvs:
            if not pv.wait_for_connection(timeout=2):
                raise RuntimeError(f"Could not connect to PV '{pv.pvname}'")

    def is_gradient_ramping(self):
        """Check if the cavity is currently ramping gradient."""
        # If the cavity is ramping is saved as the 11th bit in the
        # R...STAT1 PV
        value = self.stat1.value

        if value is None:
            raise RuntimeError(f"Error retrieving PV '{self.stat1.pvname}'")

        # We're ramping if the bit is not 0
        is_ramping = int(value) & 0x0800 > 0

        return is_ramping

    def is_rf_on(self):
        """Check if the cavity currently has RF on."""
        value = self.rf_on.value

        if value is None:
            raise RuntimeError(f"Error retrieving PV '{self.rf_on.pvname}'")

        is_on = value == 1
        return is_on

    def is_stable_running(self):
        not_ramping = not self.is_gradient_ramping()
        rf_on = self.is_rf_on()
        is_stable = not_ramping and rf_on
        return is_stable

    def get_waveforms(self):
        deta2 = self.deta2.get(use_monitor=False)
        crfp = self.crfp.get(use_monitor=False)
        if deta2 is None:
            raise RuntimeError("Error getting DETA2 waveform")
        if crfp is None:
            raise RuntimeError("Error getting CRFP waveform")

        return deta2, crfp

    def get_tdoff(self):
        value = self.tdoff.value
        if value is None:
            raise RuntimeError(f"Error retrieving PV '{self.rf_on.pvname}'")

        return value
