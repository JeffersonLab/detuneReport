import time
import epics
import threading
from datetime import datetime
from contextlib import contextmanager


class Cavity:
    class_lock = threading.Lock()
    cavities = {}

    @classmethod
    def get_cavity(cls, epics_name):
        with Cavity.class_lock:
            if epics_name in Cavity.cavities.keys():
                cavity = Cavity.cavities[epics_name]
            else:
                cavity = Cavity(epics_name)
                Cavity.cavities[epics_name] = cavity

        return cavity

    def _data_ready_cb(self, pvname=None, value=None, char_value=None, timestamp=None, **kwargs):
        """This callback should only be used to monitor the R...STAT2b.B3 PV to see when the FCC has updated data."""

        with self.data_ready_lock:

            # Handle the first time through.  We won't do anything later in this method since old and current values are
            # equal.
            if value >= 2048:
                self.scope_reached_read_step = True
            else:
                self.scope_reached_read_step = False

            # Update the appropriate window end point.
            if value == 2048:
                # 2048 is when the FPGA is done collecting the data.
                self.window_start = timestamp
            elif value == 8192:
                # 8192 is when the IOC is done processing all of the waveform records
                self.window_end = timestamp

            # This makes sure that we should have data, and that we have a window start timestamp as we saw the read
            # step.
            if value == 8192 and self.scope_reached_read_step:
                self.data_ready = True
            else:
                self.data_ready = False

    def __init__(self, epics_name):
        """Initialize a Cavity object and connect to it's PVs.  Exception raised if unable to connect."""
        self.epics_name = epics_name

        # Cavity status PVs
        self.rf_on = epics.PV(f"{epics_name}RFONr")  # Is the cavity RF on
        self.stat1 = epics.PV(f"{epics_name}STAT1")  # Is the cavity currently ramping gradient
        self.cntl2mode = epics.PV(f"{epics_name}CNTL2MODE")  # The RF control mode. 4 or 64 == stable operations)

        # Data PVs
        self.tdoff = epics.PV(f"{epics_name}TDOFF")  # The current tuner offset
        self.deta2 = epics.PV(f"{epics_name}WFSDETA2", form='time', auto_monitor=False)  # The detune angle waveform
        self.crfp = epics.PV(f"{epics_name}WFSCRFP", form='time', auto_monitor=False)  # The forward power waveform

        # Control the waveform mode.  Not sure why we have two separate PVs, but this is the API.
        self.scope_setting = epics.PV(f"{epics_name}WFSCOPrun")  # 0 = No scope, 1 = stop, 2 = single, 3 = periodic
        self.harvester_setting = epics.PV(f"{epics_name}WFTRIPrun")  # 0 = No harvester, 1 = harvester

        # This PV shows the progress of the sequencer associated with processing periodic data collection.
        # Info taken from EDM screen on 2021-10-25.  For periodic, we see 8, 64, 512, 2048, 4096, 8192, and 1024 (error)
        # 1 - reboot/init
        # 2 - Begin SINGLE mode
        # 4 - Begin RUN mode
        # 8 - Begin PERIODIC mode
        # 16 - Set take data for SINGLE
        # 32 - Set take data for RUN
        # 64 - Set take data for PERIODIC
        # 128 - Done taking data for SINGLE
        # 256 - Done taking data for RUN
        # 512 - Done taking data for PERIODIC
        # 1024 - Timeout for done taking data for PERIODIC
        # 2048 - Took the semaphore, now read the scope mode buffers
        # 4096 - Reading all 17 buffers
        # 8192 - Done reading buffers (give back semaphore)
        # 16384 - Abort
        self.scope_seq_step = epics.PV(f"{epics_name}WFSCOPstp", form='time', callback=self._data_ready_cb)
        self.scope_reached_read_step = False  # Is the sequencer currently at or past the read step (2048) this cycle.

        # The data_ready flag will be updated from callbacks and work thread.  Indicates that the data set is ready to
        # harvest.
        self.data_ready_lock = threading.Lock()
        self.data_ready = False

        # These will be float type timestamps indicating the acceptable start and end times of the waveforms for them
        # to be a valid set.  Access should be synchronized using data_ready_lock.
        self.window_start = None
        self.window_end = None

        # Track the PVs used.
        self.pvs = []
        self.pvs.append(self.rf_on)
        self.pvs.append(self.stat1)
        self.pvs.append(self.cntl2mode)
        self.pvs.append(self.scope_seq_step)
        self.pvs.append(self.tdoff)
        self.pvs.append(self.deta2)
        self.pvs.append(self.crfp)
        self.pvs.append(self.scope_setting)
        self.pvs.append(self.harvester_setting)

        # Wait for things to connect.  If the IOC isn't available at the start, raise an exception for the worker thread
        # to handle.
        for pv in self.pvs:
            if not pv.wait_for_connection(timeout=2):
                raise RuntimeError(f"Could not connect to PV '{pv.pvname}'")

    def is_gradient_ramping(self):
        """Check if the cavity is currently ramping gradient."""
        # If the cavity is ramping is saved as the 11th bit in the
        # R...STAT1 PV
        value = self.__get_pv(self.stat1)

        # We're ramping if the bit is not 0
        is_ramping = int(value) & 0x0800 > 0

        return is_ramping

    def is_rf_on(self):
        """Check if the cavity currently has RF on."""
        value = self.__get_pv(self.rf_on)
        is_on = value == 1
        return is_on

    def is_valid_control_mode(self):
        """Check that the cavity is in a valid control mode for this measurement."""
        value = self.__get_pv(self.cntl2mode)
        valid = value == 4 or value == 64
        return valid

    def is_stable_running(self):
        not_ramping = not self.is_gradient_ramping()
        rf_on = self.is_rf_on()
        valid_mode = self.is_valid_control_mode()

        is_stable = all((not_ramping, rf_on, valid_mode))
        return is_stable

    def get_waveforms(self, timeout=20, sleep_dur=0.01):
        """Waits for the FCC to have reported data is ready, then grabs those waveforms.  Checks for valid timestamps"""
        count = 0
        while True:
            # Check that the sequencer-related PV is still connected since that is what drives this whole process.
            if not self.scope_seq_step.connected:
                raise RuntimeError(f"{self.epics_name}: Scope sequencer PV ({self.scope_seq_step.pvname}) "
                                   f"disconnected.")

            # Wait for to be ready, but timeout eventually
            with self.data_ready_lock:
                # Check if the FCC has gathered all of the data we need.  Get it if so.
                if self.data_ready:
                    # Get the waveforms
                    print(f"data ready time: {datetime.now()}")
                    deta2 = self.__get_pv(self.deta2, use_monitor=False)
                    crfp = self.__get_pv(self.crfp, use_monitor=False)

                    # Make sure that they look like a synchronous grouping.  These should throw if not.
                    self.__pv_in_window(self.deta2)
                    self.__pv_in_window(self.crfp)

                    # Exit the infinite loop now that we have our data.
                    break

            # Sleep for a little bit before we check again if data is ready.
            time.sleep(sleep_dur)
            count += 1
            if count * sleep_dur > timeout:
                raise RuntimeError(
                    f"{self.epics_name}: Timed out waiting for good data. (> {timeout}s)")

        return deta2, crfp

    def __pv_in_window(self, pv):
        """Check that the provided timestamp is within the acquisition window.  Raise exception if not.

        Should be called within a 'data_ready_lock'ed context.  Also raises if we have an invalid window
        """
        if self.window_end < self.window_start:
            raise RuntimeError(f"{self.epics_name}: Invalid data acquisition window")
        if not self.window_start <= pv.timestamp <= self.window_end:
            raise RuntimeError(f"{self.epics_name}: {pv.pvname} timestamp ({pv.timestamp}) outside acquisition window "
                               f"({self.window_start}, {self.window_end}).")

    def get_tdoff(self):
        return self.__get_pv(self.tdoff)

    @staticmethod
    def __get_pv(pv, **kwargs):
        """Get the current value of the PV and raise an exception if it is None.  kwargs passed to PV.get()"""
        value = pv.get(**kwargs)
        if value is None:
            raise RuntimeError(f"Error retrieving PV value '{pv.pvname}")
        return value

    @contextmanager
    def scope_mode(self, mode=3):
        """Allows convenient flip to scope mode via context manager.  Restores original values on exiting context."""
        # Cache the values so we can restore
        old_harvester = self.harvester_setting.get()
        old_scope = self.scope_setting.get()

        # Put the cavity into scope mode when called with 'with'.  Make sure we disable the first mode before updating
        # the second.
        try:
            self.harvester_setting.put(0, wait=True)
            self.scope_setting.put(mode, wait=True)

            # I'm not sure why, but I don't get an exception if I don't have write access to these settings.
            # Maybe there is a better way to handle this check, but suffices.
            if self.harvester_setting.get(use_monitor=False) != 0 and self.scope_setting.get(use_monitor=False) != mode:
                raise RuntimeError(f"{self.epics_name}: Failed to enter scope mode '{mode}'")

            # yield so we can run statements from body of 'with' statement with cavity in scope mode.
            yield

        finally:
            # When that context exits, we put it back in harvester mode.  Even if there was an exception.
            self.scope_setting.put(old_scope, wait=True)
            self.harvester_setting.put(old_harvester, wait=True)
