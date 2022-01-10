import time
import epics
import threading
from datetime import datetime
from contextlib import contextmanager
from epics.ca import ChannelAccessException

from ced import CED


class Cavity:

    @classmethod
    def get_cavity(cls, epics_name, cavity_type):
        supported_types = ('C75', 'C100', 'P1R')
        if cavity_type in ('C75',):
            cavity = C75Cavity(epics_name)
        elif cavity_type in ('C100', 'P1R'):
            cavity = C100Cavity(epics_name)
        else:
            raise ValueError(f"{epics_name}: Unsupported cavity type '{cavity_type}'.  Supported types: "
                             f"{supported_types}.")

        return cavity

    def _data_ready_cb(self, seq_start_val, seq_end_val):
        # Track if we have reached the read step of the sequencer.
        scope_reached_read_step = False

        def data_ready_cb(pvname=None, value=None, char_value=None, timestamp=None, **kwargs):
            """This should only be used to monitor the scope sequencer PV to see when the FCC has updated data."""

            nonlocal scope_reached_read_step
            with self.data_ready_lock:

                # Handle the first time through.  We won't do anything later in this method since old and current values
                # are equal.
                if value >= seq_start_val:
                    scope_reached_read_step = True
                else:
                    scope_reached_read_step = False

                # Update the appropriate window end point.
                if value == seq_start_val:
                    # 2048 is when the FPGA is done collecting the data.
                    self.window_start = timestamp
                    self.window_end = None
                elif value == seq_end_val:
                    # 8192 is when the IOC is done processing all of the waveform records
                    self.window_end = timestamp

                # This makes sure that we should have data, and that we have a window start timestamp as we saw the read
                # step.
                if (value == seq_end_val and scope_reached_read_step and self.window_start is not None
                        and self.window_end is not None):
                    self.data_ready = True
                else:
                    self.data_ready = False

        return data_ready_cb

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

        # The data_ready flag will be updated from callbacks and work thread.  Indicates that the data set is ready to
        # harvest.
        self.data_ready_lock = threading.Lock()
        self.data_ready = False

        # These will be float type timestamps indicating the acceptable start and end times of the waveforms for them
        # to be a valid set.  Access should be synchronized using data_ready_lock.
        self.window_start = None
        self.window_end = None

        # This should be a PV eventually that is defined in Child classes.  Each cavity type uses it's own PV, but the
        # basic logic for getting waveforms is the same so I've captured that in a common method which checks for this
        # PV.
        self.scope_seq_step = None

        # Track the PVs used.
        self.pvs = []
        self.pvs.append(self.rf_on)
        self.pvs.append(self.stat1)
        self.pvs.append(self.cntl2mode)
        self.pvs.append(self.tdoff)
        self.pvs.append(self.deta2)
        self.pvs.append(self.crfp)

    def _wait_for_pvs_to_connect(self):
        # Wait for things to connect.  If the IOC isn't available at the start, raise an exception for the worker thread
        # to handle.
        for pv in self.pvs:
            if not pv.wait_for_connection(timeout=2):
                raise RuntimeError(f"Could not connect to PV '{pv.pvname}'")

    def is_gradient_ramping(self):
        """Check if the cavity is currently ramping gradient."""
        # If the cavity is ramping is saved as the 11th bit in the
        # R...STAT1 PV
        value = self._get_pv(self.stat1)

        # We're ramping if the bit is not 0
        is_ramping = int(value) & 0x0800 > 0

        return is_ramping

    def is_rf_on(self):
        """Check if the cavity currently has RF on."""
        value = self._get_pv(self.rf_on)
        is_on = value == 1
        return is_on

    def is_valid_control_mode(self):
        """Check that the cavity is in a valid control mode for this measurement."""
        value = self._get_pv(self.cntl2mode)
        valid = value == 4 or value == 64
        return valid

    def is_stable_running(self):
        not_ramping = not self.is_gradient_ramping()
        rf_on = self.is_rf_on()
        valid_mode = self.is_valid_control_mode()

        is_stable = all((not_ramping, rf_on, valid_mode))
        return is_stable

    # TODO: Figure out why no multiple samples
    def get_waveforms(self, timeout=20, sleep_dur=0.01):
        """Waits for the FCC to have reported data is ready, then grabs those waveforms.  Checks for valid timestamps"""

        # This is like an assert, so I've probably gone wrong somewhere in my design.  Still we need to check since the
        # base class does not have an actual PV defined.
        if self.scope_seq_step is None:
            raise RuntimeError(f"{self.epics_name}: Scope sequencer PV has not been initialized.")

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
                    deta2 = self._get_pv(self.deta2, use_monitor=False)
                    crfp = self._get_pv(self.crfp, use_monitor=False)

                    # Make sure that they look like a synchronous grouping.  These should throw if not.
                    self._pv_in_window(self.deta2)
                    self._pv_in_window(self.crfp)

                    # Exit the infinite loop now that we have our data.  Flip the flag to false so we don't try to get
                    # more data until the sequencer says we're ready.  It's possible we process faster than the IOC
                    # sequencer.
                    self.data_ready = False
                    break

            # Sleep for a little bit before we check again if data is ready.
            time.sleep(sleep_dur)
            count += 1
            if count * sleep_dur > timeout:
                raise RuntimeError(f"{self.epics_name}: Timed out waiting for good data. (> {timeout}s)")

        return deta2, crfp

    def _pv_in_window(self, pv):
        """Check that the provided timestamp is within the acquisition window.  Raise exception if not.

        Should be called within a 'data_ready_lock'ed context.  Also raises if we have an invalid window
        """
        if self.window_end < self.window_start:
            raise RuntimeError(f"{self.epics_name}: Invalid data acquisition window")
        if not self.window_start <= pv.timestamp <= self.window_end:
            raise RuntimeError(f"{self.epics_name}: {pv.pvname} timestamp ({pv.timestamp}) outside acquisition window "
                               f"({self.window_start}, {self.window_end}).")

    def get_tdoff(self):
        return self._get_pv(self.tdoff)

    @staticmethod
    def _get_pv(pv, **kwargs):
        """Get the current value of the PV and raise an exception if it is None.  kwargs passed to PV.get()"""
        value = pv.get(**kwargs)
        if value is None:
            raise RuntimeError(f"Error retrieving PV value '{pv.pvname}")
        return value


class C100Cavity(Cavity):

    def __init__(self, epics_name):
        """Initialize a Cavity object and connect to it's PVs.  Exception raised if unable to connect."""

        super(C100Cavity, self).__init__(epics_name=epics_name)

        # Control the waveform mode.  Not sure why we have two separate PVs, but this is the API.
        self.scope_setting = epics.PV(f"{epics_name}WFSCOPrun")  # 0 = No scope, 1 = stop, 2 = single, 3 = periodic
        self.harvester_setting = epics.PV(f"{epics_name}WFTRIPrun")  # 0 = No harvester, 1 = harvester
        self.sample_rate = epics.PV(f"{epics_name}TRGS1")  # Sample interval in milliseconds

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
        self.scope_seq_step = epics.PV(f"{epics_name}WFSCOPstp", form='time',
                                       callback=self._data_ready_cb(seq_start_val=2048, seq_end_val=8192))

        # Track the PVs used.
        self.pvs.append(self.scope_seq_step)
        self.pvs.append(self.scope_setting)
        self.pvs.append(self.harvester_setting)
        self.pvs.append(self.sample_rate)

        self._wait_for_pvs_to_connect()

        if not self.scope_setting.write_access:
            raise ChannelAccessException(f"User lacks write permission for {self.scope_setting.pvname}")
        if not self.harvester_setting.write_access:
            raise ChannelAccessException(f"User lacks write permission for {self.harvester_setting.pvname}")
        if not self.sample_rate.write_access:
            raise ChannelAccessException(f"User lacks write permission for {self.sample_rate.pvname}")


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


class C75Cavity(Cavity):

    def __init__(self, epics_name):
        """Initialize a Cavity object and connect to it's PVs.  Exception raised if unable to connect."""
        super(C75Cavity, self).__init__(epics_name)

        # Control the waveform mode.  Not sure why we have two separate PVs, but this is the API.
        # TODO: Verify the 0/1 meaning and values
        self.scope_setting = epics.PV(f"{epics_name}WFSrunCmd")  # 0 = Scope off, 1 = Scope on

        # PV controlling how long the interval between samples will be in milliseconds
        self.sample_rate = epics.PV(f"{epics_name}WAVSTM")

        # This PV shows the progress of the C75 sequencer associated with processing periodic data collection.
        # Info taken from EDM screen on 2021-11-04.  Typical sequence goes 1, 2, 8, 48 (16+32)
        # 1 - Wait for Run Command
        # 2 - Wait for Waveform Data
        # 4 - No Waveform data
        # 8 - Process Waveform Data
        # 16 - Done processing
        # 32 - Loop Delay
        self.scope_seq_step = epics.PV(f"{epics_name}WFSstep", form='time',
                                       callback=self._data_ready_cb(seq_start_val=8, seq_end_val=48))

        self.pvs.append(self.scope_seq_step)
        self.pvs.append(self.scope_setting)
        self.pvs.append(self.sample_rate)
        self._wait_for_pvs_to_connect()

        if not self.scope_setting.write_access:
            raise ChannelAccessException(f"User lacks write permission for {self.scope_setting.pvname}")
        if not self.sample_rate.write_access:
            raise ChannelAccessException(f"User lacks write permission for {self.sample_rate.pvname}")


    @contextmanager
    def scope_mode(self, mode=1):
        """Allows convenient flip to scope mode via context manager.  Restores original values on exiting context."""


        # Cache the values so we can restore
        old_scope = self.scope_setting.get()

        # TODO: C75 has dual buffer, so we don't need to toggle harvester/scope mode.
        # TODO: Include sample interval at 0.2ms for both C75 and C100

        # Put the cavity into scope mode when called with 'with'.  Make sure we disable the first mode before updating
        # the second.
        try:
            self.scope_setting.put(mode, wait=True)

            # I'm not sure why, but I don't get an exception if I don't have write access to these settings.
            # Maybe there is a better way to handle this check, but suffices.
            if self.scope_setting.get(use_monitor=False) != mode:
                raise RuntimeError(f"{self.epics_name}: Failed to enter scope mode '{mode}'")

            # yield so we can run statements from body of 'with' statement with cavity in scope mode.
            yield

        finally:
            # When that context exits, we put it back in harvester mode.  Even if there was an exception.
            self.scope_setting.put(old_scope, wait=True)
