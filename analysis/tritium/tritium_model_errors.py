from libra_toolbox.tritium.model import ureg, Model, quantity_to_activity
import numpy as np
import json
from libra_toolbox.tritium.lsc_measurements import (
    LIBRARun,
    LSCFileReader,
    GasStream,
    LSCSample,
    LIBRASample,
)

import warnings
from scipy.interpolate import interp1d
from datetime import datetime


all_file_readers = []
all_quench = []
# Dictionary to store 2sigma errors for each sample
# Key: sample label, Value: 2sigma error in Bq
all_sample_errors = {}

# Background uncertainty (2sigma as a fraction)
BACKGROUND_UNCERTAINTY_PERCENT = 5.0  # 20% 2sigma uncertainty on background

# Confidence interval conversion factors (multiply 1-sigma by these values)
# All internal calculations use 2sigma (95.45%), then convert to desired confidence level
CONFIDENCE_INTERVALS = {
    "68.3%": 1.0,  # 1-sigma
    "90%": 1.645,  # 1.645-sigma
    "95%": 1.96,  # 1.96-sigma (commonly used)
    "95.45%": 2.0,  # 2-sigma (internal storage)
    "99%": 2.576,  # 2.576-sigma
    "99.7%": 3.0,  # 3-sigma
}

# Set your desired confidence interval here
DESIRED_CONFIDENCE_INTERVAL = (
    "95.45%"  # Change this to adjust error bar confidence level
)


def create_sample(label: str, filename: str, background_curve=None) -> LSCSample:
    """
    Create a LSCSample from a LSC file with background substracted.
    Also reads and stores the 2sigma% error from the CSV file.

    Args:
        label: the label of the sample in the LSC file
        filename: the filename of the LSC file

    Returns:
        the LSCSample object
    """
    is_none_sample = filename == "NONE" or filename.endswith("/NONE")
    if is_none_sample:
        label_none = "NONE"
        if "/" in filename:
            filename = f"{filename.rsplit('/', 1)[0]}/NO_SAMPLE.csv"
        else:
            filename = "NO_SAMPLE.csv"

    # check if a LSCFileReader has been created for this filename
    found = False
    for file_reader in all_file_readers:
        if file_reader.filename == filename:
            found = True
            break

    # if not, create it and add it to the list of LSCFileReaders
    if not found:
        file_reader = LSCFileReader(filename, labels_column="SMPL_ID")

    file_reader.read_file()

    # create the sample
    if is_none_sample:
        sample = LSCSample.from_file(file_reader, label_none)
        sample_data = get_row_by_label(file_reader, label_none)
        sample.name = label
        sample.background_substracted = True
    else:
        sample = LSCSample.from_file(file_reader, label)
        sample_data = get_row_by_label(file_reader, label)

    # Read the 2sigma% error from the CSV file (this is on the RAW measurement)
    if is_none_sample:
        two_sigma_percent = 0.0
    else:
        two_sigma_percent = float(sample_data["A:2S%"])

    # Calculate the absolute error from the RAW activity (before background subtraction)
    activity_bq_raw = sample.activity.magnitude
    two_sigma_error_bq_raw = activity_bq_raw * two_sigma_percent / 100.0

    # Now perform background subtraction and propagate uncertainties
    background_bq = 0.0  # Will be updated below

    if is_none_sample:
        pass
    elif background_curve:
        tSIE = float(sample_data["tSIE"])
        background_bq = background_curve(tSIE)
        substract_scalar_background(sample, background_bq)
    else:
        # try to find the background sample from the file
        background_labels = ["1L-BL-1", "1L-BL-2", "1L-BL-3", "1L-BL-4"]
        background_sample = None

        for background_label in background_labels:
            try:
                background_sample = LSCSample.from_file(file_reader, background_label)
                break
            except ValueError:
                continue

        if background_sample is None:
            raise ValueError(f"Background sample not found in {filename}")

        # Get background activity before subtraction
        background_bq = background_sample.activity.magnitude

        # substract background
        sample.substract_background(background_sample)

    # Propagate uncertainties: when subtracting, add in quadrature
    # σ_net = √(σ_raw² + σ_background²)
    if is_none_sample:
        two_sigma_error_bq_net = 0.0
    else:
        two_sigma_error_bq_background = (
            background_bq * BACKGROUND_UNCERTAINTY_PERCENT / 100.0
        )
        two_sigma_error_bq_net = np.sqrt(
            two_sigma_error_bq_raw**2 + two_sigma_error_bq_background**2
        )

    # Store the final error (after background subtraction) in the global dictionary
    all_sample_errors[label] = two_sigma_error_bq_net

    # read quench set
    all_quench.append(file_reader.quench_set)

    return sample


def get_row_by_label(reader: LSCFileReader, label: str) -> dict:
    if reader.data is None:
        raise ValueError("Data not loaded. Call reader.read_file() first.")
    if reader.labels_column is None:
        raise ValueError("labels_column is not set in reader.")
    row = reader.data[reader.data[reader.labels_column] == label]
    if row.empty:
        raise ValueError(f"Label '{label}' not found in data.")
    return row.iloc[0].to_dict()


def substract_scalar_background(sample: LSCSample, background_bq: float) -> None:
    if sample.background_substracted:
        raise ValueError("Background already substracted")
    sample.activity -= background_bq * ureg.Bq
    if sample.activity.magnitude < 0.07:  # MDA Correction
        warnings.warn(
            f"Activity of {sample.name} is negative after substracting background. Setting to zero."
        )
        sample.activity = sample.activity  # 0 * ureg.Bq
    sample.background_substracted = True


def build_background_curve_from_file(reader: LSCFileReader, blank_labels: list[str]):
    tSIE_values = []
    Bq_values = []

    for label in blank_labels:
        sample_data = get_row_by_label(reader, label)
        tSIE = float(sample_data["tSIE"])
        sample = LSCSample.from_file(reader, label)
        Bq = sample.activity.magnitude
        tSIE_values.append(tSIE)
        Bq_values.append(Bq)

    interpolator = interp1d(
        tSIE_values, Bq_values, bounds_error=False, fill_value="extrapolate"
    )
    return interpolator


lsc_data_folder = "../../data/tritium_detection"
with open("../../data/general.json", "r") as f:
    general_data = json.load(f)

run_nb = general_data["general_data"]["run_nb"]

if "tritium_blank_set" in general_data:
    curved_bkgr = True
    blank_info = general_data["tritium_blank_set"]
    background_file = f"{lsc_data_folder}/{blank_info['filename']}"
    background_reader = LSCFileReader(background_file, labels_column="SMPL_ID")
    background_reader.read_file()

    blank_labels = list(blank_info["blanks"].keys())
    background_curve = build_background_curve_from_file(background_reader, blank_labels)
else:
    curved_bkgr = False
    background_curve = None

# read start time from general.json
all_start_times = []
for generator in general_data["generators"]:
    if generator["enabled"] is False:
        continue
    for irradiation_period in generator["periods"]:
        start_time = datetime.strptime(irradiation_period["start"], "%m/%d/%Y %H:%M")
        all_start_times.append(start_time)
start_time = min(all_start_times)


# create gas streams
gas_streams = {}
for stream, samples in general_data["tritium_detection"].items():
    stream_samples = []
    for sample_nb, sample_dict in samples.items():
        libra_samples = []
        if sample_dict["actual_sample_time"] is None:
            continue
        for vial_nb, filename in sample_dict["lsc_vials_filenames"].items():
            sample = create_sample(
                label=f"1L-FLB-{stream}_{run_nb}-{sample_nb}-{vial_nb}",
                filename=f"{lsc_data_folder}/{filename}",
                background_curve=background_curve,
            )
            libra_samples.append(sample)

        time_sample = datetime.strptime(
            sample_dict["actual_sample_time"], "%m/%d/%Y %H:%M"
        )
        stream_samples.append(LIBRASample(libra_samples, time=time_sample))
    gas_streams[stream] = GasStream(stream_samples, start_time=start_time)


# create run
run = LIBRARun(streams=list(gas_streams.values()), start_time=start_time)

# check that only one quench set is used
assert len(np.unique(all_quench)) == 1

# check that background is always substracted  # TODO this should be done automatically in LIBRARun
for stream in run.streams:
    for sample in stream.samples:
        for lsc_vial in sample.samples:
            assert lsc_vial.background_substracted, (
                f"Background not substracted for {sample}"
            )

IV_stream = gas_streams["IV"]
OV_stream = gas_streams["OV"]


# ============================================================================
# ERROR PROPAGATION CLASSES AND METHODS
# ============================================================================


class GasStreamWithErrors:
    """
    Wrapper class for GasStream that adds error tracking and propagation.
    """

    def __init__(self, gas_stream: GasStream, stream_label: str):
        self.gas_stream = gas_stream
        self.stream_label = stream_label
        self._error_cache = {}

    def get_cumulative_activity(self, form: str = "total"):
        """
        Get cumulative activity (same as original GasStream method).

        Args:
            form: "total", "soluble", or "insoluble"

        Returns:
            Cumulative activity as a pint Quantity array
        """
        return self.gas_stream.get_cumulative_activity(form)

    def get_cumulative_activity_errors(self, form: str = "total"):
        """
        Calculate errors for cumulative activity at each sampling point.

        For cumulative sums, errors propagate in quadrature:
        σ_sum = sqrt(σ₁² + σ₂² + ... + σₙ²)

        Individual samples are stored with 2sigma errors, which are converted to
        the desired confidence interval set by DESIRED_CONFIDENCE_INTERVAL.

        Args:
            form: "total", "soluble", or "insoluble"

        Returns:
            Array of errors in Bq for each cumulative point at the desired confidence level
        """
        cache_key = f"{form}_{DESIRED_CONFIDENCE_INTERVAL}"
        if cache_key in self._error_cache:
            return self._error_cache[cache_key]

        # Calculate conversion factor from 2sigma to desired confidence interval
        # All stored errors are 2sigma, so convert: 1sigma = 2sigma / 2
        # Then multiply by the desired confidence factor
        two_sigma_factor = CONFIDENCE_INTERVALS["95.45%"]  # = 2.0
        desired_factor = CONFIDENCE_INTERVALS[DESIRED_CONFIDENCE_INTERVAL]
        conversion_factor = desired_factor / two_sigma_factor

        cumulative_errors = []
        squared_errors_sum = 0.0

        # Iterate through all samples in the stream
        for libra_sample in self.gas_stream.samples:
            # Each LIBRASample contains multiple LSCSample vials
            # First two vials (indices 0, 1) are soluble
            # Third and fourth vials (indices 2, 3) are insoluble
            for vial_index, lsc_sample in enumerate(libra_sample.samples):
                # Determine if this vial should be included based on form
                include_vial = False
                if form == "total":
                    include_vial = True
                elif form == "soluble" and vial_index in [0, 1]:
                    include_vial = True
                elif form == "insoluble" and vial_index in [2, 3]:
                    include_vial = True
                
                if not include_vial:
                    continue
                
                # Find the error for this specific sample
                sample_label = lsc_sample.name
                if sample_label in all_sample_errors:
                    error_bq_2sigma = all_sample_errors[sample_label]
                    # Convert from 2sigma to desired confidence interval
                    error_bq_desired = error_bq_2sigma * conversion_factor
                    squared_errors_sum += error_bq_desired**2
                else:
                    warnings.warn(
                        f"No error found for sample {sample_label}, assuming 0"
                    )

            # After processing all vials in this LIBRA sample, compute cumulative error
            cumulative_error = np.sqrt(squared_errors_sum)
            cumulative_errors.append(cumulative_error)

        cumulative_errors = np.array(cumulative_errors)
        self._error_cache[cache_key] = cumulative_errors

        return cumulative_errors

    @property
    def samples(self):
        return self.gas_stream.samples

    @property
    def relative_times_as_pint(self):
        return self.gas_stream.relative_times_as_pint


# Create error-aware wrappers for the gas streams
IV_stream_with_errors = GasStreamWithErrors(IV_stream, "IV")
OV_stream_with_errors = GasStreamWithErrors(OV_stream, "OV")

# Also create a combined wrapper dictionary for convenience
gas_streams_with_errors = {
    "IV": IV_stream_with_errors,
    "OV": OV_stream_with_errors,
}


sampling_times = {
    "IV": sorted(IV_stream.relative_times_as_pint),
    "OV": sorted(OV_stream.relative_times_as_pint),
}

replacement_times_top = sampling_times["IV"]
replacement_times_walls = sampling_times["OV"]

# read cover gas change times
cover_gas_switch_deltatimes = []
if (
    "switched_to" in general_data["cover_gas"]
    and general_data["cover_gas"]["switched_to"]
):
    switched_to_list = general_data["cover_gas"]["switched_to"]

    # Handle both old format (single dict) and new format (list of dicts)
    if isinstance(switched_to_list, dict):
        switched_to_list = [switched_to_list]

    for gas_switch in switched_to_list:
        if gas_switch.get("gas_switch_time"):
            gas_switch_time = datetime.strptime(
                gas_switch["gas_switch_time"], "%m/%d/%Y %H:%M"
            )
            gas_switch_deltatime = gas_switch_time - start_time
            gas_switch_deltatime = gas_switch_deltatime.total_seconds() * ureg.s
            gas_switch_deltatime = gas_switch_deltatime.to(ureg.day)
            cover_gas_switch_deltatimes.append(gas_switch_deltatime)

# read secondary gas change times
secondary_gas_switch_deltatimes = []
if (
    "switched_to" in general_data["secondary_gas"]
    and general_data["secondary_gas"]["switched_to"]
):
    switched_to_list = general_data["secondary_gas"]["switched_to"]

    # Handle both old format (single dict) and new format (list of dicts)
    if isinstance(switched_to_list, dict):
        switched_to_list = [switched_to_list]

    for gas_switch in switched_to_list:
        if gas_switch.get("gas_switch_time"):
            gas_switch_time = datetime.strptime(
                gas_switch["gas_switch_time"], "%m/%d/%Y %H:%M"
            )
            gas_switch_deltatime = gas_switch_time - start_time
            gas_switch_deltatime = gas_switch_deltatime.total_seconds() * ureg.s
            gas_switch_deltatime = gas_switch_deltatime.to(ureg.day)
            secondary_gas_switch_deltatimes.append(gas_switch_deltatime)

# tritium model

baby_diameter = 14 * ureg.cm  # TODO confirm with CAD
baby_radius = 0.5 * baby_diameter
baby_volume = 1 * ureg.L
baby_cross_section = np.pi * baby_radius**2
baby_height = baby_volume / baby_cross_section

# read irradiation times from general.json

irradiations = []
for generator in general_data["generators"]:
    if generator["enabled"] is False:
        continue
    for irradiation_period in generator["periods"]:
        irr_start_time = (
            datetime.strptime(irradiation_period["start"], "%m/%d/%Y %H:%M")
            - start_time
        )
        irr_stop_time = (
            datetime.strptime(irradiation_period["end"], "%m/%d/%Y %H:%M") - start_time
        )
        irr_start_time = irr_start_time.total_seconds() * ureg.second
        irr_stop_time = irr_stop_time.total_seconds() * ureg.second
        irradiations.append([irr_start_time, irr_stop_time])

# Neutron rate

# TODO replace by neutron rate measured with HPGe
neutron_rate = (
    # 2.628e8 * ureg.neutron * ureg.s**-1
    1.900e9 * ureg.neutron * ureg.s**-1
    # 1.869e09 * ureg.neutron * ureg.s**-1
)  # based on manufacturer test data for generator settings
# neutron_rate_uncertainty = 1.546e08 * ureg.neutron * ureg.s**-1
neutron_rate_uncertainty = 1.567e08 * ureg.neutron * ureg.s**-1
neutron_rate_relative_uncertainty = (neutron_rate_uncertainty / neutron_rate).to(
    ureg.dimensionless
)

# TBR from OpenMC

from pathlib import Path

filename = "../neutron/statepoint.100.h5"
filename = Path(filename)

if not filename.exists():
    raise FileNotFoundError(f"{filename} does not exist, run OpenMC first")

import openmc

sp = openmc.StatePoint(filename)
tally_df = sp.get_tally(name="TBR").get_pandas_dataframe()
calculated_TBR = tally_df["mean"].iloc[0] * ureg.particle * ureg.neutron**-1
calculated_TBR_std_dev = (
    tally_df["std. dev."].iloc[0] * ureg.particle * ureg.neutron**-1
)

# TBR from measurements

total_irradiation_time = sum([irr[1] - irr[0] for irr in irradiations])

T_consumed = neutron_rate * total_irradiation_time

# to calculate the measured TBR we ignore the last samples for which
# we have some contribution from other sources (nGen, cyclotron, etc.)
T_produced_IV = IV_stream.get_cumulative_activity("total")[-1]
T_produced_OV = OV_stream.get_cumulative_activity("total")[-1]

T_produced_IV_error = IV_stream_with_errors.get_cumulative_activity_errors("total")[-1]
T_produced_OV_error = OV_stream_with_errors.get_cumulative_activity_errors("total")[-1]

print(T_produced_IV, T_produced_OV)
print(T_produced_IV_error, T_produced_OV_error)

measured_TBR = ((T_produced_IV + T_produced_OV) / quantity_to_activity(T_consumed)).to(
    ureg.particle * ureg.neutron**-1
)

# Calculate total tritium produced and its error (sum in quadrature)
T_produced_total = T_produced_IV + T_produced_OV
T_produced_total_error = np.sqrt(T_produced_IV_error**2 + T_produced_OV_error**2)

# For measured_TBR = T_produced / T_consumed, propagate errors in quadrature
# Relative error: σ_TBR/TBR = sqrt((σ_T_produced/T_produced)² + (σ_T_consumed/T_consumed)²)
measured_TBR_error = measured_TBR * np.sqrt(
    (T_produced_total_error / T_produced_total.magnitude) ** 2
    + neutron_rate_relative_uncertainty.magnitude ** 2
)

# Run 1 transport coeff and measured TBR for overlay
optimised_ratio = 0
k_top = 2.5 * 12 * 1.45 * 8.9e-8 * ureg.m * ureg.s**-1
k_wall = optimised_ratio * k_top

baby_model = Model(
    radius=baby_radius,
    height=baby_height,
    TBR=calculated_TBR,
    neutron_rate=neutron_rate,
    irradiations=irradiations,
    k_top=k_top,
    k_wall=k_wall,
)


# store processed data
processed_data = {
    "modelled_baby_radius": {
        "value": baby_radius.magnitude,
        "unit": str(baby_radius.units),
    },
    "modelled_baby_height": {
        "value": baby_height.magnitude,
        "unit": str(baby_height.units),
    },
    "irradiations": [
        {
            "start_time": {
                "value": irr[0].magnitude,
                "unit": str(irr[0].units),
            },
            "stop_time": {
                "value": irr[1].magnitude,
                "unit": str(irr[1].units),
            },
        }
        for irr in irradiations
    ],
    "neutron_rate_used_in_model": {
        "value": baby_model.neutron_rate.magnitude,
        "unit": str(baby_model.neutron_rate.units),
    },
    "measured_TBR": {
        "value": measured_TBR.magnitude,
        "unit": str(measured_TBR.units),
    },
    "measured_TBR_error": {
        "value": measured_TBR_error.magnitude,
        "unit": str(measured_TBR_error.units),
    },
    "TBR_used_in_model": {
        "value": baby_model.TBR.magnitude,
        "unit": str(baby_model.TBR.units),
    },
    "k_top": {
        "value": baby_model.k_top.magnitude,
        "unit": str(baby_model.k_top.units),
    },
    "k_wall": {
        "value": baby_model.k_wall.magnitude,
        "unit": str(baby_model.k_wall.units),
    },
    "cumulative_tritium_release": {
        label: {
            **{
                form: {
                    "value": gas_stream.get_cumulative_activity(
                        form
                    ).magnitude.tolist(),
                    "unit": str(gas_stream.get_cumulative_activity(form).units),
                }
                for form in ["total", "soluble", "insoluble"]
            },
            "sampling_times": {
                "value": gas_stream.relative_times_as_pint.magnitude.tolist(),
                "unit": str(gas_stream.relative_times_as_pint.units),
            },
        }
        for label, gas_stream in gas_streams.items()
    },
}

# check if the file exists and load it

processed_data_file = "../../data/processed_data.json"

try:
    with open(processed_data_file, "r") as f:
        existing_data = json.load(f)
except FileNotFoundError:
    print(f"Processed data file not found, creating it in {processed_data_file}")
    existing_data = {}

existing_data.update(processed_data)

with open(processed_data_file, "w") as f:
    json.dump(existing_data, f, indent=4)

print(f"Processed data stored in {processed_data_file}")
