from pathlib import Path
from libra_toolbox.neutron_detection.activation_foils.calibration import (
    CheckSource,
    ba133,
    co60,
    cs137,
    mn54,
    na22,
    ActivationFoil,
    nb93_n2n,
    zr90_n2n,
)
import libra_toolbox.neutron_detection.activation_foils.compass as compass
from libra_toolbox.neutron_detection.activation_foils.compass import (
    Measurement,
    CheckSourceMeasurement,
    SampleMeasurement,
)
from libra_toolbox.tritium.model import ureg
from datetime import date, datetime
import json
from zoneinfo import ZoneInfo
from download_raw_foil_data import download_and_extract_foil_data
import copy
import numpy as np

#####################################################################
##################### CHANGE THIS FOR EVERY RUN #####################
#####################################################################

# Path to save the extracted files
output_path = Path("../../data/neutron_detection/")
activation_foil_path = output_path / "activation_foils"  


################ Check Source Calibration Information ###################


def build_check_source_from_dict(check_source_dict: dict):
    """Build a CheckSource object from a dictionary."""
    if check_source_dict["nuclide"].lower() == "co60":
        nuclide = co60
    elif check_source_dict["nuclide"].lower() == "cs137":
        nuclide = cs137
    elif check_source_dict["nuclide"].lower() == "mn54":
        nuclide = mn54
    elif check_source_dict["nuclide"].lower() == "na22":
        nuclide = na22
    elif check_source_dict["nuclide"].lower() == "ba133":
        nuclide = ba133
    elif (check_source_dict["energies"] is not None and
          check_source_dict["intensities"] is not None and
          check_source_dict["half_life"] is not None):
        nuclide = compass.Nuclide(
            energies=check_source_dict["energies"],
            intensities=check_source_dict["intensities"],
            half_life=(check_source_dict["half_life"]["value"] 
                       * ureg.parse_units(check_source_dict["half_life"]["unit"])
                       ).to(ureg.s).magnitude
        )
    else:
        raise ValueError(
            f"Unknown nuclide: {check_source_dict['nuclide']}. "
            "Please provide a valid nuclide or energies/intensities/half_life."
        )
    activity_date = datetime.strptime(
            check_source_dict["activity"]["date"], "%Y-%m-%d")
    # Set the timezone to America/New_York
    activity_date = activity_date.replace(tzinfo=ZoneInfo("America/New_York"))
    check_source = CheckSource(
        nuclide=nuclide,
        activity=(check_source_dict["activity"]["value"] 
                  * ureg.parse_units(check_source_dict["activity"]["unit"])
                  ).to(ureg.Bq).magnitude,
        activity_date=activity_date
    )
    return check_source


def read_check_source_data_from_json(json_data: dict, measurement_directory_path: Path, key=None):
    """Read check source data from the general.json file."""
    check_source_dict = {}
    if key is not None:
        source_json_data = json_data["check_sources"][key]
    else:
        source_json_data = json_data["check_sources"]
    for check_source_name in source_json_data:
        check_source_data = source_json_data[check_source_name]
        directory = measurement_directory_path / check_source_data["directory"]
        check_source = build_check_source_from_dict(check_source_data)
        check_source_dict[check_source_name] = {
            "directory": directory,
            "check_source": check_source,
        }
    return check_source_dict


################# Background Information ###################

def read_background_data_from_json(json_data: dict, measurement_directory_path: Path, key=None):
    """Read background data from the general.json file."""
    if key is None:
        background_dir = measurement_directory_path / json_data["background_directory"]
    else:
        background_dir = measurement_directory_path / json_data["background_directory"][key]
    return background_dir



################ Foil Information ###################

def get_distance_to_source_from_dict(foil_dict: dict):
    distance_to_source_dict = foil_dict["distance_to_source"]
    # unit from string with pint
    unit = ureg.parse_units(distance_to_source_dict["unit"])
    return (distance_to_source_dict["value"] * unit).to(ureg.cm).magnitude
    

def get_mass_from_dict(foil_dict: dict):
    foil_mass = foil_dict["mass"]["value"]
    # unit from string with pint
    unit = ureg.parse_units(foil_dict["mass"]["unit"])
    return (foil_mass * unit).to(ureg.g).magnitude
    

def get_thickness_from_dict(foil_dict: dict):
    foil_thickness = foil_dict["thickness"]["value"]
    # unit from string with pint
    unit = ureg.parse_units(foil_dict["thickness"]["unit"])
    return (foil_thickness * unit).to(ureg.cm).magnitude


def interpolate_mass_attenuation_coefficient(foil_element_symbol, energy):
    """Interpolate the mass attenuation coefficient for 
    a given foil element symbol and energy (keV)."""

    # Data from Table 3 of https://dx.doi.org/10.18434/T4D01F
    # data is in the form of [energy (MeV), mass attenuation coefficient (cm^2/g)]
    if foil_element_symbol == "Zr":
        data = [
            [1.00000E-01,  9.658E-01],
            [1.50000E-01,  3.790E-01], 
            [2.00000E-01,  2.237E-01], 
            [3.00000E-01,  1.318E-01], 
            [4.00000E-01,  1.018E-01], 
            [5.00000E-01,  8.693E-02], 
            [6.00000E-01,  7.756E-02], 
            [8.00000E-01,  6.571E-02], 
            [1.00000E+00,  5.810E-02], 
            [1.25000E+00,  5.150E-02], 
            [1.50000E+00,  4.700E-02], 
            [2.00000E+00,  4.146E-02]
        ]
    elif foil_element_symbol == "Nb":
        data = [
            [1.00000E-01,	1.037E+00],
            [1.50000E-01,	4.023E-01],
            [2.00000E-01,	2.344E-01],
            [3.00000E-01,	1.357E-01],
            [4.00000E-01,	1.040E-01],
            [5.00000E-01,	8.831E-02],
            [6.00000E-01,	7.858E-02],
            [8.00000E-01,	6.642E-02],
            [1.00000E+00,	5.866E-02],
            [1.25000E+00,	5.196E-02],
            [1.50000E+00,	4.741E-02],
            [2.00000E+00,	4.185E-02]
        ]
    else:
        raise ValueError(f"Unsupported foil element symbol: {foil_element_symbol}")

    data = np.array(data)
    # Interpolate the mass attenuation coefficient
    mass_attenuation_coefficient = np.interp(
        energy, 
        data[:, 0] * 1e3,  # energy values converted to keV
        data[:, 1]   # mass attenuation coefficient values
    )
    return mass_attenuation_coefficient  # in cm^2/g


def get_foil(foil_dict: dict):
    """Get information about a specific foil from the general data file.
    Args:
        json_data (dict): The loaded JSON data from the general.json file.
    Returns:
        ActivationFoil: An ActivationFoil object containing the foil's properties.
        distance_to_source (float): The distance from the foil to the neutron source in cm.
    """
    foil_element_symbol = foil_dict["material"]
    foil_designator = foil_dict.get("designator", None)

    # Get distance to generator
    distance_to_source = get_distance_to_source_from_dict(foil_dict)

    # Get mass
    foil_mass = get_mass_from_dict(foil_dict)

    # get foil thickness
    foil_thickness = get_thickness_from_dict(foil_dict)

    # Get foil name
    foil_name = foil_dict["designator"]
    if foil_name is None:
        foil_name = foil_element_symbol

    if foil_element_symbol == "Zr":
        # density in g/cm^
        # Source: 
        # Arblaster, John W. (2018). 
        # Selected Values of the Crystallographic Properties of Elements.
        #  Materials Park, Ohio: ASM International. ISBN 978-1-62708-155-9.
        foil_density = 6.505

        foil_reaction = zr90_n2n

    elif foil_element_symbol == "Nb":
        # density in g/cm^
        # Source: 
        # Arblaster, John W. (2018). 
        # Selected Values of the Crystallographic Properties of Elements.
        #  Materials Park, Ohio: ASM International. ISBN 978-1-62708-155-9.
        foil_density = 8.582

        foil_reaction = nb93_n2n

    else:
        raise ValueError(f"Unsupported foil element symbol: {foil_element_symbol}")
    
    foil_mass_attenuation_coefficient = interpolate_mass_attenuation_coefficient(
            foil_element_symbol, foil_reaction.product.energy)
    
    foil = ActivationFoil(
        reaction=foil_reaction,
        mass=foil_mass,
        name=foil_name,
        density=foil_density,
        thickness=foil_thickness,  # in cm
    )
    foil.mass_attenuation_coefficient = foil_mass_attenuation_coefficient
    print(f"Read in properties of {foil.name} foil")
    return foil, distance_to_source


def get_foil_source_dict_from_json(json_data: dict, measurement_directory_path: Path, key=None):
    """Read foil source data from the general.json file."""
    foils = json_data["materials"]
    foil_source_dict = {}
    for foil_dict in foils:
        foil, distance_to_source = get_foil(foil_dict)
        measurement_paths = {}
        if key is not None:
            measurement_subdirectories = foil_dict["measurement_directory"][key]
        else:
            measurement_subdirectories = foil_dict["measurement_directory"]
        for count_num, measurement_subdirectory in enumerate(measurement_subdirectories, start=1):
            measurement_paths[count_num] = (
                measurement_directory_path / measurement_subdirectory
            )
        # foil.name should be the same as the designator if it exists.
        # Otherwise is set to the element symbol. 
        foil_source_dict[foil.name] = {
            "measurement_paths": measurement_paths,
            "foil": foil,
            "distance_to_source": distance_to_source,
        }
    return foil_source_dict



def get_data(download_from_raw=False, 
             data_url=None,
             check_source_dict=None,
             background_dir=None,
             foil_source_dict=None,
             h5_filename="activation_data.h5",
             detector_type="NaI"):
    with open("../../data/general.json", "r") as f:
        general_data = json.load(f)
        json_data = general_data["neutron_detection"]["foils"]
    
    # get detector type
    detector_types = json_data.get("detector_type", "NaI")
    # print("detector_type: ", detector_types)

    if not isinstance(detector_types, list):
        detector_type = [detector_types]
    print("Available detector types: ", detector_types)
    if detector_type not in detector_types:
        raise ValueError(f"Detector type {detector_type} not found in general.json file. Available types: {detector_types}")
    
    
    # get measurement directory path
    if isinstance(json_data["data_directory"], dict):
        if detector_type not in json_data["data_directory"].keys():
            raise ValueError(f"Detector type {detector_type} not found in data_directory of general.json file. Available types: {json_data['data_directory'].keys()}")
        measurement_directory_path = activation_foil_path / json_data["data_directory"][detector_type]
    else:
        measurement_directory_path = activation_foil_path / json_data["data_directory"]

    # get data download url
    if isinstance(data_url, str):
        pass
    elif isinstance(json_data["data_url"], dict):
        if detector_type not in json_data["data_url"].keys():
            raise ValueError(f"Detector type {detector_type} not found in data_url of general.json file. Available types: {json_data['data_url'].keys()}")
        data_url = json_data["data_url"][detector_type]
    else:
        data_url = json_data["data_url"]


    # Get the dictionaries for check sources, background, and foils
    if check_source_dict is None:
        check_source_dict = read_check_source_data_from_json(json_data, measurement_directory_path, key=detector_type)
    if background_dir is None:
        background_dir = read_background_data_from_json(json_data, measurement_directory_path, key=detector_type)
    if foil_source_dict is None:
        foil_source_dict = get_foil_source_dict_from_json(json_data, measurement_directory_path, key=detector_type)
    if download_from_raw:
        # Download and extract foil data if not already done
        download_and_extract_foil_data(data_url, activation_foil_path, measurement_directory_path)
        # Process data
        check_source_measurements, background_meas = read_checksources_from_directory(
                                        check_source_dict, 
                                        background_dir, 
                                        detector_type=detector_type
                                        )
        foil_measurements = read_foil_measurements_from_dir(foil_source_dict, 
                                                            detector_type=detector_type)

        for measurement in check_source_measurements.values():
            measurement.detector_type = detector_type
        background_meas.detector_type = detector_type
        for foil_name in foil_measurements.keys():
            for measurement in foil_measurements[foil_name]["measurements"].values():
                measurement.detector_type = detector_type

        # save spectra to h5 for future, faster use
        print("Saving processed measurements to h5 file for future use...\n", 
                activation_foil_path,
                detector_type + '_' + h5_filename)
        save_measurements(check_source_measurements,
                        background_meas,
                        foil_measurements,
                        filepath=activation_foil_path / (detector_type + '_' + h5_filename))
    else:
        # Read measurements from h5 file
        measurements = Measurement.from_h5(activation_foil_path / (detector_type + '_' + h5_filename))
        foil_measurements = copy.deepcopy(foil_source_dict)
        check_source_measurements = {}
        # Get list of foil measurement names
        foil_measurement_names = []
        for foil_name in foil_source_dict.keys():
            for count_num in foil_source_dict[foil_name]["measurement_paths"]:
                foil_measurement_names.append(f"{foil_name} Count {count_num}")

            # Add empty measurements dictionary to foil_source_dict copy
            foil_measurements[foil_name]["measurements"] = {}
            
        for measurement in measurements:
            print(f"Processing {measurement.name} from h5 file...")
            # check if measurement is a check source measurement
            if measurement.name in check_source_dict.keys():
                # May want to change CheckSourceMeasurement in libra-toolbox to make this more seemless
                check_source_meas = CheckSourceMeasurement(measurement.name)
                check_source_meas.__dict__.update(measurement.__dict__)
                check_source_meas.check_source = check_source_dict[measurement.name]["check_source"]
                check_source_meas.detector_type = detector_type
                check_source_measurements[measurement.name] = check_source_meas
            elif measurement.name == "Background":
                background_meas = measurement
                background_meas.detector_type = detector_type
            elif measurement.name in  foil_measurement_names:
                # Extract foil name and count number from measurement name
                split_name = measurement.name.split(' ')
                count_num = int(split_name[-1])
                foil_name = " ".join(split_name[:-2])

                foil_meas = SampleMeasurement(measurement)
                foil_meas.__dict__.update(measurement.__dict__)
                foil_meas.foil = foil_source_dict[foil_name]["foil"]
                foil_meas.detector_type = detector_type
                foil_measurements[foil_name]["measurements"][count_num] = foil_meas
            else:
                print(f"Extra measurement included in h5 file: {measurement.name}")
            measurement.detector_type = detector_type   
        
    return check_source_measurements, background_meas, foil_measurements


def save_measurements(check_source_measurements,
                      background_meas,
                      foil_measurements,
                      filepath=activation_foil_path / "activation_data.h5"):
    """Save measurements to an h5 file."""
    print(f"Saving measurements to {filepath}...")
    # Ensure the directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    measurements = list(check_source_measurements.values())
    # Add background measurement to the list
    measurements.append(background_meas)
    # Add foil measurements to the list
    for foil_name in foil_measurements.keys():
        for count_num in foil_measurements[foil_name]["measurements"].keys():
            measurements.append(foil_measurements[foil_name]["measurements"][count_num])
    
    for i,measurement in enumerate(measurements):
        if i==0:
            mode = 'w'
        else:
            mode = 'a'
        measurement.to_h5(
            filename= filepath,
            mode=mode,
            spectrum_only=True
        )


def read_checksources_from_directory(
    check_source_measurements: dict, 
    background_dir: Path,
    detector_type="NaI"
):

    measurements = {}
    for name, values in check_source_measurements.items():
        print(f"Processing {name}...")
        meas = CheckSourceMeasurement.from_directory(values["directory"], name=name)
        meas.check_source = values["check_source"]
        meas.detector_type = detector_type
        measurements[name] = meas

    print(f"Processing background...")
    background_meas = Measurement.from_directory(
        background_dir,
        name="Background",
        info_file_optional=True,
    )
    background_meas.detector_type = detector_type
    return measurements, background_meas


def read_foil_measurements_from_dir(
    foil_measurements: dict,
    detector_type="NaI"
):

    for foil_name in foil_measurements.keys():
        foil_measurements[foil_name]["measurements"] = {}
        foil = foil_measurements[foil_name]["foil"]
        for count_num, measurement_path in foil_measurements[foil_name]["measurement_paths"].items():
            measurement_name = f"{foil_name} Count {count_num}"
            print(f"Processing {measurement_name}...")
            measurement = SampleMeasurement.from_directory(
                source_dir=measurement_path,
                name=measurement_name
            )
            measurement.foil = foil
            measurement.detector_type = detector_type
            foil_measurements[foil_name]["measurements"][count_num] = measurement

    return foil_measurements


# Get the irradiation schedule

with open("../../data/general.json", "r") as f:
    general_data = json.load(f)
irradiations = []
for generator in general_data["generators"]:
    if generator["enabled"] is False:
        continue
    for i, irradiation_period in enumerate(generator["periods"]):
        if i == 0:
            overall_start_time = datetime.strptime(
                irradiation_period["start"], "%m/%d/%Y %H:%M"
            )
        start_time = datetime.strptime(irradiation_period["start"], "%m/%d/%Y %H:%M")
        end_time = datetime.strptime(irradiation_period["end"], "%m/%d/%Y %H:%M")
        irradiations.append(
            {
                "t_on": (start_time - overall_start_time).total_seconds(),
                "t_off": (end_time - overall_start_time).total_seconds(),
            }
        )
time_generator_off = end_time
time_generator_off = time_generator_off.replace(tzinfo=ZoneInfo("America/New_York"))



def calculate_neutron_rate_from_foil(foil_measurements, 
                                     foil_name,
                                     background_meas,
                                     calibration_coeffs,
                                     efficiency_coeffs,
                                     search_width=330,
                                     irradiations=irradiations,
                                     time_generator_off=time_generator_off):
    neutron_rates = {}
    neutron_rate_errs = {}

    for count_num, measurement in foil_measurements[foil_name]["measurements"].items():

        neutron_rates[f"Count {count_num}"] = {}
        neutron_rate_errs[f"Count {count_num}"] = {}

        for detector in measurement.detectors:
            ch = detector.channel_nb

            gamma_emitted, gamma_emitted_err = measurement.get_gamma_emitted(
                background_measurement=background_meas,
                calibration_coeffs=calibration_coeffs[ch],
                efficiency_coeffs=efficiency_coeffs[ch],
                channel_nb=ch,
                search_width=search_width)
            
            neutron_rate = measurement.get_neutron_rate(
                channel_nb=ch,
                photon_counts=gamma_emitted,
                irradiations=irradiations,
                distance=foil_measurements[foil_name]["distance_to_source"],
                time_generator_off=time_generator_off,
                branching_ratio=foil_measurements[foil_name]["foil"].reaction.product.intensity
            )

            neutron_rate_err = measurement.get_neutron_rate(
                channel_nb=ch,
                photon_counts=gamma_emitted_err,
                irradiations=irradiations,
                distance=foil_measurements[foil_name]["distance_to_source"],
                time_generator_off=time_generator_off,
                branching_ratio=foil_measurements[foil_name]["foil"].reaction.product.intensity
            )
            neutron_rates[f"Count {count_num}"][ch] = neutron_rate
            neutron_rate_errs[f"Count {count_num}"][ch] = neutron_rate_err

    return neutron_rates, neutron_rate_errs
