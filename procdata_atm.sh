#!/bin/bash

# Check if a parameter is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <parameter>"
  echo "Example: $0 prc"
  exit 1
fi

# Parameter to use (e.g., pr, prc, prcs, prcn, qlfx)
parameter=$1

# Define scenarios and directories
declare -A scenario_dirs=(
  ["pa-futArcSIC-ext"]="/home/esp-shared-a/Distribution/Workshops/PolarClimate_2024/atm_oc_seaice_projects/Project_1_2/pa-futArcSIC-ext/WACCM/atm/Amon/${parameter}"
  ["pa-pdSIC-ext"]="/home/esp-shared-a/Distribution/Workshops/PolarClimate_2024/atm_oc_seaice_projects/Project_1_2/pa-pdSIC-ext/WACCM/atm/Amon/${parameter}"
  ["pa-futSIC-2XCO2-ext"]="/home/esp-shared-a/Distribution/Workshops/PolarClimate_2024/atm_oc_seaice_projects/Project_1_2/pa-futSIC-2XCO2-ext/WACCM/atm/Amon/${parameter}"
  ["pa-pdSIC-2XCO2-ext"]="/home/esp-shared-a/Distribution/Workshops/PolarClimate_2024/atm_oc_seaice_projects/Project_1_2/pa-pdSIC-2XCO2-ext/WACCM/atm/Amon/${parameter}"
  ["pa-futAntSIC-ext"]="/home/esp-shared-a/Distribution/Workshops/PolarClimate_2024/atm_oc_seaice_projects/Project_1_2/pa-futAntSIC-ext/WACCM/atm/Amon/${parameter}"
)

# Define the output directory
output_dir="/scratch/lvillanu/data"
mkdir -p "$output_dir"

# Define file suffix
file_suffix=".nc"

for scenario in "${!scenario_dirs[@]}"; do
  echo "Processing scenario: $scenario with parameter: $parameter"
  
  # Define the file directory for the current scenario
  file_dir="${scenario_dirs[$scenario]}"
  
  # List all .nc files in the directory
  file_paths=("${file_dir}"/*.nc)
  
  # Check if the array of file paths is not empty
  if (( ${#file_paths[@]} > 0 )); then
    # Combine all files into one file
    combined_file="${output_dir}/${scenario}_${parameter}_combined.nc"
    echo "Combining all files into $combined_file"
    cdo mergetime "${file_paths[@]}" "$combined_file"
    
    # Mean of the entire period
    mean_50years="${output_dir}/${scenario}_${parameter}_mean_50years.nc"
    echo "Calculating mean over the entire period for $combined_file"
    cdo timmean "$combined_file" "$mean_50years"
    
    # Mean for each year
    mean_per_year="${output_dir}/${scenario}_${parameter}_mean_per_year.nc"
    echo "Calculating mean for each year for $combined_file"
    cdo yearmean "$combined_file" "$mean_per_year"
    
    # Mean of each month over the years
    mean_each_month="${output_dir}/${scenario}_${parameter}_mean_each_month.nc"
    echo "Calculating mean of each month over the years for $combined_file"
    cdo ymonmean "$combined_file" "$mean_each_month"
    
    # Clean up the combined file if not needed
    # rm "$combined_file"
    
    echo "Processing complete for scenario: $scenario with parameter: $parameter"
  else
    echo "No .nc files found in directory for scenario: $scenario with parameter: $parameter"
  fi
done

echo "All scenarios processed."

