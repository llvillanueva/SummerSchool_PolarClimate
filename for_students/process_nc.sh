#!/bin/bash

# Define the directory containing the files
file_dir="/home/esp-shared-a/Distribution/Workshops/PolarClimate_2024/atm_oc_seaice_projects/Project_1_2/pa-pdSIC-ext/WACCM/atm/Amon/prc"

# Define file prefix and suffix
file_prefix="prc_Amon_CESM1-WACCM-SC_pa-pdSIC-ext_r1i1p1f1_gn_"
file_suffix=".nc"

# List of the first 10 years' files
files=(
  "028501-028512" "028601-028612" "028701-028712" "028801-028812" "028901-028912"
  "029001-029012" "029101-029112" "029201-029212" "029301-029312" "029401-029412"
)

# Construct the full paths of the files
file_paths=()
for file_year in "${files[@]}"; do
  file="${file_dir}/${file_prefix}${file_year}${file_suffix}"
  if [[ -f "$file" ]]; then
    file_paths+=("$file")
  else
    echo "File not found: $file"
  fi
done

# Check if the array of file paths is not empty
if (( ${#file_paths[@]} > 0 )); then
  # Combine all files into one file
  echo "Combining all files into combined.nc"
  cdo mergetime "${file_paths[@]}" combined.nc

  # Mean of the entire 10 years
  echo "Calculating mean over the entire 10 years"
  cdo timmean combined.nc mean_10years.nc

  # Mean for each year
  echo "Calculating mean for each year"
  cdo yearmean combined.nc mean_per_year.nc

  # Mean of each month over the years
  echo "Calculating mean of each month over the years"
  cdo ymonmean combined.nc mean_each_month.nc

  # Clean up the combined file if not needed
  # rm combined.nc

  echo "Processing complete. Mean files created: mean_10years.nc, mean_per_year.nc, mean_each_month.nc"
else
  echo "No valid files found for processing."
fi

