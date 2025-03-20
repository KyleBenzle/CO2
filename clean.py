import csv
import os

def clean_data(input_files, output_file):
    """
    Cleans data from multiple input files and writes it into a single CSV file.

    Args:
        input_files: List of text files containing the raw data.
        output_file: The CSV file to save the cleaned data.
    """
    # Prepare the output CSV file
    with open(output_file, "w") as outfile:
        csv_writer = csv.writer(outfile)
        # Write the CSV header
        csv_writer.writerow(["Date/Time", "Type", "CO2 (ppm)", "Temp (C)", "Humidity (%)", "Source File"])

        # Process each input file
        for input_file in input_files:
            with open(input_file, "r") as infile:
                current_type = None
                file_name = os.path.basename(input_file)
                
                for line in infile:
                    line = line.strip()
                    if not line:
                        continue

                    # If the line is a type label (Baseline, End of Breath IN/OUT)
                    if line in ["Baseline", "End of Breath OUT", "End of Breath IN"]:
                        current_type = line
                        continue

                    # If the line contains data
                    if current_type and "CO2:" in line:
                        try:
                            # Parse the data line
                            parts = line.split(", ")
                            datetime = parts[0]
                            co2 = parts[1].split(": ")[1].split(" ppm")[0]
                            temp = parts[2].split(": ")[1].split(" C")[0]
                            humidity = parts[3].split(": ")[1].split(" %")[0]

                            # Write the cleaned data to the CSV
                            csv_writer.writerow([datetime, current_type, co2, temp, humidity, file_name])
                        except Exception as e:
                            print("Error processing line: {} | Error: {}".format(line, e))

# Run the cleaning function
input_files = [
    "1.txt",
    "1_mask_12_1_24.txt",
    "2_masks_12_1_24.txt"
]
output_file = "cleaned_data.csv"

# Use absolute paths
base_dir = "/home/iii/MEGA/Python/CO2/"
input_files = [os.path.join(base_dir, f) for f in input_files]
output_file = os.path.join(base_dir, output_file)

clean_data(input_files, output_file)
print("Data from all files has been cleaned and saved to {}".format(output_file))

