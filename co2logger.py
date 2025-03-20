
import time
import sys
import termios
import tty
from datetime import datetime
import board
import busio
import adafruit_scd30

def take_measurements(sensor, label, file):
    """Take 3 measurements and log them with a label."""
    measurements = []
    for i in range(3):
        try:
            co2 = sensor.CO2
            temp = sensor.temperature
            humidity = sensor.relative_humidity
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            measurements.append(f"{timestamp}, CO2: {co2:.2f} ppm, Temp: {temp:.2f} C, Hum>

            # Delay only for the 2nd and 3rd measurements
            if i < 2:
                time.sleep(1)  # Short delay for faster measurements
        except OSError as e:
            print(f"I2C Error: {e}. Skipping this measurement.")
            continue

    # Print and save the label and measurements
    print(f"\n{label}")
    file.write(f"\n{label}\n")
    for entry in measurements:
        print(entry)
        file.write(f"{entry}\n")
    file.flush()

def get_key():
    """Wait for a single key press and return the key."""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return key

def main():
    # Initialize I2C bus and sensor
    i2c = busio.I2C(board.SCL, board.SDA)
    scd30 = adafruit_scd30.SCD30(i2c)

    # Open file to save data
    with open("co2logger.txt", "a") as file:
        try:
            print("Press 'i' for INHALE, 'o' for EXHALE, ENTER for BASELINE, and 'q' to qu>
            while True:
                # Wait for a single key press
                user_input = get_key()
                if user_input.lower() == 'q':
                    print("Exiting...")
                    break
                elif user_input.lower() == 'i':
                    take_measurements(scd30, "End of Breath IN", file)
                elif user_input.lower() == 'o':
                    take_measurements(scd30, "End of Breath OUT", file)
                elif user_input == '\r':  # ENTER key
                    take_measurements(scd30, "Baseline", file)
                else:
                    print("Invalid input. Please press 'i', 'o', ENTER, or 'q'.")
        except KeyboardInterrupt:
            print("Program interrupted. Exiting...")
        finally:
            print("Program exited safely.")

if __name__ == "__main__":
    main()


