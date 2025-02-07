import os
import sys

script_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.abspath(os.path.join(script_path, "../PenguinPi-robot/software/python/client/")))

from pibot_client import PiBot

# Create a connection to the PiBot
bot = PiBot("192.168.1.179")

# Request the voltage
voltage = bot.getVoltage()

# Print the voltage value
print(f"Current PiBot voltage: {voltage:.2f}V")

# Calculate battery percentage for 2 series 18650 cells (min 6.0V, max 8.4V)
min_voltage = 6.0  # 3.0V per cell
max_voltage = 8.4  # 4.2V per cell
voltage_percent = (voltage - min_voltage) / (max_voltage - min_voltage) * 100
voltage_percent = max(0, min(voltage_percent, 100))  # Clamp to 0-100%

# Print the battery percentage
print(f"Battery percentage: {voltage_percent:.0f}%")

# Close the connection
bot.stop()