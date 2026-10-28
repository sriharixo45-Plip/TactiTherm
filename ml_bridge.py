import time
from pathlib import Path

import joblib
import pandas as pd
import serial

from serial.serialutil import SerialException


# ==================================================
# CONFIGURATION
# ==================================================

PORT = "rfc2217://localhost:4000"
BAUD_RATE = 115200
MODEL_PATH = "tactile_thermal_model_v3.pkl"
RECONNECT_DELAY = 3

GRID_SIZE = 5

TRAINING_RANGES = {
    "force": (31.567, 69.989),
    "temperature": (22.102, 31.679),
    "deformation": (1.918, 9.754),
    "stiffness": (6.365, 17.820),
    "tactile_score": (15.027, 53.798),
    "thermal_score": (9.873, 58.458),
    "fusion": (14.552, 49.366),
}

CLASS_SYMBOLS = {
    "NORMAL": "N",
    "TACTILE": "T",
    "THERMAL": "H",
    "COMBINED": "C",
}


# ==================================================
# LOAD MODEL
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
model_file = BASE_DIR / MODEL_PATH

if not model_file.exists():
    raise FileNotFoundError(
        f"Model file not found: {model_file}"
    )

artifact = joblib.load(model_file)

model = artifact["model"]
features = artifact["features"]

EXPECTED_FEATURES = [
    "force",
    "temperature",
    "deformation",
    "stiffness",
    "tactile_score",
    "thermal_score",
    "fusion",
]

if features != EXPECTED_FEATURES:
    raise ValueError(
        "Unexpected model feature configuration.\n"
        f"Expected: {EXPECTED_FEATURES}\n"
        f"Received: {features}"
    )

print("=" * 65)
print("TACTILE-THERMAL RANDOM FOREST BRIDGE")
print("=" * 65)
print("Model:", model_file.name)
print("Classes:", list(model.classes_))
print("Feature calculations: PROVISIONAL")
print("Predictions are experimental, not clinical diagnoses.")


# ==================================================
# SCAN MAP
# ==================================================

scan_map = {}
active_scan_id = None


def reset_scan():
    """Clear the map before displaying a new scan."""

    global scan_map

    scan_map = {
        (row, col): None
        for row in range(GRID_SIZE)
        for col in range(GRID_SIZE)
    }


def display_scan_map():
    """Display predictions at their reported grid positions."""

    print("\n5 x 5 PREDICTION MAP")
    print("N=NORMAL T=TACTILE H=THERMAL C=COMBINED")
    print("Rows and columns are zero-indexed.\n")

    print("     " + "  ".join(str(c) for c in range(GRID_SIZE)))

    for row in range(GRID_SIZE):
        symbols = []

        for col in range(GRID_SIZE):
            prediction = scan_map.get((row, col))

            if prediction is None:
                symbols.append("-")
            else:
                symbols.append(
                    CLASS_SYMBOLS.get(prediction, "?")
                )

        print(f"{row:>3}  " + "  ".join(symbols))

    print()


# ==================================================
# FEATURE PREPARATION
# ==================================================

def prepare_features(force_percent, temperature_c):
    """
    Prepare model inputs from the two simulated sensor readings.

    IMPORTANT:
    The derived-feature equations are unverified estimates.
    They are not recovered from the original dataset generator.
    """

    force = float(force_percent)
    temperature = float(temperature_c)

    if not 0.0 <= force <= 100.0:
        raise ValueError(
            f"Force reading outside ADC percentage range: {force}"
        )

    if not 0.0 <= temperature <= 100.0:
        raise ValueError(
            f"Temperature reading outside plausible input range: "
            f"{temperature}"
        )

    # PROVISIONAL EQUATIONS — NOT PHYSICALLY VALIDATED.

    deformation = 3.171 * (force / 42.521)

    stiffness = max(
        0.0,
        13.801 - 0.35 * (deformation - 3.171),
    )

    tactile_score = max(
        0.0,
        23.857 + 1.5 * (force - 42.521),
    )

    thermal_score = max(
        0.0,
        26.614 + 5.0 * (temperature - 25.323),
    )

    fusion = (tactile_score + thermal_score) / 2.0

    values = {
        "force": force,
        "temperature": temperature,
        "deformation": deformation,
        "stiffness": stiffness,
        "tactile_score": tactile_score,
        "thermal_score": thermal_score,
        "fusion": fusion,
    }

    return pd.DataFrame(
        [[values[name] for name in features]],
        columns=features,
    )


# ==================================================
# INPUT VALIDATION
# ==================================================

def report_range_warnings(X):
    """Report inputs outside the observed dataset ranges."""

    warnings = []

    for name in features:
        low, high = TRAINING_RANGES[name]
        value = float(X.iloc[0][name])

        if value < low or value > high:
            warnings.append(
                f"{name}={value:.3f}; "
                f"observed range [{low:.3f}, {high:.3f}]"
            )

    for warning in warnings:
        print("RANGE WARNING:", warning)

    return warnings


# ==================================================
# PROCESS ONE SCAN POINT
# ==================================================

def process_scan_point(fields):
    global active_scan_id

    if len(fields) != 8 or fields[0] != "SCAN_POINT":
        return

    point = int(fields[1])
    row = int(fields[2])
    col = int(fields[3])

    scenario = fields[4]
    force_percent = float(fields[5])
    temperature_c = float(fields[6])
    arduino_class = fields[7]

    if not 1 <= point <= 25:
        raise ValueError(f"Invalid scan point: {point}")

    if not 0 <= row < GRID_SIZE or not 0 <= col < GRID_SIZE:
        raise ValueError(
            f"Invalid grid position: row={row}, col={col}"
        )

    if not 0.0 <= force_percent <= 100.0:
        raise ValueError(f"Invalid force reading: {force_percent}")

    # Reset when a new scan starts at point 1.
    if point == 1:
        reset_scan()

    X = prepare_features(
        force_percent,
        temperature_c,
    )

    warnings = report_range_warnings(X)

    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]

    scan_map[(row, col)] = prediction

    print("\n" + "-" * 60)
    print(f"Point: {point}/25")
    print(f"Position: ({row}, {col})")
    print(f"Scenario: {scenario}")
    print(f"Force: {force_percent:.2f}%")
    print(f"Temperature: {temperature_c:.2f} C")

    print("\nModel features:")
    for name in features:
        print(f"  {name}: {X.iloc[0][name]:.3f}")

    print("\nArduino classification:", arduino_class)
    print("Random Forest prediction:", prediction)

    print("\nClass probabilities:")
    for label, probability in zip(model.classes_, probabilities):
        print(f"  {label}: {probability * 100:.2f}%")

    if warnings:
        print(
            "\nWARNING: One or more model inputs are outside "
            "the observed training ranges."
        )

    display_scan_map()
    print("-" * 60)


# ==================================================
# SERIAL CONNECTION
# ==================================================

def process_connection():
    print(f"\nConnecting to {PORT}...")

    with serial.serial_for_url(
        PORT,
        baudrate=BAUD_RATE,
        timeout=2,
    ) as ser:

        print("Connected.")
        print("Start a scan using the Arduino START button.")

        while True:
            raw_line = ser.readline().decode(
                "utf-8",
                errors="ignore",
            ).strip()

            if not raw_line:
                continue

            if raw_line == "SCAN COMPLETED":
                print("\nScan completed.")
                display_scan_map()
                continue

            fields = raw_line.split(",")

            if not fields or fields[0] != "SCAN_POINT":
                continue

            try:
                process_scan_point(fields)

            except (ValueError, KeyError, IndexError) as error:
                print("Could not process scan point:", error)


# ==================================================
# RECONNECTION LOOP
# ==================================================

try:
    while True:
        try:
            process_connection()

        except (SerialException, OSError) as error:
            print("\nSerial connection lost:", error)
            print(f"Retrying in {RECONNECT_DELAY} seconds...")
            time.sleep(RECONNECT_DELAY)

except KeyboardInterrupt:
    print("\nML bridge stopped.")

