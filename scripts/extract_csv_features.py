import os
import glob
import pandas as pd
import numpy as np
from flirimageextractor import FlirImageExtractor

# --- Configuration ---
INPUT_FOLDER = r'C:\Users\simto\OneDrive\Desktop\ניסוי שני אלכוהול, מאי-20251119T073944Z-1-001\ניסוי שני אלכוהול, מאי\יום שבת 24.05\Thermal Images'
OUTPUT_FOLDER = r'C:\Users\simto\OneDrive\Desktop\ניסוי שני אלכוהול, מאי-20251119T073944Z-1-001\ניסוי שני אלכוהול, מאי\יום שבת 24.05\CSV'

# --- Calibration Offset ---
CALIBRATION_OFFSET = 0.017443

# --- Decimal truncation ---
DECIMAL_PLACES = 3

# --- Path format option ---
PATH_FORMAT = 'full'  # 'filename', 'relative', or 'full'

# ---------------------

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

search_path = os.path.join(INPUT_FOLDER, '*.jpg')
file_paths = glob.glob(search_path)

if not file_paths:
    print(f"No .jpg files found in {INPUT_FOLDER}")
    exit()

print(f"Found {len(file_paths)} JPG files. Starting thermal data extraction...")

flir = FlirImageExtractor()

for img_path in file_paths:
    base_name = os.path.basename(img_path)

    if '_X' in base_name:
        print(f"Skipping bad file: {base_name}")
        continue

    try:
        print(f"Processing: {base_name}")

        short_path = img_path
        try:
            import ctypes
            from ctypes import wintypes

            GetShortPathNameW = ctypes.windll.kernel32.GetShortPathNameW
            GetShortPathNameW.argtypes = [wintypes.LPCWSTR, wintypes.LPWSTR, wintypes.DWORD]
            GetShortPathNameW.restype = wintypes.DWORD

            buffer_size = 500
            short_path_buffer = ctypes.create_unicode_buffer(buffer_size)
            ret = GetShortPathNameW(img_path, short_path_buffer, buffer_size)

            if ret and ret <= buffer_size:
                short_path = short_path_buffer.value
                print(f"  Using short path for processing")
        except:
            pass

        flir.process_image(short_path)
        thermal_data = flir.get_thermal_np()

        if thermal_data is None or thermal_data.size == 0:
            print(f"  WARNING: No thermal data extracted. Skipping file.")
            continue

        thermal_data = thermal_data - CALIBRATION_OFFSET
        thermal_data = np.floor(thermal_data * (10 ** DECIMAL_PLACES)) / (10 ** DECIMAL_PLACES)

        base_name_no_ext = os.path.splitext(base_name)[0]
        output_path = os.path.join(OUTPUT_FOLDER, f"{base_name_no_ext}.csv")

        if PATH_FORMAT == 'filename':
            csv_path = base_name
        elif PATH_FORMAT == 'relative':
            csv_path = os.path.join("Thermal Images", base_name)
        else:
            csv_path = img_path

        # Write to CSV in exact format with quoted path
        with open(output_path, 'w', encoding='utf-8-sig') as f:
            # Write header with file path wrapped in quotes
            f.write(f'File: ,"{csv_path}"\n')  # ← Changed this line
            f.write("\n")

            rows, cols = thermal_data.shape
            for i in range(rows):
                if i == 0:
                    f.write("Frame 1")
                for j in range(cols):
                    f.write(f",{thermal_data[i, j]:.{DECIMAL_PLACES}f}")
                f.write("\n")

        print(f"  → Successfully extracted temperatures -> {base_name_no_ext}.csv")

    except Exception as e:
        print(f"  CRITICAL ERROR processing {base_name}: {e}")
        continue

print("\n=== Processing Complete ===")
