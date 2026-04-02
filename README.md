# anello-haneheld-scripts

This repository contains supporting calibration scripts for ANELLO Handheld, specifically `pdr_speed_calibration.py` for calibrating PDR walking-speed parameters from a Handheld log.

The script:
- extracts the required topics from the `.ulg` file
- detects the walking interval automatically from GPS speed and detected steps
- shows a GPS speed preview so you can accept or override the detected start and stop times
- writes the four resulting calibration parameters to text files

## Repository Contents

- `pdr_speed_calibration.py`
- `requirements.txt`

## Requirements

- Python 3.9 or newer
- `ulog2csv` available on your `PATH`

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

If `ulog2csv` is not already available after installing dependencies, install `pyulog` in a way that exposes its command-line tools in your environment.

## Usage

Run the script with the full path to the log file:

```bash
python pdr_speed_calibration.py /path/to/log_file.ulg
```

## What To Expect

When you run the script:

1. The log is exported to CSV into a folder next to the log file.
2. A GPS speed preview plot is generated and shown.
3. The script automatically detects a walking interval.
4. You can press Enter to accept the detected start and stop times, or enter your own.

The start time should be when walking motion starts.

The automatic detection uses this logic:

- start when filtered horizontal GPS speed is above `1.0 m/s` for at least `1.0 s`
- require at least `3` valid detected steps in that region
- begin fitting `1.0 s` after that trigger
- stop at the last sustained moving segment
- trim the final `1.0 s` before that segment ends

## Outputs

The script creates a folder named after the input log and writes plots plus these text files:

- `<log_name>_analytics.txt`
- `<log_name>_param_set.txt`

Both files contain the same four calibration results:

```text
EKF2_PDR_COEFF_F -0.310
EKF2_PDR_COEFF_V 0.011
EKF2_PDR_COEFF_B 1.198
EKF2_PDR_YAW_OFF -4.077
```

`_param_set.txt` includes the same values prefixed with `param set` so they can be copied directly into a PX4 parameter workflow.

## Notes

- The script uses GPS-derived speed plus IMU step detection, so the selected interval should contain clear walking motion.
- If the automatic interval is not appropriate, enter custom start and stop times when prompted.
- If no plot window appears, your Python environment may be using a non-interactive Matplotlib backend. The preview image is still saved in the output folder.

## Troubleshooting

### `ulog2csv was not found on PATH`

Install the dependencies and make sure the `ulog2csv` command is available in the same shell where you run the script.

### Calibration does not produce parameters

Check that:

- the log includes a clear walking segment
- GPS speed is valid during that walking segment
- the selected time range covers only the intended walking motion

