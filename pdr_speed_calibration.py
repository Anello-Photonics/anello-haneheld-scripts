import argparse
import os
import shutil
import subprocess
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
CONSTANTS_ONE_G = 9.80665

EXPORT_MARKER = ".ulog_export_complete"

SAMPLE_FREQUENCY = 200 # Hz
cutoff_frequency = 5 # Hz for low pass filter on accel norm
gps_offset = 0.1 # adds an offset to the raw gps speeds
nf_initialized = True
speed_replay = True
gps_filter_alpha = 0.1
speed_threshold_m_s = 1.0
speed_threshold_hold_s = 1.0
fit_margin_s = 1.0
min_steps_for_start = 3
step_variance_threshold = 0.3


class RunningVariance:
    def __init__(self):
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, sample):
        value = float(np.asarray(sample).reshape(-1)[0])
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        delta2 = value - self.mean
        self.m2 += delta * delta2

    @property
    def var_p(self):
        if self.count == 0:
            return 0.0
        return self.m2 / self.count

class GPSAlphaFilter:
    def __init__(self, alpha, initial_value=0.0):
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be between 0 and 1")

        self.alpha = alpha
        self.y = initial_value
        self.initialized = False

    def update(self, x):
        if not self.initialized:
            self.y = x
            self.initialized = True
        else:
            self.y = self.alpha * x + (1.0 - self.alpha) * self.y
        return self.y

    def reset(self, value=0.0):
        self.y = value
        self.initialized = False

class AlphaFilter:
    """
    PX4-style first-order low-pass ("alpha") filter.
    Works on 3D numpy vectors (shape: (3,))
    Example: 5 Hz cutoff, 200 Hz sampling → alpha ≈ 0.152
    """

    def __init__(self, sample_freq, cutoff_freq):
        self.sample_freq = sample_freq
        self.cutoff_freq = cutoff_freq
        self._alpha = self._compute_alpha()
        self._state = np.zeros(3)   # 3D vector state
        self._initialized = False

    def _compute_alpha(self):
        T_s = 1.0 / self.sample_freq
        tau = 1.0 / (2.0 * np.pi * self.cutoff_freq)
        return T_s / (T_s + tau)

    def reset(self, value):
        """
        Initialize filter state.
        value must be a numpy array of shape (3,)
        """
        self._state = np.array(value, dtype=float)
        self._initialized = True

    def update(self, sample):
        """
        Feed one 3D sample (np array shape (3,))
        Returns filtered 3D output
        """
        sample = np.array(sample, dtype=float)

        if not self._initialized:
            self.reset(sample)
            return sample

        # Vectorized alpha update
        self._state = self._state + self._alpha * (sample - self._state)
        return self._state

    def get_state(self):
        return self._state.copy()

    def get_alpha(self):
        return self._alpha

class StepDetectorPX4:
    """
    PX4-style step detector using a_norm_lpf and accel_z.
    Mirrors PdrAhrs::identifyStep() state machine.
    """

    class StepState:
        IDLE = 0
        RISING = 1
        FALLING = 2
        REFRACTORY = 3

    def __init__(self, low_thresh=9.7, high_thresh=10.5, refractory_s=0.1):
        self.low_thresh = low_thresh
        self.high_thresh = high_thresh
        self.refractory_s = refractory_s

        self._state = self.StepState.IDLE
        self._step_accel_z_max = -CONSTANTS_ONE_G
        self._step_accel_z_min = -CONSTANTS_ONE_G
        self._fall_count = 0
        self._peak_val = 0.0
        self._peak_time = 0.0

        self._step_detected = False

        self._variance = RunningVariance()

    def _ready(self, t_now):
        return (t_now - self._peak_time) > self.refractory_s
    
    def resetStepVariables(self): 
        self._step_accel_z_max = -CONSTANTS_ONE_G
        self._step_accel_z_min = -CONSTANTS_ONE_G
        self._fall_count = 0
        self._peak_val = 0.0
        self._peak_time = 0.0
        self._step_detected = False
        self._variance = RunningVariance()

    def update(self, a_norm_lpf: float, accel_z: float, t_now: float):
        """
        Process one sample. Returns (step_detected, peak_val, peak_time).
        Args:
            a_norm_lpf: filtered accel norm [m/s²]
            accel_z: vertical accel (Z-axis) [m/s²]
            t_now: timestamp in seconds
        """
        if (self._step_detected):
            self.resetStepVariables()

        # --- Update accel extrema for step length estimation ---
        self._step_accel_z_max = max(self._step_accel_z_max, accel_z)
        self._step_accel_z_min = min(self._step_accel_z_min, accel_z)

        self._variance.add(a_norm_lpf)

        # --- State machine ---
        if self._state == self.StepState.IDLE:
            if a_norm_lpf > self.high_thresh and self._ready(t_now):
                self._state = self.StepState.RISING
                self._peak_val = a_norm_lpf
                self._peak_time = t_now

        elif self._state == self.StepState.RISING:
            if a_norm_lpf > self._peak_val:
                self._peak_val = a_norm_lpf
                self._peak_time = t_now

            if (self._peak_val - a_norm_lpf) > 1.0:
                # start fall count
                self._fall_count += 1
            else:
                self._fall_count = 0

            if self._fall_count >= 3:
                self._state = self.StepState.FALLING
                self._fall_count = 0

        elif self._state == self.StepState.FALLING:
            if a_norm_lpf > self._peak_val:
                self._peak_val = a_norm_lpf
                self._peak_time = t_now

            if a_norm_lpf < self.low_thresh:
                self._step_detected = True
                self._state = self.StepState.REFRACTORY

        elif self._state == self.StepState.REFRACTORY:
            if self._ready(t_now):
                self._state = self.StepState.IDLE

        return self._step_detected, self._peak_val, self._peak_time, self._variance.var_p


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run pdr_speed_calibration.py directly from a ULog file."
    )
    parser.add_argument("ulog_path", help="Full path to the input .ulg or .ulog file")
    return parser.parse_args()


def validate_ulog_path(ulog_path_str):
    ulog_path = Path(ulog_path_str).expanduser().resolve()
    if not ulog_path.is_file():
        raise FileNotFoundError(f"ULog file not found: {ulog_path}")
    if ulog_path.suffix.lower() not in {".ulg", ".ulog"}:
        raise ValueError(
            f"Unsupported log extension '{ulog_path.suffix}'. Expected .ulg or .ulog."
        )
    return ulog_path


def get_required_topics():
    required_topics = [
        "vehicle_imu",
        "vehicle_gps_position",
    ]

    if nf_initialized:
        required_topics.append("estimator_gyro_compass_apins")

    if speed_replay:
        required_topics.extend([
            "estimator_aid_src_gnss_pos",
            "estimator_gyro_compass_ins_debug",
        ])

    return required_topics


def get_export_dir(ulog_path):
    export_dir = ulog_path.parent / ulog_path.stem
    export_dir.mkdir(exist_ok=True)
    return export_dir


def export_ulog_to_csv(ulog_path, export_dir):
    ulog2csv_path = shutil.which("ulog2csv")
    if ulog2csv_path is None:
        raise RuntimeError(
            "ulog2csv was not found on PATH. Install dependencies first, for example with "
            "'pip install -r requirements.txt'."
        )

    csv_args = [
        ulog2csv_path,
        "-m",
        ",".join(get_required_topics()),
        "-o",
        str(export_dir),
        str(ulog_path),
    ]
    subprocess.run(csv_args, check=True)
    (export_dir / EXPORT_MARKER).touch(exist_ok=True)


def ensure_csv_exports(ulog_path, export_dir):
    gps_csv = glob(str(export_dir / "*_vehicle_gps_position_0.csv"))
    imu_csv = glob(str(export_dir / "*_vehicle_imu_0.csv"))
    apins_csv = glob(str(export_dir / "*_estimator_gyro_compass_apins_0.csv"))
    gnss_pos_csv = glob(str(export_dir / "*_estimator_aid_src_gnss_pos_0.csv"))
    pdr_og_csv = glob(str(export_dir / "*_estimator_gyro_compass_ins_debug_0.csv"))
    export_marker = export_dir / EXPORT_MARKER

    optional_topics_ready = True
    if nf_initialized and len(apins_csv) == 0:
        optional_topics_ready = False
    if speed_replay and (len(gnss_pos_csv) == 0 or len(pdr_og_csv) == 0):
        optional_topics_ready = False

    if len(gps_csv) > 0 and len(imu_csv) > 0 and (optional_topics_ready or export_marker.exists()):
        return

    export_ulog_to_csv(ulog_path, export_dir)


def prepare_gps_dataframe(gps_csv_path):
    gps_df = pd.read_csv(gps_csv_path)
    gps_df["timestamp"] = gps_df["timestamp"] / 1e6  # convert to seconds
    gps_df = gps_df.sort_values("timestamp").reset_index(drop=True)
    gps_df["vel_h_m_s"] = np.sqrt(
        gps_df["vel_n_m_s"]**2 + gps_df["vel_e_m_s"]**2
    ) + gps_offset

    gps_filter = GPSAlphaFilter(alpha=gps_filter_alpha)
    gps_filter_arr = np.empty(gps_df.shape[0])
    for i, speed in enumerate(gps_df["vel_h_m_s"].to_numpy(dtype=float)):
        gps_filter_arr[i] = gps_filter.update(speed)
    gps_df["vel_h_m_s_filt"] = gps_filter_arr
    return gps_df


def prepare_imu_dataframe(imu_csv_path):
    imu_df = pd.read_csv(imu_csv_path)
    imu_df["timestamp"] = imu_df["timestamp"] / 1e6  # convert to seconds
    imu_df = imu_df.sort_values("timestamp").reset_index(drop=True)
    imu_df["accel_x"] = imu_df["delta_velocity[0]"] / (imu_df["delta_velocity_dt"] / 1e6)
    imu_df["accel_y"] = imu_df["delta_velocity[1]"] / (imu_df["delta_velocity_dt"] / 1e6)
    imu_df["accel_z"] = imu_df["delta_velocity[2]"] / (imu_df["delta_velocity_dt"] / 1e6)
    return imu_df


def compute_step_detection(imu_df, min_freq, max_freq):
    accel_lpf = AlphaFilter(sample_freq=SAMPLE_FREQUENCY, cutoff_freq=cutoff_frequency)
    det = StepDetectorPX4(low_thresh=9.0, high_thresh=10.8, refractory_s=0.1)

    raw_accel = imu_df[["accel_x", "accel_y", "accel_z"]].to_numpy(dtype=float)
    t = imu_df["timestamp"].to_numpy(dtype=float)
    raw_accel_z = imu_df["accel_z"].to_numpy(dtype=float)
    raw_accel_y = imu_df["accel_y"].to_numpy(dtype=float)
    raw_accel_x = imu_df["accel_x"].to_numpy(dtype=float)
    og_norm = np.linalg.norm(raw_accel, axis=1)
    sample_count = raw_accel.shape[0]

    accel_lpf_arr = np.empty_like(raw_accel)
    accel_norm_arr = np.empty(sample_count)
    accel_z_arr = np.empty(sample_count)
    accel_x_arr = np.empty(sample_count)
    accel_y_arr = np.empty(sample_count)

    step_detected_peak_vals = []
    step_detected_peak_times = []
    step_detected_vars = []

    for i in range(sample_count):
        filtered_accel = accel_lpf.update(raw_accel[i])
        accel_lpf_arr[i] = filtered_accel
        accel_norm_arr[i] = np.linalg.norm(filtered_accel)
        accel_z_arr[i] = raw_accel_z[i]
        accel_x_arr[i] = raw_accel_x[i]
        accel_y_arr[i] = raw_accel_y[i]

        step_detected, peak_val, peak_time, var = det.update(
            accel_norm_arr[i],
            raw_accel_z[i],
            t[i],
        )
        if step_detected:
            step_detected_peak_vals.append(peak_val)
            step_detected_peak_times.append(peak_time)
            step_detected_vars.append(var)

    step_detected_peak_vals = np.asarray(step_detected_peak_vals, dtype=float)
    step_detected_peak_times = np.asarray(step_detected_peak_times, dtype=float)
    step_detected_vars = np.asarray(step_detected_vars, dtype=float)

    if step_detected_peak_times.size == 0:
        step_freq = np.asarray([], dtype=float)
        valid_mask = np.asarray([], dtype=bool)
    else:
        step_freq = np.full(step_detected_peak_times.shape, np.nan, dtype=float)
        if step_detected_peak_times.size > 1:
            step_freq[1:] = 1.0 / np.diff(step_detected_peak_times)
        valid_mask = (
            (step_freq >= min_freq)
            & (step_freq <= max_freq)
            & (step_detected_vars > step_variance_threshold)
        )

    return {
        "t": t,
        "og_norm": og_norm,
        "accel_norm_arr": accel_norm_arr,
        "accel_z_arr": accel_z_arr,
        "accel_x_arr": accel_x_arr,
        "accel_y_arr": accel_y_arr,
        "step_detected_peak_vals": step_detected_peak_vals,
        "step_detected_peak_times": step_detected_peak_times,
        "step_freq_valid": step_freq[valid_mask],
        "step_var_valid": step_detected_vars[valid_mask],
        "step_times_valid": step_detected_peak_times[valid_mask],
        "detector_low_thresh": det.low_thresh,
        "detector_high_thresh": det.high_thresh,
    }


def find_sustained_regions(timestamps, values, threshold, min_duration_s):
    regions = []
    region_start_index = None

    for i, is_above_threshold in enumerate(values > threshold):
        if is_above_threshold and region_start_index is None:
            region_start_index = i
        elif not is_above_threshold and region_start_index is not None:
            region_start_time = float(timestamps[region_start_index])
            region_end_time = float(timestamps[i - 1])
            if (region_end_time - region_start_time) >= min_duration_s:
                regions.append((region_start_time, region_end_time))
            region_start_index = None

    if region_start_index is not None:
        region_start_time = float(timestamps[region_start_index])
        region_end_time = float(timestamps[-1])
        if (region_end_time - region_start_time) >= min_duration_s:
            regions.append((region_start_time, region_end_time))

    return regions


def detect_time_window(gps_df, step_times_valid):
    timestamps = gps_df["timestamp"].to_numpy(dtype=float)
    filtered_speed = gps_df["vel_h_m_s_filt"].to_numpy(dtype=float)
    sustained_regions = find_sustained_regions(
        timestamps,
        filtered_speed,
        speed_threshold_m_s,
        speed_threshold_hold_s,
    )

    if not sustained_regions:
        return float(timestamps[0]), float(timestamps[-1]), False

    start_time = None
    for region_start, region_end in sustained_regions:
        steps_in_region = np.count_nonzero(
            (step_times_valid >= region_start) & (step_times_valid <= region_end)
        )
        if steps_in_region >= min_steps_for_start:
            start_time = region_start + fit_margin_s
            break

    if start_time is None:
        return float(timestamps[0]), float(timestamps[-1]), False

    stop_time = sustained_regions[-1][1] - fit_margin_s
    if start_time >= stop_time:
        return float(timestamps[0]), float(timestamps[-1]), False

    return start_time, stop_time, True


def show_preview_plot():
    backend = plt.get_backend().lower()
    if "agg" in backend:
        return
    plt.show(block=False)
    plt.pause(0.001)


def prompt_time_window(gps_df, step_times_valid, export_dir, file_name):
    detected_start, detected_end, used_auto_detection = detect_time_window(gps_df, step_times_valid)
    preview_plot_path = export_dir / f"{file_name}_gps_speed_preview.png"

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(gps_df["timestamp"], gps_df["vel_h_m_s"], label="GPS Horizontal Speed", alpha=0.5)
    ax.plot(gps_df["timestamp"], gps_df["vel_h_m_s_filt"], label="Filtered GPS Speed", linewidth=2)
    ax.axhline(speed_threshold_m_s, color="r", linestyle="--", label="Walking Threshold")

    if step_times_valid.size > 0:
        step_marker_y = np.interp(
            step_times_valid,
            gps_df["timestamp"].to_numpy(dtype=float),
            gps_df["vel_h_m_s_filt"].to_numpy(dtype=float),
        )
        ax.plot(step_times_valid, step_marker_y, "kx", label="Valid Detected Steps")

    if used_auto_detection:
        ax.axvspan(detected_start, detected_end, color="g", alpha=0.15, label="Detected Fit Window")

    ax.set_title("GPS Speed Preview")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(preview_plot_path)
    show_preview_plot()

    min_time = float(gps_df["timestamp"].min())
    max_time = float(gps_df["timestamp"].max())

    while True:
        if used_auto_detection:
            start_input = input(
                f"Review the GPS speed preview. The start time should be when walking motion starts.\n"
                f"Press Enter to use the detected start time of {detected_start:.1f} s, "
                f"or enter a custom start time [s]: "
            ).strip()
            end_input = input(
                f"Press Enter to use the detected stop time of {detected_end:.1f} s, "
                f"or enter a custom stop time [s]: "
            ).strip()
        else:
            start_input = input(
                "Automatic walking-window detection did not find a usable interval.\n"
                "Enter the start time [s] when walking motion starts: "
            ).strip()
            end_input = input(
                "Enter the stop time [s] when walking motion ends: "
            ).strip()

        try:
            start_time = float(start_input) if start_input else detected_start
            end_time = float(end_input) if end_input else detected_end
        except ValueError:
            continue

        if start_time < min_time or end_time > max_time or start_time >= end_time:
            continue

        plt.close(fig)
        return start_time, end_time


def build_parameter_lines(a1, a2, a3, yaw_offset):
    return [
        f"EKF2_PDR_COEFF_F {a1:.3f}",
        f"EKF2_PDR_COEFF_V {a2:.3f}",
        f"EKF2_PDR_COEFF_B {a3:.3f}",
        f"EKF2_PDR_YAW_OFF {yaw_offset:.3f}",
    ]


def build_param_set_lines(a1, a2, a3, yaw_offset):
    return [f"param set {line}" for line in build_parameter_lines(a1, a2, a3, yaw_offset)]


def write_parameter_outputs(output_dir, file_name, a1, a2, a3, yaw_offset):
    analytics_path = Path(output_dir) / f"{file_name}_analytics.txt"
    param_set_path = Path(output_dir) / f"{file_name}_param_set.txt"

    analytics_path.write_text("\n".join(build_parameter_lines(a1, a2, a3, yaw_offset)) + "\n")
    param_set_path.write_text("\n".join(build_param_set_lines(a1, a2, a3, yaw_offset)) + "\n")


def print_parameter_summary(a1, a2, a3, yaw_offset):
    for line in build_parameter_lines(a1, a2, a3, yaw_offset):
        print(line)


args = parse_args()
ulog_path = validate_ulog_path(args.ulog_path)
export_dir = get_export_dir(ulog_path)
ensure_csv_exports(ulog_path, export_dir)

final_folder_path = str(export_dir)

gps_csv = glob(final_folder_path + "/*_vehicle_gps_position_0.csv")
imu_csv = glob(final_folder_path + "/*_vehicle_imu_0.csv")

if len(gps_csv) == 0 or len(imu_csv) == 0:
    raise FileNotFoundError(
        f"Required GPS/IMU CSVs were not found in {final_folder_path} after conversion."
    )

if nf_initialized:
    apins_csv = glob(final_folder_path + "/*_estimator_gyro_compass_apins_0.csv")

if speed_replay:
    gnss_pos_csv = glob(final_folder_path + "/*_estimator_aid_src_gnss_pos_0.csv")
    pdr_og_csv = glob(final_folder_path + "/*_estimator_gyro_compass_ins_debug_0.csv")


max_freq = 4
min_freq = 1

preview_file_name = os.path.basename(gps_csv[0]).split("_vehicle_gps_position_0.csv")[0]
preview_gps_df = prepare_gps_dataframe(gps_csv[0])
preview_imu_df = prepare_imu_dataframe(imu_csv[0])
preview_step_detection = compute_step_detection(preview_imu_df, min_freq, max_freq)
start_us, end_us = prompt_time_window(
    preview_gps_df,
    preview_step_detection["step_times_valid"],
    export_dir,
    preview_file_name,
)

# log from EKF solution
# post_processed = "log10_ekf_sol" + "/" + "log10_ekf_sol" + "_estimator_states_0.csv"

for file_index in range(len(gps_csv)):
    file_name = os.path.basename(gps_csv[file_index]).split("_vehicle_gps_position_0.csv")[0]
    gps_df = prepare_gps_dataframe(gps_csv[file_index])
    imu_df = prepare_imu_dataframe(imu_csv[file_index])

    # cut the time section we want
    gps_df = gps_df[(gps_df["timestamp"] >= start_us) & (gps_df["timestamp"] <= end_us)].copy()
    imu_df = imu_df[(imu_df["timestamp"] >= start_us) & (imu_df["timestamp"] <= end_us)].copy()

    step_detection = compute_step_detection(imu_df, min_freq, max_freq)
    t = step_detection["t"]
    og_norm = step_detection["og_norm"]
    accel_norm_arr = step_detection["accel_norm_arr"]
    accel_z_arr = step_detection["accel_z_arr"]
    accel_x_arr = step_detection["accel_x_arr"]
    accel_y_arr = step_detection["accel_y_arr"]
    step_detected_peak_vals = step_detection["step_detected_peak_vals"]
    step_detected_peak_times = step_detection["step_detected_peak_times"]
    step_freq_valid = step_detection["step_freq_valid"]
    step_var_valid = step_detection["step_var_valid"]
    step_times_valid = step_detection["step_times_valid"]

    # plot the results
    plt.figure(figsize=(12, 8))
    plt.plot(t, accel_norm_arr, label='Accel Norm LPF')
    # plt.plot(t, og_norm, label='Accel Norm', alpha=0.5)
    # # also plot a star at peak_time, peak_val for each detected step
    plt.plot(step_detected_peak_times,step_detected_peak_vals, 'rx', label='Detected Steps')
    plt.axhline(step_detection["detector_low_thresh"], color='r', linestyle='--', label='Low Threshold')
    plt.axhline(step_detection["detector_high_thresh"], color='g', linestyle='--', label='High Threshold')
    plt.title('Low-pass Filtered Accel Norm with Step Detection Thresholds')
    plt.xlabel('Time (s)')
    plt.ylabel('Accel Norm (m/s²)')
    plt.legend()
    plt.savefig(final_folder_path + "/" + file_name + "_accel_norm_step_detection.png")
    # plt.show()

    mid_time = (start_us + end_us) / 2

    # plot the results
    plt.figure(figsize=(12, 8))
    plt.subplot(4, 1, 1)
    plt.plot(t, accel_norm_arr, label='Accel Norm LPF')
    # plt.plot(t, og_norm, label='Accel Norm', alpha=0.5)
    # # also plot a star at peak_time, peak_val for each detected step
    plt.plot(step_detected_peak_times,step_detected_peak_vals, 'rx', label='Detected Steps')
    plt.axhline(step_detection["detector_low_thresh"], color='r', linestyle='--', label='Low Threshold')
    plt.axhline(step_detection["detector_high_thresh"], color='g', linestyle='--', label='High Threshold')
    plt.title('Low-pass Filtered Accel Norm with Step Detection Thresholds')
    plt.xlabel('Time (s)')
    plt.ylabel('Accel Norm (m/s²)')
    plt.xlim(mid_time - 5, mid_time + 5)
    plt.legend()
    # only show x-axis from start_us to end_us


    plt.subplot(4, 1, 2)
    plt.plot(t, accel_z_arr, label='Accel Z Raw')
    plt.xlabel('Time (s)')
    plt.ylabel('Accel Z (m/s²)')
    plt.xlim(mid_time - 5, mid_time + 5)
    plt.legend()

    plt.subplot(4, 1, 3)
    plt.plot(t, accel_x_arr, label='Accel X Raw')
    plt.xlabel('Time (s)')
    plt.ylabel('Accel X (m/s²)')
    plt.xlim(mid_time - 5, mid_time + 5)

    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(t, accel_y_arr, label='Accel Y Raw')
    plt.xlabel('Time (s)')
    plt.ylabel('Accel Y (m/s²)')
    plt.legend()
    plt.xlim(mid_time - 5, mid_time + 5)
    plt.savefig(final_folder_path + "/" + file_name + "_step_detection_xyz.png")



    # plot the step frequency over time
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.scatter(step_times_valid, step_freq_valid, label='Step Frequency (s)')
    plt.title('Step Frequency Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Step Metrics')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.scatter(step_times_valid, step_var_valid, label='Step Variance (m/s²)^2')
    plt.title('Step Variance Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Step Metrics')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(gps_df['timestamp'], gps_df['vel_h_m_s'], label='GPS Horizontal Speed')
    plt.plot(gps_df['timestamp'], gps_df['vel_h_m_s_filt'], label='GPS Speed LPF')
    plt.plot()
    plt.title(f'GPS Horizontal Speed with offset: {gps_offset} m/s')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(final_folder_path + "/" + file_name + "_step_metrics.png")

    # plt.show()

    df_steps_valid = pd.DataFrame({
        "valid_time": step_times_valid,
        "step_freq": step_freq_valid,
        "step_var": step_var_valid
    }).sort_values("valid_time")

    merged_df = pd.merge_asof(
        df_steps_valid,
        gps_df[['timestamp', 'vel_h_m_s', 'vel_h_m_s_filt']].sort_values('timestamp'),
        left_on='valid_time',
        right_on='timestamp',
        direction = "nearest",
    )

    plt.figure(figsize=(12, 8))
    plt.scatter(merged_df["valid_time"],merged_df["step_freq"], label='Step Frequency (s)')
    plt.scatter(merged_df["valid_time"],merged_df["vel_h_m_s"], label='GPS Horizontal Speed')
    plt.scatter(merged_df["valid_time"],merged_df["vel_h_m_s_filt"], label='GPS Speed LPF')
    plt.title(f'Step Frequency and GPS Speed Over Time with offset {gps_offset} m/s')
    plt.xlabel('Time (s)')
    plt.ylabel('Metrics')
    plt.legend()
    plt.savefig(final_folder_path + "/" + file_name + "_stepfrequency_vs_gpsspeed.png")

    plt.figure(figsize=(12, 8))
    plt.scatter(merged_df["valid_time"],merged_df["step_var"], label='Step Variance (m/s²)^2')
    plt.scatter(merged_df["valid_time"],merged_df["vel_h_m_s"], label='GPS Horizontal Speed')
    plt.scatter(merged_df["valid_time"],merged_df["vel_h_m_s_filt"], label='GPS Speed LPF')
    plt.title(f'Step Variance and GPS Speed Over Time with offset {gps_offset} m/s')
    plt.xlabel('Time (s)')
    plt.ylabel('Metrics')
    plt.legend()
    plt.savefig(final_folder_path + "/" + file_name + "_stepvariance_vs_gpsspeed.png")


    # Apply optimization to find shin's parameters
    # shin's equation: a1 * step_freq + a2 * step_var + a3 = gps_speed

    f = np.asarray(merged_df["step_freq"], dtype=float)
    v = np.asarray(merged_df["step_var"], dtype=float)
    s = np.asarray(merged_df["vel_h_m_s_filt"], dtype=float)

    # drop NaNs / infs
    mask = np.isfinite(f) & np.isfinite(v) & np.isfinite(s)
    f, v, s = f[mask], v[mask], s[mask]

    # design matrix: [f, v, 1]
    X = np.column_stack([f**2, v*f, f])
    # solve min ||X a - s||_2
    a1, a2, a3 = np.linalg.lstsq(X, s, rcond=None)[0]

    step_length_hat = a1*f + a2*v + a3
    step_speed_hat  = step_length_hat * f

    plt.figure(figsize=(12, 8))
    plt.scatter(gps_df['timestamp'], gps_df['vel_h_m_s'], label='GPS Horizontal Speed')
    plt.scatter(gps_df['timestamp'], gps_df['vel_h_m_s_filt'], label='GPS Speed LPF')
    plt.scatter(merged_df["valid_time"], step_speed_hat, label='Predicted Speed')
    plt.title(f'GPS Speed (with offset {gps_offset} m/s) vs Predicted Speed')
    plt.xlabel('Time (s)')
    plt.ylabel('Speed (m/s)')
    plt.legend()
    plt.savefig(final_folder_path + "/" + file_name + "_gpsspeed_vs_predictedspeed.png")


    # plot the step frequency over time
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 1, 1)
    plt.scatter(step_times_valid, step_freq_valid, label='Step Frequency (s)')
    plt.title('Step Frequency Over Time')
    plt.xlabel('Time (s)')
    plt.ylabel('Step Metrics')
    plt.legend()

    # if NF sequence is not performed, don't do heading offsets
    if not nf_initialized or len(apins_csv) == 0:
        continue
    
    apins_df = pd.read_csv(apins_csv[file_index])
    apins_df["timestamp"] = apins_df["timestamp"]/ 1e6  # convert to seconds
    apins_df = apins_df[(apins_df['timestamp'] >= start_us) & (apins_df['timestamp'] <= end_us)]

    gps_df["cog_deg"] = (np.degrees(gps_df["cog_rad"]) + 180) % 360 - 180


    valid_mask = gps_df["vel_m_s"] >= 1.5
    gps_valid_spd = (
        gps_df.loc[valid_mask, ["timestamp", "vel_m_s", "cog_deg"]]
        .rename(columns={"timestamp": "valid_time"})
        .sort_values("valid_time")
    )

    merged_df2 = pd.merge_asof(
        gps_valid_spd,
        apins_df[['timestamp','yaw_deg']].sort_values("timestamp"),
        left_on='valid_time',
        right_on='timestamp',
        direction = "nearest",
    )
    
    plt.close('all')
    plt.figure(figsize=(12, 8))
    # plot the two headings in degrees (-180 to 180)
    plt.scatter(merged_df2["timestamp"], merged_df2["yaw_deg"], label = "APINS Yaw Deg")
    plt.scatter(merged_df2["timestamp"], merged_df2["cog_deg"], label = "GPS Yaw Deg")
    plt.legend()
    plt.title("Heading vs COG Heading")
    plt.savefig(final_folder_path + "/" + file_name + "_heading_vs_cog_heading.png")


    plt.close('all')
    plt.figure(figsize=(12,8))
    plt.subplot(2, 1, 1)
    merged_df2["yaw_diff"] = (merged_df2["cog_deg"] - merged_df2["yaw_deg"] + 180) % 360 - 180
    plt.scatter(merged_df2["timestamp"], merged_df2["yaw_diff"], label = "Yaw diff. deg")
    plt.legend()
    plt.title("Yaw difference between COG with EKF Heading")

    plt.subplot(2,1,2)
    window_size = 100
    merged_df2["yaw_moving_avg"] = merged_df2["yaw_diff"].rolling(window=window_size).mean()
    plt.plot(merged_df2["timestamp"],merged_df2["yaw_moving_avg"], label = f"Yaw diff {window_size}")
    plt.legend()
    plt.title("Yaw difference LPF between COG with EKF Heading")
    plt.savefig(final_folder_path + "/" + file_name + "_heading_diff.png")

    avg_yaw_diff = merged_df2["yaw_moving_avg"].mean()

    write_parameter_outputs(final_folder_path, file_name, a1, a2, a3, avg_yaw_diff)
    print_parameter_summary(a1, a2, a3, avg_yaw_diff)



# Speed Replays 
    if not speed_replay or len(gnss_pos_csv) == 0:
        continue

    gnss_pos_df = pd.read_csv(gnss_pos_csv[file_index])
    gnss_pos_df["timestamp"] = gnss_pos_df["timestamp"]/ 1e6  # convert to seconds
    gnss_pos_df = gnss_pos_df[(gnss_pos_df['timestamp'] >= start_us) & (gnss_pos_df['timestamp'] <= end_us)]

    pdr_og_df = pd.read_csv(pdr_og_csv[file_index])
    pdr_og_df["timestamp"] = pdr_og_df["timestamp"]/ 1e6  # convert to seconds
    pdr_og_df = pdr_og_df[(pdr_og_df['timestamp'] >= start_us) & (pdr_og_df['timestamp'] <= end_us)]

    # TODO: change coefficients if we want to try new ones, or ideally pull from the log params
    old_step_freq_coefficient = -0.24449058092062614
    old_step_variance_coefficient = 0.025659494314694498
    old_step_bias = 0.8930891134609088

    step_freq_coefficient = a1
    step_variance_coefficient = a2
    step_bias = a3
    yaw_offset = avg_yaw_diff

    merged_df3 = pd.merge_asof(
        merged_df,
        apins_df[['timestamp','yaw_deg']].sort_values("timestamp"),
        left_on='valid_time',
        right_on='timestamp',
        direction = "nearest",
    )


    steps_taken = len(merged_df)
    pdr_x_list = []
    pdr_y_list = []
    old_pdr_speed_list = []
    pdr_speed_list = []
    pdr_y = 0
    pdr_x = 0

    for step_index in range(steps_taken):
        # new parameters 
        step_time = merged_df3["valid_time"][step_index]
        if (step_index == 0):
            # reset to original X and Y for fair comparison
            idx = (pdr_og_df['timestamp'] - step_time).abs().idxmin()
            pdr_y = pdr_og_df["pdr_y"][idx]
            pdr_x = pdr_og_df["pdr_x"][idx]

        step_freq = merged_df3["step_freq"][step_index]
        step_var = merged_df3["step_var"][step_index]

        step_length = step_freq_coefficient * step_freq + step_variance_coefficient * step_var + step_bias
        hdg = (merged_df3["yaw_deg"].iloc[step_index] + yaw_offset + 180) % 360 - 180
        pdr_y +=step_length * np.sin(np.deg2rad(hdg))
        pdr_x += step_length * np.cos(np.deg2rad(hdg))
        pdr_y_list.append(pdr_y)
        pdr_x_list.append(pdr_x)

        step_vel = step_length * step_freq
        old_step_length = old_step_freq_coefficient * step_freq + old_step_variance_coefficient * step_var + old_step_bias
        old_step_vel = old_step_length * step_freq

        pdr_speed_list.append(step_vel)
        old_pdr_speed_list.append(old_step_vel)

    pdr_y_arr = np.asarray(pdr_y_list)
    pdr_x_arr = np.asarray(pdr_x_list)

    plt.close('all')
    plt.figure(figsize=(12,8))
    plt.plot(pdr_y_arr, pdr_x_arr, label = "New Parameters")
    plt.plot(pdr_og_df["pdr_y"], pdr_og_df["pdr_x"], label = "Old Parameters")
    plt.plot(gnss_pos_df["observation[1]"], gnss_pos_df["observation[0]"], label = "GPS Track")
    plt.title("PDR Tracks with GPS Comparison")
    plt.legend()
    plt.savefig(final_folder_path + "/" + file_name + "_pos_comparison.png")

    plt.close('all')
    plt.figure(figsize=(12,8))
    plt.scatter(gps_df['timestamp'], gps_df['vel_h_m_s'], label='GPS Horizontal Speed',alpha = 0.5)
    plt.scatter(gps_df['timestamp'], gps_df['vel_h_m_s_filt'], label='GPS Speed LPF',alpha = 0.5)
    plt.scatter(merged_df3["valid_time"], pdr_speed_list, label = "New Speed")
    plt.scatter(merged_df3["valid_time"], old_pdr_speed_list, label = "Old Speed")
    plt.title("Speed Comparison")
    plt.legend()
    plt.savefig(final_folder_path + "/" + file_name + "_all_speed_comparison.png")

    merged_df_old = pd.merge_asof(
        merged_df,
        pdr_og_df[['timestamp',"pdr_x","pdr_y"]].sort_values("timestamp"),
        left_on='valid_time',
        right_on='timestamp',
        direction = "nearest",
    )

    merged_df_new = pd.merge_asof(
        merged_df_old,
        gnss_pos_df[['timestamp',"observation[1]","observation[0]"]].sort_values("timestamp"),
        left_on='valid_time',
        right_on='timestamp',
        direction = "nearest",
    )

    merged_df_new["h_error_new"] = np.hypot(merged_df_new["observation[1]"] - pdr_y_arr,merged_df_new["observation[0]"] - pdr_x_arr)
    merged_df_new["h_error_old"] = np.hypot(merged_df_new["observation[1]"] - merged_df_new["pdr_y"],merged_df_new["observation[0]"] - merged_df_new["pdr_x"])

    merged_df_new["h_error_cumsum_new"] = merged_df_new["h_error_new"].cumsum()
    merged_df_new["h_error_cumsum_old"] = merged_df_new["h_error_old"].cumsum()
    plt.close('all')
    plt.figure(figsize=(12,8))
    plt.scatter(merged_df_new['valid_time'], merged_df_new['h_error_new'], label='New Position Error')
    plt.scatter(merged_df_new['valid_time'], merged_df_new['h_error_old'], label='Old Position Error')
    plt.legend()
    plt.title("Position Error (m)")
    plt.savefig(final_folder_path + "/" + file_name + "_position_error.png")

    plt.close('all')
    plt.figure(figsize=(12,8))
    plt.scatter(merged_df_new['valid_time'], merged_df_new['h_error_cumsum_new'], label='New Cumulative Position Error')
    plt.scatter(merged_df_new['valid_time'], merged_df_new['h_error_cumsum_old'], label='Old Cumulative Position Error')
    plt.legend()
    plt.title("Cumulative Position Error (m)")
    plt.savefig(final_folder_path + "/" + file_name + "_cumulative_position_error.png")

    max_error_old = merged_df_new["h_error_old"].max()
    max_error_new = merged_df_new["h_error_new"].max()
