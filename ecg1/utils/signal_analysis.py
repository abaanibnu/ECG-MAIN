import numpy as np
from scipy.signal import find_peaks, butter, filtfilt


def _butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = max(0.001, lowcut / nyq)
    high = min(0.98, highcut / nyq)
    if low >= high:
        high = min(0.98, low + 0.1)
    b, a = butter(order, [low, high], btype='band')
    return b, a


def analyze_ecg_signal(signal_data, fs=100.0):
    """
    Analyze ECG signal: detect R-peaks and compute heart rate.
    
    Key insight: The R-peak is always the TALLEST positive deflection per beat.
    P-waves and T-waves are shorter. We exploit this by:
      1. Finding all positive peaks
      2. Keeping only the top-percentile peaks by amplitude (= R-peaks only)
      3. Computing BPM from those peaks via median RR interval
    """
    if not signal_data or len(signal_data) < 20:
        return _inconclusive("Signal too short.")

    sig = np.array(signal_data, dtype=float)

    # --- 1. Bandpass filter: 0.5 – 40 Hz ---
    if len(sig) > 100 and fs > 5:
        try:
            b, a = _butter_bandpass(0.5, min(40.0, fs * 0.45), fs, order=3)
            sig = filtfilt(b, a, sig)
        except Exception:
            pass

    # --- 2. Normalize to [0, 1] based on the positive half ---
    # This ensures negative deflections don't affect the percentile calculation.
    sig_min = sig.min()
    sig_max = sig.max()
    sig_range = sig_max - sig_min
    if sig_range == 0:
        return _inconclusive("Signal is flat.")
    sig_norm = (sig - sig_min) / sig_range  # now in [0, 1]

    # --- 3. Find all candidate peaks with loose criteria ---
    # Use 300ms min distance – this prevents multiple detections within same QRS
    min_dist = max(int(fs * 0.30), 3)
    all_peaks, _ = find_peaks(sig_norm, distance=min_dist, height=0.1)

    if len(all_peaks) < 2:
        # Try a very loose detection
        all_peaks, _ = find_peaks(sig_norm, distance=max(int(fs * 0.20), 2))

    if len(all_peaks) < 2:
        return _inconclusive("Could not detect enough peaks in the signal.")

    # --- 4. R-peak isolation: keep only the TOP 33% of peaks by height ---
    # R-peaks are always significantly taller than P and T waves.
    # By keeping only the tallest third, we exclude P/T waves reliably.
    peak_heights = sig_norm[all_peaks]
    threshold_33 = np.percentile(peak_heights, 67)   # top 33% = above 67th percentile
    r_peaks = all_peaks[peak_heights >= threshold_33]

    # Ensure we still have at least 2 R-peaks
    if len(r_peaks) < 2:
        r_peaks = all_peaks  # Fall back to all peaks

    # --- 5. Re-merge peaks that are too close together ---
    # After filtering, some consecutive R-peaks might still be too close (if
    # two nearby peaks pass the threshold). Keep the taller one in each pair.
    min_rr_samples = int(fs * 0.30)   # 300ms minimum
    r_peaks_final = [r_peaks[0]]
    for p in r_peaks[1:]:
        if p - r_peaks_final[-1] >= min_rr_samples:
            r_peaks_final.append(p)
        else:
            # Keep the taller one
            if sig_norm[p] > sig_norm[r_peaks_final[-1]]:
                r_peaks_final[-1] = p

    r_peaks_final = np.array(r_peaks_final)

    if len(r_peaks_final) < 2:
        return _inconclusive("Only one R-peak found after filtering.")

    # --- 6. Median RR interval → BPM ---
    rr = np.diff(r_peaks_final) / fs   # in seconds
    # Filter physiologically impossible values
    rr = rr[(rr >= 0.20) & (rr <= 3.0)]

    if len(rr) == 0:
        return _inconclusive("RR intervals out of physiological range.")

    bpm = 60.0 / float(np.median(rr))

    # --- 7. Classify ---
    if bpm < 60:
        abnormality = "Bradycardia"
        recommendation = (
            "Heart rate is lower than normal (below 60 BPM). "
            "Consult a doctor if you experience dizziness, fatigue, or fainting."
        )
    elif bpm > 100:
        abnormality = "Tachycardia"
        recommendation = (
            "Heart rate is higher than normal (above 100 BPM). "
            "Avoid caffeine and stress. Consult a doctor if persistent."
        )
    else:
        abnormality = "Normal"
        recommendation = "Your heart rate is within the normal range. Maintain a healthy lifestyle."

    stress = "High" if abnormality != "Normal" else "Low"

    return {
        'heart_rate': round(bpm, 1),
        'abnormality': abnormality,
        'stress_level': stress,
        'recommendation': recommendation
    }


def _inconclusive(reason=""):
    return {
        'heart_rate': 0,
        'abnormality': 'Inconclusive',
        'stress_level': 'Unknown',
        'recommendation': f'Signal analysis inconclusive: {reason}'
    }
