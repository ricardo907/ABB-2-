"""Map video times to the nearest timestamped robot-angle samples."""

import numpy as np


def nearest_sample_indices(video_times, sample_times):
    """Return nearest sample indices, treating the first sample as video time zero.

    A known camera/robot start offset must be corrected before calling this function.
    """
    video_times = np.asarray(video_times, dtype=np.float64)
    sample_times = np.asarray(sample_times, dtype=np.float64)
    if sample_times.ndim != 1 or sample_times.size == 0:
        raise ValueError("Sample times must be a non-empty one-dimensional array")
    if not np.isfinite(sample_times).all() or np.any(np.diff(sample_times) <= 0):
        raise ValueError("Sample times must be finite and strictly increasing")
    if not np.isfinite(video_times).all():
        raise ValueError("Video times must be finite")

    relative_times = sample_times - sample_times[0]
    right = np.clip(np.searchsorted(relative_times, video_times), 0, len(relative_times) - 1)
    left = np.maximum(right - 1, 0)
    use_left = np.abs(video_times - relative_times[left]) <= np.abs(
        relative_times[right] - video_times
    )
    return np.where(use_left, left, right)
