import os
import math
import pickle
import numpy as np
import cv2 as cv
import mediapipe as mp
import pandas as pd
from scipy.signal import find_peaks
from numpy.fft import fft, ifft
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.stats import iqr

# Initialize MediaPipe hand detector
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2)
mp_draw = mp.solutions.drawing_utils

# Helper functions
def distance(x0, y0, x1, y1):
    return math.sqrt((x0 - x1)**2 + (y0 - y1)**2)

def denoise(D, WINDOW_SIZE=3, THRESHOLD=3):
    D = list(D)
    Y = []
    for i in range(len(D)):
        if D[i] != -1.0:
            Y.append(D[i])
        else:
            Y.append(np.nan)
    
    Y = pd.Series(Y)
    Y = Y.interpolate(method="polynomial", order=3)
    
    for i in range(len(Y)):
        if Y[i] < 0:
            Y[i] = -1.0
    
    for i in range(len(D)):
        if D[i] == -1.0:
            if i >= WINDOW_SIZE:
                vals_before = D[(i - WINDOW_SIZE):(i - 1)]
            else:
                vals_before = D[0:(i - 1)]
                for j in range(WINDOW_SIZE - len(vals_before)):
                    vals_before = [-1.0] + vals_before

            if i < (len(D) - WINDOW_SIZE):
                vals_after = D[(i + 1):(i + WINDOW_SIZE)]
            else:
                vals_after = D[(i + 1):]
                for j in range(WINDOW_SIZE - len(vals_after)):
                    vals_after = vals_after + [-1.0]

            vals = vals_before + [-1.0] + vals_after
            if len(np.argwhere(np.asarray(vals) == -1.0)) <= THRESHOLD:
                D[i] = Y[i]
                if np.isnan(D[i]):
                    D[i] = -1.0
    
    return np.asarray(D)

def custom_peaks(D, distance):
    peaks, _ = find_peaks(D, distance=(int)(distance))
    n_peaks = len(peaks)
    middle_peaks = D[peaks[(int)(np.floor((n_peaks-1)/4)):(int)(np.floor(3*(n_peaks-1)/4))]]
    high_peak = np.percentile(middle_peaks, 80)
    height = np.floor(high_peak / 2)
    peaks, _ = find_peaks(D, distance=(int)(distance), height=(int)(height))
    return peaks

def custom_bottoms(D, distance):
    D = np.asarray(list(D))
    DI = 180 - D
    return custom_peaks(DI, distance)

def get_stats(series):
    from statistics import median, quantiles, StatisticsError
    if len(series) == 0:
        return {
            'median': None,
            'quartile_range': None,
            'mean': None,
            'min': None,
            'max': None,
            'stdev': None
        }
    stats = {
        'median': median(series),
        'quartile_range': iqr(series),
        'mean': np.mean(series),
        'min': np.min(series),
        'max': np.max(series),
        'stdev': np.std(series)
    }
    return stats

def linear_regression_fit(series_x, series_y):
    fit = {}
    series_x = np.asarray(series_x).reshape((-1, 1))
    series_y = np.asarray(series_y)
    model = LinearRegression()
    model.fit(series_x, series_y)
    fit["fitness_r2"] = model.score(series_x, series_y)
    fit["slope"] = model.coef_[0]
    return fit

def degree_for_good_fit(series_x, series_y, fitness_threshold=0.90):
    d = 0
    r2 = 0
    while r2 < fitness_threshold:
        d += 1
        x = np.asarray(series_x)
        y = np.asarray(series_y)
        z = np.polyfit(x, y, d)
        p = np.poly1d(z)
        r2 = r2_score(y, p(x))
        if d >= 10:
            return d
    return d

def entropy(p):
    return -(p * np.log(p)).sum()

def angular_amplitude_entropy(values, min_val=0, max_val=90, n_buckets=18):
    dA = (max_val - min_val) / (n_buckets - 1)
    buckets = np.arange(min_val, max_val + 1, dA)
    n = np.histogram(values, buckets)[0]
    p = n / n.sum()
    p[p == 0] = 1
    lp = np.log(p)
    ppe = -np.multiply(p, lp).sum() / np.log(n_buckets)
    return ppe

def period_entropy(values, min_val=0, max_val=2, n_buckets=50):
    dA = (max_val - min_val) / (n_buckets - 1)
    buckets = np.arange(min_val, max_val + 1, dA)
    n = np.histogram(values, buckets)[0]
    p = n / n.sum()
    p[p == 0] = 1
    lp = np.log(p)
    ppe = -np.multiply(p, lp).sum() / np.log(n_buckets)
    return ppe

def get_final_features(data):
    signal = Signal(data['D_raw'], data['W_raw'], data['num_frames'], data['duration'])
    features = signal.wrist_movements()
    features['aperiodicity_denoised'] = signal.aperiodicity('d')
    features['aperiodicity_trimmed'] = signal.aperiodicity('t')
    features['periodEntropy_denoised'] = period_entropy(signal.periods_denoised)
    features['periodEntropy_trimmed'] = period_entropy(signal.periods_trimmed)
    features['periodVarianceNorm_denoised'] = np.var(signal.periods_denoised) / np.max(signal.periods_denoised)
    features['periodVarianceNorm_trimmed'] = np.var(signal.periods_trimmed) / np.max(signal.periods_trimmed)
    features['numInterruptions_denoised'] = signal.interruption_count('d')
    features['numInterruptions_trimmed'] = signal.interruption_count('t')
    features['numFreeze_denoised'] = signal.freeze_count('d')
    features['numFreeze_trimmed'] = signal.freeze_count('t')
    features['maxFreezeDuration_denoised'] = signal.max_freeze_duration('d')
    features['maxFreezeDuration_trimmed'] = signal.max_freeze_duration('t')
    period_stats_denoised = get_stats(signal.periods_denoised)
    for k in period_stats_denoised.keys():
        features['period_' + k + "_denoised"] = period_stats_denoised[k]
    period_stats_trimmed = get_stats(signal.periods_trimmed)
    for k in period_stats_trimmed.keys():
        features['period_' + k + "_trimmed"] = period_stats_trimmed[k]
    features['period_entropy_denoised'] = period_entropy(signal.periods_denoised)
    features['period_entropy_trimmed'] = period_entropy(signal.periods_trimmed)
    frequency_stats_denoised = get_stats(1.0 / np.asarray(signal.periods_denoised))
    for k in frequency_stats_denoised.keys():
        features['frequency_' + k + "_denoised"] = frequency_stats_denoised[k]
    frequency_stats_trimmed = get_stats(1.0 / np.asarray(signal.periods_trimmed))
    for k in frequency_stats_trimmed.keys():
        features['frequency_' + k + "_trimmed"] = frequency_stats_trimmed[k]
    frequency_fit_denoised = linear_regression_fit(np.arange(0, len(signal.periods_denoised)), 1.0 / np.asarray(signal.periods_denoised))
    for k in frequency_fit_denoised.keys():
        features['frequency_lr_' + k + '_denoised'] = frequency_fit_denoised[k]
    frequency_fit_trimmed = linear_regression_fit(np.arange(0, len(signal.periods_trimmed)), 1.0 / np.asarray(signal.periods_trimmed))
    for k in frequency_fit_trimmed.keys():
        features['frequency_lr_' + k + '_trimmed'] = frequency_fit_trimmed[k]
    features['frequency_fit_min_degree_denoised'] = degree_for_good_fit(np.arange(0, len(signal.periods_denoised)), 1.0 / np.asarray(signal.periods_denoised))
    features['frequency_fit_min_degree_trimmed'] = degree_for_good_fit(np.arange(0, len(signal.periods_trimmed)), 1.0 / np.asarray(signal.periods_trimmed))
    amp_stats = signal.amplitude_stats('d')
    for k in amp_stats:
        features[k] = amp_stats[k]
    amp_stats = signal.amplitude_stats('t')
    for k in amp_stats:
        features[k] = amp_stats[k]
    amp_dec_denoised = signal.amplitude_decrement('d')
    for k in amp_dec_denoised.keys():
        features['amplitude_decrement_' + k + '_denoised'] = amp_dec_denoised[k]
    amp_dec_trimmed = signal.amplitude_decrement('t')
    for k in amp_dec_trimmed.keys():
        features['amplitude_decrement_' + k + '_trimmed'] = amp_dec_trimmed[k]
    features['num_peaks_trimmed'] = len(signal.peaks_trimmed)
    features['num_peaks_denoised'] = len(signal.peaks_denoised)
    features['num_interruptions_norm_denoised'] = features['numInterruptions_denoised'] / features['num_peaks_denoised']
    features['num_freeze_norm_denoised'] = features['numFreeze_denoised'] / features['num_peaks_denoised']
    features['num_interruptions_norm_trimmed'] = features['numInterruptions_trimmed'] / features['num_peaks_trimmed']
    features['num_freeze_norm_trimmed'] = features['numFreeze_trimmed'] / features['num_peaks_trimmed']
    speed_stats_denoised = get_stats(np.abs(signal.speed_denoised))
    for k in speed_stats_denoised.keys():
        features['speed_' + k + "_denoised"] = speed_stats_denoised[k]
    speed_stats_trimmed = get_stats(np.abs(signal.speed_trimmed))
    for k in speed_stats_trimmed.keys():
        features['speed_' + k + "_trimmed"] = speed_stats_trimmed[k]
    acceleration_stats_denoised = get_stats(np.abs(signal.acceleration_denoised))
    for k in acceleration_stats_denoised.keys():
        features['acceleration_' + k + "_denoised"] = acceleration_stats_denoised[k]
    acceleration_stats_trimmed = get_stats(np.abs(signal.acceleration_trimmed))
    for k in acceleration_stats_trimmed.keys():
        features['acceleration_' + k + "_trimmed"] = acceleration_stats_trimmed[k]
    return features

class Signal:
    NOT_FOUND = -1
    INTERRUPTION_SPEED_THRESHOLD = 50  # degree per second
    INTERRUPTION_MIN_DURATION = 0.20  # second
    FREEZE_SPEED_THRESHOLD = 50  # degree per second
    FREEZE_MIN_DURATION = 0.30  # second

    def __init__(self, raw, wrist_raw, num_frames, duration):
        self.raw_signal = raw
        self.wrist_raw = wrist_raw
        self.raw_fft = fft(self.raw_signal) if len(self.raw_signal) > 0 else np.array([])
        self.NUM_FRAMES = num_frames
        self.DURATION = duration
        self.PER_FRAME_DURATION = self.DURATION / self.NUM_FRAMES if self.NUM_FRAMES > 0 else 0
        self.denoised_signal, self.wrist_denoised = self.interpolation_and_denoise()

        self.peaks_denoised = self.peak_detection(self.denoised_signal)
        
        if len(self.peaks_denoised) > 1:
            self.peaks_trimmed = np.asarray(self.peaks_denoised[1:-1] - self.peaks_denoised[1])
        else:
            self.peaks_trimmed = np.asarray([])

        if len(self.peaks_denoised) > 1:
            self.trimmed_signal = np.asarray(self.denoised_signal[self.peaks_denoised[1]:(self.peaks_denoised[-2] + 1)])
            self.wrist_trimmed = self.wrist_denoised[self.peaks_denoised[1]:(self.peaks_denoised[-2] + 1)]
        else:
            self.trimmed_signal = np.asarray([])
            self.wrist_trimmed = np.asarray([])

        self.signals = {'r': self.raw_signal, 'd': self.denoised_signal, 't': self.trimmed_signal}
        self.peaks = {'d': self.peaks_denoised, 't': self.peaks_trimmed}
        self.periods_denoised = []
        self.periods_trimmed = []
        self.speed_denoised = []
        self.speed_trimmed = []
        self.acceleration_denoised = []
        self.acceleration_trimmed = []

        for i in range(1, len(self.peaks_denoised)):
            self.periods_denoised.append((self.peaks_denoised[i] - self.peaks_denoised[i - 1]) * self.PER_FRAME_DURATION)

        for i in range(1, len(self.peaks_trimmed)):
            self.periods_trimmed.append((self.peaks_trimmed[i] - self.peaks_trimmed[i - 1]) * self.PER_FRAME_DURATION)

        for i in range(len(self.denoised_signal) - 1):
            self.speed_denoised.append(self.denoised_signal[i + 1] - self.denoised_signal[i])

        self.speed_denoised = np.asarray(self.speed_denoised)  # degree per frame
        self.speed_denoised = self.speed_denoised / self.PER_FRAME_DURATION if self.PER_FRAME_DURATION > 0 else np.array([])  # degree per second

        for i in range(len(self.trimmed_signal) - 1):
            self.speed_trimmed.append(self.trimmed_signal[i + 1] - self.trimmed_signal[i])

        self.speed_trimmed = np.asarray(self.speed_trimmed)  # degree per frame
        self.speed_trimmed = self.speed_trimmed / self.PER_FRAME_DURATION if self.PER_FRAME_DURATION > 0 else np.array([])  # degree per second
        self.speeds = {'d': self.speed_denoised, 't': self.speed_trimmed}

        for i in range(len(self.speed_denoised) - 1):
            self.acceleration_denoised.append(self.speed_denoised[i + 1] - self.speed_denoised[i])

        self.acceleration_denoised = np.asarray(self.acceleration_denoised)  # degree per frame*second
        self.acceleration_denoised = self.acceleration_denoised / self.PER_FRAME_DURATION if self.PER_FRAME_DURATION > 0 else np.array([])  # degree per second^2

        for i in range(len(self.speed_trimmed) - 1):
            self.acceleration_trimmed.append(self.speed_trimmed[i + 1] - self.speed_trimmed[i])

        self.acceleration_trimmed = np.asarray(self.acceleration_trimmed)  # degree per frame*second
        self.acceleration_trimmed = self.acceleration_trimmed / self.PER_FRAME_DURATION if self.PER_FRAME_DURATION > 0 else np.array([])  # degree per second^2

    def interpolation_and_denoise(self):
        D = denoise(self.raw_signal)
        first_frame = self.NOT_FOUND
        last_frame = self.NOT_FOUND
        max_first_frame = -1
        max_last_frame = -1
        max_num_frames = 0
        for i in range(len(D)):
            if D[i] != self.NOT_FOUND:
                if first_frame == self.NOT_FOUND:
                    first_frame = i
                else:
                    last_frame = i
                    num_frames = last_frame - first_frame + 1
                    if num_frames > max_num_frames:
                        max_num_frames = num_frames
                        max_first_frame = first_frame
                        max_last_frame = last_frame
            else:
                first_frame = self.NOT_FOUND
                last_frame = self.NOT_FOUND

        D = D[max_first_frame:max_last_frame + 1]
        W = self.wrist_raw[max_first_frame:max_last_frame + 1]
        return D, W

    def peak_detection(self, D):
        X = np.arange(len(D))
        MIN_PERIOD = 0.15  # in seconds
        d_min = (int)(MIN_PERIOD / self.PER_FRAME_DURATION) if self.PER_FRAME_DURATION > 0 else 1
        peaks = custom_peaks(D, distance=d_min)
        n_peaks = len(peaks)
        bottoms = custom_bottoms(D, d_min)
        n_bottoms = len(bottoms)
        middle_bottoms = D[bottoms[(int)(np.floor((n_bottoms - 1) / 4)):(int)(np.floor(3 * (n_bottoms - 1) / 4))]]
        BOTTOM_MAX_HEIGHT = 10
        peaks_denoised = [peaks[0]]
        for i in range(len(peaks) - 1):
            min_val = 180.0
            for j in range(peaks[i], peaks[i + 1]):
                if D[j] < min_val:
                    min_val = D[j]

            if min_val < BOTTOM_MAX_HEIGHT:
                peaks_denoised.append(peaks[i + 1])

        return peaks_denoised

    def aperiodicity(self, signal_version):
        signal = self.signals[signal_version.lower()]
        if len(signal) == 0:
            return None
        X = fft(signal)
        power_spectrum = np.square(np.abs(X))
        power_spectrum = power_spectrum / power_spectrum.sum()
        return entropy(power_spectrum)

    def interruption_count(self, signal_version):
        n = 0
        S = np.abs(self.speeds[signal_version.lower()])
        t = 0
        for i in range(len(S)):
            if S[i] <= self.INTERRUPTION_SPEED_THRESHOLD:
                t += 1
            else:
                if (t * self.PER_FRAME_DURATION) >= self.INTERRUPTION_MIN_DURATION:
                    n += 1
                t = 0
        return n

    def freeze_count(self, signal_version):
        n = 0
        S = np.abs(self.speeds[signal_version.lower()])
        t = 0
        for i in range(len(S)):
            if S[i] <= self.FREEZE_SPEED_THRESHOLD:
                t += 1
            else:
                if (t * self.PER_FRAME_DURATION) >= self.FREEZE_MIN_DURATION:
                    n += 1
                t = 0
        return n

    def max_freeze_duration(self, signal_version):
        S = np.abs(self.speeds[signal_version.lower()])
        t = 0
        t_max = 0
        for i in range(len(S)):
            if S[i] <= self.FREEZE_SPEED_THRESHOLD:
                t += 1
            else:
                if t > t_max:
                    t_max = t
                t = 0
        return t_max * self.PER_FRAME_DURATION

    def amplitude_decrement(self, signal_version):
        D = self.signals[signal_version]
        t = self.peaks[signal_version]
        A = D[t]
        n = len(A)
        assert (n >= 2), "Not enough peaks to analyze"
        n1 = round(n / 2)
        feats = linear_regression_fit(t, -A)
        feats['end_to_mean'] = np.mean(A) - A[-1]
        feats['fit_min_degree'] = degree_for_good_fit(t, A)
        feats['last_to_first_half'] = np.mean(A[:(n1 + 1)]) - np.mean(A[(n1 + 1):])
        return feats

    def amplitude_stats(self, signal_version):
        D = self.signals[signal_version]
        t = self.peaks[signal_version]
        A = D[t]
        texts = {"d": "denoised", "t": "trimmed"}
        feats = {}
        amp_stats = get_stats(A)
        for k in amp_stats.keys():
            feats["amplitude_" + k + "_" + texts[signal_version]] = amp_stats[k]
        feats['amplitude_entropy_' + texts[signal_version]] = angular_amplitude_entropy(A)
        return feats

    def wrist_movements(self):
        W = self.wrist_trimmed
        n = len(W)
        movements_x = []
        movements_y = []
        movements_d = []
        for i in range(1, n):
            (x1, y1) = W[i]
            (x0, y0) = W[i - 1]
            if x1 == self.NOT_FOUND or x0 == self.NOT_FOUND:
                movements_x.append(0)
                movements_y.append(0)
                movements_d.append(0)
            else:
                movements_x.append((x1 - x0) / self.PER_FRAME_DURATION)
                movements_y.append((y1 - y0) / self.PER_FRAME_DURATION)
                movements_d.append(distance(x0, y0, x1, y1) / self.PER_FRAME_DURATION)

        feats = {}
        feats_x = get_stats(np.abs(movements_x))
        for k in feats_x.keys():
            feats['wrist_mvmnt_x_' + k] = feats_x[k]
        feats_y = get_stats(np.abs(movements_y))
        for k in feats_y.keys():
            feats['wrist_mvmnt_y_' + k] = feats_y[k]
        feats_d = get_stats(np.abs(movements_d))
        for k in feats_d.keys():
            feats['wrist_mvmnt_dist_' + k] = feats_d[k]
        return feats

class HandTrackerCustomized():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def handsFinder(self, image, draw=True):
        imageRGB = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)
        landmarks = {'left': {}, 'right': {}}

        if self.results.multi_handedness:
            for hand_landmarks, handedness in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
                hand_label = handedness.classification[0].label.lower()
                confidence_score = handedness.classification[0].score

                if confidence_score > 0.9:
                    for id, lm in enumerate(hand_landmarks.landmark):
                        h, w, c = image.shape
                        cx, cy = int(lm.x * w), int(lm.y * h)
                        landmarks[hand_label][id] = (cx, cy)
                    if draw:
                        self.mpDraw.draw_landmarks(image, hand_landmarks, self.mpHands.HAND_CONNECTIONS)
        return image, landmarks

# Set up webcam
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

D_left, D_right = [], []
W_left, W_right = [], []
NOT_FOUND = -1
NUM_FRAMES = 0

tracker = HandTrackerCustomized()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Flip the frame horizontally
    frame = cv.flip(frame, 1)

    NUM_FRAMES += 1
    frame, landmarks = tracker.handsFinder(frame)

    if landmarks['left']:
        wrist_x, wrist_y = landmarks['left'][0]
        thumb_x, thumb_y = landmarks['left'][4]
        index_x, index_y = landmarks['left'][8]
        thumb_cmc_x, thumb_cmc_y = landmarks['left'][1]

        vector_wt = (thumb_x - wrist_x, thumb_y - wrist_y)
        vector_wi = (index_x - wrist_x, index_y - wrist_y)
        dot = vector_wt[0] * vector_wi[0] + vector_wt[1] * vector_wi[1]
        cosx = dot / (math.sqrt(vector_wt[0]**2 + vector_wt[1]**2) * math.sqrt(vector_wi[0]**2 + vector_wi[1]**2))
        cosx = np.minimum(cosx, 1.0)
        angle = (math.acos(cosx) * 180) / math.pi

        D_left.append(angle)
        w_norm = distance(wrist_x, wrist_y, thumb_cmc_x, thumb_cmc_y)
        W_left.append((wrist_x / w_norm, wrist_y / w_norm))
        print("Left Hand: Angle:", angle)

    if landmarks['right']:
        wrist_x, wrist_y = landmarks['right'][0]
        thumb_x, thumb_y = landmarks['right'][4]
        index_x, index_y = landmarks['right'][8]
        thumb_cmc_x, thumb_cmc_y = landmarks['right'][1]

        vector_wt = (thumb_x - wrist_x, thumb_y - wrist_y)
        vector_wi = (index_x - wrist_x, index_y - wrist_y)
        dot = vector_wt[0] * vector_wi[0] + vector_wt[1] * vector_wi[1]
        cosx = dot / (math.sqrt(vector_wt[0]**2 + vector_wt[1]**2) * math.sqrt(vector_wi[0]**2 + vector_wi[1]**2))
        cosx = np.minimum(cosx, 1.0)
        angle = (math.acos(cosx) * 180) / math.pi

        D_right.append(angle)
        w_norm = distance(wrist_x, wrist_y, thumb_cmc_x, thumb_cmc_y)
        W_right.append((wrist_x / w_norm, wrist_y / w_norm))
        print("Right Hand: Angle:", angle)

    if not landmarks['left'] and not landmarks['right']:
        print("No hands detected.")

    cv.imshow('Webcam', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

# Process the captured data
DURATION = NUM_FRAMES / 30.0  # assuming 30 FPS
data_left = {'D_raw': np.asarray(D_left), 'W_raw': W_left, 'num_frames': NUM_FRAMES, 'duration': DURATION}
data_right = {'D_raw': np.asarray(D_right), 'W_raw': W_right, 'num_frames': NUM_FRAMES, 'duration': DURATION}

features_left = get_final_features(data_left)
features_right = get_final_features(data_right)

# Print or save the extracted features
print("Left Hand Features:", features_left)
print("Right Hand Features:", features_right)
