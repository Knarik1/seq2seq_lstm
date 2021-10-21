sample_rate = 16000
classes_num = 88    # Number of notes of piano
begin_note = 21     # MIDI note of A0, the lowest note of a piano.
segment_seconds = 4.09	# Training segment duration
hop_seconds = 1.
frames_per_second = 100
velocity_scale = 128

# Spectogram configs
window_size = 2048
hop_size = 128
fmin = 30
fmax = sample_rate // 2
window = 'hann'
center = True
pad_mode = 'reflect'
ref = 1.0
amin = 1e-10
top_db = None
