# --- PIPELINE CONFIGURATION ---
DIURNAL_WINDOW = 20    
DIURNAL_MIN_PERIODS = 5 

# HAR Lags (Geometric Sequence)
HAR_LAGS = [1, 5, 25, 125, 625, 3125] 

START_DATE = '2005-01-01'

# 1. Define Segments with Overlaps
SEGMENT_DEFINITIONS = {
    'morning':   {'start': 510, 'end': 660},   # 08:30 - 11:00
    'midday':    {'start': 630, 'end': 870},   # 10:30 - 14:30
    'closing':   {'start': 840, 'end': 960},   # 14:00 - 16:00
    'overnight': {'start': 990, 'end': 510}    # 16:30 - 08:30 (Wraps)
}