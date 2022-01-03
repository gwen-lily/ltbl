import pathlib

# hue bridge settings
HUE_BRIDGE_IP = '192.168.1.100'
HUE_BRIDGE_USERNAME = 'HLtg5osJJGjwVXKt2jmdZuC-UnEe1d1kUmrCPeLq'

# directory settings
OUTPUT_DIR = pathlib.Path('output')
PALETTE_DIR = OUTPUT_DIR.joinpath('palettes')
GAUSS_DIR = OUTPUT_DIR.joinpath('gauss')
QUANT_DIR = OUTPUT_DIR.joinpath('quant')

# Pillow settings
QUANTIZED_DIMENSION = 128   # pixels
# FONT = 'DroidSans.ttf'

# pandas
CSV_SEP = '\t'          # sep char to use for imported and exported csv's
PALETTE_COLUMNS = [
	'r',
	'g',
	'b',
	'x',
	'y'
]

ACTIVE_LIGHTS_NAMES = [
	'terpsichore',
]

# exceptions
# phue.PhueRequestTimeout
