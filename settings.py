import pathlib

EXAMPLE_MODE = True

# hue bridge settings
HUE_BRIDGE_IP = '192.168.1.100'
HUE_BRIDGE_USERNAME = 'HLtg5osJJGjwVXKt2jmdZuC-UnEe1d1kUmrCPeLq'

# directory settings
INPUT_DIR = pathlib.Path('example-input') if EXAMPLE_MODE else pathlib.Path('input')
OUTPUT_DIR = pathlib.Path('example-output') if EXAMPLE_MODE else pathlib.Path('output')
PALETTE_DIR = OUTPUT_DIR.joinpath('palettes')
GAUSS_DIR = OUTPUT_DIR.joinpath('gauss')
QUANT_DIR = OUTPUT_DIR.joinpath('quant')

PALETTE_DIR_STEM = 'palettes'
GAUSS_DIR_STEM = 'gauss'
QUANT_DIR_STEM = 'quant'

# Pillow settings
QUANTIZED_DIMENSION = 128   # pixels
MAX_COLORS = 16     # feel free to change, this already feels like trolling
MAX_RADIUS = 100    # I don't even know if this is valid
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
