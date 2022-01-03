import pathlib

# hue bridge settings
HUE_BRIDGE_IP = '192.168.1.100'
HUE_BRIDGE_USERNAME = 'HLtg5osJJGjwVXKt2jmdZuC-UnEe1d1kUmrCPeLq'

# directory settings
AKIKO_DIR = pathlib.Path('album-covers-akiko')
REIMU_DIR = pathlib.Path('reimus')
GENERAL_AC_DIR = pathlib.Path('album-covers-general')
ANIME_OPED_DIR = pathlib.Path('anime-oped')
PALETTE_DIR = pathlib.Path('palettes')
GAUSS_DIR = pathlib.Path('gauss')
QUANT_DIR = pathlib.Path('quant')
PINKS_DIR = pathlib.Path('pinks')

# Pillow settings
QUANTIZED_DIMENSION = 128   # pixels
FONT = 'DroidSans.ttf'

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
