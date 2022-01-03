from phue import Bridge
import phue
import logging
import pathlib
import random
import json
from PIL import Image, ImageFilter, ImageDraw, ImageFont
import PIL
import cv2
from sklearn.cluster import MiniBatchKMeans
import numpy as np
from typing import List, Tuple, Iterable, Union
import matplotlib.pyplot as plt
from rgbxy import Converter, GamutB
from time import sleep
import datetime as dt
import pandas as pd
import sys

from colorthief import ColorThief
import extcolors
import colorgram

from settings import *

logging.basicConfig()
b = Bridge(HUE_BRIDGE_IP)
b.connect()     # register the app
converter = Converter(GamutB)   # A19 bulbs I think I'm right here


def is_image(filepath: pathlib.Path) -> bool:
    try:
        with Image.open(filepath):
            return True
    except PIL.UnidentifiedImageError:
        return False


def is_unprocessed_image(filepath: pathlib.Path) -> bool:
    return is_image(filepath) and '-p-' not in filepath.stem


def is_rgb_256_color(rgb256_color):
    return all(isinstance(channel, int) and 0 <= channel <= 255 for channel in rgb256_color)


def is_rgb_01_color(rgb01_color):
    return all(isinstance(channel, float) and 0 <= channel <= 1 for channel in rgb01_color)


def is_xy_color(xy_color) -> bool:
    return isinstance(xy_color, tuple) and len(xy_color) == 2 and all(isinstance(c, float) for c in xy_color)


def filepath_plus_n(filepath: pathlib.Path, join_char: str = '-', additional_label: str = None) -> pathlib.Path:

    counter = 0
    while True:
        counter += 1
        filename_elements = [filepath.stem, 'p', str(counter)]

        if additional_label:
            filename_elements.append(additional_label)

        filename_out = join_char.join(filename_elements) + filepath.suffix
        filepath_out = filepath.parent.joinpath(filename_out)

        if not filepath_out.exists():
            return filepath_out


def load_dataframe(filepath) -> pd.DataFrame:
    return pd.read_csv(filepath, sep=CSV_SEP)


def save_dataframe(filepath, dataframe: pd.DataFrame):
    dataframe.to_csv(
        path_or_buf=filepath,
        sep=CSV_SEP,
        index=False
    )


def get_empty_palette_df():
    dataframe = pd.DataFrame(columns=PALETTE_COLUMNS)
    data_types = {
        'r': int,
        'g': int,
        'b': int,
        'x': float,
        'y': float,
        # 'active': bool,
    }
    dataframe = dataframe.astype(data_types)
    dataframe['active'] = True
    # dataframe['r'] = pd.to_numeric(dataframe['r'], downcast='integer')
    # dataframe['g'] = pd.to_numeric(dataframe['g'], downcast='integer')
    # dataframe['b'] = pd.to_numeric(dataframe['b'], downcast='integer')
    return dataframe


def extract_palette(filepath: pathlib.Path, k: int, radius: int) -> Iterable[Tuple[int]]:
    # pil_image = Image.open(filepath).convert('RGB').resize(2*(QUANTIZED_DIMENSION,))
    # image = np.array(pil_image)[:, :, ::-1].copy()

    def sanitize_rgb_tuple(rgb: Tuple[int]) -> Tuple[int]:
        return tuple(map(lambda x: int(min([255, max([0, x])])), rgb))

    filepath_gauss = filepath_plus_n(GAUSS_DIR.joinpath(filepath.stem + '.png'), additional_label='gauss')
    filepath_quant = filepath_plus_n(QUANT_DIR.joinpath(filepath.stem + '.png'), additional_label='quant')

    image_raw = Image.open(filepath)
    image_gauss = image_raw.filter(ImageFilter.GaussianBlur(radius))
    image_gauss.save(filepath_gauss)

    # read gauss and convert from RGB >> L*a*b space and then reshape to feature vector
    image = cv2.imread(str(filepath_gauss))
    (h, w) = image.shape[:2]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # apply k-means using the specified number of clusters and
    # then create the quantized image based on the predictions
    clt = MiniBatchKMeans(n_clusters=k)
    labels = clt.fit_predict(image)
    quant = clt.cluster_centers_.astype("uint8")[labels]

    # vectors > L*a*b > RGB
    quant = quant.reshape((h, w, 3))
    image = image.reshape((h, w, 3))

    quant = cv2.cvtColor(quant, cv2.COLOR_LAB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_LAB2BGR)
    cv2.imwrite(str(filepath_quant), quant)

    # display
    # cv2.imshow("image", np.hstack([image, quant]))
    # cv2.waitKey(0)

    ct = ColorThief(filepath_quant)
    ct_palette = ct.get_palette(color_count=k, quality=1)
    sanitized_palette = [sanitize_rgb_tuple(t) for t in ct_palette]
    return sanitized_palette


def remove_greys(colors: Iterable[Tuple[int]], sensitivity: float = 0.1):
    adjusted_sensitivity = min([1, max([0, sensitivity])])
    no_greys = [c for c in colors if distance_from_grey(c) > adjusted_sensitivity]
    return no_greys


def rgb_to_hex(color_rgb: tuple):
    return '%02x%02x%02x' % color_rgb


def rgb256_to_rgb01(rgb_256: Tuple[int]) -> Tuple[float]:
    return tuple(map(lambda channel: float(channel/255), rgb_256))


def rgb01_to_rgb256(rgb_01: Tuple[float]) -> Tuple[int]:
    return tuple(map(lambda channel: int(channel*255), rgb_01))


def distance_from_grey(rgb_color: Tuple[Union[int, float]]) -> float:
    if is_rgb_256_color(rgb_color):
        p0 = rgb256_to_rgb01(rgb_color)
    elif is_rgb_01_color(rgb_color):
        p0 = rgb_color
    else:
        raise Exception('bad color')

    x1 = np.asarray([0, 0, 0])
    x2 = np.asarray([1, 1, 1])
    x0 = np.asarray(p0)
    max_distance = np.sqrt(2/3)     # distance from grey line to any corner of R, G, B

    # calculate distance using classic formula and normalize to [0, 1]
    cross_vector = np.cross(x0 - x1, x0 - x2)
    distance = np.dot(cross_vector, cross_vector)**(1/2) / np.sqrt(3)
    normalized_distance = distance / max_distance
    return normalized_distance


def distance_between_colors(c1: Tuple[Union[int, float]], c2: Tuple[Union[int, float]]) -> float:
    if is_rgb_256_color(c1):
        color_1 = c1
    elif is_rgb_01_color(c1):
        color_1 = rgb01_to_rgb256(c1)
    else:
        raise Exception('bad color 1')

    if is_rgb_256_color(c2):
        color_2 = c2
    elif is_rgb_01_color(c2):
        color_2 = rgb01_to_rgb256(c2)
    else:
        raise Exception('bad color 2')

    rmean = 0.5 * (color_1[0] + color_2[0])
    dr, dg, db = np.asarray(color_1) - np.asarray(color_2)
    dc1 = np.sqrt((2 + rmean/256)*rmean**2 + 4*dg**2 + (2 + (255-rmean)/256)*db**2)
    # >>8 binary bitshift divides by 256
    # dc2 = np.sqrt((((512+rmean)*dr*dr)>>8) + 4*dg*dg + (((767-rmean)*db*db)>>8))
    dc = dc1
    return dc


def plot_palette(rgb_colors: Iterable[tuple]):

    # create figure
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    for palette_color in rgb_colors:
        palette_rgb01 = rgb256_to_rgb01(palette_color)
        ax.scatter3D(*palette_rgb01, color=rgb256_to_rgb01(palette_color))

    plt.show()


def get_lights():
    return [light for light in b.lights if light.name in ACTIVE_LIGHTS_NAMES]

def hue_sandbox(light_id: int):
    b.set_light(light_id, 'xy', [random.random(), random.random()])
    data = {'on': True, 'transitiontime': 10}
    # b.create_schedule('wake up grab a brush', '2021-12-07T13:22:00', 1, data, 'wake me up inside')


def cycle_color_palette(colors: List[Tuple[int, int, int]], cycle_time: int, transition_time: int, brightness: int):

    # light prep
    adjusted_trans_time = min([300, max([0, transition_time])])
    adjusted_brightness = min([254, max([0, brightness])])
    lights = get_lights()

    for light in lights:
        light.on = True
        light.brightness = adjusted_brightness
        light.transitiontime = adjusted_trans_time

    for cycle_color in colors:
        cycle_color_xy = converter.rgb_to_xy(*cycle_color)

        for light in lights:
            light.xy = cycle_color_xy

        print('color: ', *cycle_color, *cycle_color_xy)
        sleep(cycle_time)


def manual_color_walk(filepath: pathlib.Path):
    palette_df = load_dataframe(filepath)
    palette = load_palette(filepath)
    lights = get_lights()

    for light in lights:
        light.on = True
        light.brightness = 254
        light.transitiontime = 0
        light.xy = palette[0]

    input_text = 'Evaluate the color and enter your response: (y)es / (n)o / (q)uit\n\t'
    bool_data = []

    for index, color in enumerate(palette):
        for light in lights:
            light.xy = color

        color_text = 'x: {:.2f} y: {:.2f}'
        print(color_text.format(*color))

        while True:
            user_input = input(input_text)
            user_input = user_input.strip().lower()

            if user_input in ['y', 'yes']:
                bool_data.append(True)
                break
            elif user_input in ['n', 'no']:
                bool_data.append(False)
                break
            elif user_input in ['q', 'quit']:
                sys.exit(0)

    palette_df = palette_df.assign(active=bool_data)
    save_dataframe(filepath, palette_df)


def random_color_walk(colors: List[Tuple[Union[int, float]]],
                      cycle_time: Union[int, Iterable[int]],
                      transition_time: Union[int, Iterable[int]],
                      brightness: Union[int, Iterable[int]],
                      time_limit: int,
                      verbose: bool = True
                      ):

    def unpack_stat_iterable(var: Union[int, Iterable[int]]) -> Iterable[int]:
        # if only a value is provided, return mean value + 0 stdev, else return as is
        if isinstance(var, int):
            return var, 0
        else:
            return var

    ct_mean, ct_stdev = unpack_stat_iterable(cycle_time)
    tt_mean, tt_stdev = unpack_stat_iterable(transition_time)
    brightness_mean, bright_stdev = unpack_stat_iterable(brightness)

    lights = get_lights()
    for light in lights:
        light.on = True

    lightshow_start = dt.datetime.now()
    lightshow_now = dt.datetime.now()

    while (lightshow_now - lightshow_start).seconds < time_limit:
        random_ct = int(random.normalvariate(ct_mean, ct_stdev))
        random_ct = max([0, random_ct])

        for light in lights:
            random_color_unk = random.choice(colors)

            if is_xy_color(random_color_unk):
                random_color_xy = random_color_unk
            else:
                random_color_xy = converter.rgb_to_xy(*random_color_unk)

            random_tt = int(random.normalvariate(tt_mean, tt_stdev))
            random_tt = min([300, max([0, random_tt])])

            random_brightness = int(random.normalvariate(brightness_mean, bright_stdev))
            random_brightness = min([254, max([0, random_brightness])])

            light.xy = random_color_xy
            light.transitiontime = random_tt
            light.brightness = random_brightness
            print(light.name, 'xy:', light.xy)

        sleep(random_ct)
        lightshow_now = dt.datetime.now()


def generate_palette_grid(grid_name: str, save_palette: bool = True, display_palette: bool = False, *colors: tuple):
    grid_dim = int(np.ceil(np.sqrt(len(colors))))
    quadrant_size = 128
    font_size = 16
    text_offset = int(1.5 * font_size)

    w, h = grid_dim * quadrant_size, grid_dim * quadrant_size

    anchors = [(i * quadrant_size, j * quadrant_size) for i in range(grid_dim) for j in range(grid_dim)]
    palette_image = Image.new('RGB', (w, h))
    palette_draw = ImageDraw.Draw(palette_image)

    for color, (x, y) in zip(colors, anchors):
        # fill square
        shape = [(x, y), (x + quadrant_size, y + quadrant_size)]
        fill_color = '#' + rgb_to_hex(color)
        palette_draw.rectangle(shape, fill=fill_color)

        # write stats
        delta_grey = distance_from_grey(color)
        delta_nn = min([distance_between_colors(color, color_i) for color_i in colors])

        xmid, ymid = (x + quadrant_size // 2, y + quadrant_size // 2)
        palette_draw.text((x, ymid - 2 * text_offset), 'rgb: ({:.0f}, {:.0f}, {:.0f}'.format(*color))
        palette_draw.text((x, ymid - 1 * text_offset), 'hex: {:}'.format(fill_color))
        palette_draw.text((x, ymid + 1 * text_offset), 'grey_delta: {:.3f}'.format(delta_grey))
        palette_draw.text((x, ymid + 2 * text_offset), 'nn_delta: {:.3f}'.format(delta_nn))

    if display_palette:
        palette_image.show()

    if save_palette:
        filepath_palette = PALETTE_DIR.joinpath(grid_name + '.png')
        # filepath_palette = filepath_plus_n(filepath_palette, additional_label='palette')
        palette_image.save(filepath_palette)


def get_palette(
        name: str,
        colors: int = 10,
        blur_radius: int = 4,
        save_palette: bool = True,
        display_palette: bool = False,
        filter_greys: bool = True,
        *image_filepaths: pathlib.Path
) -> List[Tuple[int]]:
    # assert all(fp.exists() for fp in filepaths)

    full_palette = []

    for img_fp in image_filepaths:
        if not img_fp.exists():
            warning_text = str(img_fp) + 'does not appear to exist, skipping...'
            # raise UserWarning(warning_text)
            print(warning_text)
            continue

        palette = extract_palette(img_fp, colors, blur_radius)
        full_palette.extend(palette)

        if save_palette or display_palette:
            generate_palette_grid(img_fp.stem, save_palette, display_palette, *palette)

    full_palette = remove_greys(full_palette) if filter_greys else full_palette

    if save_palette:
        generate_palette_grid(name, save_palette, display_palette, *full_palette)

        full_palette_csv_filepath = PALETTE_DIR.joinpath(name + '.csv')
        # full_palette_csv_filepath = filepath_plus_n(full_palette_csv_filepath, additional_label='palette')
        full_palette_df = get_empty_palette_df()

        for rgb in full_palette:
            x, y = converter.rgb_to_xy(*rgb)
            data = {'r': rgb[0], 'g': rgb[1], 'b': rgb[2], 'x': x, 'y': y, 'active': True}
            full_palette_df = full_palette_df.append(data, ignore_index=True)

        save_dataframe(full_palette_csv_filepath, full_palette_df)

    return full_palette


def load_palette(filepath: pathlib.Path) -> List[Tuple[float]]:
    palette_df = load_dataframe(filepath)
    active_df = palette_df.loc[(palette_df['active'] == True)]

    palette_xy = [tuple(r[3:5]) for r in palette_df.to_numpy()]
    return palette_xy


if __name__ == '__main__':

    palette_name = 'akiko-utau-oka'
    images_dir = pathlib.Path('general')
    filepaths = [f for f in images_dir.iterdir() if is_unprocessed_image(f)]     # and palette_name in f.stem]

    colors_val = 5
    blur_radius_val = 4
    save_palette_val = True
    display_palette_val = False
    filter_greys_val = True

    transition_settings = 'fast'    # see if-else tree below
    brightness_settings = 'dim'
    time_limit_val = 7200           # seconds

    try:
        palette_colors = load_palette(PALETTE_DIR.joinpath(palette_name + '.csv'))

    except FileNotFoundError:
        palette_colors = get_palette(palette_name, colors_val, blur_radius_val, save_palette_val, display_palette_val,
                                     filter_greys_val, *filepaths)

    if transition_settings == 'fast':
        cycle_time_vals = (10, 2)
        trans_time_vals = (200, 30)
    else:
        # default values
        cycle_time_vals = (60, 10)      # seconds
        trans_time_vals = (200, 30)     # deciseconds

    if brightness_settings == 'bright':
        brightness_vals = (254, 0)
    elif brightness_settings == 'mid':
        brightness_vals = (128, 20)
    elif brightness_settings == 'dim':
        brightness_vals = (64, 10)
    else:
        brightness_vals = (200, 15)     # 0-254 scale

    random_color_walk(palette_colors, cycle_time_vals, trans_time_vals, brightness_vals, int(time_limit_val))
