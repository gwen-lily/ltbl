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
from rgbxy import Converter, GamutC
from time import sleep
import datetime as dt
import pandas as pd
import sys
import argparse

from colorthief import ColorThief
import extcolors
import colorgram

from settings import *

logging.basicConfig()

converter = Converter(GamutC)   # A19 bulbs I think I'm right here


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

    def sanitize_rgb_tuple(rgb: Tuple[int]) -> Tuple[int]:
        return tuple(map(lambda x: int(min([255, max([0, x])])), rgb))

    filepath_gauss = gauss_dir.joinpath(filepath.stem + '-gauss' + '.png')
    filepath_quant = quant_dir.joinpath(filepath.stem + '-quant' + '.png')

    image_raw = Image.open(filepath)
    image_gauss = image_raw.filter(ImageFilter.GaussianBlur(radius))

    if save_palette_val:
        image_gauss.save(filepath_gauss)

    # read gauss and convert from RGB >> L*a*b space and then reshape to feature vector
    try:
        image = cv2.imread(str(filepath_gauss))
        (h, w) = image.shape[:2]
    except AttributeError:
        print(filepath_gauss)
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


def generate_palette_grid(grid_name: str, *colors: tuple):
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

    if display_palette_val:
        palette_image.show()

    if save_palette_val:
        filepath_palette = palette_dir.joinpath(grid_name + '.png')
        palette_image.save(filepath_palette)


def get_palette(
        name: str,
        colors: int = 10,
        blur_radius: int = 4,
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

        if save_palette_val or display_palette_val:
            generate_palette_grid(img_fp.stem, *palette)

    full_palette = remove_greys(full_palette) if filter_greys_val else full_palette

    if save_palette_val:
        generate_palette_grid(name, *full_palette)

        full_palette_csv_filepath = palette_dir.joinpath(name + '.csv')
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


def load_palette_rgb(filepath: pathlib.Path) -> List[Tuple[int]]:
    palette_df = load_dataframe(filepath)
    active_df = palette_df.loc[(palette_df['active'] == True)]

    palette_rgb = [tuple(r[:3]) for r in palette_df.to_numpy()]
    return palette_rgb


if __name__ == '__main__':
    program_start_timestamp = str(int(dt.datetime.utcnow().timestamp()))

    # arg parsing ######################################################################################################
    parser = argparse.ArgumentParser(
        description="bedevere's palette analyzer and philips hue light controller: let-there-be-light",
    )
    parser.add_argument('-name', '-n',
                        type=str, metavar='N', default='palette-' + program_start_timestamp,
                        help="""Provide a name for the palette to be saved, the default is the utc timestamp at 
                        program start.""")
    parser.add_argument('-input', '-i',
                        type=str, metavar='IN', default='input',
                        help='Provide a file or directory containing files, the default is "input"')
    parser.add_argument('-colors', '-c',
                        type=int, metavar='C', default='10',
                        help='Choose the number of colors to pick per image, the default is 10.')
    parser.add_argument('-radius', '-r',
                        type=int, metavar='R', default=4,
                        help='Choose the gaussian blur radius applied to the image before clustering, the default is 4')
    parser.add_argument('-output', '-o',
                        type=str, metavar='OUT', default='output',
                        help='Provide the output directory, default is "output"')
    parser.add_argument('--save', '--s',
                        action='store_true',
                        help='Enable to save palette information in the specified output directory')
    parser.add_argument('--filter-grey', '--fg',
                        action='store_true',
                        help='Enable to filter out sufficiently grey colors')
    parser.add_argument('--display', '--d',
                        action='store_true',
                        help='Enable to display palette images during operation')
    parser.add_argument('-transition', '-t',
                        type=str.lower, metavar='T', default='fast',
                        help='Specify transition speed: fast / slow')
    parser.add_argument('-brightness', '-b',
                        type=str.lower, metavar='B', default='bright',
                        help='Specify brightness value: bright / mid / dim')
    parser.add_argument('-time-limit', '-tl',
                        type=int, metavar='TL', default=3600,
                        help='Specify the time limit in seconds')
    parser.add_argument('-bridge-ip',
                        type=str, metavar='IP', default=HUE_BRIDGE_IP)
    args = parser.parse_args()

    palette_name = args.name

    b = Bridge(args.bridge_ip)
    b.connect()  # register the app


    try:
        input_dir = pathlib.Path(args.input)
        assert input_dir.is_dir()
        image_files = [f for f in input_dir.iterdir() if is_unprocessed_image(f)]
    except AssertionError:
        image_files = []
        if input_dir.is_file() and is_unprocessed_image(input_dir):
            image_files = [input_dir]
        else:
            sys.exit(-1)

    try:
        output_dir = pathlib.Path(args.output)
        assert output_dir.is_dir()
    except AssertionError:
        if output_dir.exists():
            sys.exit(-1)
        else:
            output_dir.mkdir()
    finally:
        palette_dir = output_dir.joinpath(PALETTE_DIR_STEM)
        quant_dir = output_dir.joinpath(QUANT_DIR_STEM)
        gauss_dir = output_dir.joinpath(GAUSS_DIR_STEM)
        output_subdirs = [palette_dir, quant_dir, gauss_dir]

        for subdir in output_subdirs:
            if not subdir.exists():
                subdir.mkdir()

    try:
        colors_val = args.colors
        assert 0 < colors_val < MAX_COLORS
    except AssertionError as exc:
        print(exc)
        sys.exit(-1)

    try:
        blur_radius_val = args.radius
        assert 0 < blur_radius_val < MAX_RADIUS
    except AssertionError as exc:
        print(exc)
        sys.exit(-1)

    save_palette_val = args.save
    display_palette_val = args.display
    filter_greys_val = args.filter_grey

    transition_options = ['fast']
    brightness_options = ['bright', 'mid', 'dim']

    if args.transition in transition_options:
        transition_settings = args.transition
    else:
        transition_settings = 'default'

    if transition_settings == 'fast':
        cycle_time_vals = (10, 2)
        trans_time_vals = (200, 30)
    else:
        # default values
        cycle_time_vals = (60, 10)      # seconds
        trans_time_vals = (200, 30)     # deciseconds

    if args.brightness in brightness_options:
        brightness_settings = args.brightness
    else:
        brightness_settings = 'default'

    if brightness_settings == 'bright':
        brightness_vals = (254, 0)
    elif brightness_settings == 'mid':
        brightness_vals = (128, 20)
    elif brightness_settings == 'dim':
        brightness_vals = (64, 10)
    else:
        brightness_vals = (200, 15)     # 0-254 scale

    time_limit_val = args.time_limit

    # input

    try:
        palette_filepath = palette_dir.joinpath(palette_name + '.csv')
        palette_colors = load_palette(palette_filepath)

        colors_rgb = load_palette_rgb(palette_filepath)
        plot_palette(colors_rgb)


    except (FileNotFoundError, AssertionError):
        # assert image_files and len(image_files) > 0
        palette_colors = get_palette(palette_name, colors_val, blur_radius_val, *image_files)

    random_color_walk(palette_colors, cycle_time_vals, trans_time_vals, brightness_vals, time_limit_val)
