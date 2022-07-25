"""K-means quantization extraction of colors.

Code source: https://towardsdatascience.com/colour-image-quantization-using-k-means-636d93887061
"""

from PIL import ImageFilter
from PIL.Image import Image
import cv2
from sklearn.cluster import MiniBatchKMeans

from ..color import Palette, colorthief_get_palette


class KMeansQuantization:
    def extract(__image: Image, /, k: int, radius: int) -> Palette:
        h, w = __image.shape[:2]
        gauss = __image.filter(ImageFilter.GaussianBlur(radius))

        # convert from RGB >> L*a*b space and then reshape to feature vector
        lab_image = cv2.cvtColor(gauss, cv2.COLOR_BGR2LAB)
        image_vector = lab_image.reshape((h * w, 3))

        # apply k-means using the specified number of clusters and
        # then create the quantized image based on the predictions
        clf = MiniBatchKMeans(n_clusters=k)
        labels = clf.fit_predict(image_vector)
        quant = clf.cluster_centers_.astype("uint8")[labels]

        # vectors > L*a*b > RGB
        lab_quant = quant.reshape((h, w, 3))
        lab_image = lab_image.reshape((h, w, 3))

        quant = cv2.cvtColor(lab_quant, cv2.COLOR_LAB2BGR)
        image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

        return colorthief_get_palette()
