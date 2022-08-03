"""K-means quantization extraction of colors.

Code source: https://towardsdatascience.com/colour-image-quantization-using-k-means-636d93887061
"""

from PIL import ImageFilter
from PIL.Image import Image
import cv2
from sklearn.cluster import MiniBatchKMeans

from ..color import Palette, colorthief_get_palette


class KMeansQuantization:
    """A K-Means Quantization algorithm implementing ExtractionProtocol."""

    def extract(__image: Image, /, k: int, radius: int) -> Palette:
        """extract k colors from a Pillow Image with a gaussian blur with radius radius.

        Parameters
        ----------
        __image : Image
            A Pillow Image object.
        k : int
            The number of colors to identify.
        radius : int
            The radius of the gaussian blur applied to the image.

        Returns
        -------
        Palette
        """

        h, w = __image.shape[:2]

        # gaussian blur to nullify edge pixels
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
        # query the above for values, use collections

        image = cv2.cvtColor(lab_image, cv2.COLOR_LAB2BGR)

        return colorthief_get_palette(image, k)
