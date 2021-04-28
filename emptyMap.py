import numpy as np
from nuscenes.prediction.input_representation.interface import StaticLayerRepresentation
from nuscenes.prediction.input_representation.utils import get_crops, get_rotation_matrix, convert_to_pixel_coords
def drawCircleAt(pixel_point, img, color):
    for x in range(-2,3):
        for y in range(-2,3):
            try:
                img[pixel_point[0] + x, pixel_point[1] + y] = color
            except:
                pass
class emptyStaticLayerRasterizer(StaticLayerRepresentation):
    def __init__(self, 
                 resolution: float = 0.1,
                 meters_ahead: float = 40,
                 meters_behind:float = 10,
                 meters_left: float = 25,
                 meters_right: float = 25):
        self.colors = [(255, 255, 255), (119, 136, 153), (0, 0, 255)]
        self.resolution = resolution
        self.meters_ahead = meters_ahead
        self.meters_behind = meters_behind
        self.meters_left = meters_left
        self.meters_right = meters_right
    def make_representation(self, token1, token2):
        image_side_length = 2 * max(self.meters_ahead, self.meters_behind,
                                    self.meters_left, self.meters_right)
        image_side_length_pixels = int(image_side_length / self.resolution)
        image = np.zeros((image_side_length_pixels, image_side_length_pixels, 3))
        row_crop, col_crop = get_crops(self.meters_ahead, self.meters_behind, self.meters_left,
                                       self.meters_right, self.resolution,
                                       int(image_side_length / self.resolution))
        return image[row_crop, col_crop, :].astype('uint8')
    def plotTrajectories(self, trajects):
        trajectories = -trajects
        image_side_length = 2 * max(self.meters_ahead, self.meters_behind,
                                    self.meters_left, self.meters_right)
        image_side_length_pixels = int(image_side_length / self.resolution)
        image = np.zeros((image_side_length_pixels, image_side_length_pixels, 3))
        row_crop, col_crop = get_crops(self.meters_ahead, self.meters_behind, self.meters_left,
                                       self.meters_right, self.resolution,
                                       int(image_side_length / self.resolution))
        central_pixels = (image_side_length_pixels / 2, image_side_length_pixels / 2)
        colors = [[255,0,0], [0,255,0], [0,0,255], [125,125,0], [125,0,125], [0,125,125]]
        for i, traject in enumerate(trajectories):
            for point in traject:
                row_pixel = int(point[1] / self.resolution + central_pixels[0])
                col_pixel = int(point[0] / self.resolution + central_pixels[1])
                drawCircleAt((row_pixel, col_pixel), image, colors[i])
        print(np.sum(image > 0 )) 
        return image[row_crop, col_crop, :].astype('uint8')

