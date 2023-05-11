import numpy as np


class Pose(object):
    def bounding_box(self, coords):

        min_x = 100000
        min_y = 100000
        max_x = -100000
        max_y = -100000

        for item in coords:
            if item[0] < min_x:
                min_x = item[0]

            if item[0] > max_x:
                max_x = item[0]

            if item[1] < min_y:
                min_y = item[1]

            if item[1] > max_y:
                max_y = item[1]
        return [(int(min_x), int(min_y)), (int(max_x), int(min_y)), (int(max_x), int(max_y)), (int(min_x), int(max_y))]

    def roi(self, imagepoints):
        coords_new_reshaped = imagepoints
        coords_new = np.asarray(coords_new_reshaped).reshape(6, 2)
        roi_coords = self.bounding_box(coords_new)
        coords_new = self.get_new_coords(coords_new, roi_coords)
        coords_new = coords_new.reshape(12,)
        # coords_new = np.concatenate((coords_new[0:34],imagepoints[34:52]))
        return coords_new

    def get_new_coords(self, coords, fun_bound):
        coords[:, :1] = coords[:, :1] - fun_bound[0][0]
        coords[:, 1:2] = coords[:, 1:2] - fun_bound[0][1]
        return coords
