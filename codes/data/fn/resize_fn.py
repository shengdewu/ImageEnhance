import cv2


class MaxEdgeResize:
    def __init__(self, max_edge_length):
        self.max_edge_length = max_edge_length * 1.0
        return

    def __call__(self, image):
        h, w, c = image.shape
        max_size = max(h, w)
        scale = self.max_edge_length / max_size
        if scale < 1.0:
            if w > h:
                w = int(self.max_edge_length)
                h = int(h * scale)
            else:
                w = int(w * scale)
                h = int(self.max_edge_length)

        return cv2.resize(image, (w, h), cv2.INTER_CUBIC)

    def __str__(self):
        return 'MaxEdgeResize'
