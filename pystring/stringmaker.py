import numpy as np
import matplotlib.pyplot as plt
import cv2

from pystring.helper import line_indices
from .helper import rot_matrix, reflect_matrix


class StringMaker:
    def __init__(self, image_path):
        self.image_path = image_path
        self.image = (
            1
            - cv2.cvtColor(cv2.imread(self.image_path), cv2.COLOR_BGR2GRAY)
            / 255
        )
        self.sanitize()

    def sanitize(self):
        self.resolution = np.min(self.image.shape)
        self.image = self.image[: self.resolution, : self.resolution]

    def resize(self, resolution):
        self.image = cv2.resize(self.image, dsize=(resolution, resolution))
        self.sanitize()

    def make_nails(self, n_nails=300):
        self.n_nails = n_nails
        radius = self.resolution // 2
        center = np.ceil(np.array([self.resolution, self.resolution]) / 2) - 1
        delta_theta = 2 * np.pi / n_nails
        coord = [np.array([0, 1])]
        for i in range(1, n_nails):
            coord.append(rot_matrix(delta_theta) @ coord[-1])
        self.nails = np.array(coord) * radius + center
        self.nail_dict = {
            idx: nail for idx, nail in enumerate(self.nails.astype(int))
        }
        self.calc_line_indices()

    def calc_line_indices(self):
        self.line_indices = {}
        for i in range(self.n_nails):
            for j in range(i):
                self.line_indices[(i, j)] = line_indices(
                    *self.nail_dict[i], *self.nail_dict[j]
                )
                self.line_indices[(j, i)] = self.line_indices[i, j]
        return self.line_indices

    def line_intensity(self, n_0, n_1):
        indices = self.line_indices[n_0, n_1]
        return indices.shape[1], self.work_image[indices[0], indices[1]].sum()

    def write_line(self, inicio, fin, brightness_reduction_factor=0.7):
        idx = self.line_indices[inicio, fin]
        self.work_image[idx[0], idx[1]] *= brightness_reduction_factor

    def run(self, n_iter=500, brightness_reduction_factor=0.8):
        """
        Run the stringmaker algorithm.

        Parameters
        ----------
        - n_iter (int): Number of iterations to run the algorithm. Default is 500.
        - brightness_reduction_factor (float): Factor by which the brightness of the written
        lines is multiplied. Default is 0.8.
        """
        seq = [0]
        self.work_image = self.image.copy()
        #     seq = [[0, y]]
        for it in range(n_iter):
            # Print iteration number every 50 iterations
            if it % 50 == 0:
                print(it, end="\t")

            weights = np.zeros(self.n_nails)
            n_0 = seq[-1]
            for n_1 in range(self.n_nails):
                if np.abs(n_0 - n_1) > 10:
                    n, suma = self.line_intensity(n_0, n_1)
                    if n > 0:
                        weights[n_1] = suma / n
            seq.append(int(np.argmax(weights)))
            self.write_line(
                seq[-2],
                seq[-1],
                brightness_reduction_factor=brightness_reduction_factor,
            )
        self.sequence = seq

    def imshow(self, ax=None, show=False):
        if ax is not None:
            plt.sca(ax)
        plt.imshow(self.image, cmap="binary", vmin=0, vmax=1)
        if hasattr(self, "nails"):
            plt.scatter(*self.nails.T, c="r", marker=".")
        if show:
            plt.show()

    def plot_seq(self, ax=None, show=False, transform=False):
        assert hasattr(self, "sequence"), "You need to run the algorithm first"
        if ax is not None:
            plt.sca(ax)
        pos = np.array([self.nail_dict[i] for i in self.sequence]).T
        if transform is True:
            pos = reflect_matrix() @ (rot_matrix(np.pi) @ pos)
        plt.plot(pos[0], pos[1], "k", lw=0.1)
        plt.gca().set_aspect("equal")
        if show:
            plt.show()

    def plot_results(self, show=False):
        fig, ax = plt.subplots(
            1,
            3,
            figsize=(14, 5),
            sharex=True,
            sharey=True,
            constrained_layout=True,
        )
        self.imshow(ax[0])
        self.plot_seq(ax[1])
        ax[2].imshow(self.work_image, cmap="binary", vmin=0, vmax=1)
        [a.axis("off") for a in ax]
        ax[0].set_title("Original Image")
        ax[1].set_title("String Sequence")
        ax[2].set_title("Remainder")
        if show:
            plt.show()
        return fig, ax

    def seq_to_png(self, resolution=1024, name="out.png"):
        new_nails = self.nails * resolution / self.resolution
        canvas = 255 * np.ones((resolution, resolution), dtype=np.uint8)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)
        nseq = np.array(
            [
                [self.sequence[i], self.sequence[i + 1]]
                for i in range(len(self.sequence) - 1)
            ]
        )
        color = 0  # black
        width = 1  # line width
        for i0, i1 in nseq:
            start = tuple(new_nails[i0].astype(int))
            end = tuple(new_nails[i1].astype(int))
            cv2.line(canvas, start, end, color, thickness=width, lineType=cv2.LINE_AA)  # type: ignore
        cv2.imwrite(name, canvas)
        return canvas

    def seq_to_movie(
        self,
        img_res=1024,
        vid_res=1024,
        fps=25,
        length=5,
        name="out.mp4",
        codec="mp4v",
    ):
        delta = int(
            len(self.sequence) / ((length - 2) * fps)
        )  # We add 2 seconds at the end
        vcode = cv2.VideoWriter_fourcc(*codec)  # type: ignore
        new_nails = self.nails * img_res / self.resolution
        canvas = 255 * np.ones((img_res, img_res), dtype=np.uint8)
        vid = cv2.VideoWriter(name, vcode, fps, (vid_res, vid_res))
        line_color = 0
        line_width = 1
        for idx in range(len(self.sequence) - 1):
            start = tuple(new_nails[self.sequence[idx]].astype(int))
            end = tuple(new_nails[self.sequence[idx + 1]].astype(int))
            cv2.line(canvas, start, end, line_color, thickness=line_width)  # type: ignore
            if idx % delta == 0:
                vid.write(
                    cv2.cvtColor(
                        cv2.resize(canvas, (vid_res, vid_res)),
                        cv2.COLOR_GRAY2RGB,
                    )
                )
        for i in range(2 * fps):
            vid.write(
                cv2.cvtColor(
                    cv2.resize(canvas, (vid_res, vid_res)),
                    cv2.COLOR_GRAY2RGB,
                )
            )
        vid.release()
