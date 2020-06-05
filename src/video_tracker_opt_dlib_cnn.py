import cv2
import dlib
import matplotlib.pyplot as plt
import os
import data_io


def lin_smooth(data, stop, iterations, return_new_list=True):
    """
    You were changing input-data. I moved the list.copy in this method,
    as it make more sense in my opinion. Anyway, there is a flag, so you can
    use it either way.

    Args:
        data:
        stop (int):
        iterations (int):
        return_new_list: Whether to copy incoming data or operate
                         on existing list

    Returns:
        list: smoothed out ``data``

    """
    if return_new_list:
        data = data.copy()

    for it in range(iterations):
        for n in range(stop - 2):
            if n > 1:
                data[n] = ((data[n - 2] + data[n - 1] + data[n] + data[n + 1] + data[n + 2]) / 5)
    return data


class StereoLandmarks:
    """
    Utility class for storing eye landmarks data. Take a look at heavy
    property usage. The syntax is a bit clunky, but you get used to it.

    To generate a "prop-skeleton" in PyCharm, try typing "prop" or "props"
    and hitting Enter. Pycharm should create a template for you to fill in your
    filthy code ;) There is a red border around your typing caret, which means,
    you are typing in multiple places simultaneously. Just hit "Tab" after
    you've finished, to apply the name across multiple occurrences.
    (Hitting enter could paste in some weired auto-complete non-sense).

    Example (Letter "I" represents the caret occurrences):

        >>> @property
        >>> def I(self):
        >>>     return

        >>> @I.setter
        >>> def (self, value):
        >>>     pass

    Python properties are just like the properties in C# but with a horrible
    syntax. Bless PyCharm for creating presets for such an non-sense

    Feel free to test few other PyCharm templates. I also use these ones:
        - main - Creates ``if __name__ == '__main__':``
        - super - Creates a super call in a class. super(ClassName, self).__init__()

    """

    class _Eye:
        """
        Utility subclass. This kind of shit is also possible.

        It has an underscore prefix to make it "protected".
        I know, we are in python, and there is no "protected" or "private",
        but! it's a **convention**, which PyCharm is making use of.
        The class will not be listed in an auto-complete list,
        until you've typed in like 70% of it's name.
        """

        def __init__(self):
            self._x = list()
            self._y = list()

        @property
        def x(self):
            return self._x

        @x.setter
        def x(self, value):
            self._x = value

        @property
        def y(self):
            return self._y

        @y.setter
        def y(self, value):
            self._y = value

    class Data:
        """
        Some hardcoded data found in the script. I have no idea, if it needs
        to stay hardcoded. Probably at some point not any more.
        """
        left = [42, 43, 44, 45, 46, 47]   # starting from inner corner -> up
        right = [36, 37, 38, 39, 40, 41]  # starting from outer corner -> up

    def __init__(self):
        self._left = self._Eye()
        self._right = self._Eye()

    def append(self, landmarks):
        self.right.x.append(landmarks.part(self.Data.right[3]).x)
        self.right.y.append(landmarks.part(self.Data.right[3]).y)

        self.left.x.append(landmarks.part(self.Data.left[0]).x)
        self.left.y.append(landmarks.part(self.Data.left[0]).y)

    @property
    def left(self):
        return self._left

    @left.setter
    def left(self, value):
        self._left = value

    @property
    def right(self):
        return self._right

    @right.setter
    def right(self, value):
        self._right = value


class TestPersonPlot:

    class Config:
        suffix_footage = '_processed.0002.avi'
        suffix_render = '_tracked.0001.avi'
        suffix_eye = '_eye_{side}_tracked.0001.avi'

        file_predictor = 'shape_predictor_68_face_landmarks.dat'
        file_plot_exp = os.path.join(os.path.dirname(__file__), 'PLOT.png')

        scale = 0.5
        iterations = 9

        # None of the following variables was used in the script at refactoring
        # time. Maybe in one of the commented lines. I've just ignored them...
        # But the static variables are still here, if you ever need them.
        # If you need to change them dynamically, I would recomend to move them
        # to the parent class. This one should only be used as
        # a static config container

        # roi_eye_offset_x = 130
        # roi_eye_offset_y = 80
        #
        # roi_eye_damping = 1.0
        #
        # is_denoised = 1
        # is_contrast_improved = 1
        # is_stabilized = 1
        #
        # is_full_frame_render = 1
        # is_eye_render = 1
        # is_full_frame_show = 0
        # is_eye_frame_show = 1

    def __init__(self, test_person, verbose=True):
        """
        Constructor method, which starts the video capture, as far as i can
        understand, what I am doining here :D

        Args:
            test_person (str): Probably a subfolder name. Probanly
                               a persons name?
            verbose: Whether the script should print it's "progress". Feel
                     free to use this variable more often down the line

        """
        plt.style.use('seaborn-whitegrid')

        self.test_person = test_person
        self.verbose = verbose
        self.landmarks = StereoLandmarks()

        self.video_cap = cv2.VideoCapture(self.footage_path)
        self.video_codec = cv2.VideoWriter_fourcc(*'MJPG')

        self.video_tracked = cv2.VideoWriter(
            self.render_path,
            self.video_codec,
            self.video_fps,
            self.video_size_scaled
        )

        if self.verbose:
            self.print_report_video_capture()

    def print_report_video_capture(self):
        """
        Prints the report about running video capture

        Note: I'm using the new formatted-string syntax here.
        It's available in Python 3.7+.

        """
        print('-' * 100)
        print(f'Video to track: {self.footage_file}\n'
              f'Input video width: {self.video_size.get("width")}\n'
              f'Input video height: {self.video_size.get("height")}')
        print('-' * 100)

    def face_landmark_detection(self):
        """
        Face landmark detection magic happens here.
        I hope, I didn't forgot anything :D
        """
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(self.predictor_path)

        # WTF? found no usage for them, but here you are :D
        frame = self.get_frame()
        roi_r_eye_frame = frame
        roi_l_eye_frame = frame

        self.reset_stream()

        for fnum in range(self.video_frame_count - 2):
            success, frame = self.video_cap.read()
            if not success:
                break

            frame = cv2.resize(frame, self.video_size_scaled)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Wooow! what a weired syntax! WTF is this ``detector``?
            # Pointer to a class or method? Same with ``predictor``?
            faces = detector(gray)
            for face in faces:
                self.landmarks.append(predictor(gray, face))

    def plot_data(self, right=True, left=True, show=True, save_img=False):
        """
        Plotting some eye data. You know better, what happens here :D

        Args:
            right (bool): Whether to plot **right** eye. Default: True
            left (bool): Whether to plot **left** eye. Default: True
            show (bool): Whether to show the graph. Default: True
            save_img (bool): Whether to write an image file. Default: False

        """
        subplots = (int(right) + int(left)) * 2

        if subplots == 0:
            raise AttributeError(
                'Neither ``right`` nor ``left`` is set to ``True``. '
                'Nothing to plot.'
            )

        fig, axs = plt.subplots(subplots)
        frame_num = self.video_frame_count - 2

        if right:
            x_smooth = lin_smooth(self.landmarks.right.x,
                                  frame_num,
                                  self.Config.iterations)
            y_smooth = lin_smooth(self.landmarks.right.y,
                                  frame_num,
                                  self.Config.iterations)

            axs[0].plot(self.landmarks.right.x, 'tab:red',
                        x_smooth, 'tab:green')
            axs[1].plot(self.landmarks.right.y, 'tab:green',
                        y_smooth, 'tab:orange')

        if left:
            x_smooth = lin_smooth(self.landmarks.left.x,
                                  frame_num,
                                  self.Config.iterations)
            y_smooth = lin_smooth(self.landmarks.left.y,
                                  frame_num,
                                  self.Config.iterations)

            # I'm using negative indices here to be able to work no matter,
            # whether the right side is being plotted or not
            axs[-2].plot(self.landmarks.left.x, 'tab:yellow',
                         x_smooth, 'tab:blur')
            axs[-1].plot(self.landmarks.left.y, 'tab:blue',
                         y_smooth, 'tab:magenta')

        # Don't know, how you use the plotter colors/tabs, what so ever ;)
        # Please double check, it's doing what it should :D

        if show:
            plt.show()

        if save_img:
            # You could generate a picture / pdf here...
            # It should work. Can't test anything here :D
            plt.legend()
            plt.grid()
            plt.tight_layout()
            plt.savefig(self.Config.file_plot_exp, dpi=300)
            plt.close()
            print(f'Plot file exported to "{self.Config.file_plot_exp}".')

    # - Video data properties and methods ----------------------------------- #
    def get_frame(self):
        return self.video_cap.read()[1]

    def reset_stream(self):
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    @property
    def video_frame_count(self):
        return int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))

    @property
    def video_fps(self):
        return int(self.video_cap.get(cv2.CAP_PROP_FPS))

    @property
    def video_size(self):
        return {
            'height': int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'width': int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        }

    @property
    def video_size_scaled(self):
        return int(self.video_size.get('width') * self.Config.scale),\
               int(self.video_size.get('height') * self.Config.scale)

    # - Path properties ----------------------------------------------------- #
    @property
    def footage_file(self):
        return self.test_person + self.Config.suffix_footage

    @property
    def footage_path(self):
        return os.path.join(
            data_io.get_paths(self.test_person).get('footage'),
            self.footage_file
        )

    @property
    def render_file(self):
        return self.test_person + self.Config.suffix_render

    @property
    def render_path(self):
        return os.path.join(
            data_io.get_paths(self.test_person).get('footage'),
            self.render_file
        )

    @property
    def predictor_file(self):
        return self.Config.file_predictor

    @property
    def predictor_path(self):
        return os.path.join(
            data_io.get_paths(self.test_person).get('config'),
            self.predictor_file
        )


if __name__ == '__main__':
    if os.name not in ('nt', 'posix'):
        raise RuntimeError('Wrong OS. Exiting')

    plotter = TestPersonPlot('me02')
    plotter.face_landmark_detection()
    plotter.plot_data()
