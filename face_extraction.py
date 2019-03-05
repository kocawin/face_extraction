import face_recognition
import cv2
from pathlib import Path
import numpy as np
import time
import datetime
import dlib
import click
from typing import Union, List, Tuple

from cv2.cv2 import VideoWriter


class FaceExtraction:

    def __init__(
            self,
            ref_images: [Path],
            input_video: Path,
            output_video: Path = None,
            output_images_dir: Path = None,
            model='cnn',
            max_frames=None,
            height=None,
            out_fps=1,
            tolerance=.4,
            match_all=False,
            max_minutes=5,
            extract_only=False,
            override=False,
            debug=False
    ):
        self.video_in = input_video
        self.video_out = Path(output_video) if output_video is not None else output_video
        self.ref_images: [Path] = ref_images
        self.images_out_dir = Path(output_images_dir) if output_images_dir is not None else output_images_dir
        if self.video_out is None and self.images_out_dir is None:
            raise Exception('No outputs are given!')
        self.model = 'cnn' if dlib.DLIB_USE_CUDA else model
        self.max_frames = max_frames
        self.height = height
        self.out_fps = out_fps
        self.tolerance = tolerance
        self.match_all = match_all
        self.max_seconds = max_minutes * 60
        self.extract_only = extract_only
        self.image_counter = -1
        self.known_faces = []
        self.override = override
        self.debug = debug
        self.images_out_counter = 0
        self._load_faces()

    def _load_faces(self):
        for image in self.ref_images:
            image = face_recognition.load_image_file(str(image))
            if image is not None:
                encodings = face_recognition.face_encodings(image, )
                if encodings:
                    self.known_faces.append(encodings[0])

    def check_image(self, img: Union[Path, np.ndarray]) -> Tuple[Tuple[int, int, int, int], List[str]]:
        """

        :param np.ndarray img:
        :return:
        :rtype: bool
        """
        if isinstance(img, np.ndarray):
            img = img[:, :, ::-1]
        else:
            img = face_recognition.load_image_file(str(img))
        face_locations = face_recognition.face_locations(img, model=self.model)
        face_encodings = face_recognition.face_encodings(img, face_locations)
        if not face_encodings:
            return tuple(), []
        face_names = []
        for face_encoding in face_encodings:
            match = face_recognition.compare_faces(self.known_faces, face_encoding, tolerance=self.tolerance)
            if self.debug:
                print('face match', match)
            name = None
            for i, m in enumerate(match):
                if m:
                    print(i, len(self.ref_images))
                    name = self.ref_images[i].name
                    break
            face_names.append(name)
        return face_locations, face_names

    def save_img_func(self):

        if self.images_out_dir is None:
            return lambda x: None

        if not self.images_out_dir.exists():
            self.images_out_dir.mkdir(exist_ok=True, parents=True)
        if not self.override:
            images = sorted(file for file in self.images_out_dir.iterdir() if file.is_file())
            if images:
                try:
                    self.images_out_counter = int(images[-1].stem) + 1
                except:
                    self.images_out_counter = 0
            else:
                self.images_out_counter = 0
                images = None

        def save_func(img: np.ndarray):
            cv2.imwrite(f'{self.images_out_dir}/{self.images_out_counter:05}.png', img)
            self.images_out_counter += 1

        return save_func

    def run(self, update_func=None):
        if self.video_out is not None and self.video_out.exists():
            if self.override:
                self.video_out.unlink()
            else:
                raise FileExistsError('Output video exists!')

        save_img = self.save_img_func() if self.images_out_dir is not None else lambda x: None

        input_movie = cv2.VideoCapture(str(self.video_in.resolve()))
        fps = int(input_movie.get(cv2.CAP_PROP_FPS))
        if fps <= 0:
            raise Exception('Bed video!')

        max_search_frames = self.max_seconds * fps
        skip = int(fps / self.out_fps)
        length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))
        rate = 1 / fps
        size = (
            int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT))
        )
        if self.height is not None and size[1] > self.height:
            scale = self.height / size[1]
            size = (int(size[0] * scale), int(self.height))
        output_movie: VideoWriter = None
        if self.video_out is not None:
            fourcc = cv2.VideoWriter_fourcc('A', 'V', 'C', '1')
            output_movie: VideoWriter = cv2.VideoWriter(
                str(self.video_out.resolve()), fourcc, 1,
                size
            )

        frame_number = 0
        font = cv2.FONT_HERSHEY_DUPLEX
        out_counter = 0
        start_time = time.time()
        while True:
            if frame_number > max_search_frames:
                break

            if not self.extract_only:
                input_movie.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = input_movie.read()
            frame_number += skip
            if not ret:
                break
            frame = cv2.resize(frame, size)
            if self.extract_only and callable(update_func):
                update_func(image=frame)
                continue
            face_locations, face_names = self.check_image(frame)
            if self.debug:
                print('face_locations', face_locations, 'face_names', face_names)
            if not face_locations:
                continue

            for (top, right, bottom, left), name in zip(face_locations, face_names):
                if name is None:
                    continue
                height = int(abs(top - bottom) * .6)
                width = int(abs(right - left) * 0.15)
                l, r, t, b = max(0, left - width), max(0, right + width), max(top - height, 0), max(bottom + height, 0)
                if self.debug:
                    print('l, r, t, b', l, r, t, b)
                # if b - t < r - l:
                #     continue
                #
                mask = np.zeros(frame.shape, np.uint8)
                mask[t:b, l:r] = frame[t:b, l:r]
                # if b - t < r - l:
                #     continue
                searched_time = ','.join(
                    map(str, (str(datetime.timedelta(seconds=int(frame_number * rate))), out_counter, frame_number)))
                print(searched_time)
                cv2.putText(
                    mask,
                    searched_time,
                    (l, b), font, 0.5,
                    (255, 255, 255),
                    1
                )
                if callable(update_func):
                    update_func(frame=out_counter, total_frame=self.max_frames, searched_time=searched_time,
                                msg=f'filtering',
                                image=mask)
                    if out_counter % 10 == 0:
                        update_func(tt=datetime.datetime.now())
                if self.debug:
                    print(f'counter: {out_counter}')

                if self.video_out is not None:
                    output_movie.write(mask)
                save_img(mask)
                out_counter += 1
            if self.max_frames is not None and out_counter > self.max_frames:
                break

        # All done!
        if self.video_out is not None:
            output_movie.release()
        input_movie.release()
        cv2.destroyAllWindows()
        print(f'total time: {time.time() - start_time}')


@click.command()
@click.option('--face', '-f', multiple=True, required=True,
              help='face image/头像图片(accept multiple images, ex: -p 1.png -p 2.png)')
@click.option('--tolerance', '-t', type=click.IntRange(1, 10, clamp=True), default=4,
              help='tolerance for comparision(1 is strictest)/匹配程度,越小越严格')
@click.option('--input-video', '-i', type=str, required=True,
              help='input video path/输入视频地址')
@click.option('--output-video', '-ov', type=str, default=None,
              help='output video path/输出视频地址')
@click.option('--output-images-dir', '-oi', type=str, default=None,
              help='output images directory path/输出图片文件夹')
@click.option('--override', '-or', is_flag=True, default=False,
              help='override existing files/覆盖现有文件')
@click.option('--debug', '-d', is_flag=True, default=False,
              help='override existing files/覆盖现有文件')
def run(**kwargs):
    kwargs['tolerance'] /= 10
    kwargs['input_video'] = Path(kwargs.get('input_video'))
    kwargs['ref_images'] = [Path(path) for path in kwargs['face']]
    del kwargs['face']
    fe = FaceExtraction(**kwargs)
    fe.run()


if __name__ == '__main__':
    run()
