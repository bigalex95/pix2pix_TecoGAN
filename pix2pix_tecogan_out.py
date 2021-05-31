# IMPORT NECCESARY LIBRARIES
import cv2
import tensorflow as tf
import numpy as np
import torch
from threading import Thread
import pyvirtualcam
import yaml
from TecoGAN_PyTorch.codes.utils import base_utils
import os.path as osp
from TecoGAN_PyTorch.codes.models import define_model
from imutils.video import FPS

# PIX2PIX AND TECOGAN CONFIGURATIONS
pix2pix_size = 512
pix2pix_norm = (pix2pix_size/2) - 0.5
pix2pix_model_path = "./model/pix2pixTF"
tecogan_size = 256
tecogan_conf_path = "./config/"
check_input_src = True
check_tecogan = False
check_denoise = True
check_display = True
check_webcam = True

# TENSORFLOW GPU CONFIGURATIONS
# tf.debugging.set_log_device_placement(True)
gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
    # Restrict TensorFlow to only use the first GPU
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        # tf.config.experimental.set_memory_growth(gpus[1], True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
    except RuntimeError as e:
        # Visible devices must be set before GPUs have been initialized
        print(e)


# LOAD PIX2PIX MODEL
generator = tf.saved_model.load(pix2pix_model_path + str(pix2pix_size))

# USE OPENCV WITH MULTITHREADING


class WebcamVideoStream:
    def __init__(self, src=0, device=None):
        # initialize the video camera stream and read the first frame
        # from the stream
        self.stream = cv2.VideoCapture(src, device)
        (self.grabbed, self.frame) = self.stream.read()
        # initialize the variable used to indicate if the thread should
        # be stopped
        self.stopped = False

    def start(self):
        # start the thread to read frames from the video stream
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        # keep looping infinitely until the thread is stopped
        while True:
            # if the thread indicator variable is set, stop the thread
            if self.stopped:
                return

            # otherwise, read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        # return the frame most recently read
        return self.frame

    def stop(self):
        # indicate that the thread should be stopped
        self.stopped = True


# TECOGAN CONFIGURATIONS
# PYTORCH GPU CONFIGURATIONS
if check_tecogan:
    device = torch.device('cuda:1')
    torch.cuda.set_device(device)
    print('Active CUDA Device: GPU', torch.cuda.current_device())
with open(osp.join(tecogan_conf_path, "live.yml"), 'r') as f:
    opt = yaml.load(f.read(), Loader=yaml.FullLoader)

# ----------------- general configs ----------------- #
# experiment dir
opt['exp_dir'] = tecogan_conf_path

# random seed
base_utils.setup_random_seed(opt['manual_seed'])

# logger
base_utils.setup_logger('base')
# opt['verbose'] = opt.get('verbose', False)
opt['verbose'] = False

# device
opt['device'] = "cpu"  # device

# setup paths
base_utils.setup_paths(opt, mode='test')

# run
opt['is_train'] = False

# FUNCTION FOR USING PIX2PIX


def pix2pix(img):
    input_image = tf.cast(img, tf.float32)
    input_image = tf.image.resize(input_image, [pix2pix_size, pix2pix_size],
                                  method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    input_image = (input_image / pix2pix_norm) - 1
    ext_image = tf.expand_dims(input_image, axis=0)
    prediction = generator(ext_image, training=True)
    # pil_image = tf.keras.preprocessing.image.array_to_img(prediction[0])
    return prediction[0].numpy()


# FONCTION FOR DENOISING WITH BILETAREL
def denoising(img):
    img = cv2.bilateralFilter(img, 15, 80, 80)
    return img


# FUNCTION FOR WEBCAM
if check_webcam:
    camera = pyvirtualcam.Camera(
        width=pix2pix_size, height=pix2pix_size, fps=30)

    def send_to_webcam(img):
        img = img[..., ::-1]
        camera.send(img)
        camera.sleep_until_next_frame()

# FUNCTION FOR USING TECOGAN


def live(opt):
    global check_webcam

    if check_input_src:
        check_input()

    cv2.namedWindow("output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("output", 1024, 1024)
    # cap = cv2.VideoCapture(0)
    cap = WebcamVideoStream(src=1, device=cv2.CAP_DSHOW).start()
    if check_tecogan:
        # logging
        logger = base_utils.get_logger('base')
        if opt['verbose']:
            logger.info('{} Configurations {}'.format('=' * 20, '=' * 20))
            base_utils.print_options(opt, logger)

        # infer and evaluate performance for each model
        for load_path in opt['model']['generator']['load_path_lst']:
            # setup model index
            model_idx = osp.splitext(osp.split(load_path)[-1])[0]

            # log
            logger.info('=' * 40)
            logger.info('Testing model: {}'.format(model_idx))
            logger.info('=' * 40)

            # create model
            opt['model']['generator']['load_path'] = load_path
            model = define_model(opt)
            fps = FPS().start()
            while True:
                image = cap.read()
                if image.any():
                    # print(image.shape)
                    image = pix2pix(image)

                    if pix2pix_size != tecogan_size:
                        image = cv2.resize(
                            image, (tecogan_size, tecogan_size))

                    image = cv2.normalize(
                        image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

                    tmp_torch = torch.from_numpy(
                        image[None, :, :, :]).cuda(1)
                    hr_image = model.infer_live(tmp_torch)
                    image = hr_image[0]
                    if check_denoise:
                        image = denoising(image)
                    if check_display:
                        cv2.imshow("output", image)
                    if check_webcam:
                        image = image * 255
                        send_to_webcam(image.astype(np.uint8))
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        cap.stop()
                        break
                    elif cv2.waitKey(1) & 0xFF == ord('w'):
                        check_webcam = not check_webcam
                    fps.update()
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        # logging
        logger.info('Finish testing')
        logger.info('=' * 40)
    else:
        fps = FPS().start()
        while True:
            image = cap.read()
            if image.any():
                # print(image.shape)

                image = pix2pix(image)

                image = cv2.normalize(
                    image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                if check_denoise:
                    image = denoising(image)

                if check_display:
                    cv2.imshow("output", image)
                if check_webcam:
                    image = image * 255
                    send_to_webcam(image.astype(np.uint8))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    cap.stop()
                    break
                elif cv2.waitKey(1) & 0xFF == ord('w'):
                    check_webcam = not check_webcam
                fps.update()
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


def check_input(src=0):
    # For Video File
    # capture=cv2.VideoCapture("sample.webm")

    # For webcam
    capture = cv2.VideoCapture(src, cv2.CAP_DSHOW)

    # showing values of the properties
    print("CV_CAP_PROP_FRAME_WIDTH: '{}'".format(
        capture.get(cv2.CAP_PROP_FRAME_WIDTH)))
    print("CV_CAP_PROP_FRAME_HEIGHT : '{}'".format(
        capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("CAP_PROP_FPS : '{}'".format(capture.get(cv2.CAP_PROP_FPS)))
    print("CAP_PROP_POS_MSEC : '{}'".format(
        capture.get(cv2.CAP_PROP_POS_MSEC)))
    print("CAP_PROP_FRAME_COUNT : '{}'".format(
        capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    print("CAP_PROP_BRIGHTNESS : '{}'".format(
        capture.get(cv2.CAP_PROP_BRIGHTNESS)))
    print("CAP_PROP_CONTRAST : '{}'".format(
        capture.get(cv2.CAP_PROP_CONTRAST)))
    print("CAP_PROP_SATURATION : '{}'".format(
        capture.get(cv2.CAP_PROP_SATURATION)))
    print("CAP_PROP_HUE : '{}'".format(capture.get(cv2.CAP_PROP_HUE)))
    print("CAP_PROP_GAIN : '{}'".format(capture.get(cv2.CAP_PROP_GAIN)))
    print("CAP_PROP_CONVERT_RGB : '{}'".format(
        capture.get(cv2.CAP_PROP_CONVERT_RGB)))

    # release window
    capture.release()
    cv2.destroyAllWindows()
