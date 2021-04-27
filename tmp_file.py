# FUNCTION FOR USING TECOGAN
import yaml
from TecoGAN_PyTorch.codes.utils import base_utils
import os.path as osp
from TecoGAN_PyTorch.codes.models import define_model
# IMPORTING NECESSARY LIRARIES
import cv2
import torch
from threading import Thread
from imutils.video import FPS

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


# PYTORCH GPU CONFIGURATIONS
device = torch.device('cuda:1')
torch.cuda.set_device(device)
print('Active CUDA Device: GPU', torch.cuda.current_device())

with open(osp.join("./TecoGAN_PyTorch/experiments_BD/TecoGAN/001", "live.yml"), 'r') as f:
    opt = yaml.load(f.read(), Loader=yaml.FullLoader)

# ----------------- general configs ----------------- #
# experiment dir
opt['exp_dir'] = "./TecoGAN-PyTorch/experiments_BD/TecoGAN/001"

# random seed
base_utils.setup_random_seed(opt['manual_seed'])

# logger
base_utils.setup_logger('base')
opt['verbose'] = opt.get('verbose', False)

# device
opt['device'] = device

# setup paths
base_utils.setup_paths(opt, mode='test')

# run
opt['is_train'] = False


def live(opt):
    cv2.namedWindow("LR image", cv2.WINDOW_NORMAL)
    cv2.namedWindow("HR image", cv2.WINDOW_NORMAL)
    # cap = cv2.VideoCapture(0)
    cap = WebcamVideoStream(src="video.mp4").start()
    image = cap.read()
    out = cv2.VideoWriter('output.avi', -1, 20.0,
                          (512*4, 512*4))

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
                image = cv2.resize(image, (512, 512))
                cv2.imshow("LR image", image)
                # cv2.imwrite("LR_image.png", image)
                norm_image = cv2.normalize(
                    image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                tmp_torch = torch.from_numpy(norm_image[None, :, :, :]).cuda()
                hr_image = model.infer_live(tmp_torch)
                cv2.imshow("HR image", hr_image[0])
                # cv2.imwrite("HR _image.png", hr_image[0])
                out.write(hr_image[0])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    cap.stop()
                    break
                fps.update()
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    # logging
    logger.info('Finish testing')
    logger.info('=' * 40)


if __name__ == '__main__':
    live(opt)
