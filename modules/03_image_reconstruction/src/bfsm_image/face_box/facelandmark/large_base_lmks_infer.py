import torch
import numpy as np
from .nets.large_base_lmks_net import LargeBaseLmksNet

BASE_LANDMARK_NUM = 106
INPUT_SIZE = 224
ENLARGE_RATIO = 1.35


class LargeBaseLmkInfer:
    @staticmethod
    def model_preload(model_path, use_gpu=True):
        model = LargeBaseLmksNet(infer=False)
        # using gpu
        if use_gpu:
            model = model.cuda()

        checkpoint = []
        if use_gpu:
            # The hash-verified historical checkpoint includes NumPy metadata.
            checkpoint = torch.load(model_path, map_location='cuda', weights_only=False)
        else:
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}, strict=False)
        model.eval()
        return model

    @staticmethod
    def process_img(model, image, use_gpu=True):
        img_resize = image

        # cv2.imshow("test", img_resize.astype(np.uint8))
        # cv2.waitKey()

        img_resize = (img_resize - [103.94, 116.78, 123.68]) / 255.0  # important
        img_resize = img_resize.transpose([2, 0, 1])

        if use_gpu:
            img_resize = torch.from_numpy(img_resize).cuda()
        else:
            img_resize = torch.from_numpy(img_resize)

        w_new = INPUT_SIZE
        h_new = INPUT_SIZE
        img_in = torch.zeros([1, 3, h_new, w_new], dtype=torch.float32)
        if use_gpu:
            img_in = img_in.cuda()

        img_in[0, :] = img_resize

        with torch.no_grad():
            output = model(img_in)
            output = output * INPUT_SIZE

        if use_gpu:
            output = output.cpu().numpy()
        else:
            output = output.numpy()

        return output


    @staticmethod
    def infer_img(img, model,use_gpu=True):
        lmks = LargeBaseLmkInfer.process_img(model, img,use_gpu)
        return lmks
