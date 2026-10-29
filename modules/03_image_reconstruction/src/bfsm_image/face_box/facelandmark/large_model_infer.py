import numpy as np
from ..retinaface.predict_single import Model
import cv2
from .large_base_lmks_infer import LargeBaseLmkInfer
import torch
import os
INPUT_SIZE = 224
ENLARGE_RATIO = 1.35








class LargeModelInfer:

    def __init__(self,ckpt,  device='cuda'):
        self.large_base_lmks_model = LargeBaseLmkInfer.model_preload(ckpt,  torch.device(device).type == "cuda")
        self.device = device.lower()
        self.detector = Model(max_size=512, device=device)
        state_dict = torch.load(os.path.join(os.path.dirname(ckpt), 'retinaface_resnet50_2020-07-20_old_torch.pth' ) , map_location="cpu", weights_only=True)
        # torch.save(state_dict, './models/retinaface_resnet50_2020-07-20_old_torch.pth', _use_new_zipfile_serialization=False)
        self.detector.load_state_dict(state_dict)
        self.detector.eval()

        
    def infer(self, img_bgr):
        landmarks = []

        # rgb_image = img_bgr[:,:,::-1]
        rgb_image = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # boxes = self.detector.predict(rgb_image)
        results = self.detector.predict_jsons(rgb_image)
        # result_img = detector.draw(rgb_image, detect_results)
        # cv2.imshow("detect result", result_img[...,::-1].astype(np.uint8))

        boxes = []
        for anno in results:
            if anno['score'] == -1:
                break
            boxes.append({'x1': anno['bbox'][0], 'y1': anno['bbox'][1], 'x2': anno['bbox'][2], 'y2': anno['bbox'][3]})

        for detect_result in boxes:
            x1 = detect_result["x1"]
            y1 = detect_result["y1"]
            x2 = detect_result["x2"]
            y2 = detect_result["y2"]

            w = x2 - x1 + 1
            h = y2 - y1 + 1

            cx = (x2 + x1) / 2
            cy = (y2 + y1) / 2

            sz = max(h, w) * ENLARGE_RATIO

            x1 = cx - sz / 2
            y1 = cy - sz / 2
            trans_x1 = x1
            trans_y1 = y1
            x2 = x1 + sz
            y2 = y1 + sz

            height, width, _ = rgb_image.shape
            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)

            crop_img = rgb_image[int(y1):int(y2), int(x1):int(x2)]
            if dx > 0 or dy > 0 or edx > 0 or edy > 0:
                crop_img = cv2.copyMakeBorder(crop_img, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT, value=(103.94, 116.78, 123.68))
            crop_img = cv2.resize(crop_img, (INPUT_SIZE, INPUT_SIZE))
            # cv2.imshow("crop resize", crop_img.astype(np.uint8))
            # cv2.waitKey()

            base_lmks = LargeBaseLmkInfer.infer_img(crop_img, self.large_base_lmks_model, torch.device(self.device).type == "cuda")

            inv_scale = sz / INPUT_SIZE

            affine_base_lmks = np.zeros((106, 2))
            for idx in range(106):
                affine_base_lmks[idx][0] = base_lmks[0][idx * 2 + 0] * inv_scale + trans_x1
                affine_base_lmks[idx][1] = base_lmks[0][idx * 2 + 1] * inv_scale + trans_y1

            x1 = np.min(affine_base_lmks[:, 0])
            y1 = np.min(affine_base_lmks[:, 1])
            x2 = np.max(affine_base_lmks[:, 0])
            y2 = np.max(affine_base_lmks[:, 1])

            w = x2 - x1 + 1
            h = y2 - y1 + 1

            cx = (x2 + x1) / 2
            cy = (y2 + y1) / 2

            sz = max(h, w) * ENLARGE_RATIO

            x1 = cx - sz / 2
            y1 = cy - sz / 2
            trans_x1 = x1
            trans_y1 = y1
            x2 = x1 + sz
            y2 = y1 + sz

            height, width, _ = rgb_image.shape
            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)

            crop_img = rgb_image[int(y1):int(y2), int(x1):int(x2)]
            if dx > 0 or dy > 0 or edx > 0 or edy > 0:
                crop_img = cv2.copyMakeBorder(crop_img, int(dy), int(edy), int(dx), int(edx), cv2.BORDER_CONSTANT,
                                              value=(103.94, 116.78, 123.68))
            crop_img = cv2.resize(crop_img, (INPUT_SIZE, INPUT_SIZE))
            # cv2.imshow("crop resize", crop_img.astype(np.uint8))
            # cv2.waitKey()

            base_lmks = LargeBaseLmkInfer.infer_img(crop_img, self.large_base_lmks_model, torch.device(self.device).type == "cuda")

            inv_scale = sz / INPUT_SIZE

            affine_base_lmks = np.zeros((106, 2))
            for idx in range(106):
                affine_base_lmks[idx][0] = base_lmks[0][idx * 2 + 0] * inv_scale + trans_x1
                affine_base_lmks[idx][1] = base_lmks[0][idx * 2 + 1] * inv_scale + trans_y1

            landmarks.append(affine_base_lmks)

        return boxes, landmarks







        

