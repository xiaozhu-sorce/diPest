import os
import torch
import cv2
import glob
import numpy as np
from PIL import Image
from skimage import io
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms
from models.u2net import U2NET
from data_loader import RescaleT, ToTensorLab, SalObjDataset


class ImageProcessor:
    def __init__(self, model_path: str):
        """
        初始化 U2NET 模型
        :param model_path: 预训练模型路径
        """
        self.model_path = model_path
        self.model = self._load_model()
        self.model.eval()

        # 目录管理
        self.base_dir = os.getcwd()
        self.upload_dir = os.path.join(self.base_dir, 'upload')
        self.mask_dir = os.path.join(self.base_dir, 'masked')
        self.final_dir = os.path.join(self.base_dir, 'final')

        os.makedirs(self.upload_dir, exist_ok=True)
        os.makedirs(self.mask_dir, exist_ok=True)
        os.makedirs(self.final_dir, exist_ok=True)

    def _load_model(self):
        """ 加载 U2NET 预训练模型 """
        model = U2NET(3, 1)
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(self.model_path))
            model.cuda()
        else:
            model.load_state_dict(torch.load(self.model_path, map_location='cpu'))
        return model

    def _norm_pred(self, d):
        """ 归一化预测结果 """
        return (d - torch.min(d)) / (torch.max(d) - torch.min(d))

    def _save_output(self, image_name, pred):
        """ 保存预测的掩码图像 """
        predict_np = pred.squeeze().cpu().data.numpy()
        im = Image.fromarray(predict_np * 255).convert('RGB')

        # 读取原图大小
        image = io.imread(image_name)
        im_resized = im.resize((image.shape[1], image.shape[0]), resample=Image.BILINEAR)

        # 生成保存路径
        img_name = os.path.basename(image_name).split('.')[0]
        save_path = os.path.join(self.mask_dir, f"{img_name}.png")
        im_resized.save(save_path)

        return save_path

    def _postprocess_leaf(self, original_img_path, mask_img_path):
        """ 叶片分割后处理，返回最终图片路径 """
        output_path = mask_img_path.replace("masked", "final").replace(".png", ".jpg")

        orig = cv2.imread(original_img_path)
        mask = cv2.imread(mask_img_path, cv2.IMREAD_GRAYSCALE)

        if orig is None or mask is None:
            print(f"无法加载图像: {original_img_path} 或 {mask_img_path}")
            return None

        if mask.shape != orig.shape[:2]:
            mask = cv2.resize(mask, (orig.shape[1], orig.shape[0]), interpolation=cv2.INTER_NEAREST)

        _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean_mask = cv2.morphologyEx(binary_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        clean_mask = cv2.GaussianBlur(clean_mask, (5, 5), 0)

        mask_3c = cv2.merge([clean_mask, clean_mask, clean_mask])
        result = np.where(mask_3c == 0, 255, orig)

        cv2.imwrite(output_path, result)
        return output_path

    def process_image(self, input_img):
        """ 处理图像，返回最终处理后的图像路径 """
        file_path = os.path.join(self.base_dir, "imgNum.txt")

        # 读取并更新编号
        with open(file_path, "r") as file:
            number = int(file.read().strip())
        with open(file_path, "w") as file:
            file.write(str(number + 1))

        # 保存原图
        filename = f"{number}.jpg"
        orig_path = os.path.join(self.upload_dir, filename)
        Image.fromarray(input_img).save(orig_path)

        # 1. 生成数据集
        test_dataset = SalObjDataset(
            img_name_list=[orig_path], lbl_name_list=[],
            transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)])
        )
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

        # 2. 运行模型
        data_test = next(iter(test_loader))
        inputs_test = Variable(data_test['image'].type(torch.FloatTensor))
        if torch.cuda.is_available():
            inputs_test = inputs_test.cuda()

        d1, *_ = self.model(inputs_test)

        # 3. 归一化 & 保存掩码
        mask_path = self._save_output(orig_path, self._norm_pred(d1[:, 0, :, :]))

        # 4. 叶片后处理
        return self._postprocess_leaf(orig_path, mask_path)
