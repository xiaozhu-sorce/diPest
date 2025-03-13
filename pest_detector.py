import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Dict, Tuple
import logging
import numpy as np

class AdvancedPestDetector:
    def __init__(self, 
                 model_path: str, 
                 class_names: list,
                 device: str = "auto"):
        """
        增强版病虫害识别器
        :param model_path: 模型文件路径(.pth)
        :param class_names: 类别名称列表，顺序与训练时一致
        :param device: 运行设备(auto/cpu/cuda)
        """
        # 初始化日志
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # 设备配置
        self.device = self._configure_device(device)
        self.logger.info(f"运行设备: {self.device}")
        
        # 加载模型
        self.model = self._load_model(model_path)
        self.model.eval()
        
        # 类别配置
        self.class_names = class_names
        if len(class_names) != 6:
            raise ValueError("类别数量必须为6，与模型输出维度一致")
        
        # 图像预处理
        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])

    def _configure_device(self, device_setting: str) -> torch.device:
        """配置运行设备"""
        if device_setting == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_setting)

    def _load_model(self, model_path: str) -> nn.Module:
        """加载训练好的模型"""
        try:
            # 初始化模型结构
            model = models.densenet121(pretrained=False)
            model.classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(model.classifier.in_features, 6)  # 输出层改为6个单元
            )
            
            # 加载权重
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict)
            return model.to(self.device)
        except FileNotFoundError:
            self.logger.error(f"模型文件未找到: {model_path}")
            raise
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            raise

    def _validate_image(self, image: Image.Image) -> bool:
        """验证图像有效性"""
        if image.mode != 'RGB':
            self.logger.warning("图像非RGB格式，已自动转换")
            image = image.convert('RGB')
        return image

    def predict(self, input_source) -> Tuple[str, Dict[str, float]]:
        """
        通用预测方法
        :param input_source: 支持文件路径/PIL图像/NumPy数组
        :return: (预测类别, {类别: 置信度})
        """
        # 处理不同输入类型
        if isinstance(input_source, str):
            img = Image.open(input_source)
        elif isinstance(input_source, Image.Image):
            img = input_source
        elif isinstance(input_source, np.ndarray):
            img = Image.fromarray(input_source)
        else:
            raise TypeError("不支持的输入类型，支持: 文件路径/PIL图像/NumPy数组")
        
        # 验证并预处理
        img = self._validate_image(img)
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        
        # 执行推理
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # 生成结果
        confidences = {self.class_names[i]: float(probs[i]) 
                      for i in range(len(self.class_names))}
        pred_class = max(confidences, key=confidences.get)
        return pred_class, confidences

    def get_top_n(self, confidences: Dict[str, float], n: int = 3) -> list:
        """获取置信度最高的前N个结果"""
        return sorted(confidences.items(), key=lambda x: x[1], reverse=True)[:n]