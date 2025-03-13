import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# 定义模型结构（必须与训练时完全一致）
def create_model(num_classes=5):
    model = models.densenet121(pretrained=False)
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.classifier.in_features, num_classes))
    return model

# 加载训练好的模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = create_model().to(device)
model.load_state_dict(torch.load("best_densenet121_model_Pretrain_20241029.pth", map_location=device))
model.eval()

# 预处理转换（与验证集相同）
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 类别名称（根据实际类别修改）
class_names = ["LI", "HE", "AN_DM", "DI", "NP"]  # 示例名称

def predict(img):
    # 转换图片并预处理
    img = Image.fromarray(img.astype('uint8'), 'RGB')
    img_tensor = preprocess(img).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        outputs = model(img_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]
    
    # 获取预测结果
    confidences = {class_names[i]: float(probs[i]) for i in range(len(class_names))}
    pred_class = max(confidences, key=confidences.get)
    return f"识别结果：{pred_class}（置信度：{confidences[pred_class]:.2%}）"