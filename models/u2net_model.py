import torch
from models.u2net import U2NET  # 引入U2-Net模型架构

model = None

def load_model():
    global model
    model = U2NET()  # 创建模型实例
    model.load_state_dict(torch.load('u2net.pth', map_location=torch.device('cpu')))  # 加载预训练模型的权重文件
    model.eval()  # 设置为评估模式，推理时需要此操作
    return model

def predict(image_tensor):
    with torch.no_grad():  # 关闭梯度计算，推理时不需要计算梯度
        output = model(image_tensor)  # 执行推理
        return output  # 输出模型预测结果（例如分割掩模）