# from recognize import predict
import gradio as gr
from image_processor import ImageProcessor
from pest_detector import AdvancedPestDetector

image_processor = ImageProcessor(model_path="u2net.pth")
# 初始化害虫检测模型
detector = AdvancedPestDetector(
    model_path="best_densenet121_model_Pretrain.pth",
    class_names=['AN', 'DI', 'DM', 'HE', 'LI', 'NP']
)

def pest_predict(image):
    _, confidences = detector.predict(image)
    return {k: v for k, v in detector.get_top_n(confidences, n=3)}

demo = gr.Blocks()
with demo:
    gr.Markdown("# 🐞 丝瓜病虫害识别系统")
    with gr.Row():
        with gr.Column():
            img_input = gr.Image(label="输入图像")
            btn1 = gr.Button("分割识别")
            btn2 = gr.Button("直接识别")
        with gr.Column():
            outputs_img = gr.Image(label="分割掩码")
            output_label = gr.Label(label="识别结果", num_top_classes=3)

    btn1.click(fn=image_processor.process_image, inputs=img_input, outputs=outputs_img).then(
        fn=pest_predict, inputs=outputs_img, outputs=output_label
    )
    btn2.click(fn=pest_predict, inputs=img_input, outputs=output_label)

    # 示例图片
    gr.Examples(
        examples=["LI.jpg", "HE.jpg", "AN_DM.jpg", "DI.jpg", "NP.jpg"],
        inputs=img_input,
        outputs=output_label,
        fn=pest_predict,
        cache_examples=False
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
