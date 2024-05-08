from flask import Flask, Response
from torchvision import transforms

from models.common import *

"""
    模版，不能动
"""

app = Flask(__name__)

LABELS = ['right', 'left', 'explosion', 'top', 'collapse', 'water', 'fire', 'person', 'red', 'blue', 'yellow']

class YOLOv5():
    def __init__(self,weights='weights/dog_best.pt'):
        self.weights = weights
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.model = DetectMultiBackend(weights=self.weights)
        self.model.to(self.device)

    def draw(self,image, x1, x2, y1, y2, cls, conf):
        # 不同列别使用不同颜色
        colors = [
            (255, 0, 0),  # 蓝色
            (0, 255, 0),  # 绿色
            (0, 0, 255),  # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 紫色
            (0, 255, 255),  # 黄色
            (128, 0, 0),  # 深蓝色
            (0, 128, 0),  # 深绿色
            (0, 0, 128),  # 深红色
            (128, 128, 0),  # 橄榄色
            (128, 0, 128)  # 粉色
        ]

        color = (0, 0, 0)  # BGR格式，这里表示黑色
        thickness = 3 # 边界框线条粗细
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1

        # 在图像上绘制边界框
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colors[int(cls)], thickness)
        text = f'Class: {LABELS[int(cls)]}, Confidence: {conf:.2f}'
        cv2.putText(image, text, (int(x1), int(y1) - 5), font, font_scale, color, font_thickness)
        return image

    def frame_to_tensor(self,frame):
        frame = cv2.resize(frame, (640, 640))
        # 将图像转换为RGB格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 将图像转换为张量并归一化
        transform = transforms.Compose([transforms.ToTensor()])
        frame_tensor = transform(frame_rgb).unsqueeze(0)  # 添加批量维度
        return frame_tensor

    def get_frame(self,frame):
        # 显示的图像上也这样操作
        frame = cv2.resize(frame, (640, 640))

        # 图片预处理
        frame_tensor = self.frame_to_tensor(frame)

        # 将图片放入cuda/cpu中
        frame_tensor = frame_tensor.to(self.device)

        # 使用模型进行推理
        with torch.no_grad():
            outputs = self.model(frame_tensor)

            # 极大值抑制 ----> 去除框的数量,筛选置信度最高的
            pred = non_max_suppression(outputs, conf_thres=0.4, iou_thres=0.5, max_det=1000)

        # 排除为空的情况
        if pred:
            for detection in pred:
                # 每个 detection 是一个 tensor，每行是一个边界框，包括 (x1, y1, x2, y2, conf, cls)
                for box in detection:
                    box_list = box.tolist()
                    # print(box_list)   # 调试
                    if len(box_list) >= 6:
                        x1, y1, x2, y2, conf, cls = box_list
                        print(f'Bounding box: ({x1}, {y1}) - ({x2}, {y2}), Confidence: {conf}, Class: {LABELS[int(cls)]}')
                        # 绘制对象
                        if cls:
                            frame = self.draw(frame, x1, x2, y1, y2, cls, conf)
        return frame


try:
    # 捕获摄像头视频流
    camera = cv2.VideoCapture(0)
except Exception as e:
    print('未获得摄像头')

def generate_frames():
    num = 0
    while True:
        num += 1
        # 读取视频帧
        success, frame = camera.read()
        if not success:
            break
        else:
            # 在这里可以对视频帧进行处理，例如添加滤镜、人脸识别等

            # 将处理后的视频帧转换为字节流
            frame = yolo.get_frame(frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()

            # 以字节流的形式发送视频帧
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# 加载模型
yolo = YOLOv5()
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=False,port=8080)