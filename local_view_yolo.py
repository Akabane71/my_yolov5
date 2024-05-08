import cv2
from torchvision import transforms
from models.common import *

"""
    解析远程的视频流，测试预测结果
"""

LABELS = ['right', 'left', 'explosion', 'top', 'collapse', 'water', 'fire', 'person', 'red', 'blue', 'yellow']
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


# YOLO类
class YOLOv5():
    def __init__(self, weights='./weights/best_yolov5m.pt'):
        self.weights = weights
        self.device = "cuda" if torch.cuda.is_available() else 'cpu'
        self.model = DetectMultiBackend(weights=self.weights)
        self.model.to(self.device)

    def draw(self, image, x1, x2, y1, y2, cls, conf):

        color = (0, 0, 0)  # BGR格式，这里表示黑色
        thickness = 3  # 边界框线条粗细
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        font_thickness = 1

        # 在图像上绘制边界框
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), colors[int(cls)], thickness)
        text = f'Class: {LABELS[int(cls)]}, Confidence: {conf:.2f}'
        cv2.putText(image, text, (int(x1), int(y1) - 5), font, font_scale, color, font_thickness)
        return image

    # 将数据转换为tensor_frame，尺寸和归一化
    def frame_to_tensor(self, frame):
        frame = cv2.resize(frame, (640, 640))
        # 将图像转换为RGB格式
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 将图像转换为张量并归一化
        transform = transforms.Compose([transforms.ToTensor()])
        frame_tensor = transform(frame_rgb).unsqueeze(0)  # 添加批量维度
        return frame_tensor

    # 获得处理后的图像
    def get_frame(self, frame):
        """
        Args:
            frame:未处理的任意图像

        Returns: 处理后的图像

        """
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
                        print(
                            f'Bounding box: ({x1}, {y1}) - ({x2}, {y2}), Confidence: {conf}, Class: {LABELS[int(cls)]}')
                        # 绘制对象
                        if cls:
                            frame = self.draw(frame, x1, x2, y1, y2, cls, conf)
        return frame


# 获取字节流数据
def get_video_capture(url):
    try:
        # 打开网络视频流
        cap = cv2.VideoCapture(url)

        # 检查视频流是否成功打开
        if not cap.isOpened():
            print("Error: Unable to open video stream.")
            return

        # 循环读取并显示视频帧
        while True:
            ret, frame = cap.read()

            # 检查帧是否成功读取
            if not ret:
                print("Error: Unable to read frame.")
                break
            frame = model.get_frame(frame)
            # 在这里可以对帧进行处理，例如显示、保存等
            cv2.imshow('Frame', frame)

            # 按下'q'键退出循环
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # 释放视频流资源
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print('Error:', e)


model = YOLOv5()



if __name__ == '__main__':
    url = 'http://192.168.1.101:5000/'
    get_video_capture(url)