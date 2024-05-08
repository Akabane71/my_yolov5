import cv2
import numpy as np

def gpt_cv2_yolo():
    # 加载 ONNX 模型
    net = cv2.dnn.readNetFromONNX("./weights/dog_best.onnx")

    print('载入成功')
    # 加载图像
    image = cv2.imread("../cap/23333.jpg")

    # 预处理图像（根据模型的要求进行调整）
    blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(640, 640), mean=(127.5, 127.5, 127.5), swapRB=True)

    # 将预处理后的图像作为输入传递给模型
    net.setInput(blob)

    # 执行推理
    output = net.forward()




def format_yolov5(image):

    # put the image in square big enough
    # col, row, _ = source.shape
    # _max = max(col, row)
    # resized = np.zeros((_max, _max, 3), np.uint8)
    # resized[0:col, 0:row] = source

    # resize to 640x640, normalize to [0,1[ and swap Red and Blue channels
    result = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(640, 640), mean=(127.5, 127.5, 127.5), swapRB=True)

    return result

def unwrap_detection(input_image, output_data):
    class_ids = []
    confidences = []
    boxes = []

    rows = output_data.shape[0]

    image_width, image_height, _ = input_image.shape

    x_factor = image_width / 640
    y_factor =  image_height / 640

    for r in range(rows):
        row = output_data[r]
        confidence = row[4]
        if confidence >= 0.4:
            classes_scores = row[5:]
            _, _, _, max_indx = cv2.minMaxLoc(classes_scores)
            class_id = max_indx[1]
            if (classes_scores[class_id] > .25):
                confidences.append(confidence)
                class_ids.append(class_id)
                x, y, w, h = row[0].item(), row[1].item(), row[2].item(), row[3].item()
                left = int((x - 0.5 * w) * x_factor)
                top = int((y - 0.5 * h) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)

    return class_ids, confidences, boxes

def nms(boxes,confidences,class_ids):
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.25, 0.45)

    result_class_ids = []
    result_confidences = []
    result_boxes = []

    for i in indexes:
        result_confidences.append(confidences[i])
        result_class_ids.append(class_ids[i])
        result_boxes.append(boxes[i])
    return result_class_ids,result_confidences,result_boxes
if __name__ == '__main__':
    net = cv2.dnn.readNetFromONNX("./weights/dog_best.onnx")
    image = cv2.imread("../cap/23333.jpg")
    # 数据预处理
    output = cv2.dnn.blobFromImage(image, scalefactor=1.0, size=(640, 640), mean=(127.5, 127.5, 127.5), swapRB=True)
    # 处理数据
    net.forward(output)
    predictions = net.setInput(output)
    class_ids, confidences, boxes = unwrap_detection(output,predictions)

    result_class_ids,result_confidences,result_boxes = nms(class_ids, confidences, boxes)
    print(result_boxes)
    print(result_confidences)
    print(result_boxes)



