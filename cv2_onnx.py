import cv2
import numpy as np

def format_yolov5(source):

    # put the image in square big enough
    col, row, _ = source.shape
    _max = max(col, row)
    resized = np.zeros((_max, _max, 3), np.uint8)
    resized[0:col, 0:row] = source

    # resize to 640x640, normalize to [0,1[ and swap Red and Blue channels
    result = cv2.dnn.blobFromImage(resized, 1/255.0, (640, 640), swapRB=True)

    return result

# 数据解析
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

def main():
    net = cv2.dnn.readNetFromONNX("./weights/dog_best.onnx")
    image = cv2.imread("../cap/23333.jpg")
    result = format_yolov5(image)
    print(result.shape)
    # 查看onnx模型的输入
    """
        输入的数据为 (1,3,640,640)
        输出数据为  (1,25200,16)
    """
    net.setInput(result)
    output = net.forward()
    print(output.shape)

    class_ids, confidences, boxes = unwrap_detection(image,output)


if __name__ == '__main__':
    main()

