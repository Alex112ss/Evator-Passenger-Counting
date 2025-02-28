import cv2
import numpy as np

# 設定捕獲影片
cap = cv2.VideoCapture("original_video/2-1.mp4")  # 替換成你的影片檔案路徑

# 載入模型
net = cv2.dnn.readNetFromCaffe("MobileNet-SSD/deploy.prototxt", "MobileNet-SSD/mobilenet_iter_73000.caffemodel")

# 定義電梯區域的固定框位置
roi_elevator_startX, roi_elevator_startY, roi_elevator_endX, roi_elevator_endY = 180, 100, 700, 400

# 定義門的標記點位置
door_marker_x, door_marker_y = 260, 130
door_marker_radius = 5

# 定義門的標記點顏色閾值
door_marker_color_threshold = 10

# 初始化變數
max_people_in_elevator = 0

# 取得影片的寬高和幀率
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# 定義輸出影片的參數
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用MP4編碼
output_video = cv2.VideoWriter('output_video.mp4', fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 在影像上畫出電梯的固定框
    cv2.rectangle(frame, (roi_elevator_startX, roi_elevator_startY), (roi_elevator_endX, roi_elevator_endY), (0, 255, 0), 2)

    # 取得電梯區域
    roi = frame[roi_elevator_startY:roi_elevator_endY, roi_elevator_startX:roi_elevator_endX]

    # 計算中心部分的亮度
    brightness = np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY))

    # 根據亮度閾值判斷
    brightness_threshold = 146
    if brightness > brightness_threshold:
        elevator_doors_open = True
    else:
        elevator_doors_open = False

    # 當電梯門打開時進行人數檢測
    if elevator_doors_open:
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        num_people_in_elevator = 0

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.2:
                box = detections[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
                (startX, startY, endX, endY) = box.astype("int")

                if roi_elevator_startX <= startX <= roi_elevator_endX and roi_elevator_startY <= startY <= roi_elevator_endY:
                    num_people_in_elevator += 1
                    cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

        max_people_in_elevator = max(max_people_in_elevator, num_people_in_elevator)

    # 在左上角顯示電梯門的狀態
    door_roi = frame[door_marker_y - door_marker_radius:door_marker_y + door_marker_radius,
                      door_marker_x - door_marker_radius:door_marker_x + door_marker_radius]
    door_mean_color = np.mean(door_roi, axis=(0, 1))

    # 判斷標記點周圍的顏色變化
    if door_mean_color[1] - door_mean_color[0] > door_marker_color_threshold:
        elevator_doors_open = True
    else:
        elevator_doors_open = False

    door_status = "Open" if elevator_doors_open else "Close"
    cv2.putText(frame, f"Door Status: {door_status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 即時檢測外部人數
    blob_external = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob_external)
    detections_external = net.forward()

    external_people_count = 0

    for i in range(detections_external.shape[2]):
        confidence_external = detections_external[0, 0, i, 2]
        if confidence_external > 0.7:
            box_external = detections_external[0, 0, i, 3:7] * np.array([frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]])
            (startX_external, startY_external, endX_external, endY_external) = box_external.astype("int")

            external_people_count += 1
            cv2.rectangle(frame, (startX_external, startY_external), (endX_external, endY_external), (255, 0, 0), 2)

    # 在影像上顯示電梯區域內的最大人數
    cv2.putText(frame, f"People in Elevator: {max_people_in_elevator}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # 在影像上顯示即時外部人數
    cv2.putText(frame, f"External People: {external_people_count}", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # 將處理後的影像寫入輸出影片
    output_video.write(frame)

    # 顯示影像
    cv2.imshow("Real-Time People Counting", frame)

    # 按 'q' 鍵退出迴圈
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 釋放資源
cap.release()
output_video.release()
cv2.destroyAllWindows()
