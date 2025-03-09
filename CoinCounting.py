import cv2
import numpy as np

# กำหนดขนาดภาพคงที่
FIXED_WIDTH = 800
FIXED_HEIGHT = 700

# ฟังก์ชันลบ Noise และลดแสงสะท้อน
def preprocess_image(image):
    image = cv2.bilateralFilter(image, 9, 75, 75)
    return image

# ฟังก์ชันใช้ Adaptive Threshold และ Canny Edge Detection
def enhance_edges(mask):
    edges = cv2.Canny(mask, 30, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    return edges

# ฟังก์ชันใช้ Watershed Algorithm แยกเหรียญที่ติดกัน
def apply_watershed(mask, original_image):
    _, thresh = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    edges = enhance_edges(thresh)

    kernel = np.ones((3, 3), np.uint8)
    sure_bg = cv2.dilate(edges, kernel, iterations=2)  # ลด iterations จาก 3 เป็น 2
    sure_bg = cv2.erode(sure_bg, kernel, iterations=2)  # เพิ่ม iterations ของ erode

    # 🔄 ปรับการใช้ Distance Transform และ Threshold
    dist_transform = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
    cv2.normalize(dist_transform, dist_transform, 0, 1.0, cv2.NORM_MINMAX)
    ret, sure_fg = cv2.threshold(dist_transform, 0.4, 1.0, cv2.THRESH_BINARY)  # 🔄 เปลี่ยนจาก 0.3 เป็น 0.4

    # 🔄 ใช้ erode ก่อน เพื่อให้จุดศูนย์กลางชัดเจนขึ้น
    sure_fg = cv2.erode(sure_fg, kernel, iterations=2)
    sure_fg = np.uint8(sure_fg * 255)
    unknown = cv2.subtract(sure_bg, sure_fg)

    ret, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(original_image, markers)
    mask[markers == -1] = 0
    return markers, mask

# ฟังก์ชันตรวจจับวงรีในเหรียญ
def detect_ellipses(markers, mask, original_image, color):
    count = 0
    index = 1  # 🔄 ตัวนับหมายเลขวงกลม
    for marker in np.unique(markers):
        if marker == 0 or marker == 1:
            continue  # ข้ามพื้นหลังและขอบที่ไม่ต้องการ

        mask_single = np.zeros(mask.shape, dtype=np.uint8)
        mask_single[markers == marker] = 255

        contours, _ = cv2.findContours(mask_single, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if len(cnt) >= 1 and 0.01 < area < 5000:
                ellipse = cv2.fitEllipse(cnt)
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = float(w) / h
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(area) / hull_area if hull_area > 0 else 0
                arc_len = cv2.arcLength(cnt, True)

                if 0.01 < aspect_ratio < 10 and solidity > 0.01 and arc_len < 500:
                    cv2.ellipse(original_image, ellipse, color, 2)
                    
                    # 🔄 วาดหมายเลขที่ศูนย์กลางวงรี
                    center = (int(ellipse[0][0]), int(ellipse[0][1]))
                    cv2.putText(original_image, str(index), center, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                    index += 1  # 🔄 เพิ่มตัวนับหมายเลข
                    
                    count += 1
    return count


# ฟังก์ชันตรวจจับเหรียญสีเหลือง
def detect_yellow_coins(hsv, original_image):
    lower_yellow = np.array([20, 90, 50])
    upper_yellow = np.array([30, 255, 255])
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    mask_yellow = cv2.erode(mask_yellow, np.ones((3, 3), np.uint8), iterations=2)
    mask_yellow = cv2.dilate(mask_yellow, np.ones((3, 3), np.uint8), iterations=2)

    markers, mask_yellow = apply_watershed(mask_yellow, original_image)
    yellow_count = detect_ellipses(markers, mask_yellow, original_image, (255, 0, 0))
    return yellow_count, mask_yellow

# ฟังก์ชันตรวจจับเหรียญสีฟ้า
def detect_blue_coins(hsv, original_image):
    lower_blue = np.array([88, 100, 50])
    upper_blue = np.array([130, 255, 255])
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)

    mask_blue = cv2.erode(mask_blue, np.ones((3, 3), np.uint8), iterations=3)
    mask_blue = cv2.dilate(mask_blue, np.ones((3, 3), np.uint8), iterations=3)

    markers, mask_blue = apply_watershed(mask_blue, original_image)
    blue_count = detect_ellipses(markers, mask_blue, original_image, (0, 255, 255))
    return blue_count, mask_blue

# ฟังก์ชันหลักในการนับเหรียญ
def coinCounting(filename):
    im = cv2.imread(filename)
    im = cv2.resize(im, (FIXED_WIDTH, FIXED_HEIGHT), interpolation=cv2.INTER_AREA)
    im = preprocess_image(im)
    
    hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    
    yellow_count, mask_yellow = detect_yellow_coins(hsv, im)
    blue_count, mask_blue = detect_blue_coins(hsv, im)
    
    cv2.putText(im, f"[Yellow: {yellow_count}, Blue: {blue_count}]", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow('Result Image', im)
    cv2.imshow('Yellow Mask', mask_yellow)
    #cv2.imshow('Blue Mask', mask_blue)
    cv2.waitKey(0)
    
    return [yellow_count, blue_count]

# ทดสอบกับภาพตัวอย่าง
for i in range(1, 11):
    filename = f"C:\\Users\\Yuki\\Downloads\\CoinCounting\\coin{i}.jpg"
    result = coinCounting(filename)
    print(f'Image {i}: Yellow = {result[0]}, Blue = {result[1]}')

cv2.destroyAllWindows()
