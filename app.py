import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
from PIL import Image

# --- 核心算法：增加了对边缘点识别的容错性 ---
def process_image(image_bytes):
    # 将上传的文件转为 OpenCV 格式
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None, 0
    
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # --- 优化1：更宽松的红点阈值 (解决最上面的点抓不到的问题) ---
    # 降低了饱和度(S)下限到 80，降低了亮度(V)下限到 60
    lower_red1 = np.array([0, 80, 60])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([165, 80, 60])
    upper_red2 = np.array([180, 255, 255])
    
    mask = cv2.add(cv2.inRange(hsv, lower_red1, upper_red1), 
                   cv2.inRange(hsv, lower_red2, upper_red2))

    # --- 优化2：动态面积过滤 ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 15 < area < 3000: # 门槛进一步放低
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                # 仅保留画面中间 30%-70% 宽度的点，排除手部大面积红色干扰
                if 0.3 * w < cX < 0.7 * w:
                    candidates.append((cX, cY))

    # --- 优化3：逻辑排序 ---
    candidates = sorted(candidates, key=lambda x: x[1])
    if len(candidates) >= 3:
        p1, p2, p3 = candidates[0], candidates[len(candidates)//2], candidates[-1]
        
        # 计算角度
        v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
        v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
        angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1.0, 1.0)))
        
        # 绘图
        cv2.line(img, p1, p2, (0, 255, 0), 4)
        cv2.line(img, p2, p3, (0, 255, 0), 4)
        for p in [p1, p2, p3]: cv2.circle(img, p, 15, (0, 255, 255), -1)
        cv2.putText(img, f"{angle:.2f} deg", (p2[0]+30, p2[1]), 1, 2, (0,255,255), 2)
        
        return img, angle
    return None, 0

# --- Streamlit UI 界面 ---
st.title("👓 眼镜面弯角自动测量系统")
st.write("上传包含图片的 Zip 压缩包，系统将自动识别红点并计算角度。")

uploaded_file = st.file_uploader("选择 Zip 压缩包", type="zip")

if uploaded_file is not None:
    results_zip = io.BytesIO()
    processed_images = []
    
    with zipfile.ZipFile(uploaded_file, "r") as z_in:
        with zipfile.ZipFile(results_zip, "w") as z_out:
            file_list = [f for f in z_in.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            progress_bar = st.progress(0)
            for i, file_name in enumerate(file_list):
                with z_in.open(file_name) as f:
                    img_bytes = f.read()
                    res_img, angle = process_image(img_bytes)
                    
                    if res_img is not None:
                        # 转回字节以存入 zip
                        _, encoded_img = cv2.imencode(".jpg", res_img)
                        z_out.writestr(f"processed_{os.path.basename(file_name)}", encoded_img.tobytes())
                        processed_images.append((file_name, res_img, angle))
                
                progress_bar.progress((i + 1) / len(file_list))

    st.success(f"处理完成！成功识别 {len(processed_images)} 张图片。")

    # 提供预览
    if processed_images:
        st.subheader("处理预览")
        cols = st.columns(3)
        for idx, (name, img, ang) in enumerate(processed_images[:6]): # 预览前6张
            with cols[idx % 3]:
                st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption=f"{name}: {ang:.1f}°")

    # 下载按钮
    st.download_button(
        label="📥 下载处理后的结果 (Zip)",
        data=results_zip.getvalue(),
        file_name="measurement_results.zip",
        mime="application/zip"
    )