import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
import pandas as pd
from datetime import datetime

# --- 核心算法：严格遵循验证过的 OpenCV-test 逻辑 ---
def process_image_v10(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None, 0, "文件损坏或无法读取"
    
    h, w = img.shape[:2]
    dyn_radius = max(3, int(w / 180))
    dyn_line = max(1, int(w / 700))
    dyn_font_scale = w / 1600
    dyn_font_thick = max(1, int(w / 900))

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 100, 70]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 120, 100]), np.array([180, 255, 255])
    mask = cv2.add(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    centers = []
    ROI_X_MIN, ROI_X_MAX = 0.35, 0.65
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 2000:
            peri = cv2.arcLength(cnt, True)
            if peri == 0: continue
            circularity = 4 * np.pi * (area / (peri * peri))
            if circularity >= 0.5:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    if w * ROI_X_MIN < cX < w * ROI_X_MAX:
                        centers.append((cX, cY))

    if len(centers) < 3:
        return img, 0, f"识别失败：仅找到 {len(centers)} 个点"
    
    centers = sorted(centers, key=lambda x: x[1])
    p1, p2, p3 = centers[0], centers[len(centers)//2], centers[-1]
    
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    angle = np.degrees(np.arccos(np.clip(np.dot(v1, v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)), -1.0, 1.0)))
    
    # 标注绘制
    cv2.line(img, p1, p2, (255, 120, 0), dyn_line, cv2.LINE_AA)
    cv2.line(img, p2, p3, (255, 120, 0), dyn_line, cv2.LINE_AA)
    for p in [p1, p2, p3]:
        cv2.circle(img, p, dyn_radius, (0, 255, 255), -1, cv2.LINE_AA)
    
    text = f"ANGLE: {angle:.2f} DEG"
    cv2.putText(img, text, (p2[0]+40, p2[1]), cv2.FONT_HERSHEY_DUPLEX, dyn_font_scale, (0,0,0), dyn_font_thick+2, cv2.LINE_AA)
    cv2.putText(img, text, (p2[0]+40, p2[1]), cv2.FONT_HERSHEY_DUPLEX, dyn_font_scale, (255,255,255), dyn_font_thick, cv2.LINE_AA)
    
    return img, angle, "成功"

# --- Streamlit 界面 ---
st.set_page_config(page_title="WrapAngle V10", layout="wide")
st.title("👓 面弯角测量系统 V10")

# 初始化 session_state 用于保存本次运行的历史数据
if 'history' not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["📸 单图上传", "📦 批量上传 (Zip)"])

# --- Tab 1: 单图上传 ---
with tab1:
    single_file = st.file_uploader("上传单张眼镜俯拍图", type=['jpg', 'jpeg', 'png'], key="single")
    if single_file:
        data = single_file.read()
        res_img, ang, status = process_image_v10(data)
        
        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption="处理结果")
        with col_info:
            st.subheader("测量信息")
            if status == "成功":
                st.success(f"测量角度: {ang:.2f}°")
                # 存入历史记录
                if not any(d['文件名'] == single_file.name for d in st.session_state.history):
                    st.session_state.history.append({
                        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "文件名": single_file.name,
                        "角度": f"{ang:.2f}°",
                        "状态": "成功"
                    })
            else:
                st.error(status)

# --- Tab 2: 批量上传 ---
with tab2:
    zip_file = st.file_uploader("上传图片 Zip 压缩包", type="zip", key="zip")
    if zip_file:
        output_zip = io.BytesIO()
        with zipfile.ZipFile(zip_file, "r") as z_in, zipfile.ZipFile(output_zip, "w") as z_out:
            files = [f for f in z_in.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for f_name in files:
                img_data = z_in.read(f_name)
                res_img, ang, status = process_image_v10(img_data)
                if res_img is not None:
                    _, buf = cv2.imencode(".jpg", res_img)
                    z_out.writestr(f"Result_{os.path.basename(f_name)}", buf.tobytes())
                    st.session_state.history.append({
                        "时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "文件名": os.path.basename(f_name),
                        "角度": f"{ang:.2f}°" if ang > 0 else "-",
                        "状态": status
                    })
        st.success(f"批量处理完成！")
        st.download_button("📥 下载结果包", output_zip.getvalue(), "batch_results.zip")

# --- 历史数据展示 ---
st.divider()
st.subheader("📜 本次操作记录")
if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history, use_container_width=True)
    
    # 导出历史数据为 CSV
    csv = df_history.to_csv(index=False).encode('utf-8-sig')
    st.download_button(
        "📊 导出历史记录表格 (CSV)",
        csv,
        "measurement_history.csv",
        "text/csv",
        key='download-csv'
    )
    if st.button("🗑️ 清空历史记录"):
        st.session_state.history = []
        st.rerun()
else:
    st.write("暂无历史记录。")