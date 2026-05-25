import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
import pandas as pd
from datetime import datetime

# --- 核心算法：黑域锚定 + 三位一体过滤 ---
def process_image_v11(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None, 0, "文件损坏或无法读取"
    
    h, w = img.shape[:2]
    
    # UI 自适应微缩化参数
    dyn_radius = max(3, int(w / 180))
    dyn_line = max(1, int(w / 700))
    dyn_font_scale = w / 1600
    dyn_font_thick = max(1, int(w / 900))

    # 转换颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ==========================================
    # 步骤 1：提取黑色镜框区域（创建“安全框蒙版”）
    # ==========================================
    # 黑色在 HSV 中的特点是：亮度（Value）非常低。
    # 这里设定亮度上限为 65（可根据光线微调），提取出画面中所有的黑色物体
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 65]) 
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # 对黑色区域进行轻微膨胀（Dilation），把镜框边缘稍微扩大几个像素，确保能包裹住贴在边缘的红点
    kernel_dilate = np.ones((7, 7), np.uint8)
    black_mask_expanded = cv2.dilate(black_mask, kernel_dilate, iterations=1)

    # ==========================================
    # 步骤 2：提取红色红点
    # ==========================================
    lower_red1, upper_red1 = np.array([0, 100, 70]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 120, 100]), np.array([180, 255, 255])
    red_mask = cv2.add(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    
    # 动态自适应形态学降噪内核
    # 小图用 3x3 避免擦除红点，大图用 5x5 增强降噪效果
    kernel_size = 3 if w < 1500 else 5
    kernel_open = np.ones((kernel_size, kernel_size), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_open)

    # ==========================================
    # 步骤 3：核心融合（关键优化点）
    # ==========================================
    # 将红点蒙版与黑色镜框蒙版做“位与（AND）”操作
    # 只有【既是红色】又【长在黑色镜框上】的像素才会被保留，长在皮肤上的红色直接变黑抹除！
    final_mask = cv2.bitwise_and(red_mask, black_mask_expanded)

    # 4. 几何约束过滤
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 50 < area < 2000:
            peri = cv2.arcLength(cnt, True)
            if peri == 0: continue
            circularity = 4 * np.pi * (area / (peri * peri))
            if circularity >= 0.5:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    centers.append((cX, cY))

    if len(centers) < 3:
        return img, 0, f"识别失败：镜框内仅识别到 {len(centers)} 个点"
    
    # 5. 点位对齐排序
    centers = sorted(centers, key=lambda x: x[1])
    best_set = None
    min_x_diff = float('inf')
    
    if len(centers) == 3:
        best_set = centers
    else:
        for i in range(len(centers)-2):
            for j in range(i+1, len(centers)-1):
                for k in range(j+1, len(centers)):
                    pts = [centers[i], centers[j], centers[k]]
                    x_range = max(p[0] for p in pts) - min(p[0] for p in pts)
                    if x_range < min_x_diff:
                        min_x_diff = x_range
                        best_set = pts
                        
    p1, p2, p3 = best_set
    
    # 6. 计算角度
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    # 7. 专业极细标注绘制
    cv2.line(img, p1, p2, (255, 120, 0), dyn_line, cv2.LINE_AA)
    cv2.line(img, p2, p3, (255, 120, 0), dyn_line, cv2.LINE_AA)
    for p in [p1, p2, p3]:
        cv2.circle(img, p, dyn_radius, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p, dyn_radius, (0, 0, 0), 1, cv2.LINE_AA)
    
    text = f"ANGLE: {angle:.2f} DEG"
    text_pos = (p2[0]+40, p2[1])
    cv2.putText(img, text, text_pos, cv2.FONT_HERSHEY_DUPLEX, dyn_font_scale, (0,0,0), dyn_font_thick+2, cv2.LINE_AA)
    cv2.putText(img, text, text_pos, cv2.FONT_HERSHEY_DUPLEX, dyn_font_scale, (255,255,255), dyn_font_thick, cv2.LINE_AA)
    
    return img, angle, "成功"

# --- Streamlit UI 控制层（单图+批量+历史记录） ---
st.set_page_config(page_title="WrapAngle V11", layout="wide")
st.title("👓 面弯角高精度测量系统 V11 (黑色镜框锚定版)")

if 'history' not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["📸 单图实时检测", "📦 压缩包批量处理"])

with tab1:
    single_file = st.file_uploader("请上传单张图片", type=['jpg', 'jpeg', 'png'], key="single")
    if single_file:
        res_img, ang, status = process_image_v11(single_file.read())
        c_img, c_info = st.columns([2, 1])
        with c_img:
            st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption="精细化测量结果")
        with c_info:
            st.subheader("📋 实时诊断")
            if status == "成功":
                st.success(f"面弯角角度: {ang:.2f}°")
                if not any(d['文件名'] == single_file.name for d in st.session_state.history):
                    st.session_state.history.append({
                        "记录时间": datetime.now().strftime("%H:%M:%S"),
                        "文件名": single_file.name,
                        "测量角度": f"{ang:.2f}°",
                        "结果": "成功"
                    })
            else:
                st.error(status)

with tab2:
    zip_file = st.file_uploader("请上传包含图片的 Zip 包", type="zip", key="zip")
    if zip_file:
        out_zip = io.BytesIO()
        with zipfile.ZipFile(zip_file, "r") as z_in, zipfile.ZipFile(out_zip, "w") as z_out:
            files = [f for f in z_in.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for f_name in files:
                res_img, ang, status = process_image_v11(z_in.read(f_name))
                if res_img is not None:
                    _, buf = cv2.imencode(".jpg", res_img)
                    z_out.writestr(f"Result_{os.path.basename(f_name)}", buf.tobytes())
                    st.session_state.history.append({
                        "记录时间": datetime.now().strftime("%H:%M:%S"),
                        "文件名": os.path.basename(f_name),
                        "测量角度": f"{ang:.2f}°" if ang > 0 else "-",
                        "结果": status
                    })
        st.success("批量数据处理完毕！")
        st.download_button("📥 导出处理后的图片包", out_zip.getvalue(), "V11_Results.zip")

# --- 汇总看板与历史导出 ---
st.divider()
st.subheader("📜 本次实验历史记录")
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df, use_container_width=True)
    st.download_button("📊 导出为 Excel/CSV 报表", df.to_csv(index=False).encode('utf-8-sig'), "data_history.csv", "text/csv")
    if st.button("🗑️ 清除缓存数据"):
        st.session_state.history = []
        st.rerun()
else:
    st.write("等待数据输入...")