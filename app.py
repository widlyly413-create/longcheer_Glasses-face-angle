import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
import pandas as pd
from datetime import datetime

# --- 核心算法层：先找圆点，再过黑域（不破坏圆度） ---
def process_image_v13(image_bytes):
    # 1. 图像解码
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: 
        return None, 0, "文件读取失败"
    
    h, w = img.shape[:2]
    
    # --- 动态比例因子 (自适应极细工业级标注) ---
    dyn_radius = max(3, int(w / 180))      
    dyn_line = max(1, int(w / 700))        
    dyn_font_scale = w / 1600              
    dyn_font_thick = max(1, int(w / 900))  

    # 2. 转换为 HSV 颜色空间
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ==========================================
    # 步骤 1：构建黑色镜框“安全参考区”
    # ==========================================
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 65]) 
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # 将黑色镜框蒙版向外膨胀 11 像素，确保有足够宽的“合法安全带”能罩住红点中心
    kernel_dilate = np.ones((11, 11), np.uint8)
    black_mask_expanded = cv2.dilate(black_mask, kernel_dilate, iterations=1)

    # ==========================================
    # 步骤 2：在全图提取纯红点 (不加黑框干扰，维持完美圆度)
    # ==========================================
    lower_red1, upper_red1 = np.array([0, 100, 70]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 120, 100]), np.array([180, 255, 255])
    red_mask = cv2.add(cv2.inRange(hsv, lower_red1, upper_red1), 
                       cv2.inRange(hsv, lower_red2, upper_red2))
    
    # 动态内核开运算：小图用 3x3 防误杀，大图用 5x5
    kernel_size = 3 if w < 1500 else 5
    kernel_open = np.ones((kernel_size, kernel_size), np.uint8)
    red_mask_opened = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_open)

    # ==========================================
    # 步骤 3：轮廓提取与“双因子”校验 (圆度 + 位置锚定)
    # ==========================================
    contours, _ = cv2.findContours(red_mask_opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 宽容的面积筛选
        if 40 < area < 3000:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            # 由于未进行提前相交，这里的红点能够展现出 0.5 以上的完美圆度
            if circularity >= 0.5:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / bh
                
                if 0.6 < aspect_ratio < 1.5:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        # --- 核心改进：中心坐标黑域检验 ---
                        # 检查红点的几何中心点是否位于膨胀后的镜框安全区内。255表示处于黑区
                        if black_mask_expanded[cY, cX] == 255:
                            # 彻底移除原先的 [0.35w - 0.65w] 水平限制
                            centers.append((cX, cY))

    # 4. 点数判定与三点共线垂直优选
    num_pts = len(centers)
    if num_pts < 3:
        return img, 0, f"识别失败：镜框内仅提取到 {num_pts} 个合规点"
    
    # 纵向对齐排序
    centers = sorted(centers, key=lambda x: x[1])
    best_set = None
    min_x_diff = float('inf')
    
    if len(centers) == 3:
        best_set = centers
    else:
        # 如果依然有干扰，找出在 X 轴最接近垂直直线的一组三点
        for i in range(len(centers)-2):
            for j in range(i+1, len(centers)-1):
                for k in range(j+1, len(centers)):
                    pts = [centers[i], centers[j], centers[k]]
                    x_range = max(p[0] for p in pts) - min(p[0] for p in pts)
                    if x_range < min_x_diff:
                        min_x_diff = x_range
                        best_set = pts
                        
    p1, p2, p3 = best_set
    
    # 5. 几何向量法角度解算
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    # 6. 精细绘制标注
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.line(img, p1, p2, (255, 120, 0), dyn_line, cv2.LINE_AA)
    cv2.line(img, p2, p3, (255, 120, 0), dyn_line, cv2.LINE_AA)
    
    for p in [p1, p2, p3]:
        cv2.circle(img, p, dyn_radius, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p, dyn_radius, (0, 0, 0), 1, cv2.LINE_AA)

    text = f"ANGLE: {angle:.2f} DEG"
    text_pos = (p2[0] + 40, p2[1])
    cv2.putText(img, text, text_pos, font, dyn_font_scale, (0,0,0), dyn_font_thick+2, cv2.LINE_AA)
    cv2.putText(img, text, text_pos, font, dyn_font_scale, (255,255,255), dyn_font_thick, cv2.LINE_AA)
    
    return img, angle, "成功"

# --- Streamlit 界面层 (支持单图/批量/历史CSV下载) ---
st.set_page_config(page_title="WrapAngle V13", layout="wide")
st.title("👓 面弯角高精度全自动测量系统 (V13)")
st.caption("最新升级：基于圆度恢复与黑域校验融合算法，完美兼容各类倾斜和低清压缩图。")

if 'history' not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["📸 单图实时测定", "📦 压缩包批量解析"])

with tab1:
    single_file = st.file_uploader("上传单张俯视图", type=['jpg', 'jpeg', 'png'], key="single")
    if single_file:
        res_img, ang, status = process_image_v13(single_file.read())
        
        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption="精细化结果展示")
        with col_info:
            st.subheader("📊 诊断结果")
            if status == "成功":
                st.success(f"计算面弯角: {ang:.2f}°")
                if not any(d['文件名'] == single_file.name for d in st.session_state.history):
                    st.session_state.history.append({
                        "测定时间": datetime.now().strftime("%H:%M:%S"),
                        "文件名": single_file.name,
                        "面弯角角度": f"{ang:.2f}°",
                        "诊断状态": "成功"
                    })
            else:
                st.error(status)

with tab2:
    zip_file = st.file_uploader("上传 Zip 图片压缩包", type="zip", key="zip")
    if zip_file:
        out_zip = io.BytesIO()
        with zipfile.ZipFile(zip_file, "r") as z_in, zipfile.ZipFile(out_zip, "w") as z_out:
            files = [f for f in z_in.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            p_bar = st.progress(0)
            for i, f_name in enumerate(files):
                res_img, ang, status = process_image_v13(z_in.read(f_name))
                if res_img is not None:
                    _, buf = cv2.imencode(".jpg", res_img)
                    z_out.writestr(f"Result_{os.path.basename(f_name)}", buf.tobytes())
                    st.session_state.history.append({
                        "操作时间": datetime.now().strftime("%H:%M:%S"),
                        "文件名": os.path.basename(f_name),
                        "面弯角角度": f"{ang:.2f}°" if ang > 0 else "-",
                        "诊断状态": status
                    })
                p_bar.progress((i + 1) / len(files))
        st.success("批量数据解析完毕！")
        st.download_button("📥 导出标注图片包 (Zip)", out_zip.getvalue(), "Measurement_Results_V13.zip")

# --- 实验报表导出 ---
st.divider()
st.subheader("📜 本次项目数据报表")
if st.session_state.history:
    df_h = pd.DataFrame(st.session_state.history)
    st.dataframe(df_h, use_container_width=True)
    st.download_button("📊 导出历史数据表格 (Excel/CSV)", df_h.to_csv(index=False).encode('utf-8-sig'), "data_history_v13.csv", "text/csv")
    if st.button("🗑️ 清空所有表格数据"):
        st.session_state.history = []
        st.rerun()
else:
    st.write("等待上传实验样本图片...")