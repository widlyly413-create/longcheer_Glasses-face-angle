import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
import pandas as pd
from datetime import datetime

# --- 核心算法层：黑框轮廓填充包裹 + 失败容错绘制 ---
def process_image_v15(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: 
        return None, 0, "文件读取失败"
    
    h, w = img.shape[:2]
    
    # 动态比例因子
    dyn_radius = max(4, int(w / 160))      
    dyn_line = max(2, int(w / 600))        
    dyn_font_scale = w / 1200              
    dyn_font_thick = max(1, int(w / 800))  
    font = cv2.FONT_HERSHEY_DUPLEX

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ==========================================
    # 步骤 1：识别黑色区域并“包裹”内部所有颜色
    # ==========================================
    # 提取黑色
    lower_black, upper_black = np.array([0, 0, 0]), np.array([180, 255, 75]) 
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # 【核心逻辑】：寻找黑色区域的外轮廓，并将其内部全部填充为白色（实心化）
    # 这样，就算红点原本在黑框里是个“洞”，现在也会被当成黑框的一部分被包裹进去
    contours_black, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    black_filled = np.zeros_like(black_mask)
    for cb in contours_black:
        if cv2.contourArea(cb) > 1000: # 过滤掉极小的黑色噪点，只填补大镜框
            cv2.drawContours(black_filled, [cb], -1, 255, thickness=cv2.FILLED)
            
    # 稍微膨胀一下实心黑框，确保边缘的红点也能被完全包裹
    kernel_dilate = np.ones((15, 15), np.uint8)
    black_filled = cv2.dilate(black_filled, kernel_dilate, iterations=1)

    # ==========================================
    # 步骤 2：识别红色点
    # ==========================================
    lower_red1, upper_red1 = np.array([0, 100, 70]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 120, 100]), np.array([180, 255, 255])
    red_mask = cv2.add(cv2.inRange(hsv, lower_red1, upper_red1), 
                       cv2.inRange(hsv, lower_red2, upper_red2))

    # ==========================================
    # 步骤 3：在黑框包裹区内提取红点
    # ==========================================
    # 只有在 black_filled 范围内的红点才会保留。
    # 因为 black_filled 是实心的，红点不会被切碎，完美保留了原始的圆形边缘！
    red_on_black = cv2.bitwise_and(red_mask, black_filled)
    
    # 对筛选后的红点进行开运算降噪（动态内核防误杀）
    kernel_size = 3 if w < 1500 else 5
    kernel_open = np.ones((kernel_size, kernel_size), np.uint8)
    red_cleaned = cv2.morphologyEx(red_on_black, cv2.MORPH_OPEN, kernel_open)

    # ==========================================
    # 步骤 4：几何特征校验
    # ==========================================
    contours_red, _ = cv2.findContours(red_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    
    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        if 40 < area < 3000:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            if circularity >= 0.45:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / bh
                if 0.6 < aspect_ratio < 1.5:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                        # 区域限制：只保留水平方向 35%-65% 范围内的点
                        if 0.35 * w < cX < 0.65 * w:
                            centers.append((cX, cY))

    # ==========================================
    # 步骤 5：【新增】失败容错绘制逻辑
    # ==========================================
    num_pts = len(centers)
    if num_pts < 3:
        # 把找到的 1 个或 2 个点用【红色】醒目标出，方便排查漏掉了哪里的点
        for p in centers:
            cv2.circle(img, p, dyn_radius, (0, 0, 255), -1, cv2.LINE_AA) # 红色实心点
            cv2.circle(img, p, dyn_radius + 2, (255, 255, 255), 2, cv2.LINE_AA) # 白色描边
            
        fail_text = f"FAIL: Only Found {num_pts}/3 Points"
        cv2.putText(img, fail_text, (int(w*0.05), int(h*0.1)), font, dyn_font_scale, (0, 0, 0), dyn_font_thick+2, cv2.LINE_AA)
        cv2.putText(img, fail_text, (int(w*0.05), int(h*0.1)), font, dyn_font_scale, (0, 0, 255), dyn_font_thick, cv2.LINE_AA)
        
        return img, 0, f"识别失败：仅提取到 {num_pts} 个合规点"
    
    # ==========================================
    # 步骤 6：成功计算与绘制 (大于等于3个点)
    # ==========================================
    centers = sorted(centers, key=lambda x: x[1])
    best_set = centers if len(centers) == 3 else None
    
    if len(centers) > 3:
        min_x_diff = float('inf')
        for i in range(len(centers)-2):
            for j in range(i+1, len(centers)-1):
                for k in range(j+1, len(centers)):
                    pts = [centers[i], centers[j], centers[k]]
                    x_range = max(p[0] for p in pts) - min(p[0] for p in pts)
                    if x_range < min_x_diff:
                        min_x_diff = x_range
                        best_set = pts
                        
    p1, p2, p3 = best_set
    
    v1, v2 = np.array([p1[0]-p2[0], p1[1]-p2[1]]), np.array([p3[0]-p2[0], p3[1]-p2[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    # 成功绘制 (青黄色点 + 橙色线)
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

# --- Streamlit UI 层 ---
st.set_page_config(page_title="WrapAngle V15", layout="wide")
st.title("👓 面弯角测量系统 (V15 拓扑包裹防切割版)")
st.caption("新增特性：黑框实心化包裹，解决圆点被切割问题；支持识别失败时标记遗漏点。")

if 'history' not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["📸 单图实时测定", "📦 压缩包批量解析"])

with tab1:
    single_file = st.file_uploader("上传单张俯视图", type=['jpg', 'jpeg', 'png'], key="single")
    if single_file:
        res_img, ang, status = process_image_v15(single_file.read())
        col_img, col_info = st.columns([2, 1])
        with col_img:
            # 即便失败，也会显示画了红点的失败图
            st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption="检测结果图 (失败会红点标记)")
        with col_info:
            st.subheader("📊 诊断结果")
            if status == "成功":
                st.success(f"计算面弯角: {ang:.2f}°")
                if not any(d['文件名'] == single_file.name for d in st.session_state.history):
                    st.session_state.history.append({
                        "测定时间": datetime.now().strftime("%H:%M:%S"),
                        "文件名": single_file.name,
                        "面弯角": f"{ang:.2f}°",
                        "状态": "成功"
                    })
            else:
                st.error(status)
                st.warning("请查看左侧图片，已用【红点】标记出成功识别到的部分贴纸，可借此判断哪个点因为光照或角度原因未达标。")

with tab2:
    zip_file = st.file_uploader("上传 Zip 图片压缩包", type="zip", key="zip")
    if zip_file:
        out_zip = io.BytesIO()
        with zipfile.ZipFile(zip_file, "r") as z_in, zipfile.ZipFile(out_zip, "w") as z_out:
            files = [f for f in z_in.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            p_bar = st.progress(0)
            for i, f_name in enumerate(files):
                res_img, ang, status = process_image_v15(z_in.read(f_name))
                if res_img is not None:
                    _, buf = cv2.imencode(".jpg", res_img)
                    # 失败的图片也会被保存在 Zip 中，文件名前加上 Fail_ 方便筛选
                    prefix = "Result_" if status == "成功" else "Fail_"
                    z_out.writestr(f"{prefix}{os.path.basename(f_name)}", buf.tobytes())
                    st.session_state.history.append({
                        "操作时间": datetime.now().strftime("%H:%M:%S"),
                        "文件名": os.path.basename(f_name),
                        "面弯角": f"{ang:.2f}°" if ang > 0 else "-",
                        "状态": status
                    })
                p_bar.progress((i + 1) / len(files))
        st.success("批量数据解析完毕！识别失败的图片已在压缩包内以 'Fail_' 命名并画出遗漏点。")
        st.download_button("📥 导出标注图片包 (Zip)", out_zip.getvalue(), "Measurement_Results_V15.zip")

st.divider()
st.subheader("📜 本次项目数据报表")
if st.session_state.history:
    df_h = pd.DataFrame(st.session_state.history)
    st.dataframe(df_h, use_container_width=True)
    st.download_button("📊 导出报表 (CSV)", df_h.to_csv(index=False).encode('utf-8-sig'), "data_history_v15.csv", "text/csv")
    if st.button("🗑️ 清空所有表格数据"):
        st.session_state.history = []
        st.rerun()