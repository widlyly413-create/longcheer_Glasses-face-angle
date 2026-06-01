import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
import pandas as pd
from datetime import datetime

# --- 核心算法层：V22 宏观物理硬约束与刚性拓扑锁死 ---
def process_image_v22(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: 
        return None, 0, "文件读取失败"
    
    h, w = img.shape[:2]
    
    # 动态参数设定
    dyn_radius = max(5, int(w / 150))      
    dyn_line = max(2, int(w / 500))        
    dyn_font_scale = w / 1200              
    dyn_font_thick = max(1, int(w / 800))  
    font = cv2.FONT_HERSHEY_DUPLEX

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ==========================================
    # 步骤 1：精确提取黑框与金属骨架（控制膨胀，杜绝蔓延）
    # ==========================================
    lower_frame, upper_frame = np.array([0, 0, 0]), np.array([180, 255, 145]) 
    frame_mask = cv2.inRange(hsv, lower_frame, upper_frame)
    
    kernel_close_frame = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    frame_mask = cv2.morphologyEx(frame_mask, cv2.MORPH_CLOSE, kernel_close_frame)
    
    contours_frame, _ = cv2.findContours(frame_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    frame_filled = np.zeros_like(frame_mask)
    for cb in contours_frame:
        if cv2.contourArea(cb) > 400: 
            cv2.drawContours(frame_filled, [cb], -1, 255, thickness=cv2.FILLED)
            
    # 【限制1】：适度膨胀（21x21），恰好包裹边缘标定点，绝对不覆盖进头发区
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    frame_filled = cv2.dilate(frame_filled, kernel_dilate, iterations=1)

    # ==========================================
    # 步骤 2：高纯度红点提取
    # ==========================================
    lower_red1, upper_red1 = np.array([0, 95, 90]), np.array([15, 255, 255])
    lower_red2, upper_red2 = np.array([165, 95, 90]), np.array([180, 255, 255])
    red_mask_hsv = cv2.add(cv2.inRange(hsv, lower_red1, upper_red1), 
                           cv2.inRange(hsv, lower_red2, upper_red2))

    b, g, r = cv2.split(img)
    rg_diff = cv2.subtract(r, g)
    _, red_mask_diff = cv2.threshold(rg_diff, 45, 255, cv2.THRESH_BINARY)
    red_mask = cv2.bitwise_and(red_mask_hsv, red_mask_diff)

    # ==========================================
    # 步骤 3 & 4：形状与掩膜硬核过滤（绞杀碎发噪点）
    # ==========================================
    red_on_frame = cv2.bitwise_and(red_mask, frame_filled)
    kernel_repair = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_cleaned = cv2.morphologyEx(red_on_frame, cv2.MORPH_CLOSE, kernel_repair)

    contours_red, _ = cv2.findContours(red_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    
    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        if 25 < area < 1200:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            # 【限制2】：严格的圆形度判定，头皮缝隙的形状极不规则，卡死在 0.25 以上
            if circularity >= 0.25:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / bh
                if 0.4 < aspect_ratio < 2.2:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        # 确保中心点必须落在安全区内
                        if 0 <= cY < h and 0 <= cX < w:
                            if frame_filled[cY, cX] > 0:
                                centers.append((cX, cY))

    # ==========================================
    # 步骤 5：宏观物理刚性拓扑鉴别器
    # ==========================================
    num_pts = len(centers)
    best_set = None

    if num_pts >= 3:
        min_geometric_error = float('inf')
        
        for i in range(len(centers)-2):
            for j in range(i+1, len(centers)-1):
                for k in range(j+1, len(centers)):
                    pA, pB, pC = np.array(centers[i]), np.array(centers[j]), np.array(centers[k])
                    
                    dAB = np.linalg.norm(pA - pB)
                    dBC = np.linalg.norm(pB - pC)
                    dCA = np.linalg.norm(pC - pA)
                    
                    dists = [dAB, dBC, dCA]
                    pts_temp = [centers[i], centers[j], centers[k]]
                    
                    max_idx = np.argmax(dists)
                    
                    # 【限制3：大跨度硬约束】
                    # 真正的两个镜腿（最长边）跨度极大，通常占画面尺寸的 35% 以上
                    # 扎堆在头顶上的噪点组合绝对无法满足这个大跨度，直接淘汰
                    min_span = min(w, h) * 0.35
                    if dists[max_idx] < min_span: 
                        continue 
                        
                    # 确定顶点（鼻梁点）
                    if max_idx == 0:   
                        p_mid = pts_temp[2]; p_side1 = pts_temp[0]; p_side2 = pts_temp[1]
                    elif max_idx == 1: 
                        p_mid = pts_temp[0]; p_side1 = pts_temp[1]; p_side2 = pts_temp[2]
                    else:              
                        p_mid = pts_temp[1]; p_side1 = pts_temp[0]; p_side2 = pts_temp[2]
                    
                    # 【限制4：钝角面弯角判定】
                    # 临时计算这三个点的夹角
                    v1 = np.array([p_side1[0]-p_mid[0], p_side1[1]-p_mid[1]])
                    v2 = np.array([p_side2[0]-p_mid[0], p_side2[1]-p_mid[1]])
                    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    temp_angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                    
                    # 智能眼镜面弯角必定是钝角，如果噪点组合出锐角（<160）或无限接近直线（>179.8），强行淘汰
                    if temp_angle < 160 or temp_angle > 179.8:
                        continue
                        
                    # 经过上述 4 道铁闸，能存活下来的绝对是真实的眼镜框三点
                    # 此时再计算它们的对称误差，选出最优解
                    len1 = np.linalg.norm(np.array(p_side1) - np.array(p_mid))
                    len2 = np.linalg.norm(np.array(p_side2) - np.array(p_mid))
                    balance_err = abs(len1 - len2) / max(len1, len2, 1)
                    
                    if balance_err < min_geometric_error:
                        min_geometric_error = balance_err
                        best_set = (p_side1, p_mid, p_side2)

    # ==========================================
    # 步骤 6：渲染输出层
    # ==========================================
    if best_set is None:
        for p in centers:
            cv2.circle(img, p, dyn_radius, (0, 0, 255), -1, cv2.LINE_AA) 
            cv2.circle(img, p, dyn_radius + 2, (255, 255, 255), 2, cv2.LINE_AA) 
        return img, 0, f"识别失败：未匹配到合规镜框三点（请检查是否有红点被完全遮挡。当前捕获噪点: {num_pts} 个）"
        
    p1, p2, p3 = best_set 
    
    # 精密计算最终夹角
    v1, v2 = np.array([p1[0]-p2[0], p1[1]-p2[1]]), np.array([p3[0]-p2[0], p3[1]-p2[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
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

# --- Streamlit UI ---
st.set_page_config(page_title="WrapAngle V22", layout="wide")
st.title("👓 面弯角测量系统 (V22 物理锁死版)")

if 'history' not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["📸 单图实时测定", "📦 压缩包批量解析"])

with tab1:
    single_file = st.file_uploader("上传单张俯视图", type=['jpg', 'jpeg', 'png'], key="single")
    if single_file:
        res_img, ang, status = process_image_v22(single_file.read())
        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption="检测结果图")
        with col_info:
            st.subheader("📊 诊断结果")
            if status == "成功":
                st.success(f"面弯角: {ang:.2f}°")
                if not any(d['文件名'] == single_file.name for d in st.session_state.history):
                    st.session_state.history.append({
                        "测定时间": datetime.now().strftime("%H:%M:%S"),
                        "文件名": single_file.name,
                        "面弯角": f"{ang:.2f}°",
                        "状态": "成功"
                    })
            else:
                st.error(status)
                st.warning("如果提示失败，画面中亮起的红圈表示当前所有落在‘框架范围内’的合规纯红标记点。")

with tab2:
    zip_file = st.file_uploader("上传 Zip 图片压缩包", type="zip", key="zip")
    if zip_file:
        out_zip = io.BytesIO()
        with zipfile.ZipFile(zip_file, "r") as z_in, zipfile.ZipFile(out_zip, "w") as z_out:
            files = [f for f in z_in.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            p_bar = st.progress(0)
            for i, f_name in enumerate(files):
                res_img, ang, status = process_image_v22(z_in.read(f_name))
                if res_img is not None:
                    _, buf = cv2.imencode(".jpg", res_img)
                    prefix = "Result_" if status == "成功" else "Fail_"
                    z_out.writestr(f"{prefix}{os.path.basename(f_name)}", buf.tobytes())
                    st.session_state.history.append({
                        "操作时间": datetime.now().strftime("%H:%M:%S"),
                        "文件名": os.path.basename(f_name),
                        "面弯角": f"{ang:.2f}°" if ang > 0 else "-",
                        "状态": status
                    })
            p_bar.progress((i + 1) / len(files))
        st.success("批量数据解析完毕！")
        st.download_button("📥 导出标注图片包 (Zip)", out_zip.getvalue(), "Measurement_Results_V22.zip")

st.divider()
st.subheader("📜 本次项目数据报表")
if st.session_state.history:
    df_h = pd.DataFrame(st.session_state.history)
    st.dataframe(df_h, use_container_width=True)
    st.download_button("📊 导出报表 (CSV)", df_h.to_csv(index=False).encode('utf-8-sig'), "data_history_v22.csv", "text/csv")
    if st.button("🗑️ 清空所有表格数据"):
        st.session_state.history = []
        st.rerun()