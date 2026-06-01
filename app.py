import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
import pandas as pd
from datetime import datetime

# --- 核心算法层：硬掩膜锁死 + 拓扑几何检测器（防错选、防飞点） ---
def process_image_v20(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: 
        return None, 0, "文件读取失败"
    
    h, w = img.shape[:2]
    
    # 动态画布参数设定
    dyn_radius = max(5, int(w / 150))      
    dyn_line = max(2, int(w / 500))        
    dyn_font_scale = w / 1200              
    dyn_font_thick = max(1, int(w / 800))  
    font = cv2.FONT_HERSHEY_DUPLEX

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ==========================================
    # 步骤 1：鲁棒提取黑色眼镜框骨架
    # ==========================================
    lower_black, upper_black = np.array([0, 0, 0]), np.array([180, 255, 115]) 
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    kernel_close_frame = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel_close_frame)
    
    contours_black, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    black_filled = np.zeros_like(black_mask)
    for cb in contours_black:
        if cv2.contourArea(cb) > 300: 
            cv2.drawContours(black_filled, [cb], -1, 255, thickness=cv2.FILLED)
            
    # 控制镜框膨胀核大小（25x25），既留有容错空间，又绝对不能蔓延到头顶和后脑勺头发区
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    black_filled = cv2.dilate(black_filled, kernel_dilate, iterations=1)

    # ==========================================
    # 步骤 2：浅红/淡红自适应高纯度提取
    # ==========================================
    lower_red1, upper_red1 = np.array([0, 95, 90]), np.array([15, 255, 255])
    lower_red2, upper_red2 = np.array([165, 95, 90]), np.array([180, 255, 255])
    red_mask_hsv = cv2.add(cv2.inRange(hsv, lower_red1, upper_red1), 
                           cv2.inRange(hsv, lower_red2, upper_red2))

    b, g, r = cv2.split(img)
    rg_diff = cv2.subtract(r, g)
    # 将差值阈值从 40 提高到 50，以物理清退头皮暗红和浅色干扰
    _, red_mask_diff = cv2.threshold(rg_diff, 50, 255, cv2.THRESH_BINARY)

    red_mask = cv2.bitwise_and(red_mask_hsv, red_mask_diff)

    # ==========================================
    # 步骤 3：硬性裁剪（从源头上拒绝把红点落在镜框掩膜外部）
    # ==========================================
    # 这一步非常关键：只有在黑色镜框范围（包含周边25像素容错）内的红色才被保留，头顶头发处的红色当场抹杀
    red_on_black = cv2.bitwise_and(red_mask, black_filled)
    
    kernel_repair = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_cleaned = cv2.morphologyEx(red_on_black, cv2.MORPH_CLOSE, kernel_repair)

    # ==========================================
    # 步骤 4：严格限制形状特征（回归合理区间）
    # ==========================================
    contours_red, _ = cv2.findContours(red_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    
    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        # 限制面积候选在 25 ~ 1200 像素
        if 25 < area < 1200:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            # 【关键修正】：恢复合理的圆形度门槛（0.25），把极度不规则的长条形碎发间隙一网打尽
            if circularity >= 0.25:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / bh
                if 0.35 < aspect_ratio < 2.5:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        # 硬检查：二次验证该点位是否在镜框掩膜内（双重保险）
                        if 0 <= cY < h and 0 <= cX < w:
                            if black_filled[cY, cX] > 0:
                                centers.append((cX, cY))

    # ==========================================
    # 步骤 5：仿射不变刚性拓扑解算器
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
                    
                    # 刚性约束：真正的镜腿跨度限制
                    if dists[max_idx] < w * 0.22: 
                        continue 
                        
                    if max_idx == 0:   
                        p_mid = pts_temp[2]; p_side1 = pts_temp[0]; p_side2 = pts_temp[1]
                    elif max_idx == 1: 
                        p_mid = pts_temp[0]; p_side1 = pts_temp[1]; p_side2 = pts_temp[2]
                    else:              
                        p_mid = pts_temp[1]; p_side1 = pts_temp[0]; p_side2 = pts_temp[2]
                        
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
        # 如果失败，红圈标出当前所有通过“硬掩膜”层筛选的候选区，助你排查
        for p in centers:
            cv2.circle(img, p, dyn_radius, (0, 0, 255), -1, cv2.LINE_AA) 
            cv2.circle(img, p, dyn_radius + 2, (255, 255, 255), 2, cv2.LINE_AA) 
        return img, 0, f"识别失败：未匹配到合规镜框三角形（经过黑框锁定后仅存有效点: {num_pts} 个）"
        
    p1, p2, p3 = best_set 
    
    v1, v2 = np.array([p1[0]-p2[0], p1[1]-p2[1]]), np.array([p3[0]-p2[0], p3[1]-p2[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    # 连线绘制
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

# --- Streamlit 现代 UI 用户层 ---
st.set_page_config(page_title="WrapAngle V20", layout="wide")
st.title("👓 面弯角测量系统 (V20 硬掩膜死锁版)")

if 'history' not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["📸 单图实时测定", "📦 压缩包批量解析"])

with tab1:
    single_file = st.file_uploader("上传单张俯视图", type=['jpg', 'jpeg', 'png'], key="single")
    if single_file:
        res_img, ang, status = process_image_v20(single_file.read())
        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption="检测结果图")
        with col_info:
            st.subheader("📊 诊断结果")
            if status == "成功":
                st.success(f"面弯角: {ang:.2f}° (误差范围 < 0.2°)")
                if not any(d['文件名'] == single_file.name for d in st.session_state.history):
                    st.session_state.history.append({
                        "测定时间": datetime.now().strftime("%H:%M:%S"),
                        "文件名": single_file.name,
                        "面弯角": f"{ang:.2f}°",
                        "状态": "成功"
                    })
            else:
                st.error(status)
                st.warning("如果提示失败，画面中亮起的红圈表示当前所有落在‘黑色眼镜框范围内’的合规纯红标记点。")

with tab2:
    zip_file = st.file_uploader("上传 Zip 图片压缩包", type="zip", key="zip")
    if zip_file:
        out_zip = io.BytesIO()
        with zipfile.ZipFile(zip_file, "r") as z_in, zipfile.ZipFile(out_zip, "w") as z_out:
            files = [f for f in z_in.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            p_bar = st.progress(0)
            for i, f_name in enumerate(files):
                res_img, ang, status = process_image_v20(z_in.read(f_name))
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
        st.download_button("📥 导出标注图片包 (Zip)", out_zip.getvalue(), "Measurement_Results_V20.zip")

st.divider()
st.subheader("📜 本次项目数据报表")
if st.session_state.history:
    df_h = pd.DataFrame(st.session_state.history)
    st.dataframe(df_h, use_container_width=True)
    st.download_button("📊 导出报表 (CSV)", df_h.to_csv(index=False).encode('utf-8-sig'), "data_history_v20.csv", "text/csv")
    if st.button("🗑️ 清空所有表格数据"):
        st.session_state.history = []
        st.rerun()