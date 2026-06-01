import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
import pandas as pd
from datetime import datetime

# --- 核心算法层：V32 虚化容错 + 红色特异性增强 + 黑框拓扑死锁器 ---
def process_image_v32(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: 
        return None, 0, "文件读取失败"
    
    h, w = img.shape[:2]
    
    # 动态画布渲染参数设定
    dyn_radius = max(5, int(w / 150))      
    dyn_line = max(2, int(w / 500))        
    dyn_font_scale = w / 1200              
    dyn_font_thick = max(1, int(w / 800))  
    font = cv2.FONT_HERSHEY_DUPLEX

    # ==========================================
    # 1. 前置红色特异性增强（对抗虚化：强行拉高正红像素的饱和度）
    # ==========================================
    hsv_enhance = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H, S, V = cv2.split(hsv_enhance)

    # 锁定正红色的色调区间（严格排除皮肤的橙黄色 15-30）
    red_mask1 = (H >= 0) & (H <= 8)
    red_mask2 = (H >= 168) & (H <= 180)
    true_red_pixel = red_mask1 | red_mask2

    # 对属于正红的像素进行非线性色彩暴击
    S = np.where(true_red_pixel, np.clip(S.astype(np.int16) * 2.2, 0, 255).astype(np.uint8), S)
    V = np.where(true_red_pixel, np.clip(V.astype(np.int16) * 1.5, 0, 255).astype(np.uint8), V)

    # 重新组合成增强后的图像
    hsv = cv2.merge([H, S, V])
    img_enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    gray = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2GRAY)

    # ==========================================
    # 2. 拓扑基座：提取黑框的“势力范围”（针对对焦错误、虚化进行深度容错）
    # ==========================================
    # 将阈值从 65 提升至 85，允许对焦模糊、发灰的镜框边缘被识别出来
    _, black_frame_mask = cv2.threshold(gray, 85, 255, cv2.THRESH_BINARY_INV)
    
    # 将膨胀核从 15 扩大到 25，形成更大的黑框包裹圈，把向外虚化晕染的红点光晕强行收纳进来
    kernel_expand = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    black_influence_zone = cv2.dilate(black_frame_mask, kernel_expand, iterations=1)

    # 采用高纯度红阈值，配合色彩暴击，直接在第一步过滤掉皮肤
    lower_red1 = np.array([0, 110, 70])      
    upper_red1 = np.array([8, 255, 255])
    lower_red2 = np.array([168, 110, 70])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    all_red_mask = cv2.bitwise_or(mask1, mask2)

    # 拓扑锁死：只有在黑框势力范围内的红点像素才有效
    valid_red_mask = cv2.bitwise_and(all_red_mask, black_influence_zone)

    # ==========================================
    # 3. 形态学修复与级联候选点提取（对抗虚化导致的散落、变形）
    # ==========================================
    cascade_thresholds = [
        # 第一档：针对虚化严重、形状极度不规则、被切碎的红点极致容错层
        {"close_k": 7, "dilate_k": 3, "circ": 0.20, "min_area": 3},  
        # 第二档：标准层
        {"close_k": 5, "dilate_k": 0, "circ": 0.40, "min_area": 8}   
    ]
    
    best_set = None
    min_geometric_error = float('inf')

    for pass_idx, th in enumerate(cascade_thresholds):
        # 闭运算自动填补虚化产生的内部空隙，把毛糙的红点黏合回一个整体
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (th["close_k"], th["close_k"]))
        red_cleaned = cv2.morphologyEx(valid_red_mask, cv2.MORPH_CLOSE, kernel_close)
        
        if th["dilate_k"] > 0:
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (th["dilate_k"], th["dilate_k"]))
            red_cleaned = cv2.dilate(red_cleaned, kernel_dilate, iterations=1)
            
        contours_red, _ = cv2.findContours(red_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        
        for cnt in contours_red:
            area = cv2.contourArea(cnt)
            if th["min_area"] < area < 2000: # 稍微放宽上限，包容因虚化而变大的红晕
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                # 允许虚化红点较低的圆形度（th["circ"] 最低放宽至 0.20）
                if circularity >= th["circ"]:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        if 0.02 * w < cX < 0.98 * w and 0.02 * h < cY < 0.98 * h:
                            candidates.append((cX, cY))
                            
        num_pts = len(candidates)
        
        # ==========================================
        # 4. 双翼刚性对称拓扑解算
        # ==========================================
        if num_pts >= 3:
            for i in range(len(candidates)-2):
                for j in range(i+1, len(candidates)-1):
                    for k in range(j+1, len(candidates)):
                        pA, pB, pC = np.array(candidates[i]), np.array(candidates[j]), np.array(candidates[k])
                        
                        dAB = np.linalg.norm(pA - pB)
                        dBC = np.linalg.norm(pB - pC)
                        dCA = np.linalg.norm(pC - pA)
                        dists = [dAB, dBC, dCA]
                        pts_temp = [candidates[i], candidates[j], candidates[k]]
                        
                        max_idx = np.argmax(dists)
                        max_dist = dists[max_idx]
                        
                        # 跨度约束
                        if max_dist < min(w, h) * 0.25 or max_dist > min(w, h) * 0.75: 
                            continue 
                            
                        if max_idx == 0:   
                            p_mid, p1, p2 = pts_temp[2], pts_temp[0], pts_temp[1]
                        elif max_idx == 1: 
                            p_mid, p1, p2 = pts_temp[0], pts_temp[1], pts_temp[2]
                        else:              
                            p_mid, p1, p2 = pts_temp[1], pts_temp[0], pts_temp[2]
                        
                        v1 = np.array([p1[0] - p_mid[0], p1[1] - p_mid[1]])
                        v2 = np.array([p2[0] - p_mid[0], p2[1] - p_mid[1]])
                        
                        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        temp_angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                        
                        # 面弯角物理区间约束
                        if temp_angle < 95 or temp_angle > 179.5:
                            continue
                            
                        # 刚体物理对称指标
                        len1 = np.linalg.norm(v1)
                        len2 = np.linalg.norm(v2)
                        balance_err = abs(len1 - len2) / max(len1, len2, 1)
                        
                        if balance_err > 0.20:
                            continue 
                        
                        if balance_err < min_geometric_error:
                            min_geometric_error = balance_err
                            best_set = (p1, p_mid, p2)
            
            if best_set is not None:
                break 

    # ==========================================
    # 5. 精密解算与渲染（始终在干净的原图上标注）
    # ==========================================
    if best_set is None:
        return img, 0, f"识别失败：已剥离皮肤干扰并进行了虚化重构。有效红点数：{len(candidates)}。请检查是否有红点被完全盖死。"
        
    p1, p_mid, p2 = best_set 
    
    v1, v2 = np.array([p1[0]-p_mid[0], p1[1]-p_mid[1]]), np.array([p2[0]-p_mid[0], p2[1]-p_mid[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    # 渲染骨架
    cv2.line(img, p1, p_mid, (255, 120, 0), dyn_line, cv2.LINE_AA)
    cv2.line(img, p_mid, p2, (255, 120, 0), dyn_line, cv2.LINE_AA)
    for p in [p1, p_mid, p2]:
        cv2.circle(img, p, dyn_radius, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p, dyn_radius, (0, 0, 0), 1, cv2.LINE_AA)

    text = f"ANGLE: {angle:.2f} DEG"
    text_pos = (p_mid[0] + 40, p_mid[1])
    cv2.putText(img, text, text_pos, font, dyn_font_scale, (0,0,0), dyn_font_thick+2, cv2.LINE_AA)
    cv2.putText(img, text, text_pos, font, dyn_font_scale, (255,255,255), dyn_font_thick, cv2.LINE_AA)
    
    return img, angle, "成功"

# --- Streamlit UI 交互层 ---
st.set_page_config(page_title="WrapAngle V32", layout="wide")
st.title("👓 面弯角精密测量系统 (V32 虚化容错+黑框拓扑双锁版)")

if 'history' not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["📸 单图实时测定", "📦 压缩包批量解析"])

with tab1:
    single_file = st.file_uploader("上传单张俯视图", type=['jpg', 'jpeg', 'png'], key="single")
    if single_file:
        res_img, ang, status = process_image_v32(single_file.read())
        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption="检测结果图（已应用虚化重建与色彩暴击算法）")
        with col_info:
            st.subheader("📊 诊断结果")
            if status == "成功":
                st.success(f"面弯角解算成功: {ang:.2f}°")
                if not any(d['文件名'] == single_file.name for d in st.session_state.history):
                    st.session_state.history.append({
                        "测定时间": datetime.now().strftime("%H:%M:%S"),
                        "文件名": single_file.name,
                        "面弯角": f"{ang:.2f}°",
                        "状态": "成功"
                    })
            else:
                st.error(status)
                st.warning("安全机制提示：非黑色镜框包裹的皮肤、衣服杂色已被切断。若依然报错，请检查中点是否被完全盖死。")

with tab2:
    zip_file = st.file_uploader("上传 Zip 图片压缩包", type="zip", key="zip")
    if zip_file:
        out_zip = io.BytesIO()
        with zipfile.ZipFile(zip_file, "r") as z_in, zipfile.ZipFile(out_zip, "w") as z_out:
            files = [f for f in z_in.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            p_bar = st.progress(0)
            for i, f_name in enumerate(files):
                res_img, ang, status = process_image_v32(z_in.read(f_name))
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
        st.download_button("📥 导出标注图片包 (Zip)", out_zip.getvalue(), "Measurement_Results_V32.zip")

st.divider()
st.subheader("📜 本次项目数据报表")
if st.session_state.history:
    df_h = pd.DataFrame(st.session_state.history)
    st.dataframe(df_h, use_container_width=True)
    st.download_button("📊 导出报表 (CSV)", df_h.to_csv(index=False).encode('utf-8-sig'), "data_history_v32.csv", "text/csv")
    if st.button("🗑️ 清空所有表格数据"):
        st.session_state.history = []
        st.rerun()
