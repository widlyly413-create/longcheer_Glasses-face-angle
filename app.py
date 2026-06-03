import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
import pandas as pd
from datetime import datetime

# --- V27 级联多层检测算法 ---
def process_image_v27(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: 
        return None, 0, "文件读取失败", "V27"
    
    h, w = img.shape[:2]
    
    dyn_radius = max(5, int(w / 150))      
    dyn_line = max(2, int(w / 500))        
    dyn_font_scale = w / 1200              
    dyn_font_thick = max(1, int(w / 800))  
    font = cv2.FONT_HERSHEY_DUPLEX

    b, g, r = cv2.split(img)
    r_16 = r.astype(np.int16)
    g_16 = g.astype(np.int16)
    b_16 = b.astype(np.int16)

    rg_diff = r_16 - g_16
    rb_diff = r_16 - b_16

    cascade_thresholds = [
        {"rg": 75, "rb": 45, "r": 120, "circ": 0.55, "area_min": 12},
        {"rg": 55, "rb": 40, "r": 100, "circ": 0.45, "area_min": 10},
        {"rg": 40, "rb": 30, "r": 80, "circ": 0.35, "area_min": 8}
    ]
    
    best_set = None
    min_geometric_error = float('inf')

    for pass_idx, th in enumerate(cascade_thresholds):
        mask = (rg_diff > th["rg"]) & (rb_diff > th["rb"]) & (r_16 > th["r"])
        mask = mask.astype(np.uint8) * 255
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        contours_red, _ = cv2.findContours(red_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        
        for cnt in contours_red:
            area = cv2.contourArea(cnt)
            if th["area_min"] < area < 2000:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                if circularity >= th["circ"]:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        if 0.02 * w < cX < 0.98 * w and 0.02 * h < cY < 0.98 * h:
                            candidates.append((cX, cY))
                            
        candidates = candidates[:15]
                            
        num_pts = len(candidates)
        
        if num_pts >= 3:
            for i in range(len(candidates)-2):
                for j in range(i+1, len(candidates)-1):
                    for k in range(j+1, len(candidates)):
                        if min_geometric_error < 0.05:
                            break
                            
                        pA, pB, pC = np.array(candidates[i]), np.array(candidates[j]), np.array(candidates[k])
                        
                        dAB = np.linalg.norm(pA - pB)
                        dBC = np.linalg.norm(pB - pC)
                        dCA = np.linalg.norm(pC - pA)
                        dists = [dAB, dBC, dCA]
                        pts_temp = [candidates[i], candidates[j], candidates[k]]
                        
                        max_idx = np.argmax(dists)
                        max_dist = dists[max_idx]
                        
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
                        
                        if temp_angle < 95 or temp_angle > 179.5:
                            continue
                            
                        len1 = np.linalg.norm(v1)
                        len2 = np.linalg.norm(v2)
                        balance_err = abs(len1 - len2) / max(len1, len2, 1)
                        
                        if balance_err > 0.15: 
                            continue 
                        
                        if balance_err < min_geometric_error:
                            min_geometric_error = balance_err
                            best_set = (p1, p_mid, p2)
                    if min_geometric_error < 0.05:
                        break
                if min_geometric_error < 0.05:
                    break
            
            if best_set is not None:
                break

    if best_set is None:
        return img, 0, "V27识别失败：多级级联区内未匹配到合规眼镜刚体", "V27"
        
    p1, p_mid, p2 = best_set
    
    v1, v2 = np.array([p1[0]-p_mid[0], p1[1]-p_mid[1]]), np.array([p2[0]-p_mid[0], p2[1]-p_mid[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    cv2.line(img, p1, p_mid, (255, 120, 0), dyn_line, cv2.LINE_AA)
    cv2.line(img, p_mid, p2, (255, 120, 0), dyn_line, cv2.LINE_AA)
    for p in [p1, p_mid, p2]:
        cv2.circle(img, p, dyn_radius, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p, dyn_radius, (0, 0, 0), 1, cv2.LINE_AA)

    text = f"ANGLE: {angle:.2f} DEG (V27)"
    text_pos = (p_mid[0] + 40, p_mid[1])
    cv2.putText(img, text, text_pos, font, dyn_font_scale, (0,0,0), dyn_font_thick+2, cv2.LINE_AA)
    cv2.putText(img, text, text_pos, font, dyn_font_scale, (255,255,255), dyn_font_thick, cv2.LINE_AA)
    
    return img, angle, "成功", "V27"

# --- V28 增强容错算法（处理轻微虚化和遮挡）---
def process_image_v28(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: 
        return None, 0, "文件读取失败", "V28"
    
    h, w = img.shape[:2]
    
    dyn_radius = max(5, int(w / 150))      
    dyn_line = max(2, int(w / 500))        
    dyn_font_scale = w / 1200              
    dyn_font_thick = max(1, int(w / 800))  
    font = cv2.FONT_HERSHEY_DUPLEX

    b, g, r = cv2.split(img)
    r_16 = r.astype(np.int16)
    g_16 = g.astype(np.int16)
    b_16 = b.astype(np.int16)

    rg_diff = r_16 - g_16
    rb_diff = r_16 - b_16

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cascade_thresholds = [
        {"rg": 75, "rb": 45, "r": 120, "circ": 0.55, "area_min": 12},
        {"rg": 55, "rb": 40, "r": 100, "circ": 0.40, "area_min": 8},
        {"rg": 40, "rb": 30, "r": 80, "circ": 0.25, "area_min": 5},
        {"mode": "hsv", "h_low": 0, "h_high": 10, "s_low": 30, "v_low": 60, "circ": 0.20, "area_min": 3},
        {"mode": "hsv", "h_low": 165, "h_high": 180, "s_low": 30, "v_low": 60, "circ": 0.20, "area_min": 3}
    ]
    
    best_set = None
    min_geometric_error = float('inf')

    for pass_idx, th in enumerate(cascade_thresholds):
        if th.get("mode") == "hsv":
            lower_red = np.array([th["h_low"], th["s_low"], th["v_low"]])
            upper_red = np.array([th["h_high"], 255, 255])
            mask = cv2.inRange(hsv, lower_red, upper_red)
        else:
            mask = (rg_diff > th["rg"]) & (rb_diff > th["rb"]) & (r_16 > th["r"])
            mask = mask.astype(np.uint8) * 255
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        red_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        red_cleaned = cv2.dilate(red_cleaned, kernel_dilate, iterations=1)
        
        contours_red, _ = cv2.findContours(red_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        
        for cnt in contours_red:
            area = cv2.contourArea(cnt)
            if th["area_min"] < area < 3000:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                if circularity >= th["circ"]:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        if 0.02 * w < cX < 0.98 * w and 0.02 * h < cY < 0.98 * h:
                            candidates.append((cX, cY, area))
                            
        candidates.sort(key=lambda x: x[2], reverse=True)
        candidates = [c[:2] for c in candidates[:20]]
                            
        num_pts = len(candidates)
        
        if num_pts >= 3:
            for i in range(len(candidates)-2):
                for j in range(i+1, len(candidates)-1):
                    for k in range(j+1, len(candidates)):
                        if min_geometric_error < 0.03:
                            break
                            
                        pA, pB, pC = np.array(candidates[i]), np.array(candidates[j]), np.array(candidates[k])
                        
                        dAB = np.linalg.norm(pA - pB)
                        dBC = np.linalg.norm(pB - pC)
                        dCA = np.linalg.norm(pC - pA)
                        dists = [dAB, dBC, dCA]
                        pts_temp = [candidates[i], candidates[j], candidates[k]]
                        
                        max_idx = np.argmax(dists)
                        max_dist = dists[max_idx]
                        
                        if max_dist < min(w, h) * 0.20 or max_dist > min(w, h) * 0.80: 
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
                        
                        if temp_angle < 85 or temp_angle > 179.5:
                            continue
                            
                        len1 = np.linalg.norm(v1)
                        len2 = np.linalg.norm(v2)
                        balance_err = abs(len1 - len2) / max(len1, len2, 1)
                        
                        if balance_err < min_geometric_error:
                            min_geometric_error = balance_err
                            best_set = (p1, p_mid, p2)
                    if min_geometric_error < 0.03:
                        break
                if min_geometric_error < 0.03:
                    break
            
            if best_set is not None:
                break

    if best_set is None:
        return img, 0, "V28识别失败：增强容错级联区内未匹配到合规眼镜刚体（轻微虚化或遮挡可能影响了检测）", "V28"
        
    p1, p_mid, p2 = best_set
    
    v1, v2 = np.array([p1[0]-p_mid[0], p1[1]-p_mid[1]]), np.array([p2[0]-p_mid[0], p2[1]-p_mid[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    cv2.line(img, p1, p_mid, (0, 180, 255), dyn_line, cv2.LINE_AA)
    cv2.line(img, p_mid, p2, (0, 180, 255), dyn_line, cv2.LINE_AA)
    for p in [p1, p_mid, p2]:
        cv2.circle(img, p, dyn_radius, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p, dyn_radius, (0, 0, 0), 1, cv2.LINE_AA)

    text = f"ANGLE: {angle:.2f} DEG (V28)"
    text_pos = (p_mid[0] + 40, p_mid[1])
    cv2.putText(img, text, text_pos, font, dyn_font_scale, (0,0,0), dyn_font_thick+2, cv2.LINE_AA)
    cv2.putText(img, text, text_pos, font, dyn_font_scale, (255,255,255), dyn_font_thick, cv2.LINE_AA)
    
    return img, angle, "成功", "V28"

# --- 级联识别主函数：V27优先，失败则V28 ---
def process_image_cascade(image_bytes):
    # 先用 V27 进行识别
    res_img_v27, angle_v27, status_v27, _ = process_image_v27(image_bytes)
    
    if status_v27 == "成功":
        return res_img_v27, angle_v27, status_v27, "V27"
    
    # V27 失败，使用 V28 进行增强识别
    res_img_v28, angle_v28, status_v28, _ = process_image_v28(image_bytes)
    
    if status_v28 == "成功":
        return res_img_v28, angle_v28, status_v28, "V28"
    
    # 两个都失败，返回 V27 的结果和失败原因
    return res_img_v27, 0, f"识别失败：V27和V28算法均未能匹配到合规眼镜刚体。可能原因：\n- 红点被严重遮挡或完全覆盖\n- 光照条件不佳导致红点特征不明显\n- 图像质量过低（严重虚化、模糊）\n- 未检测到足够的红色候选点", "失败"

# --- Streamlit UI 交互层 ---
st.set_page_config(page_title="WrapAngle V27+V28", layout="wide")
st.title("👓 面弯角精密测量系统 (V27+V28 级联识别版)")

if 'history' not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["📸 单图实时测定", "📦 压缩包批量解析"])

with tab1:
    single_file = st.file_uploader("上传单张俯视图", type=['jpg', 'jpeg', 'png'], key="single")
    if single_file:
        res_img, ang, status, algo_version = process_image_cascade(single_file.read())
        col_img, col_info = st.columns([2, 1])
        with col_img:
            caption = f"检测结果图"
            if algo_version == "V27":
                caption += "（V27算法成功识别，黄色点为检测到的红点）"
            elif algo_version == "V28":
                caption += "（V27失败后V28算法成功识别，紫色点为检测到的红点）"
            st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption=caption)
        with col_info:
            st.subheader("📊 诊断结果")
            if status == "成功":
                st.success(f"面弯角解算成功: {ang:.2f}° (算法版本: {algo_version})")
                if not any(d['文件名'] == single_file.name for d in st.session_state.history):
                    st.session_state.history.append({
                        "测定时间": datetime.now().strftime("%H:%M:%S"),
                        "文件名": single_file.name,
                        "面弯角": f"{ang:.2f}°",
                        "算法版本": algo_version,
                        "状态": "成功"
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
                res_img, ang, status, algo_version = process_image_cascade(z_in.read(f_name))
                if res_img is not None:
                    _, buf = cv2.imencode(".jpg", res_img)
                    prefix = "Result_" if status == "成功" else "Fail_"
                    z_out.writestr(f"{prefix}{os.path.basename(f_name)}", buf.tobytes())
                    st.session_state.history.append({
                        "操作时间": datetime.now().strftime("%H:%M:%S"),
                        "文件名": os.path.basename(f_name),
                        "面弯角": f"{ang:.2f}°" if ang > 0 else "-",
                        "算法版本": algo_version,
                        "状态": status
                    })
                p_bar.progress((i + 1) / len(files))
        st.success("批量数据解析完毕！")
        st.download_button("📥 导出标注图片包 (Zip)", out_zip.getvalue(), "Measurement_Results_V27_V28.zip")

st.divider()
st.subheader("📜 本次项目数据报表")
if st.session_state.history:
    df_h = pd.DataFrame(st.session_state.history)
    st.dataframe(df_h, use_container_width=True)
    st.download_button("📊 导出报表 (CSV)", df_h.to_csv(index=False).encode('utf-8-sig'), "data_history_v27_v28.csv", "text/csv")
    if st.button("🗑️ 清空所有表格数据"):
        st.session_state.history = []
        st.rerun()