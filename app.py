import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
import pandas as pd
from datetime import datetime
from streamlit_image_coordinates import streamlit_image_coordinates

MULTIPLE_COLORS = [
    (255, 120, 0),  
    (0, 180, 255),  
    (0, 255, 0),    
    (0, 0, 255),    
    (255, 0, 255),  
    (255, 255, 0)   
]

def calculate_angle_from_three_points(p1, p_mid, p2):
    v1 = np.array([p1[0]-p_mid[0], p1[1]-p_mid[1]])
    v2 = np.array([p2[0]-p_mid[0], p2[1]-p_mid[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def pixel_level_reconstruct_mask_v34(img):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    lower_red1 = np.array([0, 12, 30])
    upper_red1 = np.array([22, 255, 255])
    lower_red2 = np.array([150, 12, 30])
    upper_red2 = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_hsv = cv2.bitwise_or(mask1, mask2)
    
    flood_mask = mask_hsv.copy()
    h_f, w_f = flood_mask.shape[:2]
    fill_contour = np.zeros((h_f + 2, w_f + 2), np.uint8)
    cv2.floodFill(flood_mask, fill_contour, (0, 0), 255)
    mask_filled = mask_hsv | cv2.bitwise_not(flood_mask)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask_filled, cv2.MORPH_CLOSE, kernel)

def process_image_v34_core(img):
    if img is None:
        return None, 0, "文件读取失败", []
    
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    pixel_mask = pixel_level_reconstruct_mask_v34(img)
    
    contours, _ = cv2.findContours(pixel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 4 < area < 3500:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                if not (0.005 * w < cX < 0.995 * w and 0.005 * h < cY < 0.995 * h):
                    continue
                
                check_offsets = [-12, -8, 8, 12]
                dark_pixel_count = 0
                for offset in check_offsets:
                    if 0 <= cX + offset < w and gray[cY, cX + offset] < 90:
                        dark_pixel_count += 1
                    if 0 <= cY + offset < h and gray[cY + offset, cX] < 90:
                        dark_pixel_count += 1
                
                if dark_pixel_count < 1:
                    continue
                
                candidates.append((cX, cY))
    
    candidates = list(set(candidates))[:40]
    num_pts = len(candidates)
    
    all_valid_combinations = []
    if num_pts >= 3:
        for i in range(num_pts - 2):
            for j in range(i + 1, num_pts - 1):
                for k in range(j + 1, num_pts):
                    pA, pB, pC = np.array(candidates[i]), np.array(candidates[j]), np.array(candidates[k])
                    dists = [np.linalg.norm(pA - pB), np.linalg.norm(pB - pC), np.linalg.norm(pC - pA)]
                    pts_temp = [candidates[i], candidates[j], candidates[k]]
                    
                    max_idx = np.argmax(dists)
                    max_dist = dists[max_idx]
                    
                    if max_dist < min(w, h) * 0.18 or max_dist > max(w, h) * 0.99:
                        continue
                    
                    if max_idx == 0:
                        p_mid, p1, p2 = pts_temp[2], pts_temp[0], pts_temp[1]
                    elif max_idx == 1:
                        p_mid, p1, p2 = pts_temp[0], pts_temp[1], pts_temp[2]
                    else:
                        p_mid, p1, p2 = pts_temp[1], pts_temp[0], pts_temp[2]
                    
                    v1 = np.array([p1[0]-p_mid[0], p1[1]-p_mid[1]])
                    v2 = np.array([p2[0]-p_mid[0], p2[1]-p_mid[1]])
                    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    temp_angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                    
                    if temp_angle < 165.0 or temp_angle > 179.8:
                        continue
                    
                    len1, len2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    balance_err = abs(len1 - len2) / max(len1, len2, 1)
                    
                    if balance_err > 0.48:
                        continue
                    
                    all_valid_combinations.append({
                        "points": (p1, p_mid, p2),
                        "angle": temp_angle,
                        "error": balance_err,
                        "span": max_dist
                    })
    
    if not all_valid_combinations:
        return img, 0, "算法已成功清除头发与皮肤反光！但图像缺陷极其严重，导致未能提取到足够的真红点外壳来合成眼镜刚体。", []
    
    all_valid_combinations.sort(key=lambda x: (-x["span"], -x["angle"]))
    
    unique_combinations = []
    seen_mids = []
    for comp in all_valid_combinations:
        mid_pt = comp["points"][1]
        if any(np.linalg.norm(np.array(mid_pt) - np.array(m)) < 35 for m in seen_mids):
            continue
        unique_combinations.append(comp)
        seen_mids.append(mid_pt)
        if len(unique_combinations) >= 6:
            break
    
    avg_angle = np.mean([c["angle"] for c in unique_combinations])
    
    return img, avg_angle, "成功", unique_combinations

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
    res_img_v27, angle_v27, status_v27, _ = process_image_v27(image_bytes)
    
    if status_v27 == "成功":
        return res_img_v27, angle_v27, status_v27, "V27"
    
    res_img_v28, angle_v28, status_v28, _ = process_image_v28(image_bytes)
    
    if status_v28 == "成功":
        return res_img_v28, angle_v28, status_v28, "V28"
    
    return res_img_v27, 0, f"识别失败：V27和V28算法均未能匹配到合规眼镜刚体。可能原因：\n- 红点被严重遮挡或完全覆盖\n- 光照条件不佳导致红点特征不明显\n- 图像质量过低（严重虚化、模糊）\n- 未检测到足够的红色候选点", "失败"

# --- Streamlit UI 交互层 ---
st.set_page_config(page_title="WrapAngle V35 Multi-Track", layout="wide")
st.title("👓 面弯角精密测量系统 (V35 智能算法+鼠标手动点击双轨版)")

if 'history' not in st.session_state:
    st.session_state.history = []

if 'out_zip_bytes' not in st.session_state:
    st.session_state.out_zip_bytes = None

if 'manual_pts' not in st.session_state:
    st.session_state.manual_pts = []

tab1, tab2 = st.tabs(["📸 单图智能测定与手动补偿", "📦 压缩包批量解析"])

with tab1:
    single_file = st.file_uploader("上传俯视图进行测定", type=['jpg', 'jpeg', 'png'], key="single")
    
    if single_file:
        file_bytes = single_file.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        orig_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        h, w = orig_img.shape[:2]
        
        mode = st.radio("🛠️ 请选择测定模式：", 
                       ["🤖 智能算法自动解析", "🖱️ 手动交互点选（当自动识别失败时激活）"], 
                       horizontal=True)
        
        col_img, col_info = st.columns([2, 1])
        
        if mode == "🤖 智能算法自动解析":
            st.session_state.manual_pts = []
            res_img, ang, status, groups = process_image_v34_core(orig_img.copy())
            
            with col_img:
                if status == "成功":
                    for idx, comp in enumerate(groups):
                        p1, p_mid, p2 = comp["points"]
                        color = MULTIPLE_COLORS[idx % len(MULTIPLE_COLORS)]
                        cv2.line(res_img, p1, p_mid, color, max(2, int(w/500)), cv2.LINE_AA)
                        cv2.line(res_img, p_mid, p2, color, max(2, int(w/500)), cv2.LINE_AA)
                        for p in [p1, p_mid, p2]:
                            cv2.circle(res_img, p, max(5, int(w/150)), color, -1, cv2.LINE_AA)
                    cv2.putText(res_img, f"AUTO AVG: {ang:.2f} DEG", (30, 60), 
                                cv2.FONT_HERSHEY_DUPLEX, w/1500, (0,0,255), 
                                max(1, int(w/1000))+2, cv2.LINE_AA)
                    st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption="算法自动识别结果")
                else:
                    st.error("算法未能自动匹配到有效眼镜刚体，请切换上方模式为【手动交互点选】。")
                    st.image(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB), caption="原始图像")
            
            with col_info:
                st.subheader("📊 自动化解算状态")
                if status == "成功":
                    st.success("🎉 自动解算成功！")
                    st.metric(label="自动面弯角", value=f"{ang:.2f}°")
                    if st.button("💾 将自动结果记入报表"):
                        if not any(d['文件名'] == single_file.name for d in st.session_state.history):
                            st.session_state.history.append({
                                "测定时间": datetime.now().strftime("%H:%M:%S"),
                                "文件名": single_file.name,
                                "测量角度": f"{ang:.2f}°",
                                "模式": "自动识别",
                                "状态": "成功"
                            })
                            st.rerun()
        
        else:
            with col_info:
                st.subheader("🖱️ 手动点选指引")
                st.markdown("""
                请依次点击图片中眼镜的 **三个红点位置**：
                1. 🟢 **第 1 下**：点击 **左侧标记点**
                2. 🔴 **第 2 下**：点击 **鼻梁中间标记点**
                3. 🔵 **第 3 下**：点击 **右侧标记点**
                """)
                if st.button("🗑️ 清空重选", key="clear_manual"):
                    st.session_state.manual_pts = []
                    st.rerun()
                curr_count = len(st.session_state.manual_pts)
                st.info(f"当前已点击：{curr_count} / 3 个点")
            
            canvas_img = orig_img.copy()
            for idx, pt in enumerate(st.session_state.manual_pts):
                pt_color = (0, 255, 0) if idx == 0 else ((0, 0, 255) if idx == 1 else (255, 0, 0))
                cv2.circle(canvas_img, pt, max(6, int(w/120)), pt_color, -1, cv2.LINE_AA)
                cv2.circle(canvas_img, pt, max(6, int(w/120)), (255,255,255), max(1, int(w/600)), cv2.LINE_AA)
                cv2.putText(canvas_img, str(idx+1), (pt[0]+15, pt[1]-15), 
                            cv2.FONT_HERSHEY_DUPLEX, w/1000, pt_color, 
                            max(1, int(w/800))+2, cv2.LINE_AA)
            
            manual_angle = 0
            if len(st.session_state.manual_pts) == 3:
                p1, p_mid, p2 = st.session_state.manual_pts
                manual_angle = calculate_angle_from_three_points(p1, p_mid, p2)
                cv2.line(canvas_img, p1, p_mid, (0, 165, 255), max(3, int(w/400)), cv2.LINE_AA)
                cv2.line(canvas_img, p_mid, p2, (0, 165, 255), max(3, int(w/400)), cv2.LINE_AA)
                cv2.putText(canvas_img, f"MANUAL: {manual_angle:.2f} DEG", (30, 60), 
                            cv2.FONT_HERSHEY_DUPLEX, w/1500, (0, 165, 255), 
                            max(1, int(w/1000))+2, cv2.LINE_AA)
                
                with col_info:
                    st.success("🎉 刚体骨架手动测定完毕！")
                    st.metric(label="手动解算角度", value=f"{manual_angle:.2f}°")
                    if st.button("💾 将手动结果记入报表", key="save_manual"):
                        st.session_state.history.append({
                            "测定时间": datetime.now().strftime("%H:%M:%S"),
                            "文件名": single_file.name,
                            "测量角度": f"{manual_angle:.2f}°",
                            "模式": "手动交互",
                            "状态": "成功"
                        })
                        st.session_state.manual_pts = []
                        st.rerun()
            
            with col_img:
                value = streamlit_image_coordinates(cv2.cvtColor(canvas_img, cv2.COLOR_BGR2RGB), key="interactive_canvas")
                if value is not None and len(st.session_state.manual_pts) < 3:
                    new_pt = (value["x"], value["y"])
                    if not st.session_state.manual_pts or np.linalg.norm(np.array(st.session_state.manual_pts[-1]) - np.array(new_pt)) > 5:
                        st.session_state.manual_pts.append(new_pt)
                        st.rerun()

with tab2:
    zip_file = st.file_uploader("上传 Zip 图片压缩包", type="zip", key="zip")
    
    if zip_file:
        zip_file.seek(0)
        
        if st.button("🚀 启动自动化批量解析", key="start_batch"):
            out_zip = io.BytesIO()
            try:
                with zipfile.ZipFile(zip_file, "r") as z_in, zipfile.ZipFile(out_zip, "w") as z_out:
                    files = [f for f in z_in.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png')) and not f.startswith('__MACOSX')]
                    
                    if len(files) == 0:
                        st.warning("压缩包内无有效图片。")
                    else:
                        p_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, f_name in enumerate(files):
                            base_name = os.path.basename(f_name)
                            status_text.text(f"自动分析中 ({i+1}/{len(files)}): {base_name}")
                            
                            img_bytes = z_in.read(f_name)
                            if not img_bytes:
                                continue
                            
                            nparr = np.frombuffer(img_bytes, np.uint8)
                            batch_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            res_img, ang, status, groups = process_image_v34_core(batch_img.copy())
                            
                            prefix = f"Auto_{ang:.0f}_" if status == "成功" else "Fail_"
                            
                            if status == "成功":
                                for idx, comp in enumerate(groups):
                                    p1, p_mid, p2 = comp["points"]
                                    cv2.line(res_img, p1, p_mid, (0,255,0), max(2, int(h/500)), cv2.LINE_AA)
                                    cv2.line(res_img, p_mid, p2, (0,255,0), max(2, int(h/500)), cv2.LINE_AA)
                                
                                _, buf = cv2.imencode(".jpg", res_img)
                                z_out.writestr(f"{prefix}{base_name}", buf.tobytes())
                            else:
                                z_out.writestr(f"Manual_Required_{base_name}", img_bytes)
                            
                            st.session_state.history.append({
                                "操作时间": datetime.now().strftime("%H:%M:%S"),
                                "文件名": base_name,
                                "测量角度": f"{ang:.2f}°" if status == "成功" else "需人工介入",
                                "模式": "批量自动",
                                "状态": status
                            })
                            
                            p_bar.progress((i + 1) / len(files))
                        
                        status_text.text("✨ 批量自动解析完毕！")
                        st.session_state.out_zip_bytes = out_zip.getvalue()
                        st.toast("全量解算完成！", icon="✅")
            
            except zipfile.BadZipFile:
                st.error("文件格式不正确。")
        
        if st.session_state.out_zip_bytes:
            st.write("---")
            st.download_button(
                "📥 导出标注图片包 (Zip)",
                st.session_state.out_zip_bytes,
                "Measurement_Results_V35.zip"
            )

st.divider()
st.subheader("📜 本次项目数据报表")
if st.session_state.history:
    df_h = pd.DataFrame(st.session_state.history)
    st.dataframe(df_h, use_container_width=True)
    st.download_button("📊 导出报表 (CSV)", df_h.to_csv(index=False).encode('utf-8-sig'), "data_history_v35.csv", "text/csv")
    if st.button("🗑️ 清空所有表格数据"):
        st.session_state.history = []
        st.session_state.out_zip_bytes = None
        st.rerun()