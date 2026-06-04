import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
import pandas as pd
import plotly.express as px

MULTIPLE_COLORS = [
    (255, 120, 0), (0, 180, 255), (0, 255, 0), 
    (0, 0, 255), (255, 0, 255), (255, 255, 0)
]

# --- 核心数据流缓存配置 ---
if 'batch_images' not in st.session_state: st.session_state.batch_images = {} 
if 'success_results' not in st.session_state: st.session_state.success_results = {} 
if 'history_log' not in st.session_state: st.session_state.history_log = []
if 'click_pts_accumulator' not in st.session_state: st.session_state.click_pts_accumulator = []
if 'current_processing_file' not in st.session_state: st.session_state.current_processing_file = ""

def calculate_angle_from_three_points(p1, p_mid, p2):
    v1 = np.array([p1[0] - p_mid[0], p1[1] - p_mid[1]])
    v2 = np.array([p2[0] - p_mid[0], p2[1] - p_mid[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def render_measurement_style(img, p1, p_mid, p2, angle, group_idx=0, mode_label="AUTO"):
    h, w = img.shape[:2]
    dyn_line = max(2, int(w / 600))        
    dyn_font_scale = w / 1500              
    dyn_font_thick = max(1, int(w / 1000))  
    font = cv2.FONT_HERSHEY_DUPLEX
    color = MULTIPLE_COLORS[group_idx % len(MULTIPLE_COLORS)]
    
    cv2.line(img, p1, p_mid, color, dyn_line, cv2.LINE_AA)
    cv2.line(img, p_mid, p2, color, dyn_line, cv2.LINE_AA)
    for p in [p1, p_mid, p2]:
        cv2.circle(img, p, int(dyn_line * 2.5), color, -1, cv2.LINE_AA)
        cv2.circle(img, p, int(dyn_line * 2.5), (0, 0, 0), 1, cv2.LINE_AA)
        
    text = f"#{group_idx+1}: {angle:.2f} DEG"
    text_pos = (p_mid[0] + 30, p_mid[1] + (group_idx * int(w / 35)) - int(w / 70))
    cv2.putText(img, text, text_pos, font, dyn_font_scale * 0.75, (0,0,0), dyn_font_thick + 1, cv2.LINE_AA)
    cv2.putText(img, text, text_pos, font, dyn_font_scale * 0.75, color, dyn_font_thick, cv2.LINE_AA)
    
    cv2.putText(img, f"V40 {mode_label} AVG: {angle:.2f} DEG", (30, 60), font, dyn_font_scale, (0, 0, 255), dyn_font_thick + 2, cv2.LINE_AA)
    return img

@st.cache_data
def load_and_resize_image(file_bytes, max_side=720):
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None, None, 1.0
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img_resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img.copy()
    return img, img_resized, scale

# --- V27 级联多层检测算法 ---
def process_image_v27(img):
    if img is None:
        return None, 0, "文件读取失败", "V27"
    
    h, w = img.shape[:2]
    
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
    best_horizontal_score = -1

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
                            candidates.append((cX, cY, area))
        
        candidates.sort(key=lambda x: x[2], reverse=True)
        candidates = [c[:2] for c in candidates[:12]]
                            
        num_pts = len(candidates)
        
        if num_pts >= 3:
            for i in range(len(candidates)-2):
                if min_geometric_error < 0.03:
                    break
                for j in range(i+1, len(candidates)-1):
                    if min_geometric_error < 0.03:
                        break
                    for k in range(j+1, len(candidates)):
                        pA, pB, pC = np.array(candidates[i]), np.array(candidates[j]), np.array(candidates[k])
                        
                        dAB = np.linalg.norm(pA - pB)
                        dBC = np.linalg.norm(pB - pC)
                        dCA = np.linalg.norm(pC - pA)
                        dists = [dAB, dBC, dCA]
                        pts_temp = [candidates[i], candidates[j], candidates[k]]
                        
                        max_idx = np.argmax(dists)
                        max_dist = dists[max_idx]
                        
                        if max_dist < min(w, h) * 0.30: 
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
                        
                        if temp_angle < 165 or temp_angle > 179.8:
                            continue
                            
                        len1 = np.linalg.norm(v1)
                        len2 = np.linalg.norm(v2)
                        balance_err = abs(len1 - len2) / max(len1, len2, 1)
                        
                        if balance_err > 0.15: 
                            continue 
                        
                        line_vec = np.array([p2[0] - p1[0], p2[1] - p1[1]])
                        horizontal_angle = np.abs(np.arctan2(line_vec[1], line_vec[0]) * 180 / np.pi)
                        horizontal_score = 1.0 - min(abs(horizontal_angle), abs(horizontal_angle-90), 
                                                     abs(horizontal_angle-180), abs(horizontal_angle-270)) / 90
                        
                        combined_score = (1.0 - balance_err) * 0.7 + horizontal_score * 0.3
                        
                        if combined_score > best_horizontal_score or \
                           (combined_score == best_horizontal_score and balance_err < min_geometric_error):
                            min_geometric_error = balance_err
                            best_horizontal_score = combined_score
                            best_set = (p1, p_mid, p2)
    
            if best_set is not None:
                break

    if best_set is None:
        return img, 0, "V27识别失败", "V27"
        
    p1, p_mid, p2 = best_set
    
    v1, v2 = np.array([p1[0]-p_mid[0], p1[1]-p_mid[1]]), np.array([p2[0]-p_mid[0], p2[1]-p_mid[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    img_rendered = render_measurement_style(img.copy(), p1, p_mid, p2, angle, 0, "AUTO")
    return img_rendered, angle, "成功", "V27"

# --- V28 增强容错算法 ---
def process_image_v28(img):
    if img is None: 
        return None, 0, "文件读取失败", "V28"
    
    h, w = img.shape[:2]

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
        candidates = [c[:2] for c in candidates[:15]]
                            
        num_pts = len(candidates)
        
        if num_pts >= 3:
            for i in range(len(candidates)-2):
                if min_geometric_error < 0.02:
                    break
                for j in range(i+1, len(candidates)-1):
                    if min_geometric_error < 0.02:
                        break
                    for k in range(j+1, len(candidates)):
                        pA, pB, pC = np.array(candidates[i]), np.array(candidates[j]), np.array(candidates[k])
                        
                        dAB = np.linalg.norm(pA - pB)
                        dBC = np.linalg.norm(pB - pC)
                        dCA = np.linalg.norm(pC - pA)
                        dists = [dAB, dBC, dCA]
                        pts_temp = [candidates[i], candidates[j], candidates[k]]
                        
                        max_idx = np.argmax(dists)
                        max_dist = dists[max_idx]
                        
                        if max_dist < min(w, h) * 0.30: 
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
                        
                        if temp_angle < 160 or temp_angle > 179.8:
                            continue
                            
                        len1 = np.linalg.norm(v1)
                        len2 = np.linalg.norm(v2)
                        balance_err = abs(len1 - len2) / max(len1, len2, 1)
                        
                        if balance_err < min_geometric_error:
                            min_geometric_error = balance_err
                            best_set = (p1, p_mid, p2)
    
            if best_set is not None:
                break

    if best_set is None:
        return img, 0, "V28识别失败", "V28"
        
    p1, p_mid, p2 = best_set
    
    v1, v2 = np.array([p1[0]-p_mid[0], p1[1]-p_mid[1]]), np.array([p2[0]-p_mid[0], p2[1]-p_mid[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    img_rendered = render_measurement_style(img.copy(), p1, p_mid, p2, angle, 0, "AUTO")
    return img_rendered, angle, "成功", "V28"

# --- 级联识别主函数：V27优先，失败则V28 ---
def process_image_cascade(img):
    MIN_ANGLE = 168.0
    
    res_img_v27, angle_v27, status_v27, _ = process_image_v27(img.copy())
    
    if status_v27 == "成功" and angle_v27 >= MIN_ANGLE:
        return res_img_v27, angle_v27, status_v27, "V27"
    
    res_img_v28, angle_v28, status_v28, _ = process_image_v28(img.copy())
    
    if status_v28 == "成功" and angle_v28 >= MIN_ANGLE:
        return res_img_v28, angle_v28, status_v28, "V28"
    
    fail_reason = "识别失败："
    if status_v27 == "成功" and angle_v27 < MIN_ANGLE:
        fail_reason += f"V27识别角度 {angle_v27:.1f}° 低于阈值 {MIN_ANGLE}°；"
    else:
        fail_reason += "V27未识别成功；"
        
    if status_v28 == "成功" and angle_v28 < MIN_ANGLE:
        fail_reason += f"V28识别角度 {angle_v28:.1f}° 低于阈值 {MIN_ANGLE}°"
    else:
        fail_reason += "V28未识别成功"
    
    return img, 0, fail_reason, "失败"

# --- 主视图渲染 ---
st.set_page_config(page_title="WrapAngle V40", layout="wide")
st.title("👓 面弯角高通量流水线测定系统 (V40 云端原生对齐版)")
st.caption("完美适配云端沙盒环境。放弃 HTML5 组件入侵，改用官方最稳健的图表点击序列。前端毫秒级响应，点满3点自动闭环保存。")

uploaded_files = st.file_uploader("📥 上传多张俯视图 或 导入 1 个 Zip 压缩包", type=['jpg', 'jpeg', 'png', 'zip'], accept_multiple_files=True)

if uploaded_files:
    new_pool = {}
    for f in uploaded_files:
        if f.name.lower().endswith('.zip'):
            try:
                with zipfile.ZipFile(io.BytesIO(f.read()), 'r') as z:
                    for name in z.namelist():
                        if name.lower().endswith(('.jpg', '.jpeg', '.png')) and not name.startswith('__MACOSX'):
                            new_pool[os.path.basename(name)] = z.read(name)
            except Exception: st.error(f"Zip包 {f.name} 解压失败")
        else:
            new_pool[f.name] = f.read()
            
    if not st.session_state.batch_images or set(new_pool.keys()) != set(st.session_state.batch_images.keys()):
        st.session_state.batch_images = new_pool
        st.session_state.success_results = {}
        st.session_state.history_log = []
        st.session_state.click_pts_accumulator = []
        
        with st.spinner("🤖 后台算法正在快速分流无需介入的合格品..."):
            for name, b_data in st.session_state.batch_images.items():
                nparr = np.frombuffer(b_data, np.uint8)
                raw_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if raw_img is None: continue
                res_img, ang, status, algo_version = process_image_cascade(raw_img.copy())
                
                if status == "成功":
                    _, buf = cv2.imencode(".jpg", res_img)
                    st.session_state.success_results[name] = {
                        "bytes": buf.tobytes(), "angle": f"{ang:.2f}°", "mode": f"自动识别({algo_version})"
                    }

if st.session_state.batch_images:
    total_count = len(st.session_state.batch_images)
    success_count = len(st.session_state.success_results)
    fail_count = total_count - success_count
    
    c1, c2, c3 = st.columns(3)
    c1.metric("📦 图像总流转量", f"{total_count} 张")
    c2.metric("🤖 后台算法通过", f"{success_count} 张")
    c3.metric("🖱️ 待手动介入干预", f"{fail_count} 张")

    # --- 数据一键打包下载区 ---
    st.write("---")
    st.subheader("📥 核心成果一键打包下载区")
    
    if st.session_state.success_results:
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as z_out:
                for f_name, data_obj in st.session_state.success_results.items():
                    prefix = "Auto_" if data_obj["mode"].startswith("自动识别") else "Manual_"
                    z_out.writestr(f"{prefix}{f_name}", data_obj["bytes"])
            st.download_button("📥 导出混合标注结果图片包 (Zip)", zip_buffer.getvalue(), f"WrapAngle_Results_V40.zip", "application/zip", use_container_width=True)
        with col_dl2:
            all_log = []
            for name in st.session_state.batch_images.keys():
                if name in st.session_state.success_results:
                    obj = st.session_state.success_results[name]
                    all_log.append({"文件名": name, "最终测量面弯角": obj["angle"], "分析模式": obj["mode"], "状态": "✅ 成功闭环"})
                else:
                    all_log.append({"文件名": name, "最终测量面弯角": "-", "分析模式": "未通过", "状态": "❌ 待手动介入"})
            df = pd.DataFrame(all_log)
            st.download_button("📊 导出完整面弯角分析报表 (CSV)", df.to_csv(index=False).encode('utf-8-sig'), "WrapAngle_Report.csv", "text/csv", use_container_width=True)
        st.dataframe(df, use_container_width=True)

    # --- 鼠标点击全自动补偿区 ---
    st.write("---")
    st.subheader("🖱️ 手动异常补偿干预区 (官方原生交互 · 零延迟)")
    
    target_file = st.selectbox("🎯 请选择需要【手动微调选点】的目标图片：", list(st.session_state.batch_images.keys()))
    
    if target_file:
        # 切图时自动清空历史点击残余
        if st.session_state.current_processing_file != target_file:
            st.session_state.click_pts_accumulator = []
            st.session_state.current_processing_file = target_file
            
        if target_file in st.session_state.success_results:
            st.warning(f"💡 提示：图片 `{target_file}` 此前已有结果（角度: {st.session_state.success_results[target_file]['angle']}），执行新点选将直接覆盖。")
        else:
            st.error(f"🔍 提示：图片 `{target_file}` 自动识别失败。请在下方画面中，直接连续单击红点。")
            
        raw_data = st.session_state.batch_images[target_file]
        orig_img, display_img, scale = load_and_resize_image(raw_data, max_side=720)
        
        col_workspace, col_control = st.columns([13, 7])
        
        with col_control:
            st.markdown(f"**当前调节目标**: `{target_file}`")
            pt_len = len(st.session_state.click_pts_accumulator)
            
            st.info(f"📍 请在右图上【直接单击鼠标左键】：\n1. 左侧红点 ({'🟢 已捕获' if pt_len>=1 else '⚪ 待点击'})\n2. 鼻梁中点 ({'🔴 已捕获' if pt_len>=2 else '⚪ 待点击'})\n3. 右侧红点 ({'🔵 已捕获' if pt_len>=3 else '⚪ 待点击'})")
            st.caption("💡 提示：鼠标会显示为十字准星，点击即可标记")
            
            # 备选方案：手动输入坐标
            st.markdown("---")
            st.subheader("🔧 备选：手动输入坐标")
            h, w = display_img.shape[:2]
            input_x = st.slider(f"X坐标 (0-{w-1})", 0, w-1, w//2, key=f"x_{target_file}")
            input_y = st.slider(f"Y坐标 (0-{h-1})", 0, h-1, h//2, key=f"y_{target_file}")
            
            if st.button(f"📍 添加点 ({input_x}, {input_y})"):
                new_pt = (input_x, input_y)
                if len(st.session_state.click_pts_accumulator) < 3:
                    if not st.session_state.click_pts_accumulator or np.linalg.norm(np.array(st.session_state.click_pts_accumulator[-1]) - np.array(new_pt)) > 6:
                        st.session_state.click_pts_accumulator.append(new_pt)
                        st.rerun()
            
            if st.button("🗑️ 清空当前点重新选"):
                st.session_state.click_pts_accumulator = []
                st.rerun()
                
            # 💡 核心自动化闭环：一满3个点，后台在静默状态下瞬间完成原图映射与对齐计算
            if pt_len == 3:
                p1_d, pm_d, p2_d = st.session_state.click_pts_accumulator
                p1_r = (int(p1_d[0] / scale), int(p1_d[1] / scale))
                pm_r = (int(pm_d[0] / scale), int(pm_d[1] / scale))
                p2_r = (int(p2_d[0] / scale), int(p2_d[1] / scale))
                
                m_angle = calculate_angle_from_three_points(p1_r, pm_r, p2_r)
                st.success(f"📐 测算面弯角: **{m_angle:.2f}°**")
                
                # 统一调用自动化测量完全相同的渲染引擎，格式100%对齐
                final_render_img = render_measurement_style(orig_img.copy(), p1_r, pm_r, p2_r, m_angle, 0, "MANUAL")
                _, out_buf = cv2.imencode(".jpg", final_render_img)
                
                st.session_state.success_results[target_file] = {
                    "bytes": out_buf.tobytes(), "angle": f"{m_angle:.2f}°", "mode": "人工选点"
                }
                st.session_state.click_pts_accumulator = []
                st.toast(f"图片 {target_file} 测量结果已自动写入压缩包！", icon="🚀")
                st.rerun()

        with col_workspace:
            # 实时渲染带连线骨架和序号的临时画布，给用户以极其顺畅的反馈
            canvas = display_img.copy()
            for idx, pt in enumerate(st.session_state.click_pts_accumulator):
                c_color = (255, 120, 0) if idx==0 else ((0, 255, 0) if idx==1 else (0, 0, 255))
                cross = 10
                cv2.line(canvas, (pt[0] - cross, pt[1]), (pt[0] + cross, pt[1]), c_color, 2, cv2.LINE_AA)
                cv2.line(canvas, (pt[0], pt[1] - cross), (pt[0], pt[1] + cross), c_color, 2, cv2.LINE_AA)
                cv2.putText(canvas, str(idx+1), (pt[0]+12, pt[1]-12), cv2.FONT_HERSHEY_DUPLEX, 0.5, c_color, 1, cv2.LINE_AA)
            
            if len(st.session_state.click_pts_accumulator) >= 2:
                cv2.line(canvas, st.session_state.click_pts_accumulator[0], st.session_state.click_pts_accumulator[1], (0, 165, 255), 2, cv2.LINE_AA)
                if len(st.session_state.click_pts_accumulator) == 3:
                    cv2.line(canvas, st.session_state.click_pts_accumulator[1], st.session_state.click_pts_accumulator[2], (0, 165, 255), 2, cv2.LINE_AA)

            # 💡 【原生 HTML5 Canvas - 真正的单点点击】
            # 使用 base64 编码图片，通过原生 JavaScript 实现真正的单击事件
            import base64
            _, buffer = cv2.imencode('.jpg', canvas)
            img_b64 = base64.b64encode(buffer).decode('utf-8')
            h, w = canvas.shape[:2]
            
            # 生成唯一的画布ID，防止不同图片之间的冲突
            canvas_id = f"paint_canvas_{target_file.replace('.', '_')}"
            
            html_code = f"""
            <style>
                #{canvas_id} {{
                    display: block;
                    cursor: crosshair;
                    max-width: 100%;
                    height: auto;
                    border: 2px solid #ddd;
                    border-radius: 4px;
                }}
            </style>
            <canvas id="{canvas_id}" width="{w}" height="{h}"></canvas>
            <script>
                const canvas = document.getElementById("{canvas_id}");
                const ctx = canvas.getContext("2d");
                const img = new Image();
                img.src = "data:image/jpeg;base64,{img_b64}";
                img.onload = function() {{
                    ctx.drawImage(img, 0, 0);
                }};
                
                canvas.addEventListener("click", function(e) {{
                    const rect = canvas.getBoundingClientRect();
                    const scaleX = canvas.width / rect.width;
                    const scaleY = canvas.height / rect.height;
                    const x = Math.round((e.clientX - rect.left) * scaleX);
                    const y = Math.round((e.clientY - rect.top) * scaleY);
                    
                    // 通过隐藏输入框传递坐标
                    const input = document.createElement('input');
                    input.type = 'hidden';
                    input.name = 'canvas_click';
                    input.value = JSON.stringify({{x: x, y: y, file: "{target_file}"}});
                    document.body.appendChild(input);
                    
                    // 模拟表单提交触发页面刷新
                    const event = new Event('submit');
                    const form = document.createElement('form');
                    form.appendChild(input);
                    document.body.appendChild(form);
                    form.dispatchEvent(event);
                    form.remove();
                    
                    // 也可以通过 URL 参数传递
                    const params = new URLSearchParams(window.location.search);
                    params.set('click_x', x);
                    params.set('click_y', y);
                    params.set('click_file', "{target_file}");
                    window.history.replaceState({{}}, '', window.location.pathname + '?' + params.toString());
                    
                    // 触发页面重新运行
                    setTimeout(() => {{
                        window.location.reload();
                    }}, 50);
                }});
            <\/script>
            """
            st.markdown(html_code, unsafe_allow_html=True)
            
            # 从 URL 参数获取点击坐标
            import urllib.parse
            try:
                query_params = st.query_params
                if 'click_x' in query_params and 'click_y' in query_params and 'click_file' in query_params:
                    click_x = int(query_params['click_x'])
                    click_y = int(query_params['click_y'])
                    click_file = urllib.parse.unquote(query_params['click_file'])
                    
                    if click_file == target_file and len(st.session_state.click_pts_accumulator) < 3:
                        new_pt = (click_x, click_y)
                        if not st.session_state.click_pts_accumulator or np.linalg.norm(np.array(st.session_state.click_pts_accumulator[-1]) - np.array(new_pt)) > 6:
                            st.session_state.click_pts_accumulator.append(new_pt)
                            # 清除 URL 参数
                            st.query_params.clear()
                            st.rerun()
            except Exception as e:
                pass

    if st.button("🗑️ 清空流水线内所有图片缓存"):
        st.session_state.batch_images = {}
        st.session_state.success_results = {}
        st.session_state.click_pts_accumulator = []
        st.rerun()