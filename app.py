import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
import pandas as pd
from datetime import datetime

MULTIPLE_COLORS = [
    (255, 120, 0), (0, 180, 255), (0, 255, 0), 
    (0, 0, 255), (255, 0, 255), (255, 255, 0)
]

# --- 核心数据流缓存配置 ---
if 'batch_images' not in st.session_state: st.session_state.batch_images = {} 
if 'success_results' not in st.session_state: st.session_state.success_results = {} 
if 'history_log' not in st.session_state: st.session_state.history_log = []

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
    
    cv2.putText(img, f"V36 {mode_label} AVG: {angle:.2f} DEG", 
                (30, 60), font, dyn_font_scale, (0, 0, 255), dyn_font_thick + 2, cv2.LINE_AA)
    return img

# --- 💡 核心升级：OpenCV系统级原生鼠标监听捕获器（免网络流阻断）---
def opencv_native_manual_picker(img_src, filename_label):
    """
    通过弹出一个零延迟的底层本地窗口完成点选，完美避开浏览器自定义组件报错。
    """
    h, w = img_src.shape[:2]
    screen_max_side = 850
    scale = 1.0
    if max(h, w) > screen_max_side:
        scale = screen_max_side / max(h, w)
        img_view = cv2.resize(img_src, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_view = img_src.copy()

    clicked_points_disp = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(clicked_points_disp) < 3:
                clicked_points_disp.append((x, y))
                c_color = (255, 120, 0) if len(clicked_points_disp)==1 else ((0, 255, 0) if len(clicked_points_disp)==2 else (0, 0, 255))
                cv2.line(img_view, (x - 8, y), (x + 8, y), c_color, 2, cv2.LINE_AA)
                cv2.line(img_view, (x, y - 8), (x, y + 8), c_color, 2, cv2.LINE_AA)
                cv2.putText(img_view, str(len(clicked_points_disp)), (x + 10, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, c_color, 1, cv2.LINE_AA)
                cv2.imshow(win_name, img_view)

    win_name = f"Manual Correction - {filename_label} (Click 3 points sequentially)"
    cv2.namedWindow(win_name, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(win_name, mouse_callback)
    
    cv2.imshow(win_name, img_view)
    cv2.setWindowProperty(win_name, cv2.WND_PROP_TOPMOST, 1)
    
    st.toast("💡 请切换到桌面查看弹出的图片窗口，顺次点击3点后窗口将自动闭环！", icon="🖥️")
    
    while True:
        key = cv2.waitKey(10) & 0xFF
        if len(clicked_points_disp) == 3 or key == 27 or key == 32:
            break
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break
            
    cv2.destroyWindow(win_name)
    
    if len(clicked_points_disp) < 3:
        return None, 0

    p1_r = (int(clicked_points_disp[0][0] / scale), int(clicked_points_disp[0][1] / scale))
    pm_r = (int(clicked_points_disp[1][0] / scale), int(clicked_points_disp[1][1] / scale))
    p2_r = (int(clicked_points_disp[2][0] / scale), int(clicked_points_disp[2][1] / scale))
    
    angle = calculate_angle_from_three_points(p1_r, pm_r, p2_r)
    final_img = render_measurement_style(img_src.copy(), p1_r, pm_r, p2_r, angle, 0, "MANUAL")
    return final_img, angle

# --- V27 级联多层检测算法（优化：三点连线尽量水平或竖直）---
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

# --- UI 视图展现 ---
st.set_page_config(page_title="WrapAngle V36 Professional", layout="wide")
st.title("👓 面弯角高通量流水线测定系统 (V36 极速交互抗卡顿版)")
st.caption("专为大规模散图和压缩包定制。自动识别失败的图片将自动进入人工补偿区，使用底层 OpenCV 原生窗口进行零延迟点选。")

uploaded_files = st.file_uploader("📥 上传俯视图 / 导入 Zip 压缩包（支持多选混投）", type=['jpg', 'jpeg', 'png', 'zip'], accept_multiple_files=True)

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
        
        with st.spinner("🤖 正在启动后台算法流水线，快速分流合格品..."):
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

# 分流展示状态看板
if st.session_state.batch_images:
    total_count = len(st.session_state.batch_images)
    success_count = len(st.session_state.success_results)
    fail_count = total_count - success_count
    
    c1, c2, c3 = st.columns(3)
    c1.metric("📦 当前流转图片总量", f"{total_count} 张")
    c2.metric("🤖 后台算法自动识别成功", f"{success_count} 张")
    c3.metric("🖱️ 需人工补偿校准", f"{fail_count} 张")

    # --- 第一步：一键打包混下载区 ---
    st.write("---")
    st.subheader("📥 核心成果数据包导出")
    
    if st.session_state.success_results:
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as z_out:
                for f_name, data_obj in st.session_state.success_results.items():
                    prefix = "Auto_" if data_obj["mode"].startswith("自动识别") else "Manual_"
                    z_out.writestr(f"{prefix}{f_name}", data_obj["bytes"])
            
            st.download_button(
                label="📥 导出已处理的混合标注图片包 (Zip) - 无需等待页面加载",
                data=zip_buffer.getvalue(),
                file_name=f"WrapAngle_V36_Combined_{datetime.now().strftime('%m%d_%H%M')}.zip",
                mime="application/zip",
                use_container_width=True
            )
        with col_dl2:
            all_log = []
            for name in st.session_state.batch_images.keys():
                if name in st.session_state.success_results:
                    obj = st.session_state.success_results[name]
                    all_log.append({"文件名": name, "最终测量面弯角": obj["angle"], "测量模式": obj["mode"], "状态": "✅ 成功闭环"})
                else:
                    all_log.append({"文件名": name, "最终测量面弯角": "-", "测量模式": "未通过", "状态": "❌ 待手动介入"})
            df = pd.DataFrame(all_log)
            st.download_button(
                label="📊 导出完整面弯角数据分析报表 (CSV)",
                data=df.to_csv(index=False).encode('utf-8-sig'),
                file_name="WrapAngle_V36_Report.csv",
                mime="text/csv",
                use_container_width=True
            )
        
        st.dataframe(df, use_container_width=True)

    # --- 第二步：原生 OpenCV 弹窗式手动选点工作区 ---
    st.write("---")
    st.subheader("🖱️ 手动异常补偿干预区 (系统窗口极速模式)")
    
    target_file = st.selectbox("🎯 请选择需要【进入手动微调】的目标图片：", list(st.session_state.batch_images.keys()))
    
    if target_file:
        is_already_success = target_file in st.session_state.success_results
        if is_already_success:
            st.warning(f"💡 提示：图片 `{target_file}` 此前已成功生成结果（角度: {st.session_state.success_results[target_file]['angle']}），重新点选将完美覆盖原纪录。")
        else:
            st.error(f"🔍 提示：图片 `{target_file}` 自动识别失败，请使用下方独立原生窗口进行纠偏。")
            
        if st.button(f"🖥️ 唤醒原生独立选点窗口：处理 {target_file}", use_container_width=True):
            raw_data = st.session_state.batch_images[target_file]
            nparr = np.frombuffer(raw_data, np.uint8)
            orig_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # 直接调用底层内存级鼠标捕获，彻底避免网络加载组件错误
            final_render_img, m_angle = opencv_native_manual_picker(orig_img, target_file)
            
            if final_render_img is not None:
                _, out_buf = cv2.imencode(".jpg", final_render_img)
                st.session_state.success_results[target_file] = {
                    "bytes": out_buf.tobytes(), "angle": f"{m_angle:.2f}°", "mode": "人工选点"
                }
                st.success(f"🎉 成功解算并注入！图片 `{target_file}` 的测量角度为：{m_angle:.2f}°")
                st.rerun()
            else:
                st.error("操作被取消或点选点数不足3个，未能成功写入。")

    if st.button("🗑️ 清空流水线内所有图片缓存（重新上传新数据前点击）"):
        st.session_state.batch_images = {}
        st.session_state.success_results = {}
        st.rerun()