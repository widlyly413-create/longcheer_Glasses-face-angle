import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
import pandas as pd
from datetime import datetime

# --- 核心算法层：V29 HSV色域 + 动态区域锁死（彻底解决飞点、错选假点问题） ---
def process_image_v29(image_bytes):
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
    # 核心升级 1：划定核心 ROI 区域（过滤手部、左侧衣物及边缘杂色）
    # 眼镜必然出现在画面中右侧的核心区域
    # ==========================================
    roi_mask = np.zeros((h, w), dtype=np.uint8)
    # 限制横向在 35% 到 95% 之间，纵向在 10% 到 90% 之间
    cv2.rectangle(roi_mask, (int(w * 0.35), int(h * 0.10)), (int(w * 0.95), int(h * 0.90)), 255, -1)

    # 转换至 HSV 颜色空间进行精准色彩提取
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 红色的色调（Hue）分布在两端，定义两档标准红阈值
    # 降低饱和度(V)和亮度(V)下限，确保暗光下被遮挡的红点能被抓到
    lower_red1 = np.array([0, 45, 45])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([165, 45, 45])
    upper_red2 = np.array([180, 255, 255])
    
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_color_mask = cv2.bitwise_or(mask1, mask2)
    
    # 将颜色掩膜与区域掩膜进行“与”操作，双重锁死
    red_mask = cv2.bitwise_and(red_color_mask, roi_mask)

    # ==========================================
    # 步骤 1：多级级联形态学修复
    # ==========================================
    cascade_thresholds = [
        {"close_k": 5, "dilate_k": 3, "circ": 0.30, "min_area": 3},  # 极度容错级
        {"close_k": 5, "dilate_k": 0, "circ": 0.45, "min_area": 8}   # 标准级
    ]
    
    best_set = None
    min_geometric_error = float('inf')

    for pass_idx, th in enumerate(cascade_thresholds):
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (th["close_k"], th["close_k"]))
        red_cleaned = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_close)
        
        if th["dilate_k"] > 0:
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (th["dilate_k"], th["dilate_k"]))
            red_cleaned = cv2.dilate(red_cleaned, kernel_dilate, iterations=1)
            
        contours_red, _ = cv2.findContours(red_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        
        for cnt in contours_red:
            area = cv2.contourArea(cnt)
            if th["min_area"] < area < 2000:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                if circularity >= th["circ"]:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        candidates.append((cX, cY))
                            
        num_pts = len(candidates)
        
        # ==========================================
        # 步骤 2：双翼刚性对称拓扑解算
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
                        
                        # 硬约束 1：眼镜两翼物理刚性总跨度限制范围（防止缩得太小或拉得太大）
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
                        
                        # 硬约束 2：眼镜面弯角物理限值
                        if temp_angle < 95 or temp_angle > 179.5:
                            continue
                            
                        # 硬约束 3：两翼刚体物理对称指标
                        len1 = np.linalg.norm(v1)
                        len2 = np.linalg.norm(v2)
                        balance_err = abs(len1 - len2) / max(len1, len2, 1)
                        
                        if balance_err > 0.25:  # 恢复严格指标，因为ROI和HSV已经滤掉了外部噪音
                            continue 
                        
                        if balance_err < min_geometric_error:
                            min_geometric_error = balance_err
                            best_set = (p1, p_mid, p2)
            
            if best_set is not None:
                break 

    # ==========================================
    # 步骤 3：精密解算与渲染
    # ==========================================
    # 绘制ROI区域方便在界面上肉眼debug查看限制范围
    cv2.rectangle(img, (int(w * 0.35), int(h * 0.10)), (int(w * 0.95), int(h * 0.90)), (0, 255, 0), max(1, dyn_line//2), cv2.LINE_AA)

    if best_set is None:
        return img, 0, f"识别失败：已强制死锁过滤背景与手部。当前核心ROI内找到的红色像素候选点数：{len(candidates)}。请确保留存的三个红点未被完全遮挡。"
        
    p1, p_mid, p2 = best_set 
    
    v1, v2 = np.array([p1[0]-p_mid[0], p1[1]-p_mid[1]]), np.array([p2[0]-p_mid[0], p2[1]-p_mid[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    # 渲染标定线
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
st.set_page_config(page_title="WrapAngle V29", layout="wide")
st.title("👓 面弯角精密测量系统 (V29 HSV空间+核心安全区锁死版)")

if 'history' not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["📸 单图实时测定", "📦 压缩包批量解析"])

with tab1:
    single_file = st.file_uploader("上传单张俯视图", type=['jpg', 'jpeg', 'png'], key="single")
    if single_file:
        res_img, ang, status = process_image_v29(single_file.read())
        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption="检测结果图（绿色框内为安全算法锁定的检测区域）")
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
                st.warning("安全机制已拦截假点。若提示失败，请确保中间红点未被配重块完全盖死，且受试者碎发没有斩断红点。")

with tab2:
    zip_file = st.file_uploader("上传 Zip 图片压缩包", type="zip", key="zip")
    if zip_file:
        out_zip = io.BytesIO()
        with zipfile.ZipFile(zip_file, "r") as z_in, zipfile.ZipFile(out_zip, "w") as z_out:
            files = [f for f in z_in.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            p_bar = st.progress(0)
            for i, f_name in enumerate(files):
                res_img, ang, status = process_image_v29(z_in.read(f_name))
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
        st.download_button("📥 导出标注图片包 (Zip)", out_zip.getvalue(), "Measurement_Results_V29.zip")

st.divider()
st.subheader("📜 本次项目数据报表")
if st.session_state.history:
    df_h = pd.DataFrame(st.session_state.history)
    st.dataframe(df_h, use_container_width=True)
    st.download_button("📊 导出报表 (CSV)", df_h.to_csv(index=False).encode('utf-8-sig'), "data_history_v29.csv", "text/csv")
    if st.button("🗑️ 清空所有表格数据"):
        st.session_state.history = []
        st.rerun()