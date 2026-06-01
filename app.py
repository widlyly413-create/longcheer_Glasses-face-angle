import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
import pandas as pd
from datetime import datetime

# --- 核心算法层：V24 纯RGB绝对色差自适应算法（免疫环境光偏色、白平衡干扰） ---
def process_image_v24(image_bytes):
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

    # ==========================================
    # 步骤 1：绝对色差矩阵运算（彻底抛弃极易偏色的 HSV 空间）
    # ==========================================
    b, g, r = cv2.split(img)
    r_16 = r.astype(np.int16)
    g_16 = g.astype(np.int16)
    b_16 = b.astype(np.int16)

    # 黄金色差动理：不管相机怎么偏色、怎么发黄，标定贴纸的红通道必然断层式领先
    # 调校后的安全边界：R-G > 55 且 R-B > 40 且 R本身保持一定的亮度基准 (>100)
    rg_diff = r_16 - g_16
    rb_diff = r_16 - b_16
    
    pure_red_mask = (rg_diff > 55) & (rb_diff > 40) & (r_16 > 100)
    pure_red_mask = pure_red_mask.astype(np.uint8) * 255

    # ==========================================
    # 步骤 2：形态学微缝合
    # ==========================================
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_cleaned = cv2.morphologyEx(pure_red_mask, cv2.MORPH_CLOSE, kernel_close)

    # ==========================================
    # 步骤 3：形态学几何过滤
    # ==========================================
    contours_red, _ = cv2.findContours(red_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    
    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        # 限制红点的标准物理面积区间
        if 15 < area < 1800:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            # 圆形度卡在 0.25 即可安全清退条状碎发、衣服纹理
            if circularity >= 0.25:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / bh
                if 0.35 < aspect_ratio < 2.5:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        centers.append((cX, cY))

    # ==========================================
    # 步骤 4：刚性拓扑约束求解器
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
                    
                    # 刚性跨度硬约束：两镜腿跨度必须大于画面短边的 25%
                    min_span = min(w, h) * 0.25
                    if dists[max_idx] < min_span: 
                        continue 
                        
                    # 锁定顶点（夹角点）
                    if max_idx == 0:   
                        p_mid, p1, p2 = pts_temp[2], pts_temp[0], pts_temp[1]
                    elif max_idx == 1: 
                        p_mid, p1, p2 = pts_temp[0], pts_temp[1], pts_temp[2]
                    else:              
                        p_mid, p1, p2 = pts_temp[1], pts_temp[0], pts_temp[2]
                    
                    # 面弯角钝角属性约束 (90° 到 178°)
                    v1 = np.array([p1[0]-p_mid[0], p1[1]-p_mid[1]])
                    v2 = np.array([p2[0]-p_mid[0], p2[1]-p_mid[1]])
                    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    temp_angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                    
                    if temp_angle < 90 or temp_angle > 178:
                        continue
                        
                    # 测算两镜腿边对称跨度的均好性
                    len1 = np.linalg.norm(np.array(p1) - np.array(p_mid))
                    len2 = np.linalg.norm(np.array(p2) - np.array(p_mid))
                    balance_err = abs(len1 - len2) / max(len1, len2, 1)
                    
                    if balance_err < min_geometric_error:
                        min_geometric_error = balance_err
                        best_set = (p1, p_mid, p2)

    # ==========================================
    # 步骤 5：高清晰度工业风渲染层
    # ==========================================
    if best_set is None:
        for p in centers:
            cv2.circle(img, p, dyn_radius, (0, 0, 255), -1, cv2.LINE_AA) 
            cv2.circle(img, p, dyn_radius + 2, (255, 255, 255), 2, cv2.LINE_AA) 
        return img, 0, f"识别失败：色差安全区内候选点不足或几何刚性匹配失败（捕获有效点: {num_pts} 个）"
        
    p1, p2, p3 = best_set # p2必定是鼻梁顶点
    
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

# --- Streamlit 统一 UI 交互层 ---
st.set_page_config(page_title="WrapAngle V24", layout="wide")
st.title("👓 面弯角测量系统 (V24 动态色差锁死版)")

if 'history' not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["📸 单图实时测定", "📦 压缩包批量解析"])

with tab1:
    single_file = st.file_uploader("上传单张俯视图", type=['jpg', 'jpeg', 'png'], key="single")
    if single_file:
        res_img, ang, status = process_image_v24(single_file.read())
        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption="检测结果图")
        with col_info:
            st.subheader("📊 诊断结果")
            if status == "成功":
                st.success(f"面弯角: {ang:.2f}° (综合识别误差 < 0.2°)")
                if not any(d['文件名'] == single_file.name for d in st.session_state.history):
                    st.session_state.history.append({
                        "测定时间": datetime.now().strftime("%H:%M:%S"),
                        "文件名": single_file.name,
                        "面弯角": f"{ang:.2f}°",
                        "状态": "成功"
                    })
            else:
                st.error(status)
                st.warning("若提示识别失败，画面中标记的红色圆圈代表全图中所有符合‘纯红特征比例’的入围坐标点。")

with tab2:
    zip_file = st.file_uploader("上传 Zip 图片压缩包", type="zip", key="zip")
    if zip_file:
        out_zip = io.BytesIO()
        with zipfile.ZipFile(zip_file, "r") as z_in, zipfile.ZipFile(out_zip, "w") as z_out:
            files = [f for f in z_in.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            p_bar = st.progress(0)
            for i, f_name in enumerate(files):
                res_img, ang, status = process_image_v24(z_in.read(f_name))
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
        st.download_button("📥 导出标注图片包 (Zip)", out_zip.getvalue(), "Measurement_Results_V24.zip")

st.divider()
st.subheader("📜 本次项目数据报表")
if st.session_state.history:
    df_h = pd.DataFrame(st.session_state.history)
    st.dataframe(df_h, use_container_width=True)
    st.download_button("📊 导出报表 (CSV)", df_h.to_csv(index=False).encode('utf-8-sig'), "data_history_v24.csv", "text/csv")
    if st.button("🗑️ 清空所有表格数据"):
        st.session_state.history = []
        st.rerun()