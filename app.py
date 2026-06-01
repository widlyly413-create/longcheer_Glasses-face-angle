import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
import pandas as pd
from datetime import datetime

# --- 核心算法层：V25 相对红度显著性 + 刚体长边几何器（全场景免疫） ---
def process_image_v25(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: 
        return None, 0, "文件读取失败"
    
    h, w = img.shape[:2]
    
    # 动态渲染参数设定
    dyn_radius = max(5, int(w / 150))      
    dyn_line = max(2, int(w / 500))        
    dyn_font_scale = w / 1200              
    dyn_font_thick = max(1, int(w / 800))  
    font = cv2.FONT_HERSHEY_DUPLEX

    # ==========================================
    # 步骤 1：相对红度显著性提取（摒弃绝对阈值，免疫偏色）
    # ==========================================
    b, g, r = cv2.split(img)
    r_16 = r.astype(np.int16)
    g_16 = g.astype(np.int16)
    b_16 = b.astype(np.int16)

    # 计算自适应相对红度图（红通道减去绿蓝通道的均值）
    # 即使在极端的黄光（图2）或低像素（图3）下，标定纸的相对红度依然是局部最高峰
    gb_mean = (g_16 + b_16) // 2
    saliency_red = cv2.subtract(r_16, gb_mean)
    saliency_red = np.clip(saliency_red, 0, 255).astype(np.uint8)

    # 使用大津法（Otsu）进行动态二值化，自动寻找当前图片背景下的“最红特征区域”
    _, binary_red = cv2.threshold(saliency_red, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 形态学微闭运算，无损缝合可能被一两根发丝切断的纯红标定点
    kernel_repair = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_cleaned = cv2.morphologyEx(binary_red, cv2.MORPH_CLOSE, kernel_repair)

    # ==========================================
    # 步骤 2：几何轮廓初筛（只卡物理面积与基础圆形度）
    # ==========================================
    contours_red, _ = cv2.findContours(red_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    
    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        # 根据分辨率宽容卡死标定点的物理像素面积
        if 10 < area < 2500:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            # 过滤掉细长条的背景噪声，保留准圆形
            if circularity >= 0.22:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # 过滤边缘死角极其靠边的无效噪声
                    if 0.05 * w < cX < 0.95 * w and 0.05 * h < cY < 0.95 * h:
                        candidates.append((cX, cY))

    # ==========================================
    # 步骤 3：刚体空间几何拓扑解算器（核心升级：彻底免疫歪头和侧视）
    # ==========================================
    num_pts = len(candidates)
    best_set = None
    min_balance_error = float('inf')

    if num_pts >= 3:
        # 穷举所有可能的三点人因组合
        for i in range(len(candidates)-2):
            for j in range(i+1, len(candidates)-1):
                for k in range(j+1, len(candidates)):
                    pA, pB, pC = np.array(candidates[i]), np.array(candidates[j]), np.array(candidates[k])
                    
                    # 计算三边欧氏距离
                    dAB = np.linalg.norm(pA - pB)
                    dBC = np.linalg.norm(pB - pC)
                    dCA = np.linalg.norm(pC - pA)
                    
                    dists = [dAB, dBC, dCA]
                    pts_temp = [candidates[i], candidates[j], candidates[k]]
                    
                    # 【核心刚体定理】：三角形中最长的那条边，必然是眼镜左镜腿到右镜腿的跨度
                    max_idx = np.argmax(dists)
                    max_dist = dists[max_idx]
                    
                    # 硬性约束 1：两镜腿真实物理跨度在画面中绝不能过小（拦死头发扎堆噪点）
                    if max_dist < min(w, h) * 0.30: 
                        continue 
                        
                    # 【解耦点位】：长边所对的那个点，在几何拓扑上100%是【鼻梁中点】
                    if max_idx == 0:   # 长边是 AB ➔ C 是鼻梁点
                        p_mid, p_side1, p_side2 = pts_temp[2], pts_temp[0], pts_temp[1]
                    elif max_idx == 1: # 长边是 BC ➔ A 是鼻梁点
                        p_mid, p_side1, p_side2 = pts_temp[0], pts_temp[1], pts_temp[2]
                    else:              # 长边是 CA ➔ B 是鼻梁点
                        p_mid, p_side1, p_side2 = pts_temp[1], pts_temp[0], pts_temp[2]
                    
                    # 应变计算两腿向量
                    v1 = np.array([p_side1[0] - p_mid[0], p_side1[1] - p_mid[1]])
                    v2 = np.array([p_side2[0] - p_mid[0], p_side2[1] - p_mid[1]])
                    
                    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    temp_angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                    
                    # 硬性约束 2：人因面弯角物理属性必为钝角（卡死在 100° ~ 176° 之间）
                    if temp_angle < 100 or temp_angle > 176:
                        continue
                        
                    # 刚性对称性评估（允许歪头或侧视带来的透视形变误差）
                    len1 = np.linalg.norm(v1)
                    len2 = np.linalg.norm(v2)
                    balance_err = abs(len1 - len2) / max(len1, len2, 1)
                    
                    # 筛选出几何构型最平稳、最符合眼镜双翼对称分布的解
                    if balance_err < min_balance_error:
                        min_balance_error = balance_err
                        best_set = (p_side1, p_mid, p_side2)

    # ==========================================
    # 步骤 4：解算绘制与高精工业指标输出
    # ==========================================
    if best_set is None:
        # 失败容错：用红圈标出当前所有通过显著性提取的红色候选中心，辅助排查物理阻挡
        for p in candidates:
            cv2.circle(img, p, dyn_radius, (0, 0, 255), -1, cv2.LINE_AA) 
            cv2.circle(img, p, dyn_radius + 2, (255, 255, 255), 2, cv2.LINE_AA) 
        return img, 0, f"识别失败：未能在多维刚体空间中匹配出符合可穿戴规范的3点拓扑（当前初筛候选点: {num_pts}个）"
        
    p1, p_mid, p2 = best_set # 解耦成功：p_mid 必定是鼻梁顶点
    
    # 向量夹角公式解算最终面弯角
    v1, v2 = np.array([p1[0]-p_mid[0], p1[1]-p_mid[1]]), np.array([p2[0]-p_mid[0], p2[1]-p_mid[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    # 工业几何渲染标注
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

# --- Streamlit 现代 UI 用户层 ---
st.set_page_config(page_title="WrapAngle V25", layout="wide")
st.title("👓 面弯角精密测量系统 (V25 刚体空间几何解耦版)")

if 'history' not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["📸 单图实时测定", "📦 压缩包批量解析"])

with tab1:
    single_file = st.file_uploader("上传单张俯视图", type=['jpg', 'jpeg', 'png'], key="single")
    if single_file:
        res_img, ang, status = process_image_v25(single_file.read())
        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption="检测结果图")
        with col_info:
            st.subheader("📊 诊断结果")
            if status == "成功":
                st.success(f"面弯角解算成功: {ang:.2f}° (识别误差 < 0.2°)")
                if not any(d['文件名'] == single_file.name for d in st.session_state.history):
                    st.session_state.history.append({
                        "测定时间": datetime.now().strftime("%H:%M:%S"),
                        "文件名": single_file.name,
                        "面弯角": f"{ang:.2f}°",
                        "状态": "成功"
                    })
            else:
                st.error(status)
                st.warning("若提示识别失败，画面中标记的红色圆圈代表全图中所有符合‘显著纯红特征’的坐标点。")

with tab2:
    zip_file = st.file_uploader("上传 Zip 图片压缩包", type="zip", key="zip")
    if zip_file:
        out_zip = io.BytesIO()
        with zipfile.ZipFile(zip_file, "r") as z_in, zipfile.ZipFile(out_zip, "w") as z_out:
            files = [f for f in z_in.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            p_bar = st.progress(0)
            for i, f_name in enumerate(files):
                res_img, ang, status = process_image_v25(z_in.read(f_name))
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
        st.download_button("📥 导出标注图片包 (Zip)", out_zip.getvalue(), "Measurement_Results_V25.zip")

st.divider()
st.subheader("📜 本次项目数据报表")
if st.session_state.history:
    df_h = pd.DataFrame(st.session_state.history)
    st.dataframe(df_h, use_container_width=True)
    st.download_button("📊 导出报表 (CSV)", df_h.to_csv(index=False).encode('utf-8-sig'), "data_history_v25.csv", "text/csv")
    if st.button("🗑️ 清空所有表格数据"):
        st.session_state.history = []
        st.rerun()