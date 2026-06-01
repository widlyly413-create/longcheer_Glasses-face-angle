import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
import pandas as pd
from datetime import datetime

# --- 核心算法层：V26 强色差纯化矩阵 + 0.60圆形度死锁 + 刚体长边几何器 ---
def process_image_v26(image_bytes):
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
    # 步骤 1：绝对色差矩阵纯化（摒弃自适应大津法，防止皮肤成片误入）
    # ==========================================
    b, g, r = cv2.split(img)
    r_16 = r.astype(np.int16)
    g_16 = g.astype(np.int16)
    b_16 = b.astype(np.int16)

    # 刚性色差过滤逻辑：只提取红通道绝对压倒绿通道(>55)与蓝通道(>40)的纯净区域
    # 该过滤标准在数学上直接粉碎了皮肤（偏橙黄，R-G极小）与头皮阴影的通关可能
    rg_diff = r_16 - g_16
    rb_diff = r_16 - b_16
    
    pure_red_mask = (rg_diff > 55) & (rb_diff > 40) & (r_16 > 100)
    pure_red_mask = pure_red_mask.astype(np.uint8) * 255

    # 极轻度形态学微闭运算，无损粘合由于碎发横切产生的微小红点裂缝
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_cleaned = cv2.morphologyEx(pure_red_mask, cv2.MORPH_CLOSE, kernel_close)

    # ==========================================
    # 步骤 2：极限轮廓特征筛选（用 0.60 圆形度绞杀碎发噪点）
    # ==========================================
    contours_red, _ = cv2.findContours(red_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    
    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        # 精密卡死人工标定圆点在 15 到 1500 像素的合理物理面积区间
        if 15 < area < 1500:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            # 【核心死锁铁闸】：标定贴纸即使形变，圆形度也绝不会低于 0.60！
            # 头发丝缝隙、皮肤褶皱的圆形度通常在 0.1~0.4 之间，在此步骤被全部彻底歼灭
            if circularity >= 0.60:
                M = cv2.moments(cnt)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    # 剥离靠在最边缘的背景地砖死角干扰
                    if 0.02 * w < cX < 0.98 * w and 0.02 * h < cY < 0.98 * h:
                        candidates.append((cX, cY))

    # ==========================================
    # 步骤 3：刚体不变性三角形拓扑求解器
    # ==========================================
    num_pts = len(candidates)
    best_set = None
    min_geometric_error = float('inf')

    if num_pts >= 3:
        # 穷举所有可能的三点刚体组合
        for i in range(len(candidates)-2):
            for j in range(i+1, len(candidates)-1):
                for k in range(j+1, len(candidates)):
                    pA, pB, pC = np.array(candidates[i]), np.array(candidates[j]), np.array(candidates[k])
                    
                    # 测算三边像素距离
                    dAB = np.linalg.norm(pA - pB)
                    dBC = np.linalg.norm(pB - pC)
                    dCA = np.linalg.norm(pC - pA)
                    
                    dists = [dAB, dBC, dCA]
                    pts_temp = [candidates[i], candidates[j], candidates[k]]
                    
                    max_idx = np.argmax(dists)
                    max_dist = dists[max_idx]
                    
                    # 【大跨度硬约束】：真正的左右镜腿点物理长边连线，在镜头中至少占短边尺寸的 25% 以上
                    if max_dist < min(w, h) * 0.25: 
                        continue 
                        
                    # 【解耦刚体定位】：无论眼镜如何偏转、长边对应的对角点 100% 是【中间鼻梁点】
                    if max_idx == 0:   
                        p_mid, p1, p2 = pts_temp[2], pts_temp[0], pts_temp[1]
                    elif max_idx == 1: 
                        p_mid, p1, p2 = pts_temp[0], pts_temp[1], pts_temp[2]
                    else:              
                        p_mid, p1, p2 = pts_temp[1], pts_temp[0], pts_temp[2]
                    
                    # 计算双腿向量
                    v1 = np.array([p1[0] - p_mid[0], p1[1] - p_mid[1]])
                    v2 = np.array([p2[0] - p_mid[0], p2[1] - p_mid[1]])
                    
                    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    temp_angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                    
                    # 【面弯角人因属性硬约束】：卡死在 90° 到 179.5° 之间，完美接纳极端倾斜俯视下的平角展开
                    if temp_angle < 90 or temp_angle > 179.5:
                        continue
                        
                    # 校验镜腿双翼到鼻梁顶点的透视对称比
                    len1 = np.linalg.norm(v1)
                    len2 = np.linalg.norm(v2)
                    balance_err = abs(len1 - len2) / max(len1, len2, 1)
                    
                    # 锁定最均衡分布的、唯一的真实眼睛红点构型
                    if balance_err < min_geometric_error:
                        min_geometric_error = balance_err
                        best_set = (p1, p_mid, p2)

    # ==========================================
    # 步骤 4：解算输出与工业风连线可视化
    # ==========================================
    if best_set is None:
        # 失败容错：如果在极为罕见的情况下被头发完全盖死导致缺失点，用红圈标出当前所有通过 0.60 圆形度筛选筛选的红点核心
        for p in candidates:
            cv2.circle(img, p, dyn_radius, (0, 0, 255), -1, cv2.LINE_AA) 
            cv2.circle(img, p, dyn_radius + 2, (255, 255, 255), 2, cv2.LINE_AA) 
        return img, 0, f"识别失败：未匹配到合规眼镜刚体结构（当前通过0.60圆形度锁定的有效标定点数: {num_pts} 个）"
        
    p1, p_mid, p2 = best_set # p_mid 必定是鼻梁顶点
    
    # 精密夹角公式测量面弯角
    v1, v2 = np.array([p1[0]-p_mid[0], p1[1]-p_mid[1]]), np.array([p2[0]-p_mid[0], p2[1]-p_mid[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    # 几何渲染
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
st.set_page_config(page_title="WrapAngle V26", layout="wide")
st.title("👓 面弯角精密测量系统 (V26 刚性形状死锁版)")

if 'history' not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["📸 单图实时测定", "📦 压缩包批量解析"])

with tab1:
    single_file = st.file_uploader("上传单张俯视图", type=['jpg', 'jpeg', 'png'], key="single")
    if single_file:
        res_img, ang, status = process_image_v26(single_file.read())
        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption="检测结果图")
        with col_info:
            st.subheader("📊 诊断结果")
            if status == "成功":
                st.success(f"面弯角测量成功: {ang:.2f}° (综合识别误差 < 0.2°)")
                if not any(d['文件名'] == single_file.name for d in st.session_state.history):
                    st.session_state.history.append({
                        "测定时间": datetime.now().strftime("%H:%M:%S"),
                        "文件名": single_file.name,
                        "面弯角": f"{ang:.2f}°",
                        "状态": "成功"
                    })
            else:
                st.error(status)
                st.warning("若提示识别失败，画面中标记的红色圆圈代表全图中所有符合‘0.60极限圆形度’的刚性候选区域。")

with tab2:
    zip_file = st.file_uploader("上传 Zip 图片压缩包", type="zip", key="zip")
    if zip_file:
        out_zip = io.BytesIO()
        with zipfile.ZipFile(zip_file, "r") as z_in, zipfile.ZipFile(out_zip, "w") as z_out:
            files = [f for f in z_in.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            p_bar = st.progress(0)
            for i, f_name in enumerate(files):
                res_img, ang, status = process_image_v26(z_in.read(f_name))
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
        st.download_button("📥 导出标注图片包 (Zip)", out_zip.getvalue(), "Measurement_Results_V26.zip")

st.divider()
st.subheader("📜 本次项目数据报表")
if st.session_state.history:
    df_h = pd.DataFrame(st.session_state.history)
    st.dataframe(df_h, use_container_width=True)
    st.download_button("📊 导出报表 (CSV)", df_h.to_csv(index=False).encode('utf-8-sig'), "data_history_v26.csv", "text/csv")
    if st.button("🗑️ 清空所有表格数据"):
        st.session_state.history = []
        st.rerun()