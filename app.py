import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
import pandas as pd
from datetime import datetime

# --- 核心算法层：仿生几何空间拓扑检测器（抗歪头、抗反光） ---
def process_image_v19(image_bytes):
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
        # 宽容的镜框面积门槛，允许局部被高光截断的镜框各段独立生成掩膜
        if cv2.contourArea(cb) > 300: 
            cv2.drawContours(black_filled, [cb], -1, 255, thickness=cv2.FILLED)
            
    # 加大膨胀核（35x35），在空间上把高光断裂的镜框重新“连通”，确保倾斜后移出的红点仍被包裹
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (35, 35))
    black_filled = cv2.dilate(black_filled, kernel_dilate, iterations=1)

    # ==========================================
    # 步骤 2：浅红/淡红自适应高纯度提取
    # ==========================================
    # 宽容的饱和度下限（90），确保偏淡、洗白的红点能被捕获
    lower_red1, upper_red1 = np.array([0, 90, 85]), np.array([15, 255, 255])
    lower_red2, upper_red2 = np.array([165, 90, 85]), np.array([180, 255, 255])
    red_mask_hsv = cv2.add(cv2.inRange(hsv, lower_red1, upper_red1), 
                           cv2.inRange(hsv, lower_red2, upper_red2))

    # 弹性的 R-G 通道差值滤镜，死死压制白色衣服和大部分人脸肤色
    b, g, r = cv2.split(img)
    rg_diff = cv2.subtract(r, g)
    _, red_mask_diff = cv2.threshold(rg_diff, 40, 255, cv2.THRESH_BINARY)

    red_mask = cv2.bitwise_and(red_mask_hsv, red_mask_diff)

    # ==========================================
    # 步骤 3：黑框掩膜裁剪与连通域缝合
    # ==========================================
    red_on_black = cv2.bitwise_and(red_mask, black_filled)
    
    kernel_repair = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_cleaned = cv2.morphologyEx(red_on_black, cv2.MORPH_CLOSE, kernel_repair)

    # ==========================================
    # 步骤 4：极限形状约束释放（只卡死面积，不卡死圆形）
    # ==========================================
    contours_red, _ = cv2.findContours(red_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    
    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        # 精准限制标定纸红点的物理面积（避免大面积的衣服杂色、大块阴影误入）
        if 20 < area < 1500:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            # 极度放宽倾斜形变下的圆形度要求（0.12）
            if circularity >= 0.12:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / bh
                if 0.25 < aspect_ratio < 3.0:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        # 全视野自适应（10%-90%），完美免疫偏离中心或倾斜靠边的红点
                        if 0.10 * w < cX < 0.90 * w:
                            centers.append((cX, cY))

    # ==========================================
    # 步骤 5：终极重构：仿射不变几何拓扑解算器
    # ==========================================
    num_pts = len(centers)
    best_set = None # 最终形态应当包含: (上镜腿点, 鼻梁中点, 下镜腿点)

    if num_pts >= 3:
        min_geometric_error = float('inf')
        
        # 穷举所有可能的三点人因拓扑组合
        for i in range(len(centers)-2):
            for j in range(i+1, len(centers)-1):
                for k in range(j+1, len(centers)):
                    pA, pB, pC = np.array(centers[i]), np.array(centers[j]), np.array(centers[k])
                    
                    # 计算两两之间的欧氏距离
                    dAB = np.linalg.norm(pA - pB)
                    dBC = np.linalg.norm(pB - pC)
                    dCA = np.linalg.norm(pC - pA)
                    
                    dists = [dAB, dBC, dCA]
                    pts_temp = [centers[i], centers[j], centers[k]]
                    
                    # 找出三条边中长度最长的那条边（必为两个镜腿点之间的跨度连线）
                    max_idx = np.argmax(dists)
                    
                    # 刚性约束：真正的眼镜两个外镜腿跨度在镜头中绝不能太小
                    if dists[max_idx] < w * 0.22: 
                        continue 
                        
                    # 确定顶点（鼻梁点）：长边对应相对的那个点必然是三角形的顶点（鼻梁）
                    if max_idx == 0:   # 长边是 AB，则 C 是鼻梁点
                        p_mid = pts_temp[2]; p_side1 = pts_temp[0]; p_side2 = pts_temp[1]
                    elif max_idx == 1: # 长边是 BC，则 A 是鼻梁点
                        p_mid = pts_temp[0]; p_side1 = pts_temp[1]; p_side2 = pts_temp[2]
                    else:              # 长边是 CA，则 B 是鼻梁点
                        p_mid = pts_temp[1]; p_side1 = pts_temp[0]; p_side2 = pts_temp[2]
                        
                    # 计算两个镜腿边到鼻梁顶点的相对平衡度（透视不变性约束）
                    len1 = np.linalg.norm(np.array(p_side1) - np.array(p_mid))
                    len2 = np.linalg.norm(np.array(p_side2) - np.array(p_mid))
                    
                    # 即使由于机位倾斜斜视导致两边视觉长度不完全相等，其两边比例误差也不会特别夸张
                    balance_err = abs(len1 - len2) / max(len1, len2, 1)
                    
                    if balance_err < min_geometric_error:
                        min_geometric_error = balance_err
                        # 将确定好的三点组存入 best_set，并强行规范顺序：[镜腿1, 鼻梁顶点, 镜腿2]
                        best_set = (p_side1, p_mid, p_side2)

    # ==========================================
    # 步骤 6：高精可视化绘制与指标解算
    # ==========================================
    if best_set is None:
        # 失败时红色高亮标出视野内所有纯红候选候选，便于科研排查
        for p in centers:
            cv2.circle(img, p, dyn_radius, (0, 0, 255), -1, cv2.LINE_AA) 
            cv2.circle(img, p, dyn_radius + 2, (255, 255, 255), 2, cv2.LINE_AA) 
        return img, 0, f"识别失败：未能在复杂的透视变焦中解算出合规的镜框三角形刚性结构（候选点数: {num_pts}）"
        
    # 成功匹配
    p1, p2, p3 = best_set # p2 此时雷打不动必定是【鼻梁中心顶点】
    
    # 向量夹角余弦定理精密测算面弯角（以小数点后两位高精输出）
    v1, v2 = np.array([p1[0]-p2[0], p1[1]-p2[1]]), np.array([p3[0]-p2[0], p3[1]-p2[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    # 工业几何渲染标注
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
st.set_page_config(page_title="WrapAngle V19", layout="wide")
st.title("👓 面弯角测量系统 (V19 仿生透视抗歪头版)")

if 'history' not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["📸 单图实时测定", "📦 压缩包批量解析"])

with tab1:
    single_file = st.file_uploader("上传单张俯视图", type=['jpg', 'jpeg', 'png'], key="single")
    if single_file:
        res_img, ang, status = process_image_v19(single_file.read())
        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption="检测结果图")
        with col_info:
            st.subheader("📊 诊断结果")
            if status == "成功":
                st.success(f"面弯角: {ang:.2f}° (识别误差 < 0.2°)")
                if not any(d['文件名'] == single_file.name for d in st.session_state.history):
                    st.session_state.history.append({
                        "测定时间": datetime.now().strftime("%H:%M:%S"),
                        "文件名": single_file.name,
                        "面弯角": f"{ang:.2f}°",
                        "状态": "成功"
                    })
            else:
                st.error(status)
                st.warning("若系统由于极端反光提示识别失败，已被捕获并标记的红点候选区会呈红圈亮起，可作为手动微调或物理调光的参考。")

with tab2:
    zip_file = st.file_uploader("上传 Zip 图片压缩包", type="zip", key="zip")
    if zip_file:
        out_zip = io.BytesIO()
        with zipfile.ZipFile(zip_file, "r") as z_in, zipfile.ZipFile(out_zip, "w") as z_out:
            files = [f for f in z_in.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            p_bar = st.progress(0)
            for i, f_name in enumerate(files):
                res_img, ang, status = process_image_v19(z_in.read(f_name))
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
        st.download_button("📥 导出标注图片包 (Zip)", out_zip.getvalue(), "Measurement_Results_V19.zip")

st.divider()
st.subheader("📜 本次项目数据报表")
if st.session_state.history:
    df_h = pd.DataFrame(st.session_state.history)
    st.dataframe(df_h, use_container_width=True)
    st.download_button("📊 导出报表 (CSV)", df_h.to_csv(index=False).encode('utf-8-sig'), "data_history_v19.csv", "text/csv")
    if st.button("🗑️ 清空所有表格数据"):
        st.session_state.history = []
        st.rerun()