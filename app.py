import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
import pandas as pd
from datetime import datetime

# --- 核心算法层：鲁棒黑框掩膜 + 强色差红点提取 + 几何拓扑过滤 ---
def process_image_v17(image_bytes):
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
    # 步骤 1：鲁棒提取黑色眼镜框（高容错率，对抗高光反光）
    # ==========================================
    # 提高亮度上限（110）并引入闭运算，防止镜框反光导致断裂
    lower_black, upper_black = np.array([0, 0, 0]), np.array([180, 255, 115]) 
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    kernel_close_frame = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel_close_frame)
    
    contours_black, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    black_filled = np.zeros_like(black_mask)
    for cb in contours_black:
        # 降低面积门槛（500），保证局部反光断裂的镜框段也能被包容
        if cv2.contourArea(cb) > 400: 
            cv2.drawContours(black_filled, [cb], -1, 255, thickness=cv2.FILLED)
            
    # 强力膨胀，确保因歪头或形变稍微移出镜框边缘的红点也能被包进掩膜中
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    black_filled = cv2.dilate(black_filled, kernel_dilate, iterations=1)

    # ==========================================
    # 步骤 2：强色差纯红提取（无视皮肤红斑与暗红压痕）
    # ==========================================
    # 过滤掉低饱和度的肉色
    lower_red1, upper_red1 = np.array([0, 130, 90]), np.array([15, 255, 255])
    lower_red2, upper_red2 = np.array([165, 130, 90]), np.array([180, 255, 255])
    red_mask_hsv = cv2.add(cv2.inRange(hsv, lower_red1, upper_red1), 
                           cv2.inRange(hsv, lower_red2, upper_red2))

    # 核心抗干扰：红绿通道差值滤镜（纯红点 R 远大于 G，而皮肤 R 和 G 很接近）
    b, g, r = cv2.split(img)
    rg_diff = cv2.subtract(r, g)
    _, red_mask_diff = cv2.threshold(rg_diff, 55, 255, cv2.THRESH_BINARY)

    # 交集运算锁定高纯度红色
    red_mask = cv2.bitwise_and(red_mask_hsv, red_mask_diff)

    # ==========================================
    # 步骤 3：骨架裁剪与发丝缝希粘合（核心进化）
    # ==========================================
    red_on_black = cv2.bitwise_and(red_mask, black_filled)
    
    # 【改动】：从原本的开运算（擦除）改为闭运算（形态学缝合）
    # 如果有细小头发丝从红点中间穿过，该操作可以强行把被切开的红点“重新缝合粘连”
    kernel_repair = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    red_cleaned = cv2.morphologyEx(red_on_black, cv2.MORPH_CLOSE, kernel_repair)

    # ==========================================
    # 步骤 4：极限放宽特征筛选（容忍严重的透视和破损）
    # ==========================================
    contours_red, _ = cv2.findContours(red_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    
    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        # 放低面积下限，被切损后变小的红点也能进
        if 25 < area < 2000:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            # 【极限放宽】：圆形度下调至 0.15（极度不规则、被半遮挡的红点也能通过）
            if circularity >= 0.15:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / bh
                # 放宽宽高比，包容倾斜视角的压扁椭圆
                if 0.3 < aspect_ratio < 2.6:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        # 放开横向限制，确保歪头导致红点靠边时不会被漏
                        if 0.12 * w < cX < 0.88 * w:
                            centers.append((cX, cY))

    # ==========================================
    # 步骤 5：自适应几何拓扑选择器（对抗歪头和杂点）
    # ==========================================
    num_pts = len(centers)
    best_set = None

    if num_pts == 3:
        # 刚好3个，直接按 Y 坐标（俯视从上到下镜腿顺序）分层
        best_set = sorted(centers, key=lambda x: x[1])
    elif num_pts > 3:
        # 当存在多个候选点时，利用眼镜框特征进行稳健几何空间组合筛选
        min_score = float('inf')
        
        for i in range(len(centers)-2):
            for j in range(i+1, len(centers)-1):
                for k in range(j+1, len(centers)):
                    tri_pts = [centers[i], centers[j], centers[k]]
                    # 按照 Y 坐标排序：分为【上镜腿点】、【中部鼻梁点】、【下镜腿点】
                    tri_pts = sorted(tri_pts, key=lambda x: x[1])
                    p_top, p_mid, p_bot = tri_pts
                    
                    # 空间几何约束 1：真正的三个标定点在 X 轴上必须具有合理的总体跨度
                    x_coords = [p_top[0], p_mid[0], p_bot[0]]
                    x_span = max(x_coords) - min(x_coords)
                    if x_span < w * 0.25: 
                        continue 
                    
                    # 空间几何约束 2：中间的鼻梁点 X 坐标不能过于偏离两镜腿的连线中点
                    # 综合计算：歪头下的相对对称得分
                    d1 = np.linalg.norm(np.array(p_top) - np.array(p_mid))
                    d2 = np.linalg.norm(np.array(p_bot) - np.array(p_mid))
                    
                    # 比例得分，越接近 0 说明两边镜腿跨度越均衡（即使整体歪斜）
                    balance_score = abs(d1 - d2) / max(d1, d2, 1)
                    
                    if balance_score < min_score:
                        min_score = balance_score
                        best_set = tri_pts

    # ==========================================
    # 步骤 6：可视化绘制与输出层
    # ==========================================
    if best_set is None or len(best_set) != 3:
        # 失败时把所有备选点标出来，方便科研复盘
        for p in centers:
            cv2.circle(img, p, dyn_radius, (0, 0, 255), -1, cv2.LINE_AA) 
            cv2.circle(img, p, dyn_radius + 2, (255, 255, 255), 2, cv2.LINE_AA) 
            
        fail_text = f"FAIL: Found {num_pts} candidates, structural matching blocked."
        cv2.putText(img, "FAIL", (int(w*0.05), int(h*0.1)), font, dyn_font_scale, (0, 0, 0), dyn_font_thick+2, cv2.LINE_AA)
        cv2.putText(img, "FAIL", (int(w*0.05), int(h*0.1)), font, dyn_font_scale, (0, 0, 255), dyn_font_thick, cv2.LINE_AA)
        return img, 0, f"识别失败：未匹配出符合镜框结构的3点组（检测到候选点: {num_pts}）"
        
    # 成功捕获 3 点
    p1, p2, p3 = best_set # p1:上镜腿, p2:中间鼻梁顶点, p3:下镜腿
    
    # 向量法精准计算面弯角夹角（顶点为鼻梁中心 p2）
    v1, v2 = np.array([p1[0]-p2[0], p1[1]-p2[1]]), np.array([p3[0]-p2[0], p3[1]-p2[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    # 工业风画线与高亮标定
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

# --- Streamlit UI 层 ---
st.set_page_config(page_title="WrapAngle V17", layout="wide")
st.title("👓 面弯角测量系统 (V17 缝合增强版)")

if 'history' not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["📸 单图实时测定", "📦 压缩包批量解析"])

with tab1:
    single_file = st.file_uploader("上传单张俯视图", type=['jpg', 'jpeg', 'png'], key="single")
    if single_file:
        res_img, ang, status = process_image_v17(single_file.read())
        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption="检测结果图")
        with col_info:
            st.subheader("📊 诊断结果")
            if status == "成功":
                st.success(f"面弯角: {ang:.2f}°")
                if not any(d['文件名'] == single_file.name for d in st.session_state.history):
                    st.session_state.history.append({
                        "测定时间": datetime.now().strftime("%H:%M:%S"),
                        "文件名": single_file.name,
                        "面弯角": f"{ang:.2f}°",
                        "状态": "成功"
                    })
            else:
                st.error(status)
                st.warning("若出现失败，系统已用红圈强制标出图像中所有符合纯红特征的候选点，您可以根据图像排查是否有极端遮挡。")

with tab2:
    zip_file = st.file_uploader("上传 Zip 图片压缩包", type="zip", key="zip")
    if zip_file:
        out_zip = io.BytesIO()
        with zipfile.ZipFile(zip_file, "r") as z_in, zipfile.ZipFile(out_zip, "w") as z_out:
            files = [f for f in z_in.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            p_bar = st.progress(0)
            for i, f_name in enumerate(files):
                res_img, ang, status = process_image_v17(z_in.read(f_name))
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
        st.download_button("📥 导出标注图片包 (Zip)", out_zip.getvalue(), "Measurement_Results_V17.zip")

st.divider()
st.subheader("📜 本次项目数据报表")
if st.session_state.history:
    df_h = pd.DataFrame(st.session_state.history)
    st.dataframe(df_h, use_container_width=True)
    st.download_button("📊 导出报表 (CSV)", df_h.to_csv(index=False).encode('utf-8-sig'), "data_history_v17.csv", "text/csv")
    if st.button("🗑️ 清空所有表格数据"):
        st.session_state.history = []
        st.rerun()
