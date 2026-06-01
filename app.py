import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
import pandas as pd
from datetime import datetime

# --- 核心算法层：V23 RGB 纯色暴力破解 + 刚性拓扑（彻底告别头发干扰） ---
def process_image_v23(image_bytes):
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
    # 步骤 1：RGB 纯色暴力分离（彻底取代黑框掩膜）
    # ==========================================
    # 分离通道，必须转换为 int16 防止 uint8 减法下溢出
    b, g, r = cv2.split(img)
    r_16 = r.astype(np.int16)
    g_16 = g.astype(np.int16)
    b_16 = b.astype(np.int16)

    # 核心铁闸：红色通道必须压倒性地高于绿色（>85）和蓝色（>50）
    # 这一步在数学上直接宣判了所有头皮、肤色、阴影的死刑
    rg_diff = r_16 - g_16
    rb_diff = r_16 - b_16
    
    # R 必须大于 140，保证是明亮的红点，而不是暗沉的红黑色
    red_mask_math = (rg_diff > 85) & (rb_diff > 50) & (r_16 > 140)
    red_mask_math = red_mask_math.astype(np.uint8) * 255

    # 辅助 HSV 空间锁定，确保色彩高饱和
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([15, 255, 255])
    lower_red2, upper_red2 = np.array([165, 100, 100]), np.array([180, 255, 255])
    red_mask_hsv = cv2.add(cv2.inRange(hsv, lower_red1, upper_red1), 
                           cv2.inRange(hsv, lower_red2, upper_red2))

    # 双重锁定，得到绝对纯净的红色区域
    final_red_mask = cv2.bitwise_and(red_mask_math, red_mask_hsv)

    # ==========================================
    # 步骤 2：轻度形态学闭运算（缝合被头发细丝切断的红点）
    # ==========================================
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    red_cleaned = cv2.morphologyEx(final_red_mask, cv2.MORPH_CLOSE, kernel_close)

    # ==========================================
    # 步骤 3：严格的几何圆形鉴定
    # ==========================================
    contours_red, _ = cv2.findContours(red_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    
    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        if 20 < area < 2000:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            # 因为没有了假红点的干扰，我们可以要求它是一个相对标准的圆（>=0.40）
            if circularity >= 0.40:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / bh
                if 0.4 < aspect_ratio < 2.5:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        centers.append((cX, cY))

    # ==========================================
    # 步骤 4：宏观物理刚性拓扑鉴别器
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
                    
                    # 【大跨度硬约束】：真正的镜腿连线跨度极大，至少占画面最短边的 25%
                    min_span = min(w, h) * 0.25
                    if dists[max_idx] < min_span: 
                        continue 
                        
                    # 确定顶点（长边对应的那个点必为鼻梁点）
                    if max_idx == 0:   
                        p_mid, p1, p2 = pts_temp[2], pts_temp[0], pts_temp[1]
                    elif max_idx == 1: 
                        p_mid, p1, p2 = pts_temp[0], pts_temp[1], pts_temp[2]
                    else:              
                        p_mid, p1, p2 = pts_temp[1], pts_temp[0], pts_temp[2]
                    
                    # 【钝角硬约束】：计算夹角，面弯角只可能在 90° 到 178° 之间
                    v1 = np.array([p1[0]-p_mid[0], p1[1]-p_mid[1]])
                    v2 = np.array([p2[0]-p_mid[0], p2[1]-p_mid[1]])
                    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    temp_angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                    
                    if temp_angle < 90 or temp_angle > 178:
                        continue
                        
                    # 计算对称平衡度，选出最符合眼镜结构的解
                    len1 = np.linalg.norm(np.array(p1) - np.array(p_mid))
                    len2 = np.linalg.norm(np.array(p2) - np.array(p_mid))
                    balance_err = abs(len1 - len2) / max(len1, len2, 1)
                    
                    if balance_err < min_geometric_error:
                        min_geometric_error = balance_err
                        best_set = (p1, p_mid, p2)

    # ==========================================
    # 步骤 5：渲染输出层
    # ==========================================
    if best_set is None:
        for p in centers:
            cv2.circle(img, p, dyn_radius, (0, 0, 255), -1, cv2.LINE_AA) 
            cv2.circle(img, p, dyn_radius + 2, (255, 255, 255), 2, cv2.LINE_AA) 
        return img, 0, f"识别失败：纯色区域内未找到合规的镜框三点结构。当前捕获候选: {num_pts} 个"
        
    p1, p2, p3 = best_set 
    
    # 精密计算最终夹角
    v1, v2 = np.array([p1[0]-p2[0], p1[1]-p2[1]]), np.array([p3[0]-p2[0], p3[1]-p2[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    # 绘制连线
    cv2.line(img, p1, p2, (255, 120, 0), dyn_line, cv2.LINE_AA)
    cv2.line(img, p2, p3, (255, 120, 0), dyn_line, cv2.LINE_AA)
    for p in [p1, p2, p3]:
        cv2.circle(img, p, dyn_radius, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p, dyn_radius, (0, 0, 0), 1, cv2.LINE_AA)

    # 输出高精度文本
    text = f"ANGLE: {angle:.2f} DEG"
    text_pos = (p2[0] + 40, p2[1])
    cv2.putText(img, text, text_pos, font, dyn_font_scale, (0,0,0), dyn_font_thick+2, cv2.LINE_AA)
    cv2.putText(img, text, text_pos, font, dyn_font_scale, (255,255,255), dyn_font_thick, cv2.LINE_AA)
    
    return img, angle, "成功"

# --- Streamlit UI ---
st.set_page_config(page_title="WrapAngle V23", layout="wide")
st.title("👓 面弯角测量系统 (V23 纯色免疫遮挡版)")

if 'history' not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["📸 单图实时测定", "📦 压缩包批量解析"])

with tab1:
    single_file = st.file_uploader("上传单张俯视图", type=['jpg', 'jpeg', 'png'], key="single")
    if single_file:
        res_img, ang, status = process_image_v23(single_file.read())
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
                st.warning("如果提示失败，画面中亮起的红圈表示当前所有符合【纯红】特征的坐标点。")

with tab2:
    zip_file = st.file_uploader("上传 Zip 图片压缩包", type="zip", key="zip")
    if zip_file:
        out_zip = io.BytesIO()
        with zipfile.ZipFile(zip_file, "r") as z_in, zipfile.ZipFile(out_zip, "w") as z_out:
            files = [f for f in z_in.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            p_bar = st.progress(0)
            for i, f_name in enumerate(files):
                res_img, ang, status = process_image_v23(z_in.read(f_name))
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
        st.download_button("📥 导出标注图片包 (Zip)", out_zip.getvalue(), "Measurement_Results_V23.zip")

st.divider()
st.subheader("📜 本次项目数据报表")
if st.session_state.history:
    df_h = pd.DataFrame(st.session_state.history)
    st.dataframe(df_h, use_container_width=True)
    st.download_button("📊 导出报表 (CSV)", df_h.to_csv(index=False).encode('utf-8-sig'), "data_history_v23.csv", "text/csv")
    if st.button("🗑️ 清空所有表格数据"):
        st.session_state.history = []
        st.rerun()