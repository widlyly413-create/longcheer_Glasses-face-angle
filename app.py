import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
import pandas as pd
from datetime import datetime

# --- 核心算法层：鲁棒黑框掩膜 + 强色差红点提取 + 几何拓扑过滤 ---
def process_image_v16(image_bytes):
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
    # 修正 1：鲁棒提取黑色眼镜框（调大V上限以容忍高光和反光）
    # ==========================================
    # 将亮度上限从 75 提升至 110，确保反射白光的灰色镜腿也能被识别进来
    lower_black, upper_black = np.array([0, 0, 0]), np.array([180, 255, 110]) 
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # 闭运算连接断裂的镜框反光部分
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel_close)
    
    # 提取外轮廓
    contours_black, _ = cv2.findContours(black_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    black_filled = np.zeros_like(black_mask)
    for cb in contours_black:
        # 适当降低面积阈值，防止镜框断裂时无法填满
        if cv2.contourArea(cb) > 500: 
            cv2.drawContours(black_filled, [cb], -1, 255, thickness=cv2.FILLED)
            
    # 适当加大膨胀半径，给镜框外侧的红点留足容错空间
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))
    black_filled = cv2.dilate(black_filled, kernel_dilate, iterations=1)

    # ==========================================
    # 修正 2：高纯度红色提取（收紧S和V，引入纯色差过滤）
    # ==========================================
    # 提高最低饱和度（90->130）和最低亮度（60->90），彻底剔除暗沉的皮肤泛红和阴影
    lower_red1, upper_red1 = np.array([0, 130, 90]), np.array([12, 255, 255])
    lower_red2, upper_red2 = np.array([165, 130, 90]), np.array([180, 255, 255])
    red_mask_hsv = cv2.add(cv2.inRange(hsv, lower_red1, upper_red1), 
                           cv2.inRange(hsv, lower_red2, upper_red2))

    # 引入 R-G 强色差辅助滤镜：纯红点的 R 通道远大于 G 通道，而皮肤的 R 和 G 靠得很近
    b, g, r = cv2.split(img)
    rg_diff = cv2.subtract(r, g)
    _, red_mask_diff = cv2.threshold(rg_diff, 60, 255, cv2.THRESH_BINARY)

    # 取双重红光滤镜的交集
    red_mask = cv2.bitwise_and(red_mask_hsv, red_mask_diff)

    # ==========================================
    # 步骤 3：掩膜裁剪与形态学去噪
    # ==========================================
    red_on_black = cv2.bitwise_and(red_mask, black_filled)
    
    kernel_size = 3 if w < 1500 else 5
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    red_cleaned = cv2.morphologyEx(red_on_black, cv2.MORPH_OPEN, kernel_open)

    # ==========================================
    # 修正 3：放宽形态学筛选与横向区域限制
    # ==========================================
    contours_red, _ = cv2.findContours(red_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    
    for cnt in contours_red:
        area = cv2.contourArea(cnt)
        # 圆点面积通常不会太大，卡死在 40~2000
        if 30 < area < 2000:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            # 真实图像中由于俯视角度，圆点可能发生轻微透视形变，将圆形度阈值稍微放宽至 0.4
            if circularity >= 0.40:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / bh
                if 0.5 < aspect_ratio < 1.8:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        
                        # 【重要修正】：放开横向限制至 15% - 85%，确保两侧镜腿的红点不被漏掉
                        if 0.15 * w < cX < 0.85 * w:
                            centers.append((cX, cY))

    # ==========================================
    # 修正 4：稳健的后验几何拓扑匹配（当备选点 > 3 时）
    # ==========================================
    num_pts = len(centers)
    best_set = None

    if num_pts == 3:
        # 刚好3个点，直接按照 Y 轴（从上到下镜腿顺序）排序
        best_set = sorted(centers, key=lambda x: x[1])
    elif num_pts > 3:
        # 如果受到其他杂点干扰导致候选点变多，利用眼镜框特征进行拓扑筛选：
        # 正确的三个点应该满足：其中两个点（左右镜腿）在上方/下方，一个点（鼻梁）在中间，且X坐标应该有明显的左、中、右区分度。
        min_topology_score = float('inf')
        
        for i in range(len(centers)-2):
            for j in range(i+1, len(centers)-1):
                for k in range(j+1, len(centers)):
                    tri_pts = [centers[i], centers[j], centers[k]]
                    # 按 Y 坐标排序（即俯视图中从屏幕上方到下方的顺序）
                    tri_pts = sorted(tri_pts, key=lambda x: x[1])
                    p_top, p_mid, p_bot = tri_pts
                    
                    # 几何约束1：中间的点（鼻梁点）的 X 坐标，理论上应该介于上下两点 X 坐标之间，或非常接近中轴线
                    x_coords = [p_top[0], p_mid[0], p_bot[0]]
                    x_span = max(x_coords) - min(x_coords)
                    
                    # 几何约束2：真正的三点不应该水平挤在一堆，X轴应当有合理的跨度
                    if x_span < w * 0.2: 
                        continue 
                    
                    # 计算对称性得分：左右两点到中间点的距离应该相对接近
                    d1 = np.linalg.norm(np.array(p_top) - np.array(p_mid))
                    d2 = np.linalg.norm(np.array(p_bot) - np.array(p_mid))
                    symmetry_ratio = abs(d1 - d2) / max(d1, d2, 1)
                    
                    if symmetry_ratio < min_topology_score:
                        min_topology_score = symmetry_ratio
                        best_set = tri_pts

    # ==========================================
    # 步骤 5 & 6：结果绘制与输出
    # ==========================================
    if best_set is None or len(best_set) != 3:
        # 失败容错绘制
        for p in centers:
            cv2.circle(img, p, dyn_radius, (0, 0, 255), -1, cv2.LINE_AA) 
            cv2.circle(img, p, dyn_radius + 2, (255, 255, 255), 2, cv2.LINE_AA) 
            
        fail_text = f"FAIL: Found {num_pts} candidates, but geometric matching failed"
        cv2.putText(img, "FAIL", (int(w*0.05), int(h*0.1)), font, dyn_font_scale, (0, 0, 0), dyn_font_thick+2, cv2.LINE_AA)
        cv2.putText(img, "FAIL", (int(w*0.05), int(h*0.1)), font, dyn_font_scale, (0, 0, 255), dyn_font_thick, cv2.LINE_AA)
        return img, 0, f"识别失败：无法匹配合规的三角形拓扑（当前候选点数: {num_pts}）"
        
    # 成功匹配
    p1, p2, p3 = best_set # 分别对应俯视图中：上镜腿点、中部鼻梁点、下镜腿点
    
    # 计算夹角（以鼻梁点 p2 为顶点）
    v1, v2 = np.array([p1[0]-p2[0], p1[1]-p2[1]]), np.array([p3[0]-p2[0], p3[1]-p2[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    # 绘制连线与标定圆点
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

# --- Streamlit UI 层 (保持不变) ---
st.set_page_config(page_title="WrapAngle V16", layout="wide")
st.title("👓 面弯角测量系统 (V16 外部轮廓包裹版)")

if 'history' not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["📸 单图实时测定", "📦 压缩包批量解析"])

with tab1:
    single_file = st.file_uploader("上传单张俯视图", type=['jpg', 'jpeg', 'png'], key="single")
    if single_file:
        res_img, ang, status = process_image_v16(single_file.read())
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
                st.warning("若出现失败，已标记部分提取出的红点，辅助判断缺失原因。")

with tab2:
    zip_file = st.file_uploader("上传 Zip 图片压缩包", type="zip", key="zip")
    if zip_file:
        out_zip = io.BytesIO()
        with zipfile.ZipFile(zip_file, "r") as z_in, zipfile.ZipFile(out_zip, "w") as z_out:
            files = [f for f in z_in.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            p_bar = st.progress(0)
            for i, f_name in enumerate(files):
                res_img, ang, status = process_image_v16(z_in.read(f_name))
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
        st.download_button("📥 导出标注图片包 (Zip)", out_zip.getvalue(), "Measurement_Results_V16.zip")

st.divider()
st.subheader("📜 本次项目数据报表")
if st.session_state.history:
    df_h = pd.DataFrame(st.session_state.history)
    st.dataframe(df_h, use_container_width=True)
    st.download_button("📊 导出报表 (CSV)", df_h.to_csv(index=False).encode('utf-8-sig'), "data_history_v16.csv", "text/csv")
    if st.button("🗑️ 清空所有表格数据"):
        st.session_state.history = []
        st.rerun()