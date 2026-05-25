import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
import pandas as pd
from datetime import datetime

# --- 核心算法层：自适应内核 + 无区域限制 + 垂直对齐过滤 ---
def process_image_v12(image_bytes):
    # 1. 图像解码 (支持中文路径与二进制流)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: 
        return None, 0, "文件读取失败"
    
    h, w = img.shape[:2]
    
    # --- 动态比例因子 (基于图片宽度的精细化标注) ---
    dyn_radius = max(3, int(w / 180))      
    dyn_line = max(1, int(w / 700))        
    dyn_font_scale = w / 1600              
    dyn_font_thick = max(1, int(w / 900))  

    # 2. 图像颜色空间转换
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # ==========================================
    # 核心步骤 1：提取黑色镜框安全区域
    # ==========================================
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 65]) 
    black_mask = cv2.inRange(hsv, lower_black, upper_black)
    
    # 对黑色镜框区域进行轻微膨胀，包裹住贴在边缘的红点
    kernel_dilate = np.ones((7, 7), np.uint8)
    black_mask_expanded = cv2.dilate(black_mask, kernel_dilate, iterations=1)

    # ==========================================
    # 核心步骤 2：提取红色贴点 (严格匹配验证过的阈值)
    # ==========================================
    lower_red1, upper_red1 = np.array([0, 100, 70]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([170, 120, 100]), np.array([180, 255, 255])
    red_mask = cv2.add(cv2.inRange(hsv, lower_red1, upper_red1), 
                   cv2.inRange(hsv, lower_red2, upper_red2))
    
    # --- 【重大更新】：动态计算形态学开运算内核大小 ---
    # 解决低分辨率图片中，固定的 5x5 内核会把变小的红点当成噪点擦除的问题
    kernel_size = 3 if w < 1500 else 5
    kernel_open = np.ones((kernel_size, kernel_size), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel_open)

    # ==========================================
    # 核心步骤 3：蒙版融合 (黑域锚定，剔除皮肤干扰)
    # ==========================================
    final_mask = cv2.bitwise_and(red_mask, black_mask_expanded)

    # 3. 几何过滤
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centers = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        # 限制面积区间
        if 40 < area < 2500:
            perimeter = cv2.arcLength(cnt, True)
            if perimeter == 0: continue
            circularity = 4 * np.pi * (area / (perimeter * perimeter))
            
            # 圆度门槛
            if circularity >= 0.45:
                x, y, bw, bh = cv2.boundingRect(cnt)
                aspect_ratio = float(bw) / bh
                # 长宽比过滤
                if 0.6 < aspect_ratio < 1.5:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        # --- 【重大更新】：完全移除对 cX 的水平视场区域限制 ---
                        centers.append((cX, cY))

    # 4. 点数校验与垂直共线优选
    num_pts = len(centers)
    if num_pts < 3:
        return img, 0, f"识别失败：镜框区域内仅找到 {num_pts} 个符合条件的点"
    
    # 坐标按 Y 轴（上下）方向排序
    centers = sorted(centers, key=lambda x: x[1])
    
    best_set = None
    min_x_diff = float('inf')
    
    # 如果刚好3个点，直接取用；如果由于干扰产生4个点以上，找出 X 轴最排成一条直线的一组
    if len(centers) == 3:
        best_set = centers
    else:
        for i in range(len(centers)-2):
            for j in range(i+1, len(centers)-1):
                for k in range(j+1, len(centers)):
                    pts = [centers[i], centers[j], centers[k]]
                    # 计算这三点 X 轴的最大离散极差
                    x_range = max(p[0] for p in pts) - min(p[0] for p in pts)
                    if x_range < min_x_diff:
                        min_x_diff = x_range
                        best_set = pts
                        
    p1, p2, p3 = best_set
    
    # 5. 几何向量计算角度
    v1 = np.array([p1[0]-p2[0], p1[1]-p2[1]])
    v2 = np.array([p3[0]-p2[0], p3[1]-p2[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    # 6. 专业级极细标注绘制
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.line(img, p1, p2, (255, 120, 0), dyn_line, cv2.LINE_AA)
    cv2.line(img, p2, p3, (255, 120, 0), dyn_line, cv2.LINE_AA)
    
    for p in [p1, p2, p3]:
        cv2.circle(img, p, dyn_radius, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p, dyn_radius, (0, 0, 0), 1, cv2.LINE_AA)

    text = f"ANGLE: {angle:.2f} DEG"
    text_pos = (p2[0] + 40, p2[1])
    # 黑色文字阴影
    cv2.putText(img, text, text_pos, font, dyn_font_scale, (0,0,0), dyn_font_thick+2, cv2.LINE_AA)
    # 白色主字体
    cv2.putText(img, text, text_pos, font, dyn_font_scale, (255,255,255), dyn_font_thick, cv2.LINE_AA)
    
    return img, angle, "成功"

# --- Streamlit UI 交互控制层 ---
st.set_page_config(page_title="WrapAngle V12", layout="wide")
st.title("👓 面弯角高精度全自动标注系统 (V12 终极版)")
st.caption("当前版本已移除画面区域限制，完美自适应大图与微信压缩小图。")

if 'history' not in st.session_state:
    st.session_state.history = []

tab1, tab2 = st.tabs(["📸 单图实时上传检测", "📦 压缩包批量处理 (Zip)"])

# --- 选项卡 1：单图处理 ---
with tab1:
    single_file = st.file_uploader("请上传单张眼镜俯拍图", type=['jpg', 'jpeg', 'png'], key="single")
    if single_file:
        res_img, ang, status = process_image_v12(single_file.read())
        
        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(cv2.cvtColor(res_img, cv2.COLOR_BGR2RGB), caption="自适应精细化测量预览")
        with col_info:
            st.subheader("📋 诊断与数据")
            if status == "成功":
                st.success(f"计算面弯角: {ang:.2f}°")
                if not any(d['文件名'] == single_file.name for d in st.session_state.history):
                    st.session_state.history.append({
                        "操作时间": datetime.now().strftime("%H:%M:%S"),
                        "文件名": single_file.name,
                        "面弯角角度": f"{ang:.2f}°",
                        "诊断状态": "成功"
                    })
            else:
                st.error(status)

# --- 选项卡 2：批量 Zip 处理 ---
with tab2:
    zip_file = st.file_uploader("请上传包含图片的 Zip 压缩包", type="zip", key="zip")
    if zip_file:
        out_zip = io.BytesIO()
        with zipfile.ZipFile(zip_file, "r") as z_in, zipfile.ZipFile(out_zip, "w") as z_out:
            files = [f for f in z_in.namelist() if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            p_bar = st.progress(0)
            for i, f_name in enumerate(files):
                res_img, ang, status = process_image_v12(z_in.read(f_name))
                if res_img is not None:
                    _, buf = cv2.imencode(".jpg", res_img)
                    z_out.writestr(f"Result_{os.path.basename(f_name)}", buf.tobytes())
                    st.session_state.history.append({
                        "操作时间": datetime.now().strftime("%H:%M:%S"),
                        "文件名": os.path.basename(f_name),
                        "面弯角角度": f"{ang:.2f}°" if ang > 0 else "-",
                        "诊断状态": status
                    })
                p_bar.progress((i + 1) / len(files))
        st.success("批量图片处理完成！")
        st.download_button("📥 点击下载处理后的图片包 (Zip)", out_zip.getvalue(), "Batch_Results_V12.zip")

# --- 实验历史数据大盘 ---
st.divider()
st.subheader("📜 大漆工艺优化研究·操作历史记录")
if st.session_state.history:
    df_history = pd.DataFrame(st.session_state.history)
    st.dataframe(df_history, use_container_width=True)
    
    # 转换为 CSV
    csv_data = df_history.to_csv(index=False).encode('utf-8-sig')
    st.download_button("📊 导出历史数据为 Excel/CSV 表格", csv_data, "measurement_history_v12.csv", "text/csv")
    
    if st.button("🗑️ 清空本次缓存数据"):
        st.session_state.history = []
        st.rerun()
else:
    st.write("暂无历史记录，等待上传图片数据...")