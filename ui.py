import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
import pandas as pd
from datetime import datetime
import plotly.express as px  # 引入官方深度支持的 Plotly 库

MULTIPLE_COLORS = [
    (255, 120, 0), (0, 180, 255), (0, 255, 0), 
    (0, 0, 255), (255, 0, 255), (255, 255, 0)
]

# --- 核心数据流缓存配置 ---
if 'batch_images' not in st.session_state: st.session_state.batch_images = {} 
if 'success_results' not in st.session_state: st.session_state.success_results = {} 
if 'history_log' not in st.session_state: st.session_state.history_log = []
# 用于暂存当前图片手动点击的原始坐标
if 'plotly_pts' not in st.session_state: st.session_state.plotly_pts = [] 
if 'last_selected_file' not in st.session_state: st.session_state.last_selected_file = ""

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
    
    cv2.putText(img, f"V38 {mode_label} AVG: {angle:.2f} DEG", 
                (30, 60), font, dyn_font_scale, (0, 0, 255), dyn_font_thick + 2, cv2.LINE_AA)
    return img

@st.cache_data
def load_and_resize_image(file_bytes, max_side=750):
    """
    轻量化缩放：限制最大边长为 750px，确保在云端网络传输时也是毫秒级响应
    """
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None, None, 1.0
    h, w = img.shape[:2]
    scale = 1.0
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        img_resized = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img.copy()
    return img, img_resized, scale

def pixel_level_reconstruct_mask_v34(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_red1, upper_red1 = np.array([0, 12, 30]), np.array([22, 255, 255])
    lower_red2, upper_red2 = np.array([150, 12, 30]), np.array([180, 255, 255])
    mask_hsv = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    
    flood_mask = mask_hsv.copy()
    h_f, w_f = flood_mask.shape[:2]
    fill_contour = np.zeros((h_f + 2, w_f + 2), np.uint8)
    cv2.floodFill(flood_mask, fill_contour, (0, 0), 255)
    mask_filled = mask_hsv | cv2.bitwise_not(flood_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(mask_filled, cv2.MORPH_CLOSE, kernel)

def process_image_v34_core(img):
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    pixel_mask = pixel_level_reconstruct_mask_v34(img)
    contours, _ = cv2.findContours(pixel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 4 < area < 3500:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                check_offsets = [-12, -8, 8, 12]
                dark_pixel_count = 0
                for offset in check_offsets:
                    if 0 <= cX + offset < w and gray[cY, cX + offset] < 90: dark_pixel_count += 1
                    if 0 <= cY + offset < h and gray[cY + offset, cX] < 90: dark_pixel_count += 1
                if dark_pixel_count < 2: continue
                candidates.append((cX, cY))
                        
    candidates = list(set(candidates))[:40] 
    num_pts = len(candidates)
    all_valid_combinations = []
    
    if num_pts >= 3:
        for i in range(num_pts - 2):
            for j in range(i + 1, num_pts - 1):
                for k in range(j + 1, num_pts):
                    pts_temp = [candidates[i], candidates[j], candidates[k]]
                    dists = [np.linalg.norm(np.array(pts_temp[0]) - np.array(pts_temp[1])),
                             np.linalg.norm(np.array(pts_temp[1]) - np.array(pts_temp[2])),
                             np.linalg.norm(np.array(pts_temp[2]) - np.array(pts_temp[0]))]
                    max_idx = np.argmax(dists)
                    if dists[max_idx] < min(w, h) * 0.28 or dists[max_idx] > max(w, h) * 0.99: continue
                    if max_idx == 0:   p_mid, p1, p2 = pts_temp[2], pts_temp[0], pts_temp[1]
                    elif max_idx == 1: p_mid, p1, p2 = pts_temp[0], pts_temp[1], pts_temp[2]
                    else:              p_mid, p1, p2 = pts_temp[1], pts_temp[0], pts_temp[2]
                    
                    if p_mid[1] < min(p1[1], p2[1]) - 10: continue
                    temp_angle = calculate_angle_from_three_points(p1, p_mid, p2)
                    if temp_angle < 165.0 or temp_angle > 179.8: continue
                    if abs(np.linalg.norm(np.array(p1)-np.array(p_mid)) - np.linalg.norm(np.array(p2)-np.array(p_mid))) / max(np.linalg.norm(np.array(p1)-np.array(p_mid)), 1) > 0.48: continue
                    all_valid_combinations.append({"points": (p1, p_mid, p2), "angle": temp_angle, "span": dists[max_idx]})

    if not all_valid_combinations: return img, 0, "失败", []
    all_valid_combinations.sort(key=lambda x: (-x["span"], -x["angle"]))
    
    unique_combinations = []
    seen_mids = []
    for comp in all_valid_combinations:
        mid_pt = comp["points"][1]
        if any(np.linalg.norm(np.array(mid_pt) - np.array(m)) < 35 for m in seen_mids): continue
        unique_combinations.append(comp)
        seen_mids.append(mid_pt)
        if len(unique_combinations) >= 1: break
        
    comp = unique_combinations[0]
    p1, p_mid, p2 = comp["points"]
    img_rendered = render_measurement_style(img.copy(), p1, p_mid, p2, comp["angle"], 0, "AUTO")
    return img_rendered, comp["angle"], "成功", unique_combinations


# --- UI 视图展现层 ---
st.set_page_config(page_title="WrapAngle V38 Cloud", layout="wide")
st.title("👓 面弯角高通量流水线测定系统 (V38 云端全兼容 Plotly 版)")
st.caption("完美适配 Streamlit Cloud 无显示器服务器环境。采用官方推荐的 Plotly 网页轻量矢量画布，彻底消灭组件超时与本地窗口报错。")

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
        st.session_state.plotly_pts = []
        
        with st.spinner("🤖 正在启动后台算法流水线，快速分流合格品..."):
            for name, b_data in st.session_state.batch_images.items():
                nparr = np.frombuffer(b_data, np.uint8)
                raw_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if raw_img is None: continue
                res_img, ang, status, _ = process_image_v34_core(raw_img.copy())
                
                if status == "成功":
                    _, buf = cv2.imencode(".jpg", res_img)
                    st.session_state.success_results[name] = {
                        "bytes": buf.tobytes(), "angle": f"{ang:.2f}°", "mode": "自动识别"
                    }

# 分流展示状态看板
if st.session_state.batch_images:
    total_count = len(st.session_state.batch_images)
    success_count = len(st.session_state.success_results)
    fail_count = total_count - success_count
    
    c1, c2, c3 = st.columns(3)
    c1.metric("📦 当前流转图片总量", f"{total_count} 张")
    c2.metric("🤖 后台算法自动识别成功", f"{success_count} 张")
    c3.metric("鼠标手动点选补偿通过", f"{success_count} 张")  # 统一合并逻辑

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
                label="📥 导出已处理的混合标注图片包 (Zip)",
                data=zip_buffer.getvalue(),
                file_name=f"WrapAngle_V38_Cloud_{datetime.now().strftime('%m%d_%H%M')}.zip",
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
                file_name="WrapAngle_V38_Report.csv",
                mime="text/csv",
                use_container_width=True
            )
        st.dataframe(df, use_container_width=True)

    # --- 第二步：Plotly 官方高能轻量级手动选点工作区 ---
    st.write("---")
    st.subheader("🖱️ 手动异常补偿干预区 (云端毫秒级不卡顿模式)")
    
    target_file = st.selectbox("🎯 请选择需要【进入手动微调】的目标图片：", list(st.session_state.batch_images.keys()))
    
    if target_file:
        # 当切换图片时，自动清洗上一张图片的点击点缓存
        if st.session_state.last_selected_file != target_file:
            st.session_state.plotly_pts = []
            st.session_state.last_selected_file = target_file
            
        is_already_success = target_file in st.session_state.success_results
        if is_already_success:
            st.warning(f"💡 提示：图片 `{target_file}` 此前已成功生成结果（角度: {st.session_state.success_results[target_file]['angle']}），重新点选将完美覆盖原纪录。")
        else:
            st.error(f"🔍 提示：图片 `{target_file}` 自动识别失败，请使用下方官方 Plotly 画布进行极速测定。")
            
        raw_data = st.session_state.batch_images[target_file]
        orig_img, display_img, scale = load_and_resize_image(raw_data)
        
        col_workspace, col_control = st.columns([2, 1])
        
        with col_control:
            st.markdown(f"**当前调节目标**: `{target_file}`")
            pt_len = len(st.session_state.plotly_pts)
            st.info(f"📍 请在左图上【左键单击】红点位置：\n1. 左侧点 ({'🟢 已捕获' if pt_len>=1 else '⚪ 待点击'})\n2. 鼻梁中点 ({'🔴 已捕获' if pt_len>=2 else '⚪ 待点击'})\n3. 右侧点 ({'🔵 已捕获' if pt_len>=3 else '⚪ 待点击'})")
            
            if st.button("🗑️ 清空当前点重新选", key="clear_points"):
                st.session_state.plotly_pts = []
                st.st.rerun()
                
            if pt_len == 3:
                # 满3点直接在原图尺寸上进行高精度换算
                p1_d, pm_d, p2_d = st.session_state.plotly_pts
                p1_r = (int(p1_d[0] / scale), int(p1_d[1] / scale))
                pm_r = (int(pm_d[0] / scale), int(pm_d[1] / scale))
                p2_r = (int(p2_d[0] / scale), int(p2_d[1] / scale))
                
                m_angle = calculate_angle_from_three_points(p1_r, pm_r, p2_r)
                st.success(f"📐 鼠标解算面弯角: **{m_angle:.2f}°**")
                
                if st.button("💾 确认并强行写入合规包", key="save_to_pool"):
                    final_render_img = render_measurement_style(orig_img.copy(), p1_r, pm_r, p2_r, m_angle, 0, "MANUAL")
                    _, out_buf = cv2.imencode(".jpg", final_render_img)
                    
                    st.session_state.success_results[target_file] = {
                        "bytes": out_buf.tobytes(), "angle": f"{m_angle:.2f}°", "mode": "人工选点"
                    }
                    st.session_state.plotly_pts = [] # 清空
                    st.toast(f"图片 {target_file} 记录已成功闭环！", icon="🚀")
                    st.rerun()

        with col_workspace:
            # 在轻量 display_img 上画线
            canvas = display_img.copy()
            for i, pt in enumerate(st.session_state.plotly_pts):
                c_color = (255, 120, 0) if i==0 else ((0, 255, 0) if i==1 else (0, 0, 255))
                cv2.circle(canvas, pt, 5, c_color, -1, cv2.LINE_AA)
                cv2.putText(canvas, str(i+1), (pt[0]+10, pt[1]-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, c_color, 1, cv2.LINE_AA)
            if len(st.session_state.plotly_pts) == 3:
                cv2.line(canvas, st.session_state.plotly_pts[0], st.session_state.plotly_pts[1], (0, 165, 255), 2, cv2.LINE_AA)
                cv2.line(canvas, st.session_state.plotly_pts[1], st.session_state.plotly_pts[2], (0, 165, 255), 2, cv2.LINE_AA)
                
            # 💡 利用官方支持的 Plotly 渲染轻量矢量画布
            # 将 OpenCV 的 BGR 转换为 RGB
            rgb_canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
            fig = px.imshow(rgb_canvas)
            fig.update_layout(
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                hovermode=False
            )
            
            # 使用 Streamlit 原生 plotly_chart 并开启点击事件监听
            # config 设置：隐藏工具栏，使得界面非常清爽干净
            click_data = st.plotly_chart(
                fig, 
                use_container_width=True, 
                config={'displayModeBar': False},
                on_select="rerun" # 监听选择/点击事件
            )
            
            # 从官方的数据流里秒级捕捉点击点的 x, y 坐标
            if click_data and "selection" in click_data and "points" in click_data["selection"]:
                pts = click_data["selection"]["points"]
                if len(pts) > 0 and len(st.session_state.plotly_pts) < 3:
                    # 拿到点击位置在缩放画布下的精确像素值
                    new_x = int(pts[0]["x"])
                    new_y = int(pts[0]["y"])
                    new_pt = (new_x, new_y)
                    
                    # 查重防抖保护
                    if not st.session_state.plotly_pts or np.linalg.norm(np.array(st.session_state.plotly_pts[-1]) - np.array(new_pt)) > 5:
                        st.session_state.plotly_pts.append(new_pt)
                        st.rerun()

    if st.button("🗑️ 清空全量图片缓存"):
        st.session_state.batch_images = {}
        st.session_state.success_results = {}
        st.session_state.plotly_pts = []
        st.rerun()