import streamlit as st
import cv2
import numpy as np
import zipfile
import io
import os
import pandas as pd
from datetime import datetime
from streamlit_image_coordinates import streamlit_image_coordinates

MULTIPLE_COLORS = [
    (255, 120, 0), (0, 180, 255), (0, 255, 0), 
    (0, 0, 255), (255, 0, 255), (255, 255, 0)
]

# --- 核心数据流缓存配置 ---
if 'batch_images' not in st.session_state: st.session_state.batch_images = {} 
if 'success_results' not in st.session_state: st.session_state.success_results = {} 
if 'manual_pts_cache' not in st.session_state: st.session_state.manual_pts_cache = [] 
if 'history_log' not in st.session_state: st.session_state.history_log = []

def calculate_angle_from_three_points(p1, p_mid, p2):
    v1 = np.array([p1[0] - p_mid[0], p1[1] - p_mid[1]])
    v2 = np.array([p2[0] - p_mid[0], p2[1] - p_mid[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

def render_measurement_style(img, p1, p_mid, p2, angle, group_idx=0, mode_label="AUTO"):
    """
    统一格式渲染引擎：确保自动识别和人工选点的文字、字体、圆圈、线宽等100%镜像一致
    """
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
    
    # 左上角大版本水印
    cv2.putText(img, f"V36 {mode_label} AVG: {angle:.2f} DEG", 
                (30, 60), font, dyn_font_scale, (0, 0, 255), dyn_font_thick + 2, cv2.LINE_AA)
    return img

@st.cache_data
def load_and_resize_image(file_bytes, max_side=800):
    """
    硬核防卡顿的核心：将大分辨率图等比缩放为前端轻量画布图，换算scale
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
        if len(unique_combinations) >= 1: break # 基础批量自动解析取最优一组
        
    comp = unique_combinations[0]
    p1, p_mid, p2 = comp["points"]
    img_rendered = render_measurement_style(img, p1, p_mid, p2, comp["angle"], 0, "AUTO")
    return img_rendered, comp["angle"], "成功", unique_combinations

# --- UI 视图展现 ---
st.set_page_config(page_title="WrapAngle V36 Professional", layout="wide")
st.title("👓 面弯角高通量流水线测定系统 (V36 极速交互抗卡顿版)")
st.caption("专为大规模散图和压缩包定制。自动识别失败的图片将自动进入人工补偿区，点击处即为绝对锚定点，格式完美对齐。")

# 图片载入总闸门
uploaded_files = st.file_uploader("📥 第一步：上传多张俯视图 或 一个 Zip 压缩包（可多选混投）", type=['jpg', 'jpeg', 'png', 'zip'], accept_multiple_files=True)

if uploaded_files:
    # 检查是否有新文件注入，若有则重置处理池
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
            
    # 如果载入了全新的数据集，触发流控池更新
    if not st.session_state.batch_images or set(new_pool.keys()) != set(st.session_state.batch_images.keys()):
        st.session_state.batch_images = new_pool
        st.session_state.success_results = {}
        st.session_state.history_log = []
        
        # 立即启动一轮全量自动化快检
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
                    st.session_state.history_log.append({
                        "文件名": name, "最终角度": f"{ang:.2f}°", "分析模式": "自动识别", "状态": "✅ 自动通过"
                    })

# 分流展示状态看板
if st.session_state.batch_images:
    total_count = len(st.session_state.batch_images)
    success_count = len(st.session_state.success_results)
    fail_count = total_count - success_count
    
    c1, c2, c3 = st.columns(3)
    c1.metric("📦 流水线总图片数", f"{total_count} 张")
    c2.metric("🤖 自动识别成功", f"{success_count} 张", delta=f"{success_count/total_count*100:.1f}%")
    c3.metric("🖱️ 需人工补偿点选", f"{fail_count} 张", delta=f"-{fail_count}" if fail_count>0 else "0", delta_color="inverse")

    st.write("---")
    
    # 建立失败品待处理队列
    fail_list = [n for n in st.session_state.batch_images.keys() if n not in st.session_state.success_results]
    
    if fail_list:
        st.subheader("🖱️ 第二步：人工高效选点补偿工作区 (零卡顿)")
        # 就像播放列表切歌一样，一次只处理一张失败的图
        selected_fail_file = st.selectbox("🎯 请选择需要补偿修正的故障图片：", fail_list)
        
        if selected_fail_file:
            raw_data = st.session_state.batch_images[selected_fail_file]
            orig_img, display_img, scale = load_and_resize_image(raw_data)
            h_orig, w_orig = orig_img.shape[:2]
            h_disp, w_disp = display_img.shape[:2]
            
            col_workspace, col_control = st.columns([2, 1])
            
            with col_control:
                st.markdown(f"**当前处理图片**: `{selected_fail_file}`")
                st.markdown(f"原始分辨率: `{w_orig}×{h_orig}` → 交互画布已被优化至: `{w_disp}×{h_disp}`")
                
                pt_len = len(st.session_state.manual_pts_cache)
                st.info(f"💡 请在左图上顺次点击红点：\n1. 左侧点 ({'已捕获' if pt_len>=1 else '待点击'}) \n2. 鼻梁中点 ({'已捕获' if pt_len>=2 else '待点击'}) \n3. 右侧点 ({'已捕获' if pt_len>=3 else '待点击'})")
                
                if st.button("🗑️ 清空重选", key="clear_points"):
                    st.session_state.manual_pts_cache = []
                    st.rerun()
                    
                if pt_len == 3:
                    # 3点捕获后直接触发换算与解算
                    p1_d, pm_d, p2_d = st.session_state.manual_pts_cache
                    # 极其平滑的无误差等比坐标映射回原图
                    p1_r = (int(p1_d[0] / scale), int(p1_d[1] / scale))
                    pm_r = (int(pm_d[0] / scale), int(pm_d[1] / scale))
                    p2_r = (int(p2_d[0] / scale), int(p2_d[1] / scale))
                    
                    m_angle = calculate_angle_from_three_points(p1_r, pm_r, p2_r)
                    st.success(f"📐 测算对面弯角: **{m_angle:.2f}°**")
                    
                    if st.button("💾 确认并强行写入合规包", key="save_to_pool"):
                        # 在高清晰度原图上采用通用渲染引擎打上相同印记
                        final_render_img = render_measurement_style(orig_img.copy(), p1_r, pm_r, p2_r, m_angle, 0, "MANUAL")
                        _, out_buf = cv2.imencode(".jpg", final_render_img)
                        
                        # 强行塞入成功池并刷新日志
                        st.session_state.success_results[selected_fail_file] = {
                            "bytes": out_buf.tobytes(), "angle": f"{m_angle:.2f}°", "mode": "人工选点"
                        }
                        st.session_state.history_log.append({
                            "文件名": selected_fail_file, "最终角度": f"{m_angle:.2f}°", "分析模式": "人工选点", "状态": "✍️ 人工补偿通过"
                        })
                        st.session_state.manual_pts_cache = [] # 释放缓存给下一张图
                        st.toast(f"{selected_fail_file} 已成功闭环！", icon="🚀")
                        st.rerun()

            with col_workspace:
                # 在缩放图上绘出当前的临时视觉点击痕迹（线宽根据 display 调整，不卡顿）
                canvas = display_img.copy()
                for i, pt in enumerate(st.session_state.manual_pts_cache):
                    c_color = (255, 120, 0) if i==0 else ((0, 255, 0) if i==1 else (0, 0, 255))
                    cv2.circle(canvas, pt, 6, c_color, -1, cv2.LINE_AA)
                    cv2.putText(canvas, str(i+1), (pt[0]+8, pt[1]-8), cv2.FONT_HERSHEY_DUPLEX, 0.5, c_color, 1, cv2.LINE_AA)
                
                if len(st.session_state.manual_pts_cache) == 3:
                    p1, pm, p2 = st.session_state.manual_pts_cache
                    cv2.line(canvas, p1, pm, (0, 165, 255), 2, cv2.LINE_AA)
                    cv2.line(canvas, pm, p2, (0, 165, 255), 2, cv2.LINE_AA)
                
                # 核心高效前端组件：绝无后台红点二次重运算，点击即记录
                coord = streamlit_image_coordinates(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB), key=f"canvas_{selected_fail_file}")
                if coord is not None and len(st.session_state.manual_pts_cache) < 3:
                    click_pt = (coord["x"], coord["y"])
                    if not st.session_state.manual_pts_cache or np.linalg.norm(np.array(st.session_state.manual_pts_cache[-1]) - np.array(click_pt)) > 3:
                        st.session_state.manual_pts_cache.append(click_pt)
                        st.rerun()
    else:
        st.balloons()
        st.success("🎉 太棒了！全量队列已全部检测完毕，没有任何失败图像！")

    # --- 第三步：一键打包混下载区 ---
    st.write("---")
    st.subheader("📥 第三步：全量混合测量数据导出包")
    
    if st.session_state.success_results:
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            # 在后台组织打包二进制
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, "w") as z_out:
                for f_name, data_obj in st.session_state.success_results.items():
                    prefix = "Auto_" if data_obj["mode"] == "自动识别" else "Manual_"
                    z_out.writestr(f"{prefix}{f_name}", data_obj["bytes"])
            
            st.download_button(
                label="📥 导出已处理的混合标注图片包 (Zip)",
                data=zip_buffer.getvalue(),
                file_name=f"WrapAngle_V36_Combined_{datetime.now().strftime('%m%d_%H%M')}.zip",
                mime="application/zip",
                use_container_width=True
            )
        with col_dl2:
            # 整理一份数据报表同步下载
            all_log = []
            for name in st.session_state.batch_images.keys():
                if name in st.session_state.success_results:
                    obj = st.session_state.success_results[name]
                    all_log.append({"文件名": name, "最终测量面弯角": obj["angle"], "测量模式": obj["mode"], "检测结果": "通过"})
                else:
                    all_log.append({"文件名": name, "最终测量面弯角": "-", "测量模式": "未检测", "检测结果": "失败/待人工选点"})
            df = pd.DataFrame(all_log)
            st.download_button(
                label="📊 导出完整面弯角数据分析报表 (CSV)",
                data=df.to_csv(index=False).encode('utf-8-sig'),
                file_name="WrapAngle_V36_Report.csv",
                mime="text/csv",
                use_container_width=True
            )
            
        st.dataframe(df, use_container_width=True)