import cv2
import numpy as np
import os
import argparse
from glob import glob

# --- V27 级联多层检测算法 ---
def process_image_v27(image_path):
    img = cv2.imread(image_path)
    if img is None: 
        return None, 0, "文件读取失败", "V27"
    
    h, w = img.shape[:2]
    
    dyn_radius = max(5, int(w / 150))      
    dyn_line = max(2, int(w / 500))        
    dyn_font_scale = w / 1200              
    dyn_font_thick = max(1, int(w / 800))  
    font = cv2.FONT_HERSHEY_DUPLEX

    b, g, r = cv2.split(img)
    r_16 = r.astype(np.int16)
    g_16 = g.astype(np.int16)
    b_16 = b.astype(np.int16)

    rg_diff = r_16 - g_16
    rb_diff = r_16 - b_16

    cascade_thresholds = [
        {"rg": 75, "rb": 45, "r": 120, "circ": 0.55, "area_min": 12},
        {"rg": 55, "rb": 40, "r": 100, "circ": 0.45, "area_min": 10},
        {"rg": 40, "rb": 30, "r": 80, "circ": 0.35, "area_min": 8}
    ]
    
    best_set = None
    min_geometric_error = float('inf')

    for pass_idx, th in enumerate(cascade_thresholds):
        mask = (rg_diff > th["rg"]) & (rb_diff > th["rb"]) & (r_16 > th["r"])
        mask = mask.astype(np.uint8) * 255
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        contours_red, _ = cv2.findContours(red_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        
        for cnt in contours_red:
            area = cv2.contourArea(cnt)
            if th["area_min"] < area < 2000:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                if circularity >= th["circ"]:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        if 0.02 * w < cX < 0.98 * w and 0.02 * h < cY < 0.98 * h:
                            candidates.append((cX, cY, area))
        
        candidates.sort(key=lambda x: x[2], reverse=True)
        candidates = [c[:2] for c in candidates[:12]]
                            
        num_pts = len(candidates)
        
        if num_pts >= 3:
            for i in range(len(candidates)-2):
                if min_geometric_error < 0.03:
                    break
                for j in range(i+1, len(candidates)-1):
                    if min_geometric_error < 0.03:
                        break
                    for k in range(j+1, len(candidates)):
                        pA, pB, pC = np.array(candidates[i]), np.array(candidates[j]), np.array(candidates[k])
                        
                        dAB = np.linalg.norm(pA - pB)
                        dBC = np.linalg.norm(pB - pC)
                        dCA = np.linalg.norm(pC - pA)
                        dists = [dAB, dBC, dCA]
                        pts_temp = [candidates[i], candidates[j], candidates[k]]
                        
                        max_idx = np.argmax(dists)
                        max_dist = dists[max_idx]
                        
                        if max_dist < min(w, h) * 0.30: 
                            continue 
                            
                        if max_idx == 0:   
                            p_mid, p1, p2 = pts_temp[2], pts_temp[0], pts_temp[1]
                        elif max_idx == 1: 
                            p_mid, p1, p2 = pts_temp[0], pts_temp[1], pts_temp[2]
                        else:              
                            p_mid, p1, p2 = pts_temp[1], pts_temp[0], pts_temp[2]
                        
                        v1 = np.array([p1[0] - p_mid[0], p1[1] - p_mid[1]])
                        v2 = np.array([p2[0] - p_mid[0], p2[1] - p_mid[1]])
                        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        temp_angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                        
                        if temp_angle < 160 or temp_angle > 180:
                            continue
                            
                        len1 = np.linalg.norm(v1)
                        len2 = np.linalg.norm(v2)
                        balance_err = abs(len1 - len2) / max(len1, len2, 1)
                        
                        if balance_err > 0.15: 
                            continue 
                        
                        if balance_err < min_geometric_error:
                            min_geometric_error = balance_err
                            best_set = (p1, p_mid, p2)
            
            if best_set is not None:
                break

    if best_set is None:
        return img, 0, "V27识别失败", "V27"
        
    p1, p_mid, p2 = best_set
    
    v1, v2 = np.array([p1[0]-p_mid[0], p1[1]-p_mid[1]]), np.array([p2[0]-p_mid[0], p2[1]-p_mid[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    cv2.line(img, p1, p_mid, (255, 120, 0), dyn_line, cv2.LINE_AA)
    cv2.line(img, p_mid, p2, (255, 120, 0), dyn_line, cv2.LINE_AA)
    for p in [p1, p_mid, p2]:
        cv2.circle(img, p, dyn_radius, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p, dyn_radius, (0, 0, 0), 1, cv2.LINE_AA)

    text = f"ANGLE: {angle:.2f} DEG (V27)"
    text_pos = (p_mid[0] + 40, p_mid[1])
    cv2.putText(img, text, text_pos, font, dyn_font_scale, (0,0,0), dyn_font_thick+2, cv2.LINE_AA)
    cv2.putText(img, text, text_pos, font, dyn_font_scale, (255,255,255), dyn_font_thick, cv2.LINE_AA)
    
    return img, angle, "成功", "V27"

# --- V28 增强容错算法 ---
def process_image_v28(image_path):
    img = cv2.imread(image_path)
    if img is None: 
        return None, 0, "文件读取失败", "V28"
    
    h, w = img.shape[:2]
    
    dyn_radius = max(5, int(w / 150))      
    dyn_line = max(2, int(w / 500))        
    dyn_font_scale = w / 1200              
    dyn_font_thick = max(1, int(w / 800))  
    font = cv2.FONT_HERSHEY_DUPLEX

    b, g, r = cv2.split(img)
    r_16 = r.astype(np.int16)
    g_16 = g.astype(np.int16)
    b_16 = b.astype(np.int16)

    rg_diff = r_16 - g_16
    rb_diff = r_16 - b_16

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cascade_thresholds = [
        {"rg": 75, "rb": 45, "r": 120, "circ": 0.55, "area_min": 12},
        {"rg": 55, "rb": 40, "r": 100, "circ": 0.40, "area_min": 8},
        {"rg": 40, "rb": 30, "r": 80, "circ": 0.25, "area_min": 5},
        {"mode": "hsv", "h_low": 0, "h_high": 10, "s_low": 30, "v_low": 60, "circ": 0.20, "area_min": 3},
        {"mode": "hsv", "h_low": 165, "h_high": 180, "s_low": 30, "v_low": 60, "circ": 0.20, "area_min": 3}
    ]
    
    best_set = None
    min_geometric_error = float('inf')

    for pass_idx, th in enumerate(cascade_thresholds):
        if th.get("mode") == "hsv":
            lower_red = np.array([th["h_low"], th["s_low"], th["v_low"]])
            upper_red = np.array([th["h_high"], 255, 255])
            mask = cv2.inRange(hsv, lower_red, upper_red)
        else:
            mask = (rg_diff > th["rg"]) & (rb_diff > th["rb"]) & (r_16 > th["r"])
            mask = mask.astype(np.uint8) * 255
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        red_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)
        
        kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        red_cleaned = cv2.dilate(red_cleaned, kernel_dilate, iterations=1)
        
        contours_red, _ = cv2.findContours(red_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        
        for cnt in contours_red:
            area = cv2.contourArea(cnt)
            if th["area_min"] < area < 3000:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                if circularity >= th["circ"]:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        if 0.02 * w < cX < 0.98 * w and 0.02 * h < cY < 0.98 * h:
                            candidates.append((cX, cY, area))
                            
        candidates.sort(key=lambda x: x[2], reverse=True)
        candidates = [c[:2] for c in candidates[:15]]
                            
        num_pts = len(candidates)
        
        if num_pts >= 3:
            for i in range(len(candidates)-2):
                if min_geometric_error < 0.02:
                    break
                for j in range(i+1, len(candidates)-1):
                    if min_geometric_error < 0.02:
                        break
                    for k in range(j+1, len(candidates)):
                        pA, pB, pC = np.array(candidates[i]), np.array(candidates[j]), np.array(candidates[k])
                        
                        dAB = np.linalg.norm(pA - pB)
                        dBC = np.linalg.norm(pB - pC)
                        dCA = np.linalg.norm(pC - pA)
                        dists = [dAB, dBC, dCA]
                        pts_temp = [candidates[i], candidates[j], candidates[k]]
                        
                        max_idx = np.argmax(dists)
                        max_dist = dists[max_idx]
                        
                        if max_dist < min(w, h) * 0.30: 
                            continue 
                            
                        if max_idx == 0:   
                            p_mid, p1, p2 = pts_temp[2], pts_temp[0], pts_temp[1]
                        elif max_idx == 1: 
                            p_mid, p1, p2 = pts_temp[0], pts_temp[1], pts_temp[2]
                        else:              
                            p_mid, p1, p2 = pts_temp[1], pts_temp[0], pts_temp[2]
                        
                        v1 = np.array([p1[0] - p_mid[0], p1[1] - p_mid[1]])
                        v2 = np.array([p2[0] - p_mid[0], p2[1] - p_mid[1]])
                        cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                        temp_angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                        
                        if temp_angle < 160 or temp_angle > 180:
                            continue
                            
                        len1 = np.linalg.norm(v1)
                        len2 = np.linalg.norm(v2)
                        balance_err = abs(len1 - len2) / max(len1, len2, 1)
                        
                        if balance_err < min_geometric_error:
                            min_geometric_error = balance_err
                            best_set = (p1, p_mid, p2)
            
            if best_set is not None:
                break

    if best_set is None:
        return img, 0, "V28识别失败", "V28"
        
    p1, p_mid, p2 = best_set
    
    v1, v2 = np.array([p1[0]-p_mid[0], p1[1]-p_mid[1]]), np.array([p2[0]-p_mid[0], p2[1]-p_mid[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
    cv2.line(img, p1, p_mid, (0, 180, 255), dyn_line, cv2.LINE_AA)
    cv2.line(img, p_mid, p2, (0, 180, 255), dyn_line, cv2.LINE_AA)
    for p in [p1, p_mid, p2]:
        cv2.circle(img, p, dyn_radius, (255, 0, 255), -1, cv2.LINE_AA)
        cv2.circle(img, p, dyn_radius, (0, 0, 0), 1, cv2.LINE_AA)

    text = f"ANGLE: {angle:.2f} DEG (V28)"
    text_pos = (p_mid[0] + 40, p_mid[1])
    cv2.putText(img, text, text_pos, font, dyn_font_scale, (0,0,0), dyn_font_thick+2, cv2.LINE_AA)
    cv2.putText(img, text, text_pos, font, dyn_font_scale, (255,255,255), dyn_font_thick, cv2.LINE_AA)
    
    return img, angle, "成功", "V28"

MULTIPLE_COLORS = [
    (255, 120, 0),  
    (0, 180, 255),  
    (0, 255, 0),    
    (0, 0, 255),    
    (255, 0, 255),  
    (255, 255, 0)   
]

# --- V29 算法：V32核心改进版 ---
def process_image_v29(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None, 0, "无法读取图片", "V29"
    
    h, w = img.shape[:2]
    
    dyn_line = max(2, int(w / 600))
    dyn_font_scale = w / 1500
    dyn_font_thick = max(1, int(w / 1000))
    font = cv2.FONT_HERSHEY_DUPLEX
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    b, g, r = cv2.split(img)
    rg_diff = cv2.absdiff(r, g)
    _, mask_rg = cv2.threshold(rg_diff, 15, 255, cv2.THRESH_BINARY)
    
    lower_red1 = np.array([0, 20, 40])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([160, 20, 40])
    upper_red2 = np.array([180, 255, 255])
    mask_hsv = cv2.bitwise_or(cv2.inRange(hsv, lower_red1, upper_red1), cv2.inRange(hsv, lower_red2, upper_red2))
    
    pixel_mask = cv2.bitwise_and(mask_rg, mask_hsv)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pixel_mask = cv2.morphologyEx(pixel_mask, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(pixel_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 2 < area < 4000:
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                
                if not (0.005 * w < cX < 0.995 * w and 0.005 * h < cY < 0.995 * h):
                    continue
                
                check_offsets = [-15, -10, 10, 15]
                dark_pixel_count = 0
                skin_pixel_count = 0
                for offset in check_offsets:
                    if 0 <= cX + offset < w:
                        val = gray[cY, cX + offset]
                        if val < 85:
                            dark_pixel_count += 1
                        if val > 125:
                            skin_pixel_count += 1
                    if 0 <= cY + offset < h:
                        val = gray[cY + offset, cX]
                        if val < 85:
                            dark_pixel_count += 1
                        if val > 125:
                            skin_pixel_count += 1
                
                if dark_pixel_count < 2:
                    continue
                
                x_s, x_e = max(0, cX-4), min(w, cX+5)
                y_s, y_e = max(0, cY-4), min(h, cY+5)
                local_s = hsv[y_s:y_e, x_s:x_e, 1]
                if np.max(local_s) < 55:
                    continue
                
                candidates.append((cX, cY))
    
    candidates = list(set(candidates))[:30]
    num_pts = len(candidates)
    
    all_valid_combinations = []
    if num_pts >= 3:
        for i in range(num_pts - 2):
            for j in range(i + 1, num_pts - 1):
                for k in range(j + 1, num_pts):
                    pA, pB, pC = np.array(candidates[i]), np.array(candidates[j]), np.array(candidates[k])
                    dists = [np.linalg.norm(pA - pB), np.linalg.norm(pB - pC), np.linalg.norm(pC - pA)]
                    pts_temp = [candidates[i], candidates[j], candidates[k]]
                    
                    max_idx = np.argmax(dists)
                    max_dist = dists[max_idx]
                    
                    if max_dist < min(w, h) * 0.20 or max_dist > max(w, h) * 0.99:
                        continue
                    
                    if max_idx == 0:
                        p_mid, p1, p2 = pts_temp[2], pts_temp[0], pts_temp[1]
                    elif max_idx == 1:
                        p_mid, p1, p2 = pts_temp[0], pts_temp[1], pts_temp[2]
                    else:
                        p_mid, p1, p2 = pts_temp[1], pts_temp[0], pts_temp[2]
                    
                    v1, v2 = np.array([p1[0]-p_mid[0], p1[1]-p_mid[1]]), np.array([p2[0]-p_mid[0], p2[1]-p_mid[1]])
                    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                    temp_angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
                    
                    if temp_angle < 165.0 or temp_angle > 179.8:
                        continue
                    
                    len1, len2 = np.linalg.norm(v1), np.linalg.norm(v2)
                    balance_err = abs(len1 - len2) / max(len1, len2, 1)
                    
                    if balance_err > 0.48:
                        continue
                    
                    all_valid_combinations.append({
                        "points": (p1, p_mid, p2),
                        "angle": temp_angle,
                        "error": balance_err,
                        "span": max_dist
                    })
    
    if not all_valid_combinations:
        return img, 0, "未组合出符合眼镜物理特征的红体骨架（额头杂质已被成功拦截）", "V29"
    
    all_valid_combinations.sort(key=lambda x: (-x["span"], -x["angle"]))
    
    unique_combinations = []
    seen_mids = []
    for comp in all_valid_combinations:
        mid_pt = comp["points"][1]
        if any(np.linalg.norm(np.array(mid_pt) - np.array(m)) < 30 for m in seen_mids):
            continue
        unique_combinations.append(comp)
        seen_mids.append(mid_pt)
        if len(unique_combinations) >= 6:
            break
    
    avg_angle = np.mean([c["angle"] for c in unique_combinations])
    
    for idx, comp in enumerate(unique_combinations):
        p1, p_mid, p2 = comp["points"]
        color = MULTIPLE_COLORS[idx % len(MULTIPLE_COLORS)]
        
        cv2.line(img, p1, p_mid, color, dyn_line, cv2.LINE_AA)
        cv2.line(img, p_mid, p2, color, dyn_line, cv2.LINE_AA)
        
        for p in [p1, p_mid, p2]:
            cv2.circle(img, p, int(dyn_line*2.5), color, -1, cv2.LINE_AA)
            cv2.circle(img, p, int(dyn_line*2.5), (0,0,0), 1, cv2.LINE_AA)
        
        text = f"#{idx+1}: {comp['angle']:.2f} DEG"
        text_pos = (p_mid[0] + 30, p_mid[1] + (idx * int(w/35)) - int(w/70))
        cv2.putText(img, text, text_pos, font, dyn_font_scale * 0.75, (0,0,0), dyn_font_thick+1, cv2.LINE_AA)
        cv2.putText(img, text, text_pos, font, dyn_font_scale * 0.75, color, dyn_font_thick, cv2.LINE_AA)
    
    cv2.putText(img, f"FINAL FIXED AVG: {avg_angle:.2f} DEG (GROUPS: {len(unique_combinations)})", 
                (30, 60), font, dyn_font_scale, (0,0,255), dyn_font_thick+2, cv2.LINE_AA)
    
    return img, avg_angle, "成功", "V29"

# --- 级联识别主函数 ---
def process_image_cascade(image_path):
    MIN_ANGLE = 168.0
    
    res_img_v27, angle_v27, status_v27, _ = process_image_v27(image_path)
    
    if status_v27 == "成功" and angle_v27 >= MIN_ANGLE:
        return res_img_v27, angle_v27, status_v27, "V27"
    
    res_img_v28, angle_v28, status_v28, _ = process_image_v28(image_path)
    
    if status_v28 == "成功" and angle_v28 >= MIN_ANGLE:
        return res_img_v28, angle_v28, status_v28, "V28"
    
    res_img_v29, angle_v29, status_v29, _ = process_image_v29(image_path)
    
    if status_v29 == "成功" and angle_v29 >= MIN_ANGLE:
        return res_img_v29, angle_v29, status_v29, "V29"
    
    fail_reason = "识别失败："
    if status_v27 == "成功" and angle_v27 < MIN_ANGLE:
        fail_reason += f"V27识别角度 {angle_v27:.1f}° 低于阈值 {MIN_ANGLE}°；"
    else:
        fail_reason += "V27未识别成功；"
        
    if status_v28 == "成功" and angle_v28 < MIN_ANGLE:
        fail_reason += f"V28识别角度 {angle_v28:.1f}° 低于阈值 {MIN_ANGLE}°；"
    else:
        fail_reason += "V28未识别成功；"
        
    if status_v29 == "成功" and angle_v29 < MIN_ANGLE:
        fail_reason += f"V29识别角度 {angle_v29:.1f}° 低于阈值 {MIN_ANGLE}°"
    else:
        fail_reason += "V29未识别成功"
    
    return res_img_v27, 0, fail_reason, "失败"

def main():
    parser = argparse.ArgumentParser(description='面弯角精密测量系统 (V27+V28 级联识别版)')
    parser.add_argument('input', help='输入图片路径或包含图片的文件夹')
    parser.add_argument('-o', '--output', default='output_v27v28', help='输出文件夹路径')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    if os.path.isdir(args.input):
        image_paths = glob(os.path.join(args.input, '*.jpg')) + \
                      glob(os.path.join(args.input, '*.jpeg')) + \
                      glob(os.path.join(args.input, '*.png'))
    elif os.path.isfile(args.input):
        image_paths = [args.input]
    else:
        print(f"错误：输入路径不存在 - {args.input}")
        return
    
    print(f"找到 {len(image_paths)} 张图片")
    
    results = []
    for img_path in image_paths:
        print(f"\n处理: {os.path.basename(img_path)}")
        res_img, angle, status, algo_version = process_image_cascade(img_path)
        
        if status == "成功":
            output_name = f"Result_{os.path.basename(img_path)}"
        else:
            output_name = f"Fail_{os.path.basename(img_path)}"
        
        output_path = os.path.join(args.output, output_name)
        cv2.imwrite(output_path, res_img)
        
        results.append({
            '文件名': os.path.basename(img_path),
            '面弯角': f"{angle:.2f}°" if angle > 0 else "-",
            '算法版本': algo_version,
            '状态': status
        })
        print(f"  -> {status}: {angle:.2f}° (算法: {algo_version})")
    
    print("\n" + "="*70)
    print("处理结果汇总:")
    print("-"*70)
    print(f"{'文件名':<20} | {'面弯角':<12} | {'算法版本':<8} | {'状态'}")
    print("-"*70)
    for r in results:
        print(f"{r['文件名']:<20} | {r['面弯角']:<12} | {r['算法版本']:<8} | {r['状态']}")
    print("="*70)
    print(f"\n结果已保存到: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()