import cv2
import numpy as np
import os
import argparse
from glob import glob

def process_image_v27(image_path):
    img = cv2.imread(image_path)
    if img is None: 
        return None, 0, "文件读取失败"
    
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
        {"rg": 75, "rb": 45, "r": 120, "circ": 0.55},
        {"rg": 55, "rb": 40, "r": 100, "circ": 0.60}
    ]
    
    best_set = None
    min_geometric_error = float('inf')

    for pass_idx, th in enumerate(cascade_thresholds):
        mask = (rg_diff > th["rg"]) & (rb_diff > th["rb"]) & (r_16 > th["r"])
        red_mask = mask.astype(np.uint8) * 255
        
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        red_cleaned = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_close)
        
        contours_red, _ = cv2.findContours(red_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        
        for cnt in contours_red:
            area = cv2.contourArea(cnt)
            if 12 < area < 1500:
                perimeter = cv2.arcLength(cnt, True)
                if perimeter == 0: continue
                circularity = 4 * np.pi * (area / (perimeter * perimeter))
                
                if circularity >= th["circ"]:
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        if 0.02 * w < cX < 0.98 * w and 0.02 * h < cY < 0.98 * h:
                            candidates.append((cX, cY))
                            
        num_pts = len(candidates)
        
        candidates.sort(key=lambda x: x[0]**2 + x[1]**2)
        candidates = candidates[:15]
        
        if num_pts >= 3:
            for i in range(len(candidates)-2):
                for j in range(i+1, len(candidates)-1):
                    for k in range(j+1, len(candidates)):
                        if min_geometric_error < 0.05:
                            break
                            
                        pA, pB, pC = np.array(candidates[i]), np.array(candidates[j]), np.array(candidates[k])
                        
                        dAB = np.linalg.norm(pA - pB)
                        dBC = np.linalg.norm(pB - pC)
                        dCA = np.linalg.norm(pC - pA)
                        dists = [dAB, dBC, dCA]
                        pts_temp = [candidates[i], candidates[j], candidates[k]]
                        
                        max_idx = np.argmax(dists)
                        max_dist = dists[max_idx]
                        
                        if max_dist < min(w, h) * 0.25: 
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
                        
                        if temp_angle < 90 or temp_angle > 179.5:
                            continue
                            
                        len1 = np.linalg.norm(v1)
                        len2 = np.linalg.norm(v2)
                        balance_err = abs(len1 - len2) / max(len1, len2, 1)
                        
                        if balance_err < min_geometric_error:
                            min_geometric_error = balance_err
                            best_set = (p1, p_mid, p2)
                    if min_geometric_error < 0.05:
                        break
                if min_geometric_error < 0.05:
                    break
            
            if best_set is not None:
                break

    if best_set is None:
        return img, 0, "识别失败：多级级联区内未匹配到合规眼镜刚体（请排查红点是否被完全盖死）"
        
    p1, p_mid, p2 = best_set
    
    v1, v2 = np.array([p1[0]-p_mid[0], p1[1]-p_mid[1]]), np.array([p2[0]-p_mid[0], p2[1]-p_mid[1]])
    cos_theta = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    angle = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
    
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

def main():
    parser = argparse.ArgumentParser(description='面弯角精密测量系统 (V27 级联多层检测版)')
    parser.add_argument('input', help='输入图片路径或包含图片的文件夹')
    parser.add_argument('-o', '--output', default='output_v27', help='输出文件夹路径')
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
        res_img, angle, status = process_image_v27(img_path)
        
        if status == "成功":
            output_name = f"Result_{os.path.basename(img_path)}"
        else:
            output_name = f"Fail_{os.path.basename(img_path)}"
        
        output_path = os.path.join(args.output, output_name)
        cv2.imwrite(output_path, res_img)
        
        results.append({
            '文件名': os.path.basename(img_path),
            '面弯角': f"{angle:.2f}°" if angle > 0 else "-",
            '状态': status
        })
        print(f"  -> {status}: {angle:.2f}°")
    
    print("\n" + "="*60)
    print("处理结果汇总:")
    print("-"*60)
    for r in results:
        print(f"{r['文件名']:25} | {r['面弯角']:12} | {r['状态']}")
    print("="*60)
    print(f"\n结果已保存到: {os.path.abspath(args.output)}")

if __name__ == "__main__":
    main()