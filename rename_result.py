import os
import argparse

def rename_result_files(folder_path):
    if not os.path.exists(folder_path):
        print(f"错误：目录不存在 - {folder_path}")
        return
    
    count = 0
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            if filename.lower().startswith('result_'):
                old_path = os.path.join(root, filename)
                new_name = filename[7:]
                new_path = os.path.join(root, new_name)
                os.rename(old_path, new_path)
                print(f'{filename} -> {new_name}')
                count += 1
    
    print(f'\n共重命名 {count} 个文件')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='批量去掉文件名中的 result_ 前缀')
    parser.add_argument('-p', '--path', default=r'D:\Gemini_Projects\glasses_face_angle\result',
                        help=r'目标文件夹路径（默认：D:\Gemini_Projects\glasses_face_angle\result）')
    args = parser.parse_args()
    
    rename_result_files(args.path)