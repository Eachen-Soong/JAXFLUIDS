import os
import argparse
from pathlib import Path

def search_files(directory: str, search_string: str, encoding: str = 'utf-8') -> list:
    """
    在指定目录下递归搜索包含特定字符串的文件
    
    参数:
        directory: 要搜索的目录路径
        search_string: 要查找的字符串
        encoding: 文件编码，默认为utf-8
    
    返回:
        包含搜索字符串的文件路径列表
    """
    matched_files = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    if search_string in f.read():
                        matched_files.append(file_path)
            except (UnicodeDecodeError, PermissionError, OSError) as e:
                print(f"无法读取文件 {file_path}: {e}")
    
    return matched_files

def main():
    parser = argparse.ArgumentParser(description='搜索目录中包含特定字符串的文件')
    parser.add_argument('directory', help='要搜索的目录路径')
    parser.add_argument('search_string', help='要查找的字符串')
    parser.add_argument('--encoding', default='utf-8', help='文件编码，默认为utf-8')
    parser.add_argument('--output', help='输出结果的文件路径')
    
    args = parser.parse_args()
    
    # 检查目录是否存在
    if not os.path.isdir(args.directory):
        print(f"错误: 指定的目录 '{args.directory}' 不存在")
        return
    
    print(f"开始在目录 '{args.directory}' 中搜索包含字符串 '{args.search_string}' 的文件...")
    
    matched_files = search_files(args.directory, args.search_string, args.encoding)
    
    if matched_files:
        print(f"找到 {len(matched_files)} 个匹配的文件:")
        for file in matched_files:
            print(file)
        
        # 如果指定了输出文件，则保存结果
        if args.output:
            output_dir = os.path.dirname(args.output)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            with open(args.output, 'w', encoding='utf-8') as f:
                f.write('\n'.join(matched_files))
            print(f"结果已保存到 {args.output}")
    else:
        print("未找到匹配的文件")

if __name__ == "__main__":
    main()    