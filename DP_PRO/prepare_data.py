"""
数据预处理脚本
用于准备选课名单和照片文件
"""
import pandas as pd
import os
import shutil
from pathlib import Path


def excel_to_csv(excel_path: str, csv_path: str = "student_list.csv"):
    """
    将 Excel 文件转换为 CSV 文件
    
    Args:
        excel_path: Excel 文件路径
        csv_path: 输出的 CSV 文件路径
    """
    print(f"正在读取 Excel 文件: {excel_path}")
    
    # 读取 Excel 文件
    # 先读取所有数据，然后手动处理B列和C列
    df_full = pd.read_excel(excel_path, header=None)
    
    print(f"Excel 文件总行数: {len(df_full)}")
    print(f"列数: {len(df_full.columns)}")
    print(f"前3行数据预览:\n{df_full.head(3)}")
    
    # 约定：B列（索引1）是学号，C列（索引2）是姓名，从第二行开始（索引1）
    if len(df_full.columns) < 3:
        raise ValueError("Excel 文件列数不足，需要至少3列（A、B、C列）")
    
    # B列是索引1，C列是索引2
    # 从第二行开始（索引1），跳过第一行（索引0）
    print("使用B列（索引1）作为学号，C列（索引2）作为姓名")
    print("从第二行开始读取数据（跳过第一行标题）")
    
    # 提取B列、C列，从第二行开始
    # 处理学号：如果是数字格式，转换为整数再转字符串（去掉 .0）
    student_ids_raw = df_full.iloc[1:, 1]  # B列，从第二行开始
    student_names = df_full.iloc[1:, 2].astype(str).str.strip()  # C列，从第二行开始
    
    # 将学号转换为字符串，如果是浮点数则先转整数
    student_ids = []
    for sid in student_ids_raw:
        if pd.isna(sid):
            student_ids.append('')
        else:
            # 尝试转换为整数（如果是浮点数如 23262010017.0）
            try:
                if isinstance(sid, float) and sid.is_integer():
                    student_ids.append(str(int(sid)))
                else:
                    student_ids.append(str(int(float(sid))))
            except (ValueError, TypeError):
                # 如果转换失败，直接转字符串
                student_ids.append(str(sid).strip())
    
    # 创建新的 DataFrame
    result_df = pd.DataFrame({
        '学号': student_ids,
        '姓名': student_names,
    })
    
    # 删除空行
    result_df = result_df.dropna(subset=['学号', '姓名'])
    result_df = result_df[result_df['学号'] != '']
    result_df = result_df[result_df['姓名'] != '']
    
    # 保存为 CSV
    result_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"\n已成功转换并保存为: {csv_path}")
    print(f"共 {len(result_df)} 个学生")
    
    return result_df


def rename_photos_by_name(photos_dir: str, student_list_df: pd.DataFrame, output_dir: str = "student_photos"):
    """
    根据学生姓名重命名照片文件为学号格式
    
    Args:
        photos_dir: 原始照片目录路径
        student_list_df: 包含学号和姓名的 DataFrame
        output_dir: 输出目录路径（照片将重命名为 {学号}.jpg 格式）
    """
    print(f"\n正在处理照片文件...")
    print(f"照片目录: {photos_dir}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 创建姓名到学号的映射
    name_to_id = {}
    for _, row in student_list_df.iterrows():
        name = str(row['姓名']).strip()
        student_id = str(row['学号']).strip()
        # 清理学号格式：如果是浮点数格式（如 23262010017.0），去掉 .0
        if student_id.endswith('.0'):
            student_id = student_id[:-2]
        name_to_id[name] = student_id
    
    print(f"已加载 {len(name_to_id)} 个学生的映射关系")
    
    # 获取所有照片文件（使用递归搜索，避免重复）
    photo_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        # 只使用递归搜索，会自动包含当前目录
        photo_files.extend(Path(photos_dir).glob(f'**/{ext}'))
    
    # 去重（转换为绝对路径后去重，避免重复处理同一文件）
    photo_files = list(set([str(p.resolve()) for p in photo_files]))
    photo_files = [Path(p) for p in photo_files]
    
    print(f"找到 {len(photo_files)} 个照片文件（已去重）")
    
    # 重命名并复制照片
    success_count = 0
    failed_files = []
    processed_students = set()  # 记录已处理的学生，避免重复
    
    for photo_path in photo_files:
        # 获取文件名（不含扩展名）
        name_without_ext = photo_path.stem
        
        # 查找对应的学号
        if name_without_ext in name_to_id:
            student_id = name_to_id[name_without_ext]
            
            # 如果该学生已经处理过，跳过（避免重复）
            if student_id in processed_students:
                print(f"  跳过: {name_without_ext} (学生 {student_id} 已有照片)")
                continue
            
            # 获取原始扩展名
            ext = photo_path.suffix.lower()
            if ext == '.jpeg':
                ext = '.jpg'
            
            # 目标文件路径
            target_path = os.path.join(output_dir, f"{student_id}{ext}")
            
            # 如果目标文件已存在，跳过（避免覆盖）
            if os.path.exists(target_path):
                print(f"  跳过: {name_without_ext} (目标文件 {student_id}{ext} 已存在)")
                processed_students.add(student_id)
                continue
            
            # 复制文件
            shutil.copy2(photo_path, target_path)
            print(f"  {name_without_ext} -> {student_id}{ext}")
            success_count += 1
            processed_students.add(student_id)
        else:
            failed_files.append(name_without_ext)
            print(f"  警告: 未找到 {name_without_ext} 对应的学号")
    
    print(f"\n处理完成:")
    print(f"  成功: {success_count} 个")
    print(f"  失败: {len(failed_files)} 个")
    
    if failed_files:
        print(f"\n未匹配的照片文件:")
        for name in failed_files[:10]:  # 只显示前10个
            print(f"  - {name}")
        if len(failed_files) > 10:
            print(f"  ... 还有 {len(failed_files) - 10} 个")
    
    return output_dir


def main():
    """主函数"""
    print("="*60)
    print("数据预处理工具")
    print("="*60)
    
    # 1. 转换 Excel 为 CSV
    excel_path = "选课名单.xlsx"
    csv_path = "student_list.csv"
    
    if not os.path.exists(excel_path):
        print(f"错误: 未找到文件 {excel_path}")
        return
    
    try:
        student_list_df = excel_to_csv(excel_path, csv_path)
    except Exception as e:
        print(f"转换 Excel 文件时出错: {e}")
        return
    
    # 2. 重命名照片文件
    # 尝试多个可能的照片目录路径
    possible_photo_dirs = [
        "学生照片目录",
        "学生照片目录/25秋深度学习应用_选课同学照片_55人",
        "学生照片目录/25秋深度学习应用_选课同学照片_55人"
    ]
    
    photos_dir = None
    for dir_path in possible_photo_dirs:
        if os.path.exists(dir_path):
            photos_dir = dir_path
            break
    
    output_dir = "student_photos"
    
    if photos_dir is None:
        print(f"警告: 未找到照片目录")
        print("尝试的路径:")
        for dir_path in possible_photo_dirs:
            print(f"  - {dir_path}")
        print("\n请手动将照片文件重命名为 {学号}.jpg 格式")
        print("或者将照片放在 '学生照片目录' 文件夹中")
        return
    
    try:
        rename_photos_by_name(photos_dir, student_list_df, output_dir)
        print(f"\n照片已处理完成，输出目录: {output_dir}")
    except Exception as e:
        print(f"处理照片文件时出错: {e}")
        return
    
    print("\n" + "="*60)
    print("预处理完成！")
    print("="*60)
    print(f"\n生成的文件:")
    print(f"  1. 选课名单: {csv_path}")
    print(f"  2. 学生照片目录: {output_dir}/")
    print(f"\n现在可以使用以下命令运行分析:")
    print(f"  python main.py --video 教室视频/视频文件名.mp4 --photos {output_dir} --list {csv_path} --task 1")


if __name__ == "__main__":
    main()

