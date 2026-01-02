"""
使用示例（仅任务1/任务2）

说明：这里不再通过旧函数调用，而是给出推荐命令。
"""


def main():
    print("任务1（出勤+seat_map）示例：")
    print('  python -u main.py --video "教室视频/1105.mp4" --task 1 --photos student_photos_processed --list student_list.csv')
    print()
    print("任务2（个人模式）示例：")
    print('  python -u main.py --video "教室视频/1105.mp4" --task 2 --t2-mode person --t2-interactive-roi --t2-save-marked-images --t2-device cuda')
    print()


if __name__ == "__main__":
    main()

