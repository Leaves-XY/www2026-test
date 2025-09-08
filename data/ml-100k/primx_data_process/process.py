import pandas as pd
import numpy as np

def process_ml100k_data():
    """
    处理MovieLens 100k数据集，执行5-core过滤、ID重映射和排序
    """
    
    # 1. 加载数据
    print("正在加载u.data文件...")
    df = pd.read_csv('u.data', sep='\t', header=None, 
                     names=['user_id', 'item_id', 'rating', 'timestamp'])
    
    print(f"原始数据: {len(df)} 条记录, {df['user_id'].nunique()} 个用户, {df['item_id'].nunique()} 个物品")
    
    # 2. 执行5-core过滤
    print("\n开始执行5-core过滤...")
    
    prev_size = len(df)
    iteration = 0
    
    while True:
        iteration += 1
        print(f"第 {iteration} 轮过滤...")
        
        # 按用户过滤：移除交互次数少于5次的用户
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= 5].index
        df = df[df['user_id'].isin(valid_users)]
        
        # 按物品过滤：移除交互次数少于5次的物品
        item_counts = df['item_id'].value_counts()
        valid_items = item_counts[item_counts >= 5].index
        df = df[df['item_id'].isin(valid_items)]
        
        current_size = len(df)
        print(f"  过滤后: {current_size} 条记录, {df['user_id'].nunique()} 个用户, {df['item_id'].nunique()} 个物品")
        
        # 如果数据大小没有变化，停止过滤
        if current_size == prev_size:
            print("5-core过滤完成！")
            break
            
        prev_size = current_size
    
    # 3. 重映射ID
    print("\n开始重映射ID...")
    
    # 获取唯一的用户和物品ID，并排序
    unique_users = sorted(df['user_id'].unique())
    unique_items = sorted(df['item_id'].unique())
    
    # 创建映射字典（从1开始）
    user_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_users, 1)}
    item_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_items, 1)}
    
    # 应用映射
    df['remapped_user_id'] = df['user_id'].map(user_mapping)
    df['remapped_item_id'] = df['item_id'].map(item_mapping)
    
    print(f"重映射完成: 用户ID 1-{len(unique_users)}, 物品ID 1-{len(unique_items)}")
    
    # 4. 排序数据
    print("\n正在排序数据...")
    df = df.sort_values(['remapped_user_id', 'timestamp'])
    
    # 5. 生成输出文件
    print("\n正在生成ml-100k.txt文件...")
    
    with open('ml-100k.txt', 'w') as f:
        for _, row in df.iterrows():
            f.write(f"{row['remapped_user_id']} {row['remapped_item_id']}\n")
    
    print(f"处理完成！输出文件包含 {len(df)} 行数据")
    print(f"最终数据集: {df['remapped_user_id'].nunique()} 个用户, {df['remapped_item_id'].nunique()} 个物品")
    
    return df

if __name__ == "__main__":
    try:
        processed_df = process_ml100k_data()
        print("\n脚本执行成功！")
    except FileNotFoundError:
        print("错误: 找不到u.data文件，请确保文件在当前目录中")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")