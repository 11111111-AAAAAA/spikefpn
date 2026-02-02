import os
import pathlib
import numpy as np
import tqdm
import argparse
from src.io.psee_loader import PSEELoader

def parse_args():
    parser = argparse.ArgumentParser(description="N-CARS framing (Batch Processing)")
    # Kaggle 默认路径
    parser.add_argument("-rp", "--root_path", type=str, default="/kaggle/input/ncars-original/Prophesee_Dataset_n_cars")
    parser.add_argument("-sp", "--save_path", type=str, default="/kaggle/working/spikefpn/ncars_framing")
    return parser.parse_args()

def process_data():
    args = parse_args()
    
    # 嵌套循环：遍历所有模式和类别
    modes = ["train", "test"]
    classes = ["cars", "background"]
    
    # 参数设置
    S = 10
    C = 1
    frame_inteval_ms = 10
    
    for mode in modes:
        for data_class in classes:
            input_dir = f"{args.root_path}/n-cars_{mode}/{data_class}"
            
            # 检查输入路径是否存在
            if not os.path.exists(input_dir):
                print(f"警告: 路径不存在 {input_dir}, 跳过...")
                continue
            
            # 获取当前文件夹下所有文件
            file_names = [pathlib.Path(item).as_posix() for item in os.scandir(input_dir) if item.is_file()]
            
            print(f"\n正在处理: {mode} - {data_class} | 文件数量: {len(file_names)}")
            
            # 设定保存路径
            save_dir = f"{args.save_path}/SBT{frame_inteval_ms}ms_S{S}C{C}/{mode}_{data_class}"
            os.makedirs(save_dir, exist_ok=True)
            
            # 开始处理当前分类下的所有文件
            for file_path in tqdm.tqdm(file_names):
                try:
                    video = PSEELoader(file_path) 
                    event_list = []
                    
                    # 1. 加载事件
                    while not video.done:
                        # 加载 (t, x, y, p) 事件
                        event = video.load_delta_t(frame_inteval_ms * 1e3)
                        if len(event) > 0:
                            event_list.append(event)
                    
                    if not event_list:
                        continue
                        
                    # 2. 确定画面尺寸
                    event_stream = np.concatenate(event_list)
                    height = event_stream["y"].max() + 1
                    width = event_stream["x"].max() + 1
                
                    # 3. SBT Framing (Stacking Based on Time)
                    frames = np.zeros(
                        shape=(len(event_list), height, width), 
                        dtype=np.int8, # 使用 int8 节省一半内存和磁盘空间
                    )
                    
                    for index, event in enumerate(event_list):
                        # 极性映射: 0->-1, 1->1
                        frames[index, event["y"], event["x"]] = 2 * event["p"].astype(np.int8) - 1
                    
                    # 调整通道数 (T, H, W) -> (T, C, H, W)
                    frames = frames.reshape(-1, C, *frames.shape[1:])
                
                    # 4. 保存文件
                    # 使用 os.path.basename 确保跨平台兼容性
                    base_name = os.path.basename(file_path)
                    # 提取原始文件名中的 ID 部分，例如 obj_0000000000000000000.npy
                    file_id = base_name.split("_")[1] if "_" in base_name else base_name
                    save_name = f"obj_{file_id.replace('.dat', '.npy')}"
                    
                    np.save(os.path.join(save_dir, save_name), frames)
                    
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {e}")
                    continue
    
    print("\n所有数据集 Framing 处理完毕！")

if __name__ == "__main__":
    process_data()
