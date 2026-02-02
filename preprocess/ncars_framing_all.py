import os
import pathlib
import numpy as np
import tqdm
import argparse
from src.io.psee_loader import PSEELoader

def parse_args():
    parser = argparse.ArgumentParser(description="N-CARS framing with Train/Val/Test Split")
    parser.add_argument("-rp", "--root_path", type=str, default="/kaggle/input/ncars-original/Prophesee_Dataset_n_cars")
    parser.add_argument("-sp", "--save_path", type=str, default="/kaggle/working/spikefpn/ncars_framing")
    return parser.parse_args()

def get_target_mode(current_mode, data_class, file_id):
    """
    根据原作者提供的索引划分逻辑确定存入哪个文件夹
    """
    if current_mode == "test":
        return "test"
    
    try:
        idx = int(file_id)
    except ValueError:
        return "train" # 如果解析失败，默认入 train
    
    if data_class == "cars":
        if 0 <= idx <= 4395: return "train"
        if 4396 <= idx <= 5983: return "validate"
        return "train" # 剩余部分
    else: # background
        if 0 <= idx <= 4210: return "train"
        if 4211 <= idx <= 5706: return "validate"
        return "train"

def process_data():
    args = parse_args()
    modes = ["train", "test"]
    classes = ["cars", "background"]
    
    S, C, frame_inteval_ms = 10, 1, 10
    
    for mode in modes:
        for data_class in classes:
            input_dir = f"{args.root_path}/n-cars_{mode}/{data_class}"
            if not os.path.exists(input_dir):
                print(f"Skipping missing path: {input_dir}")
                continue
            
            file_names = [pathlib.Path(item).as_posix() for item in os.scandir(input_dir) if item.is_file()]
            print(f"\nProcessing {mode} - {data_class}...")
            
            for file_path in tqdm.tqdm(file_names):
                # 1. 解析 File ID 以确定划分
                base_name = os.path.basename(file_path)
                # 假设文件名格式为 obj_0000000000000000000.dat
                file_id_str = base_name.split("_")[1].split(".")[0] if "_" in base_name else "0"
                
                # 确定最终保存的子文件夹 (train/validate/test)
                target_mode = get_target_mode(mode, data_class, file_id_str)
                
                # 2. 准备保存路径
                save_dir = f"{args.save_path}/SBT{frame_inteval_ms}ms_S{S}C{C}/{target_mode}_{data_class}"
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"obj_{file_id_str}.npy")

                # 3. 如果文件已存在则跳过（方便断点续跑）
                if os.path.exists(save_path):
                    continue

                try:
                    video = PSEELoader(file_path) 
                    event_list = []
                    while not video.done:
                        event = video.load_delta_t(frame_inteval_ms * 1e3)
                        if len(event) > 0: event_list.append(event)
                    
                    if not event_list: continue
                        
                    event_stream = np.concatenate(event_list)
                    height, width = event_stream["y"].max() + 1, event_stream["x"].max() + 1
                
                    frames = np.zeros(shape=(len(event_list), height, width), dtype=np.int8)
                    for index, event in enumerate(event_list):
                        frames[index, event["y"], event["x"]] = 2 * event["p"].astype(np.int8) - 1
                    
                    frames = frames.reshape(-1, C, *frames.shape[1:])
                    np.save(save_path, frames)
                    
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")

    print("\n[Done] All splits (train, validate, test) are ready!")

if __name__ == "__main__":
    process_data()
