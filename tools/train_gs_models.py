import os
import argparse
from queue import Queue
import subprocess
from tqdm import tqdm
import time
import shutil


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train the 3DGS models for all objects')
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--threads", type=int, required=True)
    
    args = parser.parse_args()
    
    obj_folders = os.listdir(args.data)
    obj_folders.sort(key=lambda x: int(x.split('-')[0]))
    
    # Build commands
    commands = []
    for obj_folder in obj_folders:
        train_data = os.path.join(args.data, obj_folder, '3DGS')
        output_path = os.path.join(args.output, obj_folder)
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path)
        # train the 3DGS model
        cmd = f'python -m src.module.3DGS.trainer default  --data_dir {train_data} \
        --data_factor 1 --result_dir {output_path} --init_type bbox --feature_rendering --disable_viewer  --strategy.prune-opa 0.1  --batch_size 4 --steps_scaler 0.25'
        # subprocess.run(cmd, shell=True)
        commands.append(cmd)

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        gpu_indexs = [int(x) for x in cuda_visible.split(",")]
    else:
        gpu_indexs = list(range(args.gpus))
    
    # Initialize command queue
    queue = Queue()
    for command in commands:
        queue.put(command)
    
    # Track the processes
    processes = []
    
    def start_command(command, gpu_id):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        process = subprocess.Popen(command, shell=True,  env=env, )
        print(f'Start command: "{command}" on GPU {gpu_id}')
        return process
    
    # progress bar
    progress = tqdm(total=len(commands), desc="Progress")
    
    for thread_id in range(args.threads):
        if not queue.empty():
            command = queue.get()
            gpu_id = thread_id%args.gpus
            process = start_command(command, gpu_indexs[gpu_id])
            processes.append((process, gpu_id))
            # gpu_assignments[gpu_id] += 1
            time.sleep(120)
            
    while not queue.empty() or any(p[0].poll() is None for p in processes):
        for i, (process, gpu_id) in enumerate(processes):
            if process.poll() is not None:  # 进程已完成
                progress.update(1)  # 更新进度条
                # gpu_assignments[gpu_id] -= 1
                if not queue.empty():  # 启动新命令
                    command = queue.get()
                    new_process = start_command(command, gpu_indexs[gpu_id])
                    processes[i] = (new_process, gpu_id)
                    # gpu_assignments[gpu_id] += 1
                    time.sleep(120)
    
        time.sleep(30)
    
    # Wait for finishing all processes
    for process, _ in processes:
        process.wait()
        progress.update(1)  # 更新进度条
    
    progress.close()  # 关闭进度条