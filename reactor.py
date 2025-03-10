import os
import re
import shutil
import logging
import subprocess
import zipfile
from extract_module import extract_data, plot_extracted_data
from fragment_module import (
    parse_species_file, create_species_timeline,
    extract_coordinates, merge_xyz_files, find_first_occurrence
)
from utils import create_job_zip

logger = logging.getLogger(__name__)

def handle_uploaded_files(files, job_id):
    """处理上传的文件,如果目录不存在则创建,如有重复文件则覆盖"""
    if not job_id:
        return "Error: Job ID is required"
    
    if not files:
        return "No files uploaded"
    
    try:
        # 创建目录(如果不存在)
        job_dir = os.path.join(os.getcwd(), job_id)
        os.makedirs(job_dir, exist_ok=True)
        
        uploaded_files = []
        # 处理上传的文件
        for file in files:
            if file is not None:
                # 获取文件名
                file_name = os.path.basename(file.name)
                dest_path = os.path.join(job_dir, file_name)
                
                # 复制文件到目标目录
                shutil.copy(file.name, dest_path)
                uploaded_files.append(file_name)
        
        if uploaded_files:
            return f"Files uploaded successfully to {job_dir}: {', '.join(uploaded_files)}"
        else:
            return "No valid files to upload"
            
    except Exception as e:
        logger.error(f"Error uploading files: {str(e)}")
        return f"Error uploading files: {str(e)}"

def run_extra(parameters_file, job_id, uploaded_files=None):
    try:
        if not parameters_file or not job_id:
            return "Error: Missing parameters file or job ID", None, None
            
        # 解析参数文件
        keyword_pairs_columns = []
        with open(parameters_file.name, 'r') as file:
            for line in file:
                parts = [part.strip() for part in line.split(',')]
                if len(parts) >= 3:
                    keyword_pairs_columns.append((parts[0], parts[1], parts[2:]))
        
        # 读取日志文件
        log_file_path = os.path.join(job_id, "cp2k.log")
        if not os.path.exists(log_file_path):
            return f"Error: Log file not found at: {log_file_path}", None, None

        with open(log_file_path, 'r', encoding='utf-8') as file:
            log_file_content = file.read()

        # 提取数据并生成图表
        result, data = extract_data(log_file_content, keyword_pairs_columns)
        
        # 保存提取的数据
        output_dir = job_id
        output_file = os.path.join(output_dir, "extracted_data.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result)

        # 创建图表
        fig = plot_extracted_data(data)
        fig_file = os.path.join(output_dir, "extracted_plot.png")
        fig.write_image(fig_file)

        # 创建结果压缩文件
        zip_file = create_job_zip(job_id, os.getcwd())
        if zip_file:
            return f"Data extracted to {output_file}\nPlot saved to {fig_file}\nResults archived to {zip_file}", fig, zip_file
        else:
            return f"Data extracted to {output_file}\nPlot saved to {fig_file}\nFailed to create zip archive", fig, None
        
    except Exception as e:
        return f"Error in run_extra: {str(e)}", None, None

def run_net(job_id, uploaded_files=None):
    """处理上传的文件并运行reacnetgenerator分析"""
    try:
        # 检查必要文件是否存在
        xyz_file = os.path.join(job_id, "B-pos-1.xyz")
        if not os.path.exists(xyz_file):
            return "Error: B-pos-1.xyz file not found in job directory", None, None
            
        # 从coord.xyz文件中提取元素类型
        coord_file = os.path.join(job_id, "coord.xyz")
        if not os.path.exists(coord_file):
            return "Error: coord.xyz file not found in job directory", None, None
            
        # 读取并提取唯一的元素类型
        elements = set()
        with open(coord_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:  # 跳过空行
                    continue
                # 获取元素符号（一个或两个字母）
                element = parts[0]
                if element.isalpha():  # 确保是元素符号而不是数字或其他字符
                    elements.add(element)
        
        # 将元素类型转换为命令行参数
        elements_str = " ".join(sorted(elements))
        
        # 从.inp文件中提取周期性盒子尺寸
        cell_params = None
        for file in os.listdir(job_id):
            if file.endswith('.inp'):
                with open(os.path.join(job_id, file), 'r') as f:
                    content = f.read()
                    # 查找CELL部分
                    cell_match = re.search(r'&SUBSYS\s+&CELL(.*?)&END CELL', content, re.DOTALL)
                    if cell_match:
                        cell_section = cell_match.group(1)
                        # 提取A、B、C向量
                        vectors = []
                        for vector in ['A', 'B', 'C']:
                            vector_match = re.search(rf'{vector}\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)', cell_section)
                            if vector_match:
                                vectors.extend([float(x) for x in vector_match.groups()])
                        if len(vectors) == 9:
                            cell_params = " ".join(f"{x:.8f}" for x in vectors)
                        break
        
        if not cell_params:
            return "Error: Could not find cell parameters in .inp file", None, None
        
        # 构建完整的reacnetgenerator命令
        command = f"cd {job_id} && reacnetgenerator -i B-pos-1.xyz -a {elements_str} --type xyz --nohmm -c {cell_params}"
        logger.info(f"Executing command: {command}")
        
        # 执行命令
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            return f"Error executing ReacNetGenerator: {stderr}", None, None
            
        # 解析species文件
        species_file = os.path.join(job_id, "B-pos-1.xyz.species")
        if not os.path.exists(species_file):
            return "Error: Species file not generated by ReacNetGenerator", None, None
            
        species_data = parse_species_file(species_file)
        
        # 创建时间步长-片段分布图
        fig = create_species_timeline(species_data)
        
        return "Analysis completed successfully", fig, species_data

    except Exception as e:
        logger.error(f"Error in run_net: {str(e)}")
        return f"Error: {str(e)}", None, None

def on_timestep_select(job_id, timestep_input, species_data):
    if timestep_input is None or species_data is None:
        return None, None
    
    timestep_input = str(timestep_input).strip()
    all_timesteps = []
    
    if timestep_input.lower() == 'all':
        # 查找每个片段首次出现的时间步
        first_occurrence = find_first_occurrence(species_data)
        all_timesteps = sorted(set(first_occurrence.values()))
        fragments_info = [f"Fragment {frag} first appears at timestep {step}" 
                         for frag, step in first_occurrence.items()]
    else:
        # 解析输入的时间步
        try:
            all_timesteps = [int(t.strip()) for t in timestep_input.split(',')]
            # 验证时间步是否有效
            invalid_steps = [t for t in all_timesteps if t not in species_data]
            if invalid_steps:
                return f"Invalid timesteps: {invalid_steps}", None
            fragments_info = []
        except ValueError:
            return "Invalid input format. Use comma-separated numbers or 'all'", None
    
    # 收集所有输出文件
    all_output_files = []
    
    for timestep in all_timesteps:
        fragments = species_data[timestep]
        output_files = extract_coordinates(job_id, timestep, fragments)
        if output_files:
            all_output_files.extend(output_files)
            fragments_info.append(f"Timestep {timestep}: {len(fragments)} fragments")
    
    if all_output_files:
        # 创建合并的xyz文件
        merged_xyz_file = os.path.join(job_id, "merged_fragments.xyz")
        radical_xyz_file = os.path.join(job_id, "radical_fragments.xyz")
        molecule_xyz_file = os.path.join(job_id, "molecule_fragments.xyz")
        
        # 获取不同类型的内容
        merged_content, radical_content, molecule_content = merge_xyz_files(all_output_files)
        
        # 写入合并的文件
        with open(merged_xyz_file, 'w') as f:
            f.writelines(merged_content)
            
        # 写入自由基文件
        if radical_content:
            with open(radical_xyz_file, 'w') as f:
                f.writelines(radical_content)
                
        # 写入分子文件
        if molecule_content:
            with open(molecule_xyz_file, 'w') as f:
                f.writelines(molecule_content)
        
        # 创建包含所有文件的压缩包
        zip_file = os.path.join(job_id, f"fragments_all_steps.zip")
        with zipfile.ZipFile(zip_file, 'w') as zf:
            # 添加单独的片段文件
            for file in all_output_files:
                zf.write(file, os.path.basename(file))
                os.remove(file)
                
            # 添加合并的xyz文件
            zf.write(merged_xyz_file, os.path.basename(merged_xyz_file))
            if radical_content:
                zf.write(radical_xyz_file, os.path.basename(radical_xyz_file))
            if molecule_content:
                zf.write(molecule_xyz_file, os.path.basename(molecule_xyz_file))
            
            # 清理文件
            os.remove(merged_xyz_file)
            if radical_content:
                os.remove(radical_xyz_file)
            if molecule_content:
                os.remove(molecule_xyz_file)
        
        summary = "\n".join(fragments_info)
        return f"Extracted fragments:\n{summary}", zip_file
    else:
        return "Failed to extract coordinates", None
