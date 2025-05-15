import pymol
from PIL import Image
import os
import tempfile
import numpy as np
import re
from typing import Tuple, List, Optional

# 从jmolss.py导入颜色方案
from jmolss import ELEMENT_COLORS, rgb_to_hex

def get_elements_from_file(file_path: str) -> List[str]:
    """从文件中识别所有独特的元素类型"""
    elements = set()
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            element_matches = re.findall(r'(?:^|\s)([A-Z][a-z]?\d*)', content, re.MULTILINE)
            for match in element_matches:
                element = re.match(r'([A-Z][a-z]?)', match).group(1)
                if element in ELEMENT_COLORS:
                    elements.add(element)
    except Exception as e:
        print(f"Error reading file: {str(e)}")
    return sorted(list(elements))

def generate_color_legend(elements: List[str]) -> str:
    """生成颜色图例HTML"""
    if not elements:
        return ""
    
    legend_html = '<div style="padding: 10px; background-color: #f5f5f5; border-radius: 5px; margin-top: 10px;">'
    legend_html += '<div style="font-weight: bold; margin-bottom: 10px;">原子类型颜色图例：</div>'
    legend_html += '<div style="display: grid; grid-template-columns: repeat(auto-fill, minmax(80px, 1fr)); gap: 10px;">'
    
    for elem in elements:
        color_hex = rgb_to_hex(ELEMENT_COLORS[elem])
        legend_html += f'''
            <div style="display: flex; align-items: center; gap: 5px;">
                <div style="width: 20px; height: 20px; background-color: {color_hex}; border: 1px solid #ccc;"></div>
                <span>{elem}</span>
            </div>
        '''
    
    legend_html += '</div></div>'
    return legend_html

def count_xyz_frames(file_path: str) -> int:
    """计算 xyz 文件中的帧数"""
    try:
        frames = 0
        with open(file_path, 'r') as f:
            while True:
                natoms_line = f.readline()
                if not natoms_line:
                    break
                    
                try:
                    n_atoms = int(natoms_line.strip())
                except ValueError:
                    print(f"Invalid number of atoms line: {natoms_line}")
                    break
                
                f.readline()  # 跳过注释行
                
                # 跳过原子坐标行
                for _ in range(n_atoms):
                    if not f.readline():
                        return frames
                
                frames += 1
                
        return frames
    except Exception as e:
        print(f"Error counting frames: {str(e)}")
        return 0

def extract_frame_to_temp(file_path: str, frame_index: int) -> Optional[str]:
    """提取指定帧到临时文件"""
    try:
        with open(file_path, 'r') as f:
            # 跳过之前的帧
            for _ in range(frame_index):
                natoms_line = f.readline()
                if not natoms_line:
                    return None
                n_atoms = int(natoms_line.strip())
                
                for _ in range(n_atoms + 1):
                    if not f.readline():
                        return None
            
            # 读取目标帧的原子数
            natoms_line = f.readline()
            if not natoms_line:
                return None
            n_atoms = int(natoms_line.strip())
            
            # 创建临时文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xyz', mode='w')
            
            # 写入原子数
            temp_file.write(natoms_line)
            
            # 写入注释行和原子坐标
            for _ in range(n_atoms + 1):
                line = f.readline()
                if not line:
                    temp_file.close()
                    os.unlink(temp_file.name)
                    return None
                temp_file.write(line)
            
            temp_file.close()
            return temp_file.name
            
    except Exception as e:
        print(f"Error extracting frame: {str(e)}")
        return None

def display_molecule_pymol(file_path: str, frame_index: int, representation: str, 
                         rotations: List[float], zoom: float) -> Tuple[Optional[Image.Image], str, int]:
    """显示分子结构的指定帧"""
    try:
        if not os.path.exists(file_path):
            return None, "", 0
            
        # 计算总帧数
        total_frames = count_xyz_frames(file_path)
        if total_frames == 0:
            return None, "", 0
            
        # 提取指定帧到临时文件
        temp_frame_file = extract_frame_to_temp(file_path, frame_index)
        if temp_frame_file is None:
            return None, "", total_frames
            
        # 启动 PyMOL
        pymol.finish_launching(["pymol", "-q"])
        
        # 清理之前的分子
        pymol.cmd.delete('all')
        
        # 加载临时文件中的帧
        pymol.cmd.load(temp_frame_file)
        
        # 获取文件中的元素
        elements = get_elements_from_file(temp_frame_file)
        
        # 清除当前显示样式
        pymol.cmd.hide("everything")
        
        # 设置分子表示方式
        if representation == "ball_and_stick":
            pymol.cmd.show("sticks")
            pymol.cmd.show("spheres")
            pymol.cmd.set("stick_radius", 0.15)
            pymol.cmd.set("sphere_scale", 0.25)
        elif representation == "spacefill":
            pymol.cmd.show("spheres")
        elif representation == "wireframe":
            pymol.cmd.show("lines")
        elif representation == "surface":
            pymol.cmd.show("surface")
        else:  # sticks
            pymol.cmd.show("sticks")
            
        # 设置元素颜色
        for element in elements:
            color = ELEMENT_COLORS[element]
            pymol.cmd.set_color(f"color_{element}", color)
            pymol.cmd.color(f"color_{element}", f"elem {element}")
        
        # 应用旋转
        pymol.cmd.rotate("x", rotations[0])
        pymol.cmd.rotate("y", rotations[1])
        pymol.cmd.rotate("z", rotations[2])
        
        # 设置缩放
        pymol.cmd.zoom('all', float(zoom))
        
        # 渲染图像
        temp_png = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name
        pymol.cmd.png(temp_png, width=800, height=800, dpi=150)
        
        # 读取生成的图像
        img = Image.open(temp_png)
        
        # 生成颜色图例
        legend_html = generate_color_legend(elements)
        
        # 清理临时文件和 PyMOL 环境
        os.unlink(temp_frame_file)
        os.unlink(temp_png)
        pymol.cmd.delete('all')
        
        return img, legend_html, total_frames
        
    except Exception as e:
        print(f"Error in display_molecule_pymol: {str(e)}")
        return None, "", 0

def export_current_image(img: Optional[Image.Image], export_format: str) -> Optional[str]:
    """导出当前图像为指定格式"""
    if img is None:
        return None
        
    try:
        # 创建临时文件
        suffix = ".png" if export_format == "PNG" else ".jpg"
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        
        # 转换并保存图像
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
        
        if export_format == "PNG":
            img.save(temp_file.name, format="PNG")
        else:
            img.save(temp_file.name, format="JPEG", quality=95)
            
        return temp_file.name
    except Exception as e:
        print(f"Error exporting image: {str(e)}")
        return None
