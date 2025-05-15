#该代码调用pymol进行可视化并提供帧选择和图像下载
import gradio as gr
import pymol
from PIL import Image
import io
import numpy as np
import re
import tempfile
import os

# 完整元素周期表的默认颜色方案 (RGB 格式, 0-1范围)
ELEMENT_COLORS = {
    'H': (0.90, 0.90, 0.90),   # 白色
    'He': (0.85, 1.00, 1.00),  # 浅青
    'Li': (0.80, 0.50, 1.00),  # 紫色
    'Be': (0.76, 1.00, 0.00),  # 深绿
    'B': (1.00, 0.71, 0.71),   # 粉色
    'C': (0.56, 0.56, 0.56),   # 灰色
    'N': (0.19, 0.31, 0.97),   # 蓝色
    'O': (1.00, 0.05, 0.05),   # 红色
    'F': (0.56, 0.88, 0.31),   # 浅绿
    'Ne': (0.70, 0.89, 0.96),  # 浅蓝
    'Na': (0.67, 0.36, 0.95),  # 紫色
    'Mg': (0.54, 1.00, 0.00),  # 亮绿
    'Al': (0.75, 0.65, 0.65),  # 灰粉
    'Si': (0.94, 0.78, 0.63),  # 金色
    'P': (1.00, 0.50, 0.00),   # 橙色
    'S': (1.00, 1.00, 0.19),   # 黄色
    'Cl': (0.12, 0.94, 0.12),  # 绿色
    'Ar': (0.50, 0.82, 0.89),  # 青色
    'K': (0.56, 0.25, 0.83),   # 紫色
    'Ca': (0.24, 1.00, 0.00),  # 亮绿
    'Sc': (0.90, 0.90, 0.90),  # 银色
    'Ti': (0.75, 0.76, 0.78),  # 灰色
    'V': (0.65, 0.65, 0.67),   # 灰色
    'Cr': (0.54, 0.60, 0.78),  # 蓝灰
    'Mn': (0.61, 0.48, 0.78),  # 紫灰
    'Fe': (0.88, 0.40, 0.20),  # 褐色
    'Co': (0.94, 0.56, 0.63),  # 粉红
    'Ni': (0.31, 0.82, 0.31),  # 绿色
    'Cu': (0.78, 0.50, 0.20),  # 铜色
    'Zn': (0.49, 0.50, 0.69),  # 深灰蓝
    'Ga': (0.76, 0.56, 0.56),  # 粉褐
    'Ge': (0.40, 0.56, 0.56),  # 灰绿
    'As': (0.74, 0.50, 0.89),  # 紫色
    'Se': (1.00, 0.63, 0.00),  # 橙色
    'Br': (0.65, 0.16, 0.16),  # 褐红
    'Kr': (0.36, 0.72, 0.82),  # 青色
    'Rb': (0.44, 0.18, 0.69),  # 深紫
    'Sr': (0.00, 1.00, 0.00),  # 绿色
    'Y': (0.58, 1.00, 1.00),   # 青色
    'Zr': (0.58, 0.88, 0.88),  # 浅青
    'Nb': (0.45, 0.76, 0.79),  # 青色
    'Mo': (0.33, 0.71, 0.71),  # 青绿
    'Tc': (0.23, 0.62, 0.62),  # 深青
    'Ru': (0.14, 0.56, 0.56),  # 深青
    'Rh': (0.04, 0.49, 0.55),  # 深青
    'Pd': (0.00, 0.41, 0.52),  # 深青
    'Ag': (0.75, 0.75, 0.75),  # 银色
    'Cd': (1.00, 0.85, 0.56),  # 金色
    'In': (0.65, 0.46, 0.45),  # 褐色
    'Sn': (0.40, 0.50, 0.50),  # 灰色
    'Sb': (0.62, 0.39, 0.71),  # 紫色
    'Te': (0.83, 0.48, 0.00),  # 褐色
    'I': (0.58, 0.00, 0.58),   # 深紫
    'Xe': (0.26, 0.62, 0.69),  # 青色
    'Cs': (0.34, 0.09, 0.56),  # 深紫
    'Ba': (0.00, 0.79, 0.00),  # 深绿
    'La': (0.44, 0.83, 1.00),  # 浅蓝
    'Ce': (1.00, 1.00, 0.78),  # 浅黄
    'Pr': (0.85, 1.00, 0.78),  # 浅绿
    'Nd': (0.78, 1.00, 0.78),  # 浅绿
    'Pm': (0.64, 1.00, 0.78),  # 浅绿
    'Sm': (0.56, 1.00, 0.78),  # 浅绿
    'Eu': (0.38, 1.00, 0.78),  # 浅绿
    'Gd': (0.27, 1.00, 0.78),  # 浅绿
    'Tb': (0.19, 1.00, 0.78),  # 浅绿
    'Dy': (0.12, 1.00, 0.78),  # 浅绿
    'Ho': (0.00, 1.00, 0.61),  # 绿色
    'Er': (0.00, 0.90, 0.46),  # 绿色
    'Tm': (0.00, 0.83, 0.32),  # 绿色
    'Yb': (0.00, 0.75, 0.22),  # 绿色
    'Lu': (0.00, 0.67, 0.14),  # 深绿
    'Hf': (0.30, 0.76, 1.00),  # 浅蓝
    'Ta': (0.30, 0.65, 1.00),  # 蓝色
    'W': (0.13, 0.58, 0.84),   # 蓝色
    'Re': (0.15, 0.49, 0.67),  # 蓝色
    'Os': (0.15, 0.40, 0.59),  # 深蓝
    'Ir': (0.09, 0.33, 0.53),  # 深蓝
    'Pt': (0.96, 0.93, 0.82),  # 金色
    'Au': (1.00, 0.82, 0.14),  # 金色
    'Hg': (0.72, 0.72, 0.82),  # 银色
    'Tl': (0.65, 0.33, 0.30),  # 褐色
    'Pb': (0.34, 0.35, 0.38),  # 深灰
    'Bi': (0.62, 0.31, 0.71),  # 紫色
    'Po': (0.67, 0.36, 0.00),  # 褐色
    'At': (0.46, 0.31, 0.27),  # 褐色
    'Rn': (0.26, 0.51, 0.59),  # 青色
    'Fr': (0.26, 0.00, 0.40),  # 深紫
    'Ra': (0.00, 0.49, 0.00),  # 深绿
    'Ac': (0.44, 0.67, 0.98),  # 浅蓝
    'Th': (0.00, 0.73, 1.00),  # 蓝色
    'Pa': (0.00, 0.63, 1.00),  # 蓝色
    'U': (0.00, 0.56, 1.00),   # 蓝色
    'Np': (0.00, 0.50, 1.00),  # 蓝色
    'Pu': (0.00, 0.42, 1.00),  # 深蓝
    'Am': (0.33, 0.36, 0.95),  # 深蓝
    'Cm': (0.47, 0.36, 0.89),  # 紫色
    'Bk': (0.54, 0.31, 0.89),  # 紫色
    'Cf': (0.63, 0.21, 0.83),  # 紫色
    'Es': (0.70, 0.12, 0.83),  # 紫色
    'Fm': (0.70, 0.12, 0.73),  # 深紫
    'Md': (0.70, 0.05, 0.65),  # 深紫
    'No': (0.74, 0.05, 0.53),  # 深紫
    'Lr': (0.78, 0.00, 0.40),  # 深紫
    'Rf': (0.80, 0.00, 0.35),  # 深紫
    'Db': (0.82, 0.00, 0.31),  # 深紫
    'Sg': (0.85, 0.00, 0.27),  # 深紫
    'Bh': (0.88, 0.00, 0.22),  # 深紫
    'Hs': (0.90, 0.00, 0.18),  # 深紫
    'Mt': (0.92, 0.00, 0.15),  # 深紫
}

def rgb_to_hex(rgb):
    """将RGB元组(0-1范围)转换为十六进制颜色代码"""
    return '#{:02x}{:02x}{:02x}'.format(
        int(rgb[0] * 255),
        int(rgb[1] * 255),
        int(rgb[2] * 255)
    )

def get_elements_from_file(file_path):
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

def generate_color_legend(elements):
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

def count_xyz_frames(file_path):
    """计算 xyz 文件中的帧数"""
    try:
        frames = 0
        with open(file_path, 'r') as f:
            while True:
                # 读取第一行（原子数）
                natoms_line = f.readline()
                if not natoms_line:  # 到达文件末尾
                    break
                    
                try:
                    n_atoms = int(natoms_line.strip())
                except ValueError:
                    print(f"Invalid number of atoms line: {natoms_line}")
                    break
                
                # 跳过注释行
                f.readline()
                
                # 跳过原子坐标行
                for _ in range(n_atoms):
                    if not f.readline():  # 确保能读到足够的原子坐标
                        return frames
                
                frames += 1
                
        return frames
    except Exception as e:
        print(f"Error counting frames: {str(e)}")
        return 0

def extract_frame_to_temp(file_path, frame_index):
    """提取指定帧到临时文件"""
    try:
        with open(file_path, 'r') as f:
            # 跳过之前的帧
            for _ in range(frame_index):
                # 读取原子数
                natoms_line = f.readline()
                if not natoms_line:
                    return None
                n_atoms = int(natoms_line.strip())
                
                # 跳过注释行和原子坐标
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

def get_elements_from_file(file_path):
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

def generate_color_legend(elements):
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

def display_molecule_pymol(file, frame_index, representation, rotations, zoom):
    """显示分子结构的指定帧"""
    try:
        if file is None:
            return None, "", 0
            
        # 计算总帧数
        total_frames = count_xyz_frames(file.name)
        if total_frames == 0:
            return None, "", 0
            
        # 提取指定帧到临时文件
        temp_frame_file = extract_frame_to_temp(file.name, frame_index)
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
        pymol.cmd.png("temp.png", width=800, height=800, dpi=150)
        
        # 读取生成的图像
        img = Image.open("temp.png")
        
        # 生成颜色图例
        legend_html = generate_color_legend(elements)
        
        # 清理临时文件和 PyMOL 环境
        os.unlink(temp_frame_file)
        pymol.cmd.delete('all')
        
        return img, legend_html, total_frames
        
    except Exception as e:
        print(f"Error in display_molecule_pymol: {str(e)}")
        return None, "", 0

def export_current_image(img, export_format):
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

def process_molecule(file, frame_index, representation, rotation_x, rotation_y, rotation_z, zoom):
    """处理分子文件并返回图像和图例"""
    if file is None:
        return None, "", gr.Slider(visible=False), None
    
    img, legend, total_frames = display_molecule_pymol(
        file=file,
        frame_index=frame_index,
        representation=representation,
        rotations=[rotation_x, rotation_y, rotation_z],
        zoom=zoom
    )
    
    # 更新帧滑块的范围
    if total_frames > 0:
        return img, legend, gr.Slider(minimum=0, maximum=total_frames-1, step=1, visible=True), img
    else:
        return img, legend, gr.Slider(visible=False), img

# 创建 Gradio 界面
with gr.Blocks(title="分子结构可视化工具") as iface:
    gr.Markdown("## 分子结构可视化工具")
    
    current_image = gr.State(None)  # 存储当前图像状态
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                label="上传分子文件（支持 .xyz 等格式）"
            )
            
            frame_slider = gr.Slider(
                minimum=0, maximum=0, step=1, value=0,
                label="选择帧", visible=False
            )
            
            representation = gr.Dropdown(
                choices=["sticks", "ball_and_stick", "spacefill", "wireframe", "surface"],
                value="sticks",
                label="分子表示方式",
                info="选择分子的显示方式"
            )
            
            with gr.Row():
                rotation_x = gr.Slider(-180, 180, value=0, label="X轴旋转")
                rotation_y = gr.Slider(-180, 180, value=0, label="Y轴旋转")
                rotation_z = gr.Slider(-180, 180, value=0, label="Z轴旋转")
            
            zoom = gr.Slider(0.1, 5.0, value=1.0, label="缩放", info="调整分子大小")
            
            with gr.Row():
                export_format = gr.Dropdown(
                    choices=["PNG", "JPG"],
                    value="PNG",
                    label="导出格式",
                    interactive=True
                )
                export_btn = gr.Button("导出图像")
            
            export_path = gr.File(label="导出的图像", interactive=False)
        
        with gr.Column(scale=2):
            output_image = gr.Image(label="分子结构图像")
            color_legend = gr.HTML(label="颜色图例")
    
    # 设置事件处理
    input_components = [
        file_input,
        frame_slider,
        representation,
        rotation_x,
        rotation_y,
        rotation_z,
        zoom
    ]
    
    # 文件上传时更新帧滑块和显示
    file_input.change(
        fn=process_molecule,
        inputs=input_components,
        outputs=[output_image, color_legend, frame_slider, current_image]
    )
    
    # 其他参数变化时更新显示
    def update_display(*args):
        img, legend, _, new_img = process_molecule(*args)
        return img, legend, new_img
    
    for component in [frame_slider, representation, rotation_x, rotation_y, rotation_z, zoom]:
        component.change(
            fn=update_display,
            inputs=input_components,
            outputs=[output_image, color_legend, current_image]
        )
    
    # 导出图像按钮事件
    export_btn.click(
        fn=export_current_image,
        inputs=[current_image, export_format],
        outputs=[export_path]
    )

# 启动界面
if __name__ == "__main__":
    iface.launch()