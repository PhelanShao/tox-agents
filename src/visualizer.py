import pymol
from PIL import Image, ImageDraw, ImageFont
import os
import tempfile
import numpy as np
import re
from typing import Tuple, List, Optional

# 从jmolss.py导入颜色方案
from jmolss import ELEMENT_COLORS, rgb_to_hex

def create_fallback_image(width=800, height=800):
    """创建备用图像，当PyMOL渲染失败时使用"""
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)

    # 绘制一个简单的分子图标
    center_x, center_y = width // 2, height // 2

    # 绘制原子（圆圈）
    atom_radius = 30
    draw.ellipse([center_x - atom_radius, center_y - atom_radius,
                  center_x + atom_radius, center_y + atom_radius],
                 fill='red', outline='black', width=2)

    # 绘制键（线条）
    bond_length = 80
    for angle in [0, 60, 120, 180, 240, 300]:
        end_x = center_x + bond_length * np.cos(np.radians(angle))
        end_y = center_y + bond_length * np.sin(np.radians(angle))
        draw.line([center_x, center_y, end_x, end_y], fill='black', width=3)

        # 在末端绘制小原子
        small_radius = 20
        draw.ellipse([end_x - small_radius, end_y - small_radius,
                      end_x + small_radius, end_y + small_radius],
                     fill='blue', outline='black', width=2)

    # 添加文字说明
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 24)
    except:
        font = ImageFont.load_default()

    text = "Molecular Structure\n(Fallback Rendering)"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (width - text_width) // 2
    text_y = height - text_height - 20

    draw.text((text_x, text_y), text, fill='black', font=font)

    return img

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
    legend_html += '<div style="font-weight: bold; margin-bottom: 10px;">Element：</div>'
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
    """计算文件中的帧数，支持XYZ和NPZ格式"""
    try:
        # 检查文件扩展名
        if file_path.lower().endswith('.npz'):
            # 处理NPZ文件
            data = np.load(file_path, allow_pickle=True)
            if 'id' in data:
                return len(data['id'])
            elif 'coord' in data:
                return len(data['coord'])
            else:
                print(f"NPZ file does not contain expected keys: {list(data.keys())}")
                return 0
        else:
            # 处理XYZ文件
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
    """提取指定帧到临时文件，支持XYZ和NPZ格式"""
    try:
        if file_path.lower().endswith('.npz'):
            # 处理NPZ文件
            data = np.load(file_path, allow_pickle=True)
            
            if frame_index >= len(data['coord']):
                print(f"Frame index {frame_index} out of range (max: {len(data['coord'])-1})")
                return None
            
            # 获取指定帧的数据
            symbols = data['symbol'][frame_index]
            coords = data['coord'][frame_index]
            
            # 创建临时XYZ文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xyz', mode='w')
            
            # 写入原子数
            temp_file.write(f"{len(symbols)}\n")
            
            # 写入注释行（使用帧ID）
            frame_id = data['id'][frame_index] if 'id' in data else frame_index
            temp_file.write(f"Frame {frame_id}\n")
            
            # 写入原子坐标
            for symbol, coord in zip(symbols, coords):
                temp_file.write(f"{symbol} {coord[0]:.6f} {coord[1]:.6f} {coord[2]:.6f}\n")
            
            temp_file.close()
            return temp_file.name
            
        else:
            # 处理XYZ文件
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

        # 设置PyMOL为无头模式，避免OpenGL问题
        import subprocess

        # 保存原始环境变量
        original_display = os.environ.get('DISPLAY')

        # 设置环境变量以支持无头渲染
        os.environ.update({
            'PYMOL_PATH': '/usr/bin/pymol',
            'LIBGL_ALWAYS_INDIRECT': '1',
            'LIBGL_ALWAYS_SOFTWARE': '1',
            'MESA_GL_VERSION_OVERRIDE': '3.3',
            'MESA_GLSL_VERSION_OVERRIDE': '330',
            'GALLIUM_DRIVER': 'llvmpipe',  # 使用软件渲染
            'LIBGL_ALWAYS_INDIRECT': '1'
        })

        # 尝试使用虚拟显示器
        xvfb_process = None
        try:
            # 检查是否有xvfb-run
            result = subprocess.run(['which', 'xvfb-run'], capture_output=True)
            if result.returncode == 0:
                # 启动虚拟显示器
                display_num = 99
                xvfb_cmd = ['Xvfb', f':{display_num}', '-screen', '0', '1024x768x24', '-ac']
                xvfb_process = subprocess.Popen(xvfb_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                os.environ['DISPLAY'] = f':{display_num}'
                import time
                time.sleep(1)  # 等待Xvfb启动
                print(f"使用虚拟显示器 :{display_num}")
        except:
            print("无法启动虚拟显示器，使用原始DISPLAY")

        # 启动 PyMOL with headless settings
        try:
            pymol.finish_launching(["pymol", "-cq"])  # -c for command line mode, -q for quiet
            print("PyMOL启动成功")
        except Exception as e:
            print(f"PyMOL启动警告: {e}")
            # 如果启动失败，尝试备用方法
            try:
                pymol.finish_launching()
                print("PyMOL备用启动成功")
            except Exception as e2:
                print(f"PyMOL启动失败: {e2}")
                # 清理虚拟显示器
                if xvfb_process:
                    xvfb_process.terminate()
                if original_display:
                    os.environ['DISPLAY'] = original_display
                elif 'DISPLAY' in os.environ:
                    del os.environ['DISPLAY']
                raise Exception("PyMOL无法启动")

        # 设置PyMOL为离屏渲染模式
        try:
            pymol.cmd.set('internal_gui', 0)    # 禁用内部GUI
            pymol.cmd.set('internal_feedback', 0) # 禁用反馈
            pymol.cmd.set('ray_trace_mode', 0)  # 禁用光线追踪
            pymol.cmd.set('ray_texture', 0)     # 禁用纹理
            pymol.cmd.set('ray_shadows', 0)     # 禁用阴影
            pymol.cmd.set('antialias', 0)       # 禁用抗锯齿
            pymol.cmd.set('direct', 1)          # 使用直接渲染
            pymol.cmd.set('bg_rgb', [1, 1, 1])  # 白色背景
            pymol.cmd.set('depth_cue', 0)       # 禁用深度提示
            pymol.cmd.set('fog', 0)             # 禁用雾效
            pymol.cmd.set('orthoscopic', 1)     # 使用正交投影
            print("PyMOL渲染设置完成")
        except Exception as e:
            print(f"PyMOL设置警告: {e}")

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

        # 渲染图像 - 使用兼容性更好的方法
        temp_png = tempfile.NamedTemporaryFile(delete=False, suffix='.png').name

        render_success = False

        # 方法1: 尝试使用基本png渲染（最兼容）
        try:
            print("尝试基本PNG渲染...")
            pymol.cmd.png(temp_png, width=800, height=800, dpi=150, ray=0)
            if os.path.exists(temp_png) and os.path.getsize(temp_png) > 1000:
                render_success = True
                print("基本PNG渲染成功")
        except Exception as e:
            print(f"基本PNG渲染失败: {e}")

        # 方法2: 如果基本渲染失败，尝试简化参数
        if not render_success:
            try:
                print("尝试简化参数渲染...")
                pymol.cmd.png(temp_png, 800, 800)
                if os.path.exists(temp_png) and os.path.getsize(temp_png) > 1000:
                    render_success = True
                    print("简化参数渲染成功")
            except Exception as e:
                print(f"简化参数渲染失败: {e}")

        # 方法3: 最后尝试ray渲染（可能在某些系统上工作）
        if not render_success:
            try:
                print("尝试ray渲染...")
                pymol.cmd.ray(800, 800)
                pymol.cmd.png(temp_png)
                if os.path.exists(temp_png) and os.path.getsize(temp_png) > 1000:
                    render_success = True
                    print("ray渲染成功")
            except Exception as e:
                print(f"ray渲染失败: {e}")

        # 检查渲染结果
        if not render_success or not os.path.exists(temp_png) or os.path.getsize(temp_png) == 0:
            print("所有PyMOL渲染方法都失败，使用备用图像")
            img = create_fallback_image()
        else:
            # 读取生成的图像
            try:
                img = Image.open(temp_png)
                print(f"PyMOL图像读取成功，尺寸: {img.size}")

                # 检查图像是否为黑色或损坏
                img_array = np.array(img)
                if len(img_array.shape) == 3:
                    mean_value = np.mean(img_array)
                    print(f"图像平均像素值: {mean_value:.2f}")

                    if mean_value < 10:  # 图像太暗，可能是黑屏
                        print("检测到PyMOL渲染黑屏，使用备用图像")
                        img = create_fallback_image()
                    else:
                        print("PyMOL渲染成功")
                else:
                    print("图像格式异常，使用备用图像")
                    img = create_fallback_image()

            except Exception as e:
                print(f"图像处理失败: {e}")
                img = create_fallback_image()
        
        # 生成颜色图例
        legend_html = generate_color_legend(elements)
        
        # 清理临时文件和 PyMOL 环境
        try:
            if os.path.exists(temp_frame_file):
                os.unlink(temp_frame_file)
                print(f"清理临时帧文件: {temp_frame_file}")
        except Exception as e:
            print(f"清理帧文件失败: {e}")

        try:
            if os.path.exists(temp_png):
                os.unlink(temp_png)
                print(f"清理临时PNG文件: {temp_png}")
        except Exception as e:
            print(f"清理PNG文件失败: {e}")

        try:
            pymol.cmd.delete('all')
        except:
            pass

        # 清理虚拟显示器
        if 'xvfb_process' in locals() and xvfb_process:
            try:
                xvfb_process.terminate()
                xvfb_process.wait(timeout=5)
                print("虚拟显示器已清理")
            except:
                try:
                    xvfb_process.kill()
                except:
                    pass

        # 恢复原始DISPLAY环境变量
        if 'original_display' in locals():
            if original_display:
                os.environ['DISPLAY'] = original_display
            elif 'DISPLAY' in os.environ and os.environ['DISPLAY'].startswith(':99'):
                del os.environ['DISPLAY']

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
