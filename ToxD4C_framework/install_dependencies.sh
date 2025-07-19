#!/bin/bash
# ToxD4C 依赖安装脚本
# 
# 使用方法: bash install_dependencies.sh
# 或者: chmod +x install_dependencies.sh && ./install_dependencies.sh

echo "🚀 开始安装 ToxD4C 依赖..."

# 检查 Python 版本
python_version=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "📋 检测到 Python 版本: $python_version"

# 检查 PyTorch 版本和 CUDA 支持
echo "📋 检查 PyTorch 安装..."
torch_info=$(python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo "✅ $torch_info"
    
    # 获取 PyTorch 和 CUDA 版本用于安装 PyTorch Geometric
    torch_version=$(python -c "import torch; print(torch.__version__.split('+')[0])")
    cuda_version=$(python -c "import torch; print(torch.__version__.split('+')[1] if '+' in torch.__version__ else 'cpu')")
    
    echo "📦 安装 PyTorch Geometric 及相关扩展..."
    if [[ $cuda_version == "cpu" ]]; then
        pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv
    else
        pip install torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv \
            -f https://data.pyg.org/whl/torch-${torch_version}+${cuda_version}.html
    fi
else
    echo "❌ PyTorch 未安装，请先安装 PyTorch"
    echo "   访问: https://pytorch.org/get-started/locally/"
    exit 1
fi

# 安装其他依赖
echo "📦 安装其他依赖包..."
pip install numpy scipy pandas scikit-learn tqdm matplotlib seaborn requests

# 尝试安装化学信息学库（可选）
echo "📦 尝试安装化学信息学库..."
pip install rdkit-pypi mordred 2>/dev/null || echo "⚠️  化学信息学库安装失败，可能需要手动安装"

# 验证安装
echo "🔍 验证关键依赖安装..."
python -c "
import torch
import torch_geometric
from torch_geometric.nn import GCN, global_mean_pool
import numpy as np
import pandas as pd
import sklearn
print('✅ 所有关键依赖安装成功！')
print(f'   - PyTorch: {torch.__version__}')
print(f'   - PyTorch Geometric: {torch_geometric.__version__}')
print(f'   - NumPy: {np.__version__}')
print(f'   - Pandas: {pd.__version__}')
print(f'   - Scikit-learn: {sklearn.__version__}')
"

if [ $? -eq 0 ]; then
    echo "🎉 ToxD4C 依赖安装完成！"
    echo "💡 现在可以运行: python test_demo.py"
else
    echo "❌ 依赖验证失败，请检查安装日志"
    exit 1
fi