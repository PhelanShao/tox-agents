# extract_module.py
import os
import re
import plotly.graph_objs as go

def extract_data(log_content, keyword_pairs_columns):
    """Extract data from log file based on keyword pairs"""
    result = []
    data = {}
    
    for line_num, line in enumerate(log_content.split('\n')):
        for start_key, end_key, columns in keyword_pairs_columns:
            if start_key in line:
                # Extract data between keywords
                data_lines = []
                for next_line in log_content.split('\n')[line_num+1:]:
                    if end_key in next_line:
                        break
                    data_lines.append(next_line.strip())
                    
                if data_lines:
                    try:
                        # Parse numeric data
                        numeric_data = []
                        for data_line in data_lines:
                            values = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', data_line)
                            if values and len(values) >= len(columns):
                                numeric_data.append([float(v) for v in values[:len(columns)]])
                                
                        if numeric_data:
                            result.append(f"\nData between '{start_key}' and '{end_key}':")
                            for col, values in zip(columns, zip(*numeric_data)):
                                result.append(f"{col}: {list(values)}")
                                if col not in data:
                                    data[col] = []
                                data[col].extend(values)
                    except Exception as e:
                        result.append(f"Error processing data: {str(e)}")
                
    return "\n".join(result), data

def plot_extracted_data(data):
    """Create plot from extracted data"""
    fig = go.Figure()
    
    for key, values in data.items():
        fig.add_trace(go.Scatter(
            y=values,
            name=key,
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title="Extracted Data Visualization",
        xaxis_title="Index",
        yaxis_title="Value",
        showlegend=True
    )
    
    return fig