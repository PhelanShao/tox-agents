'use client'

import { useState } from 'react'

// 模拟多分子预测数据
const mockResults = {
  predictions: [
    {
      properties: [
        { name: "分子量", value: 180.16, unit: "g/mol" },
        { name: "LogP", value: 2.34, unit: "" },
        { name: "极性表面积", value: 46.53, unit: "Ų" },
        { name: "氢键供体", value: 1, unit: "" },
        { name: "氢键受体", value: 2, unit: "" },
        { name: "可旋转键", value: 3, unit: "" }
      ],
      summary: {
        toxicity_score: 0.3,
        risk_level: "low",
        recommendations: ["分子1风险较低", "建议进一步验证", "关注ADMET性质"]
      }
    },
    {
      properties: [
        { name: "分子量", value: 194.19, unit: "g/mol" },
        { name: "LogP", value: 3.12, unit: "" },
        { name: "极性表面积", value: 52.60, unit: "Ų" },
        { name: "氢键供体", value: 2, unit: "" },
        { name: "氢键受体", value: 3, unit: "" },
        { name: "可旋转键", value: 4, unit: "" }
      ],
      summary: {
        toxicity_score: 0.7,
        risk_level: "high",
        recommendations: ["分子2风险较高", "需要结构优化", "考虑替代方案"]
      }
    },
    {
      properties: [
        { name: "分子量", value: 156.14, unit: "g/mol" },
        { name: "LogP", value: 1.89, unit: "" },
        { name: "极性表面积", value: 38.77, unit: "Ų" },
        { name: "氢键供体", value: 0, unit: "" },
        { name: "氢键受体", value: 2, unit: "" },
        { name: "可旋转键", value: 2, unit: "" }
      ],
      summary: {
        toxicity_score: 0.5,
        risk_level: "medium",
        recommendations: ["分子3风险中等", "可接受的安全性", "监控副作用"]
      }
    }
  ],
  total_predictions: 3
}

export default function TestPage() {
  const [currentMoleculeIndex, setCurrentMoleculeIndex] = useState(0)
  const totalMolecules = mockResults.predictions.length

  const getCurrentMoleculeResult = () => {
    console.log('获取当前分子结果:', currentMoleculeIndex)
    return mockResults.predictions[currentMoleculeIndex] || mockResults.predictions[0]
  }

  const handleMoleculeIndexChange = (index: number) => {
    console.log('分子索引改变:', index)
    setCurrentMoleculeIndex(index)
  }

  const goToPreviousMolecule = () => {
    const newIndex = currentMoleculeIndex > 0 ? currentMoleculeIndex - 1 : totalMolecules - 1
    handleMoleculeIndexChange(newIndex)
  }

  const goToNextMolecule = () => {
    const newIndex = (currentMoleculeIndex + 1) % totalMolecules
    handleMoleculeIndexChange(newIndex)
  }

  const currentResult = getCurrentMoleculeResult()

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="max-w-6xl mx-auto px-4 space-y-6">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">分子序列导航测试</h1>
          <p className="text-gray-600">测试分子切换时数据是否正确刷新</p>
        </div>

        {/* 分子序列导航 */}
        <div className="bg-white p-6 rounded-lg shadow-md">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-lg font-medium text-gray-900">分子序列导航</h4>
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-600">
                {currentMoleculeIndex + 1} / {totalMolecules}
              </span>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <button
              onClick={goToPreviousMolecule}
              className="p-2 rounded-lg bg-gray-200 hover:bg-gray-300 transition-colors"
            >
              ← 上一个
            </button>
            
            <div className="flex-1">
              <input
                type="range"
                min="0"
                max={totalMolecules - 1}
                value={currentMoleculeIndex}
                onChange={(e) => handleMoleculeIndexChange(parseInt(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                style={{
                  background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${(currentMoleculeIndex / (totalMolecules - 1)) * 100}%, #e5e7eb ${(currentMoleculeIndex / (totalMolecules - 1)) * 100}%, #e5e7eb 100%)`
                }}
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>1</span>
                <span>{totalMolecules}</span>
              </div>
            </div>
            
            <button
              onClick={goToNextMolecule}
              className="p-2 rounded-lg bg-gray-200 hover:bg-gray-300 transition-colors"
            >
              下一个 →
            </button>
          </div>
        </div>

        {/* 属性预测结果 */}
        <div key={`property-${currentMoleculeIndex}`} className="bg-white p-6 rounded-lg shadow-md">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-lg font-medium text-gray-900">
              分子属性
              <span className="text-sm text-gray-500 ml-2">
                (分子 {currentMoleculeIndex + 1})
              </span>
            </h4>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6 gap-4">
            {currentResult.properties.map((prop: any, index: number) => (
              <div key={`${currentMoleculeIndex}-${index}`} className="p-4 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors">
                <div className="text-sm font-medium text-gray-600 mb-1">
                  {prop.name}
                </div>
                <div className="text-lg font-bold text-gray-900">
                  {typeof prop.value === 'number' ? prop.value.toFixed(4) : prop.value}
                  {prop.unit && <span className="text-sm text-gray-500 ml-1">{prop.unit}</span>}
                </div>
              </div>
            ))}
          </div>

          {/* 分析摘要 */}
          {currentResult.summary && (
            <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg">
              <h5 className="font-medium mb-2">分析摘要</h5>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div>
                  <span className="text-sm text-gray-600">毒性评分: </span>
                  <span className="font-medium">{currentResult.summary.toxicity_score}</span>
                </div>
                <div>
                  <span className="text-sm text-gray-600">风险等级: </span>
                  <span className={`font-medium ${
                    currentResult.summary.risk_level === 'high' ? 'text-red-600' :
                    currentResult.summary.risk_level === 'medium' ? 'text-yellow-600' : 'text-green-600'
                  }`}>
                    {currentResult.summary.risk_level}
                  </span>
                </div>
              </div>
              {currentResult.summary.recommendations && (
                <div className="mt-3">
                  <span className="text-sm text-gray-600">建议:</span>
                  <ul className="mt-1 text-sm space-y-1">
                    {currentResult.summary.recommendations.map((rec: string, index: number) => (
                      <li key={index} className="text-gray-700">• {rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}
        </div>

        {/* 调试信息 */}
        <div className="bg-gray-100 p-4 rounded-lg">
          <h5 className="font-medium mb-2">调试信息</h5>
          <div className="text-sm space-y-1">
            <div>当前分子索引: <span className="font-mono">{currentMoleculeIndex}</span></div>
            <div>总分子数: <span className="font-mono">{totalMolecules}</span></div>
            <div>当前分子量: <span className="font-mono">{currentResult.properties[0].value}</span></div>
            <div>当前风险等级: <span className="font-mono">{currentResult.summary.risk_level}</span></div>
          </div>
        </div>

        <div className="text-center">
          <p className="text-gray-600">
            如果这个页面的数据能正确切换，说明逻辑是正确的。
            <br />
            问题可能在于实际API返回的数据结构与预期不符。
          </p>
        </div>
      </div>
    </div>
  )
}