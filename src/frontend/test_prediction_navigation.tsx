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
        { name: "氢键供体", value: 1, unit: "" }
      ],
      summary: {
        toxicity_score: 0.3,
        risk_level: "low",
        recommendations: ["分子1的建议"]
      }
    },
    {
      properties: [
        { name: "分子量", value: 194.19, unit: "g/mol" },
        { name: "LogP", value: 3.12, unit: "" },
        { name: "极性表面积", value: 52.60, unit: "Ų" },
        { name: "氢键供体", value: 2, unit: "" }
      ],
      summary: {
        toxicity_score: 0.7,
        risk_level: "high",
        recommendations: ["分子2的建议"]
      }
    },
    {
      properties: [
        { name: "分子量", value: 156.14, unit: "g/mol" },
        { name: "LogP", value: 1.89, unit: "" },
        { name: "极性表面积", value: 38.77, unit: "Ų" },
        { name: "氢键供体", value: 0, unit: "" }
      ],
      summary: {
        toxicity_score: 0.5,
        risk_level: "medium",
        recommendations: ["分子3的建议"]
      }
    }
  ],
  total_predictions: 3
}

export default function TestPredictionNavigation() {
  const [currentMoleculeIndex, setCurrentMoleculeIndex] = useState(0)
  const totalMolecules = mockResults.predictions.length

  const getCurrentMoleculeResult = () => {
    return mockResults.predictions[currentMoleculeIndex] || mockResults.predictions[0]
  }

  const handleMoleculeIndexChange = (index: number) => {
    console.log('分子索引改变:', index)
    setCurrentMoleculeIndex(index)
  }

  const currentResult = getCurrentMoleculeResult()

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <h1 className="text-2xl font-bold">预测导航测试</h1>
      
      {/* 分子序列导航 */}
      <div className="bg-white p-6 rounded-lg shadow">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-lg font-medium">分子序列导航</h4>
          <span className="text-sm text-gray-600">
            {currentMoleculeIndex + 1} / {totalMolecules}
          </span>
        </div>
        
        <div className="flex items-center space-x-4">
          <button
            onClick={() => handleMoleculeIndexChange(currentMoleculeIndex > 0 ? currentMoleculeIndex - 1 : totalMolecules - 1)}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
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
              className="w-full"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>1</span>
              <span>{totalMolecules}</span>
            </div>
          </div>
          
          <button
            onClick={() => handleMoleculeIndexChange((currentMoleculeIndex + 1) % totalMolecules)}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            下一个 →
          </button>
        </div>
      </div>

      {/* 属性预测结果 */}
      <div key={`property-${currentMoleculeIndex}`} className="bg-white p-6 rounded-lg shadow">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-lg font-medium">
            分子属性
            <span className="text-sm text-gray-500 ml-2">
              (分子 {currentMoleculeIndex + 1})
            </span>
          </h4>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {currentResult.properties.map((prop: any, index: number) => (
            <div key={index} className="p-4 bg-gray-50 rounded-lg">
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
          <div className="mt-6 p-4 bg-blue-50 rounded-lg">
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
        <pre className="text-sm">
          {JSON.stringify({
            currentMoleculeIndex,
            totalMolecules,
            currentMoleculeWeight: currentResult.properties[0].value
          }, null, 2)}
        </pre>
      </div>
    </div>
  )
}