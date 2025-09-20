'use client'

import { useState, useEffect, useRef } from 'react'
import { motion } from 'framer-motion'
import {
  ChartBarIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon,
  ArrowDownTrayIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  PlayIcon,
  PauseIcon,
  BeakerIcon
} from '@heroicons/react/24/outline'
import { apiClient } from '@/lib/api'

interface PredictionResultsProps {
  file: File | null
  predictionType: 'binary' | 'property' | null
  onResultsChange?: (results: any) => void
  onMoleculeIndexChange?: (index: number) => void
  onSendToChat?: (data: any) => void
}

export function PredictionResults({ file, predictionType, onResultsChange, onMoleculeIndexChange, onSendToChat }: PredictionResultsProps) {
  const [isLoading, setIsLoading] = useState(false)
  const [results, setResults] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [currentMoleculeIndex, setCurrentMoleculeIndex] = useState(0)
  const [currentPropertyPage, setCurrentPropertyPage] = useState(0)
  const [resultsCache, setResultsCache] = useState<Map<string, any>>(new Map())
  const [isAutoPlay, setIsAutoPlay] = useState(false)
  const [autoPlayInterval, setAutoPlayInterval] = useState<NodeJS.Timeout | null>(null)
  
  // 分页设置
  const PROPERTIES_PER_PAGE = 12
  const totalMolecules = results?.predictions ? results.predictions.length : (results?.total_predictions || 1)
  const totalPropertyPages = results?.properties ? Math.ceil(results.properties.length / PROPERTIES_PER_PAGE) : 1

  // 使用 useRef 来跟踪上次的预测参数，避免重复预测
  const lastPredictionRef = useRef<{file: File | null, type: string | null}>({file: null, type: null})

  useEffect(() => {
    if (file && predictionType) {
      // 检查是否是相同的预测参数
      const isSameParams = lastPredictionRef.current.file === file &&
                          lastPredictionRef.current.type === predictionType

      if (!isSameParams) {
        // 检查缓存，如果有缓存就直接使用，避免重复预测
        const cacheKey = getCacheKey(file, predictionType, 'unimol')
        const cachedResult = resultsCache.get(cacheKey)
        if (cachedResult) {
          console.log('使用缓存的预测结果 (useEffect):', cacheKey)
          setResults(cachedResult)
          onResultsChange?.(cachedResult)
        } else {
          console.log('缓存未命中，开始预测 (useEffect):', cacheKey)
          handlePrediction()
        }

        // 更新上次预测参数
        lastPredictionRef.current = {file, type: predictionType}
      }
    }
  }, [file, predictionType])

  // 自动播放功能
  useEffect(() => {
    if (isAutoPlay && totalMolecules > 1) {
      const interval = setInterval(() => {
        setCurrentMoleculeIndex(prev => (prev + 1) % totalMolecules)
      }, 2000) // 每2秒切换一次
      setAutoPlayInterval(interval)
      return () => clearInterval(interval)
    } else if (autoPlayInterval) {
      clearInterval(autoPlayInterval)
      setAutoPlayInterval(null)
    }
  }, [isAutoPlay, totalMolecules])

  // 清理定时器
  useEffect(() => {
    return () => {
      if (autoPlayInterval) {
        clearInterval(autoPlayInterval)
      }
    }
  }, [])

  // 当分子索引改变时重置属性页面
  useEffect(() => {
    setCurrentPropertyPage(0)
    console.log('分子索引改变:', currentMoleculeIndex)
  }, [currentMoleculeIndex])

  // 强制重新渲染当分子索引改变时
  useEffect(() => {
    if (results) {
      console.log('强制重新渲染，当前分子索引:', currentMoleculeIndex)
    }
  }, [currentMoleculeIndex, results])

  const getCacheKey = (file: File, type: string, modelType: string = 'unimol') => {
    return `${file.name}_${file.size}_${type}_${modelType}`
  }

  const handlePrediction = async () => {
    if (!file || !predictionType) return

    // 检查缓存 - 现在包含模型类型
    const cacheKey = getCacheKey(file, predictionType, 'unimol')
    const cachedResult = resultsCache.get(cacheKey)
    if (cachedResult) {
      console.log('使用缓存的预测结果:', cacheKey)
      setResults(cachedResult)
      onResultsChange?.(cachedResult)
      return
    }

    setIsLoading(true)
    setError(null)
    setResults(null)
    setCurrentMoleculeIndex(0)
    setCurrentPropertyPage(0)

    try {
      let response
      if (predictionType === 'binary') {
        response = await apiClient.predictBinary(file)
      } else {
        response = await apiClient.predictProperty(file)
      }

      if (response.success) {
        // 缓存结果
        setResultsCache(prev => new Map(prev.set(cacheKey, response.data)))
        console.log('缓存新的预测结果:', cacheKey)
        console.log('API响应数据结构:', response.data)
        setResults(response.data)
        onResultsChange?.(response.data)
      } else {
        setError(response.message || '预测失败')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : '预测过程中发生错误')
    } finally {
      setIsLoading(false)
    }
  }

  const downloadResults = () => {
    if (!results) return

    const dataStr = JSON.stringify(results, null, 2)
    const dataBlob = new Blob([dataStr], { type: 'application/json' })
    const url = URL.createObjectURL(dataBlob)
    const link = document.createElement('a')
    link.href = url
    link.download = `prediction_results_${Date.now()}.json`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
    URL.revokeObjectURL(url)
  }

  // 发送结果到聊天机器人
  const sendResultsToChat = (data: any) => {
    // 创建简化的预测结果数据，只包含核心预测信息
    let formattedData: any
    
    if (predictionType === 'binary') {
      // Binary Classification - 只包含预测结果，不包含颜色等UI相关信息
      const currentResult = getCurrentMoleculeResult()
      formattedData = {
        prediction: currentResult?.prediction,
        probability: currentResult?.probability,
        confidence: currentResult?.confidence,
        interpretation: currentResult?.interpretation
      }
    } else {
      // Property Prediction - 只包含属性数据
      const currentResult = getCurrentMoleculeResult()
      formattedData = {
        properties: currentResult?.properties || []
      }
    }
    
    if (!onSendToChat) {
      // 如果没有回调函数，则复制到剪贴板
      const jsonString = JSON.stringify(formattedData, null, 2)
      navigator.clipboard.writeText(jsonString).then(() => {
        alert('预测结果已复制到剪贴板！您可以粘贴到聊天框中。')
      }).catch(() => {
        // 如果剪贴板API失败，显示数据让用户手动复制
        const newWindow = window.open('', '_blank')
        if (newWindow) {
          newWindow.document.write(`
            <html>
              <head><title>Prediction Results JSON</title></head>
              <body>
                <h3>请复制以下JSON数据到聊天框：</h3>
                <textarea style="width:100%;height:80vh;font-family:monospace;">${jsonString}</textarea>
              </body>
            </html>
          `)
        }
      })
    } else {
      // 使用回调函数发送数据，传递格式化的数据和提示词
      const promptText = "This is the predicted property information for this molecule"
      onSendToChat({
        prompt: promptText,
        data: formattedData
      })
    }
  }

  // 分子导航控制
  const goToPreviousMolecule = () => {
    const newIndex = currentMoleculeIndex > 0 ? currentMoleculeIndex - 1 : totalMolecules - 1
    setCurrentMoleculeIndex(newIndex)
    onMoleculeIndexChange?.(newIndex)
  }

  const goToNextMolecule = () => {
    const newIndex = (currentMoleculeIndex + 1) % totalMolecules
    setCurrentMoleculeIndex(newIndex)
    onMoleculeIndexChange?.(newIndex)
  }

  const handleMoleculeIndexChange = (index: number) => {
    setCurrentMoleculeIndex(index)
    onMoleculeIndexChange?.(index)
  }

  const toggleAutoPlay = () => {
    setIsAutoPlay(prev => !prev)
  }

  // 属性分页控制 - 修复版本
  const goToPreviousPropertyPage = () => {
    console.log('点击上一页，当前页:', currentPropertyPage)
    if (currentPropertyPage > 0) {
      setCurrentPropertyPage(currentPropertyPage - 1)
    }
  }

  const goToNextPropertyPage = () => {
    console.log('点击下一页，当前页:', currentPropertyPage)
    setCurrentPropertyPage(currentPropertyPage + 1)
  }

  // 获取当前页的属性
  const getCurrentPageProperties = () => {
    if (!results?.properties) return []
    const startIndex = currentPropertyPage * PROPERTIES_PER_PAGE
    return results.properties.slice(startIndex, startIndex + PROPERTIES_PER_PAGE)
  }

  // 获取当前分子的结果
  const getCurrentMoleculeResult = () => {
    if (!results) return null
    
    console.log('getCurrentMoleculeResult:', {
      currentMoleculeIndex,
      hasPredictions: Array.isArray(results.predictions),
      predictionsLength: results.predictions?.length,
      totalMolecules
    })
    
    // 如果是单个分子结果，直接返回
    if (!Array.isArray(results.predictions)) {
      return results
    }
    
    // 如果是多个分子结果，返回当前索引的结果
    const currentResult = results.predictions[currentMoleculeIndex] || results
    console.log('Current result:', currentResult)
    return currentResult
  }

  if (!file || !predictionType) {
    return (
      <div className="glass-card p-6 text-center">
        <ChartBarIcon className="w-12 h-12 mx-auto text-gray-400 mb-4" />
        <p className="text-gray-500 dark:text-gray-400">
          Please upload a file and select prediction type to start analysis
        </p>
      </div>
    )
  }

  if (isLoading) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card p-6"
      >
        <div className="flex items-center justify-center space-x-3">
          <ClockIcon className="w-6 h-6 text-blue-500 animate-spin" />
          <span className="text-lg font-medium">
            Performing {predictionType === 'binary' ? 'binary classification' : 'property'} prediction...
          </span>
        </div>
        <div className="mt-4 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
          <div className="bg-blue-500 h-2 rounded-full animate-pulse" style={{ width: '60%' }}></div>
        </div>
      </motion.div>
    )
  }

  if (error) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card p-6 border-red-200 dark:border-red-800"
      >
        <div className="flex items-center space-x-3 text-red-600 dark:text-red-400">
          <ExclamationTriangleIcon className="w-6 h-6" />
          <span className="font-medium">Prediction Failed</span>
        </div>
        <p className="mt-2 text-red-500 dark:text-red-400">{error}</p>
        <button
          onClick={handlePrediction}
          className="mt-4 btn-primary"
        >
          Retry
        </button>
      </motion.div>
    )
  }

  if (!results) return null

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* 预测结果标题 */}
      <div className="glass-card p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <CheckCircleIcon className="w-6 h-6 text-green-500" />
            <h3 className="text-xl font-semibold">
              {predictionType === 'binary' ? 'Binary Classification Results' : 'Property Prediction Results'}
            </h3>
          </div>
          <button
            onClick={downloadResults}
            className="btn-glass flex items-center space-x-2"
          >
            <ArrowDownTrayIcon className="w-4 h-4" />
            <span>Download Results</span>
          </button>
        </div>
      </div>

      {/* 分子序列导航 */}
      {totalMolecules > 1 && (
        <div className="glass-card p-6">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-lg font-medium">Molecular Sequence Navigation</h4>
            <div className="flex items-center space-x-2">
              <span className="text-sm text-gray-600 dark:text-gray-400">
                {currentMoleculeIndex + 1} / {totalMolecules}
              </span>
              <button
                onClick={toggleAutoPlay}
                className={`p-2 rounded-lg transition-colors ${
                  isAutoPlay
                    ? 'bg-blue-500 text-white'
                    : 'bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
                }`}
              >
                {isAutoPlay ? <PauseIcon className="w-4 h-4" /> : <PlayIcon className="w-4 h-4" />}
              </button>
            </div>
          </div>
          
          <div className="flex items-center space-x-4">
            <button
              onClick={goToPreviousMolecule}
              className="p-2 rounded-lg bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
            >
              <ChevronLeftIcon className="w-5 h-5" />
            </button>
            
            <div className="flex-1">
              <input
                type="range"
                min="0"
                max={totalMolecules - 1}
                value={currentMoleculeIndex}
                onChange={(e) => handleMoleculeIndexChange(parseInt(e.target.value))}
                className="w-full h-2 bg-gray-200 dark:bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
              />
              <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                <span>1</span>
                <span>{totalMolecules}</span>
              </div>
            </div>
            
            <button
              onClick={goToNextMolecule}
              className="p-2 rounded-lg bg-gray-200 dark:bg-gray-700 hover:bg-gray-300 dark:hover:bg-gray-600 transition-colors"
            >
              <ChevronRightIcon className="w-5 h-5" />
            </button>
          </div>
        </div>
      )}

      {/* 二元分类结果 */}
      {predictionType === 'binary' && (() => {
        const currentResult = getCurrentMoleculeResult()
        return (
          <div key={`binary-${currentMoleculeIndex}`} className="glass-card p-6">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-lg font-medium">Toxicity Prediction</h4>
              {totalMolecules > 1 && (
                <span className="text-sm text-gray-500 dark:text-gray-400">
                  Molecule {currentMoleculeIndex + 1}
                </span>
              )}
            </div>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="text-center p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-2xl font-bold text-blue-600 dark:text-blue-400">
                  {currentResult?.prediction === 1 ? 'Toxic' : 'Non-toxic'}
                </div>
                <div className="text-sm text-gray-500 dark:text-gray-400">Prediction Result</div>
              </div>
              <div className="text-center p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                  {currentResult?.probability ? (currentResult.probability * 100).toFixed(1) : '0.0'}%
                </div>
                <div className="text-sm text-gray-500 dark:text-gray-400">Toxicity Probability</div>
              </div>
              <div className="text-center p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {currentResult?.confidence || 'N/A'}
                </div>
                <div className="text-sm text-gray-500 dark:text-gray-400">Confidence</div>
              </div>
            </div>
            
            {currentResult?.interpretation && (
              <div className="mt-4 p-4 bg-blue-50 dark:bg-blue-900/20 rounded-lg">
                <p className="text-blue-800 dark:text-blue-200">{currentResult.interpretation}</p>
              </div>
            )}

            {/* 概率可视化图表 */}
            {currentResult?.probability !== undefined && (
              <div className="mt-6">
                <h5 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">Probability Distribution</h5>
                <div className="relative h-8 bg-gray-200 dark:bg-gray-700 rounded-full overflow-hidden">
                  <div
                    className="absolute left-0 top-0 h-full bg-gradient-to-r from-green-500 to-red-500 transition-all duration-500"
                    style={{ width: `${currentResult.probability * 100}%` }}
                  />
                  <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-xs font-medium text-white drop-shadow">
                      {(currentResult.probability * 100).toFixed(1)}% Toxicity Probability
                    </span>
                  </div>
                </div>
                <div className="flex justify-between text-xs text-gray-500 dark:text-gray-400 mt-1">
                  <span>Safe</span>
                  <span>Dangerous</span>
                </div>
              </div>
            )}
          </div>
        )
      })()}

      {/* 属性预测结果 */}
      {predictionType === 'property' && (() => {
        // 只使用真实的API返回数据
        const currentResult = getCurrentMoleculeResult()
        const currentProperties = currentResult?.properties || results?.properties || []
        const currentTotalPropertyPages = Math.ceil(currentProperties.length / PROPERTIES_PER_PAGE)
        
        console.log('分页调试信息:', {
          currentProperties: currentProperties.length,
          PROPERTIES_PER_PAGE,
          currentTotalPropertyPages,
          currentPropertyPage
        })
        
        // 获取当前页的属性（基于当前分子）
        const getCurrentMoleculePageProperties = () => {
          const startIndex = currentPropertyPage * PROPERTIES_PER_PAGE
          const pageProperties = currentProperties.slice(startIndex, startIndex + PROPERTIES_PER_PAGE)
          console.log('当前页属性:', pageProperties.length, '起始索引:', startIndex)
          return pageProperties
        }
        
        return (
          <div key={`property-${currentMoleculeIndex}`} className="glass-card p-6">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-lg font-medium">
                Molecular Properties
                <span className="text-sm text-gray-500 dark:text-gray-400 ml-2">
                  (Molecule {currentMoleculeIndex + 1})
                </span>
              </h4>
              {currentTotalPropertyPages > 1 && (
                <div className="flex items-center space-x-2">
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    Page {currentPropertyPage + 1} of {currentTotalPropertyPages}
                  </span>
                  <button
                    onClick={goToPreviousPropertyPage}
                    disabled={currentPropertyPage <= 0}
                    className="p-1 rounded bg-gray-200 dark:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-300 dark:hover:bg-gray-600"
                  >
                    <ChevronLeftIcon className="w-4 h-4" />
                  </button>
                  <button
                    onClick={goToNextPropertyPage}
                    disabled={currentPropertyPage >= currentTotalPropertyPages - 1}
                    className="p-1 rounded bg-gray-200 dark:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-300 dark:hover:bg-gray-600"
                  >
                    <ChevronRightIcon className="w-4 h-4" />
                  </button>
                </div>
              )}
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {getCurrentMoleculePageProperties().map((prop: any, index: number) => (
                <div key={`${currentMoleculeIndex}-${index}`} className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors">
                  <div className="text-sm font-medium text-gray-600 dark:text-gray-400 mb-1">
                    {prop.name || `Property ${currentPropertyPage * PROPERTIES_PER_PAGE + index + 1}`}
                  </div>
                  <div className="text-lg font-bold text-gray-900 dark:text-white">
                    {typeof prop.value === 'number' ? prop.value.toFixed(4) : prop.value || prop}
                    {prop.unit && <span className="text-sm text-gray-500 ml-1">{prop.unit}</span>}
                  </div>
                  {prop.description && (
                    <div className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {prop.description}
                    </div>
                  )}
                </div>
              ))}
            </div>

            {/* 属性统计 */}
            {currentProperties.length > 0 && (
              <div className="mt-6 p-4 bg-gradient-to-r from-purple-50 to-pink-50 dark:from-purple-900/20 dark:to-pink-900/20 rounded-lg">
                <h5 className="font-medium mb-2">Property Statistics</h5>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">Total Properties: </span>
                    <span className="font-medium">{currentProperties.length}</span>
                  </div>
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">Current Page: </span>
                    <span className="font-medium">{currentPropertyPage + 1} / {currentTotalPropertyPages}</span>
                  </div>
                  <div>
                    <span className="text-gray-600 dark:text-gray-400">Per Page: </span>
                    <span className="font-medium">{PROPERTIES_PER_PAGE}</span>
                  </div>
                </div>
              </div>
            )}
            
            {currentResult?.summary && (
              <div className="mt-6 p-4 bg-gradient-to-r from-blue-50 to-purple-50 dark:from-blue-900/20 dark:to-purple-900/20 rounded-lg">
                <h5 className="font-medium mb-2">Analysis Summary</h5>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <span className="text-sm text-gray-600 dark:text-gray-400">Toxicity Score: </span>
                    <span className="font-medium">{currentResult.summary.toxicity_score}</span>
                  </div>
                  <div>
                    <span className="text-sm text-gray-600 dark:text-gray-400">Risk Level: </span>
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
                    <span className="text-sm text-gray-600 dark:text-gray-400">Recommendations:</span>
                    <ul className="mt-1 text-sm space-y-1">
                      {currentResult.summary.recommendations.map((rec: string, index: number) => (
                        <li key={index} className="text-gray-700 dark:text-gray-300">• {rec}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        )
      })()}

      {/* 统计信息 */}
      {results.total_predictions && (
        <div className="glass-card p-6">
          <div className="flex items-center justify-between mb-4">
            <h4 className="text-lg font-medium">Statistics</h4>
            <button
              onClick={() => sendResultsToChat(results)}
              className="btn-glass flex items-center space-x-2 text-sm"
              title="Send prediction results to AI chat"
            >
              <BeakerIcon className="w-4 h-4" />
              <span>Send to Chat</span>
            </button>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="text-center p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
              <div className="text-xl font-bold text-gray-900 dark:text-white">
                {results.total_predictions}
              </div>
              <div className="text-sm text-gray-500 dark:text-gray-400">Total Predictions</div>
            </div>
            {results.positive_predictions !== undefined && (
              <div className="text-center p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-xl font-bold text-red-600 dark:text-red-400">
                  {results.positive_predictions}
                </div>
                <div className="text-sm text-gray-500 dark:text-gray-400">Toxic Predictions</div>
              </div>
            )}
            {results.negative_predictions !== undefined && (
              <div className="text-center p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-xl font-bold text-green-600 dark:text-green-400">
                  {results.negative_predictions}
                </div>
                <div className="text-sm text-gray-500 dark:text-gray-400">Non-toxic Predictions</div>
              </div>
            )}
          </div>
        </div>
      )}
    </motion.div>
  )
}