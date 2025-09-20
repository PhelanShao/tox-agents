'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  BeakerIcon, 
  DocumentTextIcon, 
  ArrowRightIcon,
  ChartBarIcon,
  CpuChipIcon,
  CloudArrowUpIcon,
  SparklesIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline'
import { useDropzone } from 'react-dropzone'
import { apiClient } from '@/lib/api'
import toast from 'react-hot-toast'

interface ToxD4CResult {
  success: boolean
  data?: {
    results: Array<{
      Identifier: string
      [key: string]: any
    }>
    interpretations: Array<any>
    input_type: 'smiles' | 'xyz'
    num_molecules: number
  }
  message?: string
  error?: string
}

interface ToxD4CInterfaceProps {
  onSendToChat?: (data: any) => void
}

export function ToxD4CInterface({ onSendToChat }: ToxD4CInterfaceProps = {}) {
  const [inputType, setInputType] = useState<'smiles' | 'file'>('smiles')
  const [smilesInput, setSmilesInput] = useState('')
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [results, setResults] = useState<ToxD4CResult | null>(null)
  const [modelInfo, setModelInfo] = useState<any>(null)

  // 文件上传配置
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'chemical/x-xyz': ['.xyz']
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        setUploadedFile(acceptedFiles[0])
        toast.success(`已选择文件: ${acceptedFiles[0].name}`)
      }
    },
    onDropRejected: (rejectedFiles) => {
      toast.error('Please upload supported file formats (.xyz)')
    }
  })

  // 获取模型信息
  const fetchModelInfo = async () => {
    try {
      const response = await apiClient.getToxD4CInfo()
      if (response.success) {
        setModelInfo(response.data)
      }
    } catch (error) {
      console.error('Failed to get model information:', error)
    }
  }

  // 组件加载时获取模型信息
  useEffect(() => {
    fetchModelInfo()
  }, [])

  // SMILES预测
  const handleSmilesPredict = async () => {
    if (!smilesInput.trim()) {
      toast.error('Please enter SMILES string')
      return
    }

    setIsLoading(true)
    try {
      const response = await apiClient.predictToxD4CSmiles(smilesInput.trim())

      if (response.success) {
        setResults(response)
        toast.success(response.message || 'Prediction successful')
      } else {
        toast.error(response.message || 'Prediction failed')
        setResults(response)
      }
    } catch (error: any) {
      console.error('SMILES prediction failed:', error)
      toast.error(error.message || 'Prediction failed')
    } finally {
      setIsLoading(false)
    }
  }

  // 文件预测
  const handleFilePredict = async () => {
    if (!uploadedFile) {
      toast.error('Please upload a file first')
      return
    }

    setIsLoading(true)
    try {
      const response = await apiClient.predictToxD4CFile(uploadedFile)

      if (response.success) {
        setResults(response)
        toast.success(response.message || 'Prediction successful')
      } else {
        toast.error(response.message || 'Prediction failed')
        setResults(response)
      }
    } catch (error: any) {
      console.error('File prediction failed:', error)
      toast.error(error.message || 'Prediction failed')
    } finally {
      setIsLoading(false)
    }
  }

  // 发送结果到聊天机器人
  const sendResultsToChat = (data: ToxD4CResult) => {
    // 创建简化的ToxD4C预测结果数据，只包含核心预测信息
    const formattedData = {
      results: data.data?.results || [],
      input_type: data.data?.input_type,
      num_molecules: data.data?.num_molecules || 0
    }
    
    if (!onSendToChat) {
      // 如果没有回调函数，则复制到剪贴板
      const jsonString = JSON.stringify(formattedData, null, 2)
      navigator.clipboard.writeText(jsonString).then(() => {
        alert('ToxD4C预测结果已复制到剪贴板！您可以粘贴到聊天框中。')
      }).catch(() => {
        // 如果剪贴板API失败，显示数据让用户手动复制
        const newWindow = window.open('', '_blank')
        if (newWindow) {
          newWindow.document.write(`
            <html>
              <head><title>ToxD4C Prediction Results JSON</title></head>
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

  // 示例SMILES
  const exampleSmiles = [
    'CC(=O)OC1=CC=CC=C1C(=O)O',  // 阿司匹林
    'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',  // 布洛芬
    'CC(=O)NC1=CC=C(O)C=C1',  // 对乙酰氨基酚
    'CN1C=NC2=C1C(=O)N(C(=O)N2C)C'  // 咖啡因
  ]

  return (
    <div className="max-w-6xl mx-auto p-6 space-y-8">
      {/* 头部信息 */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card text-sm font-medium text-primary-600 dark:text-primary-400 mb-6">
          <CpuChipIcon className="w-4 h-4" />
          ToxD4C - Deep Learning Toxicity Prediction
        </div>
        
        <h1 className="text-4xl font-bold mb-4">
          <span className="text-gradient-primary">ToxD4C</span>
          <span className="text-gradient-cyber"> Toxicity Prediction</span>
        </h1>
        
        <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
          Deep learning-based molecular toxicity prediction system supporting SMILES strings and XYZ file format,
          providing multi-dimensional toxicity assessment and uncertainty quantification
        </p>
      </motion.div>

      {/* 输入类型选择 */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="glass-card p-6"
      >
        <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
          <BeakerIcon className="w-6 h-6 text-primary-500" />
          Input Method Selection
        </h2>
        
        <div className="flex gap-4 mb-6">
          <button
            onClick={() => setInputType('smiles')}
            className={`flex-1 p-4 rounded-lg border-2 transition-all ${
              inputType === 'smiles'
                ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                : 'border-gray-200 dark:border-gray-700 hover:border-primary-300'
            }`}
          >
            <DocumentTextIcon className="w-8 h-8 mx-auto mb-2 text-primary-500" />
            <div className="font-medium">SMILES Input</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Direct SMILES string input
            </div>
          </button>
          
          <button
            onClick={() => setInputType('file')}
            className={`flex-1 p-4 rounded-lg border-2 transition-all ${
              inputType === 'file'
                ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                : 'border-gray-200 dark:border-gray-700 hover:border-primary-300'
            }`}
          >
            <CloudArrowUpIcon className="w-8 h-8 mx-auto mb-2 text-primary-500" />
            <div className="font-medium">File Upload</div>
            <div className="text-sm text-gray-600 dark:text-gray-400">
              Upload .xyz files
            </div>
          </button>
        </div>

        {/* SMILES输入界面 */}
        {inputType === 'smiles' && (
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium mb-2">
                SMILES Strings (one per line)
              </label>
              <textarea
                value={smilesInput}
                onChange={(e) => setSmilesInput(e.target.value)}
                className="w-full h-32 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent resize-none bg-white dark:bg-gray-800"
                placeholder="Please enter SMILES strings, one molecule per line..."
              />
            </div>
            
            <div>
              <label className="block text-sm font-medium mb-2">
                Example SMILES
              </label>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                {exampleSmiles.map((smiles, index) => (
                  <button
                    key={index}
                    onClick={() => setSmilesInput(prev => 
                      prev ? prev + '\n' + smiles : smiles
                    )}
                    className="text-left p-2 text-sm font-mono bg-gray-100 dark:bg-gray-700 rounded hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
                  >
                    {smiles}
                  </button>
                ))}
              </div>
            </div>
            
            <button
              onClick={handleSmilesPredict}
              disabled={isLoading || !smilesInput.trim()}
              className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <div className="flex items-center justify-center gap-2">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Predicting...
                </div>
              ) : (
                <div className="flex items-center justify-center gap-2">
                  <SparklesIcon className="w-5 h-5" />
                  Start Prediction
                </div>
              )}
            </button>
          </div>
        )}

        {/* 文件上传界面 */}
        {inputType === 'file' && (
          <div className="space-y-4">
            <div
              {...getRootProps()}
              className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
                isDragActive
                  ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                  : 'border-gray-300 dark:border-gray-600 hover:border-primary-400'
              }`}
            >
              <input {...getInputProps()} />
              <CloudArrowUpIcon className="w-12 h-12 mx-auto mb-4 text-gray-400" />
              {uploadedFile ? (
                <div>
                  <p className="text-lg font-medium text-green-600">
                    Selected: {uploadedFile.name}
                  </p>
                  <p className="text-sm text-gray-500">
                    Size: {(uploadedFile.size / 1024).toFixed(1)} KB
                  </p>
                </div>
              ) : (
                <div>
                  <p className="text-lg font-medium mb-2">
                    {isDragActive ? 'Drop files here' : 'Drag files here or click to select'}
                  </p>
                  <p className="text-sm text-gray-500">
                    Supported formats: .xyz
                  </p>
                </div>
              )}
            </div>
            
            <button
              onClick={handleFilePredict}
              disabled={isLoading || !uploadedFile}
              className="btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <div className="flex items-center justify-center gap-2">
                  <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                  Predicting...
                </div>
              ) : (
                <div className="flex items-center justify-center gap-2">
                  <SparklesIcon className="w-5 h-5" />
                  Start Prediction
                </div>
              )}
            </button>
          </div>
        )}
      </motion.div>

      {/* 结果显示 */}
      {results && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="glass-card p-6"
        >
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-2xl font-semibold flex items-center gap-2">
              <ChartBarIcon className="w-6 h-6 text-primary-500" />
              Prediction Results
            </h2>
            {results.success && (
              <button
                onClick={() => sendResultsToChat(results)}
                className="btn-glass flex items-center space-x-2 text-sm"
                title="Send ToxD4C results to AI chat"
              >
                <BeakerIcon className="w-4 h-4" />
                <span>Send to Chat</span>
              </button>
            )}
          </div>
          
          {results.success ? (
            <div className="space-y-6">
              {/* 结果概览 */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div className="bg-green-50 dark:bg-green-900/20 p-4 rounded-lg">
                  <div className="text-green-600 dark:text-green-400 font-medium">
                    Prediction Successful
                  </div>
                  <div className="text-2xl font-bold text-green-700 dark:text-green-300">
                    {results.data?.num_molecules || 0}
                  </div>
                  <div className="text-sm text-green-600 dark:text-green-400">
                    molecules
                  </div>
                </div>
                
                <div className="bg-blue-50 dark:bg-blue-900/20 p-4 rounded-lg">
                  <div className="text-blue-600 dark:text-blue-400 font-medium">
                    Input Type
                  </div>
                  <div className="text-2xl font-bold text-blue-700 dark:text-blue-300">
                    {results.data?.input_type?.toUpperCase() || 'N/A'}
                  </div>
                  <div className="text-sm text-blue-600 dark:text-blue-400">
                    format
                  </div>
                </div>
                
                <div className="bg-purple-50 dark:bg-purple-900/20 p-4 rounded-lg">
                  <div className="text-purple-600 dark:text-purple-400 font-medium">
                    Task Count
                  </div>
                  <div className="text-2xl font-bold text-purple-700 dark:text-purple-300">
                    {results.data?.results && results.data.results.length > 0 
                      ? Object.keys(results.data.results[0]).length - 1 
                      : 0}
                  </div>
                  <div className="text-sm text-purple-600 dark:text-purple-400">
                    tasks
                  </div>
                </div>
              </div>

              {/* 详细结果表格 */}
              {results.data?.results && results.data.results.length > 0 && (
                <div className="overflow-x-auto">
                  <table className="w-full border-collapse border border-gray-300 dark:border-gray-600">
                    <thead>
                      <tr className="bg-gray-50 dark:bg-gray-700">
                        {Object.keys(results.data.results[0]).map((key) => {
                          // 为回归任务添加单位信息 - 所有回归任务都使用 mg/kg 单位
                          const getColumnHeader = (columnName: string) => {
                            const regressionTasks: { [key: string]: string } = {
                              'Acute oral toxicity (LD50)': 'Acute oral toxicity (LD50) [mg/kg]',
                              'LC50DM': 'LC50DM [mg/kg]',
                              'BCF': 'BCF [mg/kg]',
                              'LC50': 'LC50 [mg/kg]',
                              'IGC50': 'IGC50 [mg/kg]'
                            }
                            return regressionTasks[columnName] || columnName
                          }

                          return (
                            <th
                              key={key}
                              className="border border-gray-300 dark:border-gray-600 px-4 py-2 text-left font-medium"
                            >
                              {getColumnHeader(key)}
                            </th>
                          )
                        })}
                      </tr>
                    </thead>
                    <tbody>
                      {results.data.results.map((result, index) => (
                        <tr key={index} className="hover:bg-gray-50 dark:hover:bg-gray-700">
                          {Object.entries(result).map(([key, value]) => (
                            <td
                              key={key}
                              className="border border-gray-300 dark:border-gray-600 px-4 py-2"
                            >
                              {/* 回归值现在已经包含单位，直接显示字符串 */}
                              {typeof value === 'number' && !String(value).includes('mg/')
                                ? value.toFixed(4)
                                : String(value)}
                            </td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          ) : (
            <div className="flex items-center gap-3 p-4 bg-red-50 dark:bg-red-900/20 rounded-lg">
              <ExclamationTriangleIcon className="w-6 h-6 text-red-500" />
              <div>
                <div className="font-medium text-red-700 dark:text-red-300">
                  Prediction Failed
                </div>
                <div className="text-sm text-red-600 dark:text-red-400">
                  {results.message || results.error || 'Unknown error'}
                </div>
              </div>
            </div>
          )}
        </motion.div>
      )}

      {/* 模型信息 */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="glass-card p-6"
      >
        <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
          <CpuChipIcon className="w-6 h-6 text-primary-500" />
          Model Information
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h3 className="font-medium mb-2">Supported Input Formats</h3>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>• SMILES strings</li>
              <li>• XYZ molecular coordinate files</li>
            </ul>
          </div>
          
          <div>
            <h3 className="font-medium mb-2">Prediction Tasks</h3>
            <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
              <li>• Multi-dimensional toxicity classification</li>
              <li>• Toxicity intensity regression</li>
              <li>• Uncertainty quantification</li>
              <li>• Molecular attention weights</li>
            </ul>
          </div>
        </div>
        
        <button
          onClick={fetchModelInfo}
          className="mt-4 btn-glass"
        >
          Get Detailed Model Information
        </button>
        
        {modelInfo && (
          <div className="mt-4 p-4 bg-gray-50 dark:bg-gray-700 rounded-lg">
            <pre className="text-sm overflow-x-auto">
              {JSON.stringify(modelInfo, null, 2)}
            </pre>
          </div>
        )}
      </motion.div>
    </div>
  )
} 