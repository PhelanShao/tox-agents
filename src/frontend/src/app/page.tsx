'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  BeakerIcon,
  ChartBarIcon,
  CpuChipIcon,
  EyeIcon,
  ArrowRightIcon,
  CloudArrowUpIcon,
  SparklesIcon,
  AcademicCapIcon,
  ArrowDownTrayIcon
} from '@heroicons/react/24/outline'
import { Header } from '@/components/layout/header'
import { FileUpload } from '@/components/ui/file-upload'
import { PredictionResults } from '@/components/prediction/prediction-results'
import { MolecularVisualization } from '@/components/visualization/molecular-visualization'
import { ChatInterface } from '@/components/chat/chat-interface'
import { ToxD4CInterface } from '@/components/toxd4c/toxd4c-interface'
import { FloatingBackground } from '@/components/ui/floating-background'
import { AnimatedChatBubbles } from '@/components/animated-chat-bubbles'

type TabType = 'unimol' | 'toxd4c'

export default function HomePage() {
  const [activeTab, setActiveTab] = useState<TabType>('unimol')
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [predictionType, setPredictionType] = useState<'binary' | 'property' | null>(null)
  const [predictionData, setPredictionData] = useState<any>(null)
  const [visualizationData, setVisualizationData] = useState<any>(null)
  const [currentMoleculeIndex, setCurrentMoleculeIndex] = useState<number>(0)
  const [chatReceiveFunction, setChatReceiveFunction] = useState<((data: any) => void) | null>(null)
  
  // 全局缓存状态 - 跨标签页保持
  const [globalCache, setGlobalCache] = useState<{
    unimol: {
      predictionData: any
      visualizationData: any
      predictionType: 'binary' | 'property' | null
      currentMoleculeIndex: number
    }
    toxd4c: {
      results: any
    }
  }>({
    unimol: {
      predictionData: null,
      visualizationData: null,
      predictionType: null,
      currentMoleculeIndex: 0
    },
    toxd4c: {
      results: null
    }
  })

  const handleFileUpload = (file: File) => {
    // 检查是否是同一个文件，如果是则不清除缓存
    if (uploadedFile && uploadedFile.name === file.name && uploadedFile.size === file.size) {
      console.log('Same file uploaded, keeping cache')
      return
    }
    
    setUploadedFile(file)
    // 只有上传新文件时才清除缓存
    setPredictionData(null)
    setVisualizationData(null)
    setCurrentMoleculeIndex(0)
    setPredictionType(null)
    
    // 清除全局缓存
    setGlobalCache({
      unimol: {
        predictionData: null,
        visualizationData: null,
        predictionType: null,
        currentMoleculeIndex: 0
      },
      toxd4c: {
        results: null
      }
    })
  }

  // 标签页切换处理
  const handleTabChange = (newTab: TabType) => {
    // 保存当前标签页的状态到全局缓存
    if (activeTab === 'unimol') {
      setGlobalCache(prev => ({
        ...prev,
        unimol: {
          predictionData,
          visualizationData,
          predictionType,
          currentMoleculeIndex
        }
      }))
    }
    
    // 切换到新标签页
    setActiveTab(newTab)
    
    // 恢复新标签页的状态
    if (newTab === 'unimol') {
      const unimolCache = globalCache.unimol
      setPredictionData(unimolCache.predictionData)
      setVisualizationData(unimolCache.visualizationData)
      setPredictionType(unimolCache.predictionType)
      setCurrentMoleculeIndex(unimolCache.currentMoleculeIndex)
    }
  }

  const handlePredictionResults = (results: any) => {
    setPredictionData(results)
  }

  const handleMoleculeIndexChange = (index: number) => {
    setCurrentMoleculeIndex(index)
  }

  const handleVisualizationChange = (data: any) => {
    setVisualizationData(data)
  }

  // 处理发送到聊天的回调
  const handleSendToChat = (data: any) => {
    if (chatReceiveFunction) {
      chatReceiveFunction(data)
    }
  }

  // 设置聊天接收函数
  const handleChatReceiveFunction = (fn: (data: any) => void) => {
    setChatReceiveFunction(() => fn)
  }

  // 下载示例文件
  const handleDownloadExample = async () => {
    try {
      const response = await fetch('/api/download/example')
      if (response.ok) {
        const blob = await response.blob()
        const url = window.URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = 'examples.xyz'
        document.body.appendChild(a)
        a.click()
        window.URL.revokeObjectURL(url)
        document.body.removeChild(a)
      } else {
        console.error('Failed to download example file')
        alert('Failed to download example file')
      }
    } catch (error) {
      console.error('Error downloading example file:', error)
      alert('Error downloading example file')
    }
  }

  const features = [
    {
      icon: CpuChipIcon,
      title: 'Multi-Modal Deep Fusion',
      description: 'Innovative integration of GNN, Transformer, geometric processing, and chemical prior knowledge for comprehensive molecular analysis',
      gradient: 'from-blue-500 to-cyan-500'
    },
    {
      icon: SparklesIcon,
      title: 'Hierarchical Representation Learning',
      description: 'Four-level architecture from atoms to molecules, mimicking chemist cognition with multi-scale receptive fields',
      gradient: 'from-purple-500 to-pink-500'
    },
    {
      icon: ChartBarIcon,
      title: 'Intelligent Uncertainty Quantification',
      description: 'Bayesian deep learning with confidence intervals and calibrated uncertainty estimation for reliable risk assessment',
      gradient: 'from-green-500 to-emerald-500'
    },
    {
      icon: BeakerIcon,
      title: 'End-to-End Multi-Task Learning',
      description: 'Simultaneous prediction of 31 toxicity endpoints with shared representation learning and knowledge transfer',
      gradient: 'from-orange-500 to-red-500'
    }
  ]

  return (
    <div className="min-h-screen relative">
      <FloatingBackground />
      <Header />
      
      {/* Hero Section */}
      <section className="relative pt-32 pb-20 overflow-hidden">
        <div className="container-wide">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8, ease: "easeOut" }}
            className="text-center max-w-4xl mx-auto"
          >
            <motion.div
              initial={{ scale: 0.9, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card text-sm font-medium text-primary-600 dark:text-primary-400 mb-8"
            >
              <SparklesIcon className="w-4 h-4" />
              Powered by Advanced AI & Machine Learning
            </motion.div>
            
            <h1 className="text-5xl md:text-7xl font-bold mb-6 leading-tight">
              <span className="text-gradient-primary">Molecular</span>
              <br />
              <span className="text-gradient-cyber">Toxicity Predictor</span>
            </h1>
            
            <p className="text-xl md:text-2xl text-gray-600 dark:text-gray-300 mb-12 leading-relaxed">
              Harness the power of artificial intelligence to predict molecular toxicity 
              with unprecedented accuracy and speed. Transform your drug discovery process.
            </p>
            
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
              className="flex flex-col sm:flex-row gap-4 justify-center items-center"
            >
              <a
                href="https://github.com/PhelanShao/tox-agents"
                target="_blank"
                rel="noopener noreferrer"
                className="btn-primary group"
              >
                Source code
                <ArrowRightIcon className="w-5 h-5 ml-2 group-hover:translate-x-1 transition-transform" />
              </a>
              <a
                href="https://bohrium.dp.tech/apps/tox-agents"
                target="_blank"
                rel="noopener noreferrer"
                className="btn-glass"
              >
                View Demo
              </a>
            </motion.div>
          </motion.div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20">
        <div className="container-wide">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-8">
              Integrated with <span className="text-gradient-cyber">Molreac</span>
            </h2>
            <div className="max-w-4xl mx-auto">
              <AnimatedChatBubbles />
            </div>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <motion.div
                key={feature.title}
                initial={{ opacity: 0, y: 30 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: index * 0.1 }}
                viewport={{ once: true }}
                className="glass-card p-8 hover-lift group"
              >
                <div className={`w-12 h-12 rounded-xl bg-gradient-to-r ${feature.gradient} p-3 mb-6 group-hover:scale-110 transition-transform duration-300`}>
                  <feature.icon className="w-full h-full text-white" />
                </div>
                <h3 className="text-xl font-semibold mb-4 text-gray-900 dark:text-white">
                  {feature.title}
                </h3>
                <p className="text-gray-600 dark:text-gray-300 leading-relaxed">
                  {feature.description}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* Main Application Interface */}
      <section className="py-20">
        <div className="container-wide">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.8 }}
            viewport={{ once: true }}
            className="text-center mb-16"
          >
            <h2 className="text-4xl md:text-5xl font-bold mb-6">
              Start Your <span className="text-gradient-primary">Analysis</span>
            </h2>
            <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
              Choose a prediction model and upload molecular data to get instant toxicity predictions and detailed analysis.
            </p>
          </motion.div>

          {/* 标签页选择 */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6 }}
            viewport={{ once: true }}
            className="flex justify-center mb-12"
          >
            <div className="glass-card p-2 rounded-xl">
              <div className="flex gap-2">
                <button
                  onClick={() => handleTabChange('unimol')}
                  className={`px-6 py-3 rounded-lg font-medium transition-all ${
                    activeTab === 'unimol'
                      ? 'bg-primary-500 text-white shadow-lg'
                      : 'text-gray-600 dark:text-gray-400 hover:text-primary-500'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <BeakerIcon className="w-5 h-5" />
                    <div>
                      <div>Pretrained Model Prediction</div>
                      <div className="text-xs text-gray-500 dark:text-gray-400 font-normal">
                        Multi-label Normalization and Molecular Property Prediction Based on UniMol
                      </div>
                    </div>
                  </div>
                </button>
                <button
                  onClick={() => handleTabChange('toxd4c')}
                  className={`px-6 py-3 rounded-lg font-medium transition-all ${
                    activeTab === 'toxd4c'
                      ? 'bg-primary-500 text-white shadow-lg'
                      : 'text-gray-600 dark:text-gray-400 hover:text-primary-500'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <AcademicCapIcon className="w-5 h-5" />
                    <div>
                      <div>D4C Toxicity Prediction</div>
                      <div className="text-xs text-gray-500 dark:text-gray-400 font-normal">
                        Toxicity Prediction Based on a Dynamic GNN-Transformer Hybrid Architecture
                      </div>
                    </div>
                  </div>
                </button>
              </div>
            </div>
          </motion.div>

          {/* 标签页内容 */}
          {activeTab === 'unimol' && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-12">
              {/* File Upload Section */}
              <motion.div
                initial={{ opacity: 0, x: -30 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.8 }}
                viewport={{ once: true }}
                className="space-y-8"
              >
                <div className="glass-card p-8">
                  <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-3">
                      <CloudArrowUpIcon className="w-6 h-6 text-primary-500" />
                      <h3 className="text-2xl font-semibold">Upload Molecular Data</h3>
                    </div>
                    <button
                      onClick={handleDownloadExample}
                      className="flex items-center gap-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-lg transition-colors duration-200 text-sm font-medium"
                      title="Download example XYZ file"
                    >
                      <ArrowDownTrayIcon className="w-4 h-4" />
                      Download Example
                    </button>
                  </div>
                  <FileUpload
                    onFileUpload={handleFileUpload}
                    acceptedFormats={['.xyz']}
                  />
                </div>

                {/* 预测类型选择 */}
                {uploadedFile && (
                  <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6 }}
                    className="glass-card p-6"
                  >
                    <h4 className="text-lg font-semibold mb-4">Select Prediction Type</h4>
                    <div className="grid grid-cols-2 gap-4">
                      <button
                        onClick={() => setPredictionType('binary')}
                        className={`p-4 rounded-lg border-2 transition-all ${
                          predictionType === 'binary'
                            ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                            : 'border-gray-200 dark:border-gray-700 hover:border-primary-300'
                        }`}
                      >
                        <div className="text-center">
                          <ChartBarIcon className="w-8 h-8 mx-auto mb-2 text-primary-500" />
                          <div className="font-medium">Binary Classification</div>
                          <div className="text-sm text-gray-600 dark:text-gray-400">
                            Toxic/Non-toxic Prediction
                          </div>
                        </div>
                      </button>
                      <button
                        onClick={() => setPredictionType('property')}
                        className={`p-4 rounded-lg border-2 transition-all ${
                          predictionType === 'property'
                            ? 'border-primary-500 bg-primary-50 dark:bg-primary-900/20'
                            : 'border-gray-200 dark:border-gray-700 hover:border-primary-300'
                        }`}
                      >
                        <div className="text-center">
                          <BeakerIcon className="w-8 h-8 mx-auto mb-2 text-primary-500" />
                          <div className="font-medium">Property Prediction</div>
                          <div className="text-sm text-gray-600 dark:text-gray-400">
                            Detailed Toxicity Properties
                          </div>
                        </div>
                      </button>
                    </div>
                  </motion.div>
                )}

                {/* 预测结果 */}
                <PredictionResults
                  file={uploadedFile}
                  predictionType={predictionType}
                  onResultsChange={handlePredictionResults}
                  onMoleculeIndexChange={handleMoleculeIndexChange}
                  onSendToChat={handleSendToChat}
                />
              </motion.div>

              {/* Visualization Section */}
              <motion.div
                initial={{ opacity: 0, x: 30 }}
                whileInView={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.8 }}
                viewport={{ once: true }}
                className="space-y-8"
              >
                <div className="glass-card p-8">
                  <div className="flex items-center gap-3 mb-6">
                    <EyeIcon className="w-6 h-6 text-primary-500" />
                    <h3 className="text-2xl font-semibold">3D Molecular Visualization</h3>
                  </div>
                  <MolecularVisualization
                    file={uploadedFile}
                    onVisualizationChange={handleVisualizationChange}
                    currentMoleculeIndex={currentMoleculeIndex}
                    onMoleculeIndexChange={handleMoleculeIndexChange}
                  />
                </div>

                <div className="glass-card p-8">
                  <div className="flex items-center justify-between mb-6">
                    <div className="flex items-center gap-3">
                      <BeakerIcon className="w-6 h-6 text-primary-500" />
                      <h3 className="text-2xl font-semibold">AI Analysis Chat</h3>
                    </div>
                    <button
                      onClick={() => window.open('https://pinkieflow.com/', '_blank')}
                      className="px-4 py-2 bg-gradient-to-r from-purple-500 to-pink-500 text-white rounded-lg hover:from-purple-600 hover:to-pink-600 transition-all duration-200 shadow-md hover:shadow-lg flex items-center gap-2 text-sm font-medium"
                    >
                      <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
                      </svg>
                      RAG with websearch api
                    </button>
                  </div>
                  <ChatInterface
                    predictionData={predictionData}
                    visualizationData={visualizationData}
                    onReceivePredictionData={handleChatReceiveFunction}
                  />
                </div>
              </motion.div>
            </div>
          )}

          {/* ToxD4C 标签页 */}
          {activeTab === 'toxd4c' && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-12">
              {/* ToxD4C Interface - 占2/3宽度 */}
              <motion.div
                initial={{ opacity: 0, x: -30 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6 }}
                className="lg:col-span-2"
              >
                <ToxD4CInterface onSendToChat={handleSendToChat} />
              </motion.div>

              {/* Chat Interface - 占1/3宽度 */}
              <motion.div
                initial={{ opacity: 0, x: 30 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ duration: 0.6, delay: 0.2 }}
                className="lg:col-span-1"
              >
                <div className="glass-card p-8">
                  <div className="flex items-center gap-3 mb-6">
                    <BeakerIcon className="w-6 h-6 text-primary-500" />
                    <h3 className="text-2xl font-semibold">AI Analysis Chat</h3>
                  </div>
                  <ChatInterface
                    predictionData={null}
                    visualizationData={null}
                    onReceivePredictionData={handleChatReceiveFunction}
                  />
                </div>
              </motion.div>
            </div>
          )}
        </div>
      </section>
    </div>
  )
}