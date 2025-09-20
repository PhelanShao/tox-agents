'use client'

import React, { useState, useRef, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  PaperAirplaneIcon, 
  CogIcon,
  TrashIcon,
  UserIcon,
  BeakerIcon
} from '@heroicons/react/24/outline'
import { apiClient } from '@/lib/api'
import { DocumentUploadCard } from './document-upload-card'

interface ChatInterfaceProps {
  predictionData?: any
  visualizationData?: any
  onReceivePredictionData?: (data: any) => void
}

interface Message {
  id: string
  type: 'user' | 'assistant'
  content: string
  timestamp: Date
}

export function ChatInterface({ predictionData, visualizationData, onReceivePredictionData }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputMessage, setInputMessage] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isConfigured, setIsConfigured] = useState(false)
  const [showConfig, setShowConfig] = useState(false)
  const [availableModels, setAvailableModels] = useState<any[]>([])
  const [isLoadingModels, setIsLoadingModels] = useState(false)
  const [config, setConfig] = useState({
    baseUrl: 'https://openrouter.ai/api/v1',
    apiKey: '',
    modelName: 'google/gemini-2.0-flash-thinking-exp:free'
  })
  const [customModelName, setCustomModelName] = useState('')
  const [useRAG, setUseRAG] = useState(false)
  const [ragMode, setRagMode] = useState('hybrid')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const handleConfigSubmit = async () => {
    if (!config.baseUrl || !config.apiKey) {
      alert('Please fill in complete API configuration information')
      return
    }

    try {
      const response = await apiClient.configureChat(config.baseUrl, config.apiKey)
      if (response.success) {
        setIsConfigured(true)
        setShowConfig(false)
        addMessage('assistant', '✅ API configuration successful! You can now start chatting.')
        
        // 配置成功后自动获取模型列表
        await loadAvailableModels()
      } else {
        alert(`Configuration failed: ${response.message}`)
      }
    } catch (error) {
      alert(`Configuration failed: ${error}`)
    }
  }

  const loadAvailableModels = async () => {
    if (!isConfigured) return
    
    setIsLoadingModels(true)
    try {
      const response = await apiClient.getAvailableModels()
      if (response.success && response.data?.models) {
        setAvailableModels(response.data.models)
        addMessage('assistant', `✅ Loaded ${response.data.models.length} available models from OpenRouter.`)
      } else {
        console.warn('Failed to load models:', response.message)
      }
    } catch (error) {
      console.error('Error loading models:', error)
    } finally {
      setIsLoadingModels(false)
    }
  }

  const addMessage = (type: 'user' | 'assistant', content: string) => {
    const newMessage: Message = {
      id: Date.now().toString(),
      type,
      content,
      timestamp: new Date()
    }
    setMessages(prev => [...prev, newMessage])
  }

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return

    if (!isConfigured) {
      setShowConfig(true)
      return
    }

    const userMessage = inputMessage.trim()
    setInputMessage('')
    addMessage('user', userMessage)
    setIsLoading(true)

    try {
      // 构建增强的上下文消息
      let contextMessage = userMessage
      
      if (predictionData || visualizationData) {
        // 创建结构化的分析上下文
        const analysisContext = {
          timestamp: new Date().toISOString(),
          molecular_analysis: {
            prediction_results: predictionData ? {
              type: predictionData.prediction !== undefined ? 'binary_classification' : 'property_prediction',
              data: predictionData
            } : null,
            visualization_data: visualizationData ? {
              image_info: visualizationData,
              molecular_structure: "3D molecular structure visualization available"
            } : null
          },
          analysis_summary: {
            has_toxicity_prediction: !!predictionData?.prediction,
            has_property_data: !!predictionData?.properties,
            has_visualization: !!visualizationData,
            total_molecules: predictionData?.predictions?.length || (predictionData?.properties ? 1 : 0)
          }
        }

        // 构建专业的上下文提示
        const contextPrompt = `You are a professional computational chemist and toxicologist. You have access to molecular analysis data from advanced AI models (UniMol-based predictions). Please analyze the following data and answer the user's question with scientific accuracy.

MOLECULAR ANALYSIS DATA:
${JSON.stringify(analysisContext, null, 2)}

INSTRUCTIONS:
- Provide scientifically accurate interpretations
- Explain toxicity predictions in terms of molecular properties
- Reference specific values from the prediction data
- Consider ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties
- Suggest potential structural modifications if relevant
- Use professional chemical terminology appropriately

USER QUESTION: ${userMessage}`

        contextMessage = contextPrompt
      }

      // 使用选择的模型名称（支持自定义模型）
      const modelToUse = config.modelName === 'custom' ? customModelName : config.modelName
      
      // 根据是否启用RAG选择不同的API端点
      const response = useRAG
        ? await sendChatMessageWithRAG(contextMessage, modelToUse, ragMode)
        : await apiClient.sendChatMessage(contextMessage, modelToUse)
      
      if (response.success && response.data?.response) {
        addMessage('assistant', response.data.response)
      } else {
        addMessage('assistant', `❌ Send failed: ${response.message || 'Unknown error'}`)
      }
    } catch (error) {
      addMessage('assistant', `❌ Send failed: ${error}`)
    } finally {
      setIsLoading(false)
    }
  }

  const clearMessages = () => {
    setMessages([])
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  // 处理从预测结果发送过来的数据
  const handleReceivePredictionData = (data: any) => {
    let promptMessage: string
    let jsonString: string
    
    if (data.prompt && data.data) {
      // 新格式：包含提示词和数据
      jsonString = JSON.stringify(data.data, null, 2)
      promptMessage = `${data.prompt}:\n\n\`\`\`json\n${jsonString}\n\`\`\``
    } else {
      // 旧格式：直接是数据
      jsonString = JSON.stringify(data, null, 2)
      promptMessage = `This is the predicted property information for this molecule:\n\n\`\`\`json\n${jsonString}\n\`\`\``
    }
    
    // 叠加内容而不是替换
    setInputMessage(prev => {
      if (prev.trim()) {
        return prev + '\n\n' + promptMessage
      } else {
        return promptMessage
      }
    })
    
    // 自动滚动到输入框
    setTimeout(() => {
      const textarea = document.querySelector('textarea')
      if (textarea) {
        textarea.focus()
        textarea.scrollIntoView({ behavior: 'smooth', block: 'center' })
      }
    }, 100)
  }

  // 暴露函数给父组件
  React.useEffect(() => {
    if (onReceivePredictionData) {
      onReceivePredictionData(handleReceivePredictionData)
    }
  }, [onReceivePredictionData])

  // RAG增强聊天消息发送
  const sendChatMessageWithRAG = async (message: string, modelName: string, mode: string) => {
    try {
      const formData = new FormData()
      formData.append('message', message)
      formData.append('model_name', modelName)
      formData.append('use_rag', 'true')
      formData.append('rag_mode', mode)

      const response = await fetch('/api/chat/message-with-rag', {
        method: 'POST',
        body: formData
      })

      return await response.json()
    } catch (error) {
      throw error
    }
  }

  return (
    <div className="min-h-96 max-h-screen flex flex-col">
      {/* 聊天头部 */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
            <BeakerIcon className="w-4 h-4 text-white" />
          </div>
          <div>
            <h4 className="font-medium">AI Analysis Assistant</h4>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              {isConfigured ? 'Connected' : 'Not configured'}
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowConfig(!showConfig)}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
            title="Configuration"
          >
            <CogIcon className="w-4 h-4" />
          </button>
          <button
            onClick={clearMessages}
            className="p-2 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
            title="Clear conversation"
          >
            <TrashIcon className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* 配置面板 */}
      {showConfig && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="p-4 bg-gray-50 dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700"
        >
          <div className="space-y-3">
            <div>
              <label className="block text-sm font-medium mb-1">API Base URL</label>
              <input
                type="text"
                value={config.baseUrl}
                onChange={(e) => setConfig(prev => ({ ...prev, baseUrl: e.target.value }))}
                placeholder="https://openrouter.ai/api/v1"
                className="w-full p-2 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-900"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">API Key</label>
              <input
                type="password"
                value={config.apiKey}
                onChange={(e) => setConfig(prev => ({ ...prev, apiKey: e.target.value }))}
                placeholder="Enter your API key"
                className="w-full p-2 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-900"
              />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">
                Model Selection
                {isLoadingModels && <span className="text-xs text-gray-500 ml-2">(Loading...)</span>}
              </label>
              <select
                value={config.modelName}
                onChange={(e) => setConfig(prev => ({ ...prev, modelName: e.target.value }))}
                className="w-full p-2 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-900 mb-2"
              >
                <option value="google/gemini-2.0-flash-thinking-exp:free">Gemini 2.0 Flash (Free)</option>
                <option value="anthropic/claude-3.5-sonnet">Claude 3.5 Sonnet</option>
                <option value="openai/gpt-4o">GPT-4o</option>
                <option value="deepseek/deepseek-r1">DeepSeek R1</option>
                <option value="qwen/qwen-2.5-72b-instruct">Qwen 2.5 72B</option>
                {availableModels.map((model) => (
                  <option key={model.id} value={model.id}>
                    {model.name} {model.pricing?.prompt === "0" ? "(Free)" : ""}
                  </option>
                ))}
                <option value="custom">Custom Model...</option>
              </select>
              
              {config.modelName === 'custom' && (
                <input
                  type="text"
                  value={customModelName}
                  onChange={(e) => setCustomModelName(e.target.value)}
                  placeholder="Enter custom model name (e.g., provider/model-name)"
                  className="w-full p-2 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-900 mb-2"
                />
              )}
              
              {availableModels.length > 0 && (
                <div className="mt-2">
                  <button
                    onClick={loadAvailableModels}
                    disabled={isLoadingModels}
                    className="text-xs text-blue-500 hover:text-blue-600 disabled:opacity-50"
                  >
                    {isLoadingModels ? 'Loading...' : 'Refresh Models'}
                  </button>
                  <span className="text-xs text-gray-500 ml-2">
                    ({availableModels.length} models available)
                  </span>
                </div>
              )}
            </div>
            
            {/* RAG设置 */}
            <div className="border-t border-gray-200 dark:border-gray-700 pt-3">
              <label className="block text-sm font-medium mb-2">RAG增强设置</label>
              <div className="space-y-2">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={useRAG}
                    onChange={(e) => setUseRAG(e.target.checked)}
                    className="rounded border-gray-300 dark:border-gray-600"
                  />
                  <span className="text-sm">启用知识库增强</span>
                </label>
                
                {useRAG && (
                  <div>
                    <label className="block text-xs text-gray-500 mb-1">检索模式</label>
                    <select
                      value={ragMode}
                      onChange={(e) => setRagMode(e.target.value)}
                      className="w-full p-2 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-900"
                    >
                      <option value="naive">简单检索</option>
                      <option value="local">本地检索</option>
                      <option value="global">全局检索</option>
                      <option value="hybrid">混合检索</option>
                    </select>
                  </div>
                )}
              </div>
            </div>
            
            <button
              onClick={handleConfigSubmit}
              className="w-full btn-primary text-sm"
            >
              Save Configuration
            </button>
          </div>
        </motion.div>
      )}

      {/* 消息列表 */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 min-h-0">
        {messages.length === 0 && (
          <div className="text-center text-gray-500 dark:text-gray-400 py-8">
            <BeakerIcon className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p>Start chatting with AI assistant</p>
            <p className="text-sm mt-2">
              {isConfigured ? 'Enter your question...' : 'Please configure API first'}
            </p>
          </div>
        )}

        {messages.map((message) => (
          <motion.div
            key={message.id}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3 }}
            className={`flex ${message.type === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div className={`flex items-start space-x-3 max-w-[80%] ${
              message.type === 'user' ? 'flex-row-reverse space-x-reverse' : ''
            }`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                message.type === 'user' 
                  ? 'bg-blue-500' 
                  : 'bg-gradient-to-r from-purple-500 to-pink-500'
              }`}>
                {message.type === 'user' ? (
                  <UserIcon className="w-4 h-4 text-white" />
                ) : (
                  <BeakerIcon className="w-4 h-4 text-white" />
                )}
              </div>
              <div className={`rounded-lg p-3 ${
                message.type === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-gray-100 dark:bg-gray-800 text-gray-900 dark:text-white'
              }`}>
                <div className="whitespace-pre-wrap text-sm">{message.content}</div>
                <div className={`text-xs mt-1 opacity-70 ${
                  message.type === 'user' ? 'text-blue-100' : 'text-gray-500 dark:text-gray-400'
                }`}>
                  {message.timestamp.toLocaleTimeString()}
                </div>
              </div>
            </div>
          </motion.div>
        ))}

        {isLoading && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="flex justify-start"
          >
            <div className="flex items-start space-x-3">
              <div className="w-8 h-8 rounded-full bg-gradient-to-r from-purple-500 to-pink-500 flex items-center justify-center">
                <BeakerIcon className="w-4 h-4 text-white" />
              </div>
              <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-3">
                <div className="flex space-x-1">
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                </div>
              </div>
            </div>
          </motion.div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* 输入区域 */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-700">
        <div className="flex space-x-3">
          <textarea
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder={isConfigured ? (useRAG ? "输入问题（将使用知识库增强）..." : "输入问题...") : "请先配置API"}
            disabled={!isConfigured || isLoading}
            className="flex-1 p-3 border border-gray-300 dark:border-gray-600 rounded-lg resize-none bg-white dark:bg-gray-900 disabled:opacity-50"
            rows={2}
          />
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || !isConfigured || isLoading}
            className="px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <PaperAirplaneIcon className="w-5 h-5" />
          </button>
        </div>
        
        {/* RAG状态指示 */}
        {useRAG && isConfigured && (
          <div className="mt-2 flex items-center space-x-2 text-xs text-green-600 dark:text-green-400">
            <div className="w-2 h-2 bg-green-500 rounded-full"></div>
            <span>知识库增强已启用 ({ragMode})</span>
          </div>
        )}
      </div>
      
      {/* 文档上传卡片 */}
      <DocumentUploadCard onDocumentUploaded={(document) => {
        // 文档上传成功时的回调
        console.log('Document uploaded:', document)
        // 可以在这里添加提示信息或更新状态
      }} />
    </div>
  )
}