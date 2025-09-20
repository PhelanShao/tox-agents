'use client'

import React, { useState, useRef, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  DocumentPlusIcon,
  TrashIcon,
  ChartBarIcon,
  CloudArrowUpIcon,
  DocumentTextIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  XCircleIcon
} from '@heroicons/react/24/outline'

interface KnowledgeBaseManagerProps {
  onKnowledgeBaseChange?: () => void
}

interface UploadedDocument {
  filename: string
  content_length: number
  upload_time: string
  file_type: 'pdf' | 'txt'
}

interface KnowledgeBaseStats {
  total_documents: number
  total_content_length: number
  last_updated: string
  storage_size: string
}

export function KnowledgeBaseManager({ onKnowledgeBaseChange }: KnowledgeBaseManagerProps) {
  const [isExpanded, setIsExpanded] = useState(true) // 默认展开以显示配置
  const [isInitialized, setIsInitialized] = useState(false)
  const [isUploading, setIsUploading] = useState(false)
  const [uploadedDocuments, setUploadedDocuments] = useState<UploadedDocument[]>([])
  const [stats, setStats] = useState<KnowledgeBaseStats | null>(null)
  const [config, setConfig] = useState({
    apiKey: '',
    baseUrl: 'https://openrouter.ai/api/v1'
  })
  const [showConfig, setShowConfig] = useState(false)
  const [message, setMessage] = useState<{ type: 'success' | 'error' | 'info', text: string } | null>(null)
  const [forceUpdate, setForceUpdate] = useState(0) // 强制更新计数器
  const fileInputRef = useRef<HTMLInputElement>(null)

  useEffect(() => {
    // 初始化时检查RAG服务状态
    checkRAGStatus()
  }, [])

  const showMessage = (type: 'success' | 'error' | 'info', text: string) => {
    setMessage({ type, text })
    setTimeout(() => setMessage(null), 5000)
  }

  const checkRAGStatus = async () => {
    try {
      const response = await fetch('/api/rag/stats')
      const data = await response.json()
      
      if (data.success && data.data && data.data.is_initialized) {
        setStats(data.data)
        setIsInitialized(true)
      } else {
        setIsInitialized(false)
        setStats(null)
      }
    } catch (error) {
      console.error('Failed to check RAG status:', error)
      setIsInitialized(false)
      setStats(null)
    }
  }

  const loadStats = async () => {
    try {
      const response = await fetch('/api/rag/stats')
      const data = await response.json()
      
      if (data.success) {
        setStats(data.data)
        setIsInitialized(data.data.is_initialized || false)
      } else {
        setIsInitialized(false)
        setStats(null)
      }
    } catch (error) {
      console.error('Failed to load RAG stats:', error)
      setIsInitialized(false)
      setStats(null)
    }
  }

  const handleInitialize = async () => {
    if (!config.apiKey.trim()) {
      showMessage('error', '请输入API密钥')
      return
    }

    try {
      const formData = new FormData()
      formData.append('api_key', config.apiKey)
      formData.append('base_url', config.baseUrl)

      const response = await fetch('/api/rag/initialize', {
        method: 'POST',
        body: formData
      })

      const data = await response.json()

      if (data.success) {
        setIsInitialized(true)
        setShowConfig(false)
        showMessage('success', 'RAG服务初始化成功！现在可以上传PDF和TXT文件了')
        // 先加载统计信息，然后确保展开状态
        await loadStats()
        // 强制组件重新渲染
        setForceUpdate(prev => prev + 1)
        // 使用setTimeout确保状态更新后再展开
        setTimeout(() => {
          setIsExpanded(true)
        }, 100)
      } else {
        showMessage('error', `初始化失败: ${data.message}`)
      }
    } catch (error) {
      showMessage('error', `初始化失败: ${error}`)
    }
  }

  const handleFileUpload = async (files: FileList) => {
    if (!files.length) return

    setIsUploading(true)
    const uploadResults: UploadedDocument[] = []

    for (const file of Array.from(files)) {
      if (!file.name.endsWith('.pdf') && !file.name.endsWith('.txt')) {
        showMessage('error', `不支持的文件格式: ${file.name}`)
        continue
      }

      try {
        const formData = new FormData()
        formData.append('file', file)
        formData.append('filename', file.name)

        const response = await fetch('/api/rag/upload', {
          method: 'POST',
          body: formData
        })

        const data = await response.json()

        if (data.success) {
          uploadResults.push(data.data)
          showMessage('success', `文档 "${file.name}" 上传成功`)
        } else {
          showMessage('error', `上传 "${file.name}" 失败: ${data.message}`)
        }
      } catch (error) {
        showMessage('error', `上传 "${file.name}" 失败: ${error}`)
      }
    }

    setUploadedDocuments(prev => [...prev, ...uploadResults])
    setIsUploading(false)
    await loadStats()
    onKnowledgeBaseChange?.()
  }

  const handleClearKnowledgeBase = async () => {
    if (!confirm('确定要清空知识库吗？此操作不可撤销。')) {
      return
    }

    try {
      const response = await fetch('/api/rag/clear', {
        method: 'POST'
      })

      const data = await response.json()

      if (data.success) {
        setUploadedDocuments([])
        setStats(null)
        showMessage('success', '知识库已清空')
        onKnowledgeBaseChange?.()
      } else {
        showMessage('error', `清空失败: ${data.message}`)
      }
    } catch (error) {
      showMessage('error', `清空失败: ${error}`)
    }
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 B'
    const k = 1024
    const sizes = ['B', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div className="border-t border-gray-200 dark:border-gray-700">
      {/* 消息提示 */}
      <AnimatePresence>
        {message && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className={`p-3 text-sm flex items-center space-x-2 ${
              message.type === 'success' ? 'bg-green-50 text-green-700 border-green-200' :
              message.type === 'error' ? 'bg-red-50 text-red-700 border-red-200' :
              'bg-blue-50 text-blue-700 border-blue-200'
            } border-b`}
          >
            {message.type === 'success' && <CheckCircleIcon className="w-4 h-4" />}
            {message.type === 'error' && <XCircleIcon className="w-4 h-4" />}
            {message.type === 'info' && <ExclamationTriangleIcon className="w-4 h-4" />}
            <span>{message.text}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 知识库头部 */}
      <div className="p-4 bg-gray-50 dark:bg-gray-800">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-r from-green-500 to-teal-500 rounded-full flex items-center justify-center">
              <DocumentTextIcon className="w-4 h-4 text-white" />
            </div>
            <div>
              <h4 className="font-medium">知识库管理</h4>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {isInitialized ? '已初始化' : '未初始化'}
              </p>
            </div>
          </div>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setIsExpanded(!isExpanded)}
              className="px-3 py-1 text-sm bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-50 dark:hover:bg-gray-600 transition-colors"
            >
              {isExpanded ? '收起' : '展开'}
            </button>
          </div>
        </div>

        {/* 统计信息 */}
        {stats && (
          <div className="mt-3 grid grid-cols-3 gap-4 text-sm">
            <div className="text-center">
              <div className="font-medium text-gray-900 dark:text-white">{stats.total_documents}</div>
              <div className="text-gray-500 dark:text-gray-400">文档数量</div>
            </div>
            <div className="text-center">
              <div className="font-medium text-gray-900 dark:text-white">{formatFileSize(stats.total_content_length)}</div>
              <div className="text-gray-500 dark:text-gray-400">内容大小</div>
            </div>
            <div className="text-center">
              <div className="font-medium text-gray-900 dark:text-white">
                {stats.last_updated ? new Date(stats.last_updated).toLocaleDateString() : '-'}
              </div>
              <div className="text-gray-500 dark:text-gray-400">最后更新</div>
            </div>
          </div>
        )}
      </div>

      {/* 展开的管理面板 */}
      <AnimatePresence>
        {isExpanded && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="border-t border-gray-200 dark:border-gray-700"
          >
            {!isInitialized ? (
              /* 初始化配置 */
              <div className="p-4 space-y-4">
                <div className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                  请先配置RAG服务以使用知识库功能
                </div>
                
                {!showConfig ? (
                  <button
                    onClick={() => setShowConfig(true)}
                    className="w-full btn-primary text-sm"
                  >
                    配置RAG服务
                  </button>
                ) : (
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
                        placeholder="输入您的API密钥"
                        className="w-full p-2 text-sm border border-gray-300 dark:border-gray-600 rounded bg-white dark:bg-gray-900"
                      />
                    </div>
                    <div className="flex space-x-2">
                      <button
                        onClick={handleInitialize}
                        className="flex-1 btn-primary text-sm"
                      >
                        初始化RAG服务
                      </button>
                      <button
                        onClick={() => setShowConfig(false)}
                        className="px-4 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                      >
                        取消
                      </button>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              /* 文档管理 */
              <div className="p-4 space-y-4">
                {/* 文件上传区域 */}
                <div className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-6 text-center">
                  <input
                    ref={fileInputRef}
                    type="file"
                    multiple
                    accept=".pdf,.txt"
                    onChange={(e) => e.target.files && handleFileUpload(e.target.files)}
                    className="hidden"
                  />
                  
                  {isUploading ? (
                    <div className="space-y-2">
                      <CloudArrowUpIcon className="w-8 h-8 mx-auto text-blue-500 animate-pulse" />
                      <p className="text-sm text-gray-600 dark:text-gray-400">正在上传文档...</p>
                    </div>
                  ) : (
                    <div className="space-y-2">
                      <DocumentPlusIcon className="w-8 h-8 mx-auto text-gray-400" />
                      <p className="text-sm text-gray-600 dark:text-gray-400">
                        点击上传或拖拽PDF/TXT文件到此处
                      </p>
                      <button
                        onClick={() => fileInputRef.current?.click()}
                        className="px-4 py-2 text-sm bg-blue-500 text-white rounded hover:bg-blue-600 transition-colors"
                      >
                        选择文件
                      </button>
                    </div>
                  )}
                </div>

                {/* 已上传文档列表 */}
                {uploadedDocuments.length > 0 && (
                  <div className="space-y-2">
                    <h5 className="text-sm font-medium">最近上传的文档</h5>
                    <div className="space-y-1 max-h-32 overflow-y-auto">
                      {uploadedDocuments.map((doc, index) => (
                        <div key={index} className="flex items-center justify-between p-2 bg-gray-50 dark:bg-gray-700 rounded text-sm">
                          <div className="flex items-center space-x-2">
                            <DocumentTextIcon className="w-4 h-4 text-gray-500" />
                            <span className="truncate">{doc.filename}</span>
                          </div>
                          <div className="text-xs text-gray-500">
                            {formatFileSize(doc.content_length)}
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* 操作按钮 */}
                <div className="flex space-x-2">
                  <button
                    onClick={() => loadStats()}
                    className="flex-1 flex items-center justify-center space-x-2 px-4 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                  >
                    <ChartBarIcon className="w-4 h-4" />
                    <span>刷新统计</span>
                  </button>
                  <button
                    onClick={handleClearKnowledgeBase}
                    className="flex items-center justify-center space-x-2 px-4 py-2 text-sm bg-red-500 text-white rounded hover:bg-red-600 transition-colors"
                  >
                    <TrashIcon className="w-4 h-4" />
                    <span>清空知识库</span>
                  </button>
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}