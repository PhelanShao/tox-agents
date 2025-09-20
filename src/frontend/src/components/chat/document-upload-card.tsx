'use client'

import React, { useState, useRef } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  DocumentPlusIcon,
  CloudArrowUpIcon,
  DocumentTextIcon,
  CheckCircleIcon,
  XCircleIcon,
  TrashIcon,
  ChartBarIcon
} from '@heroicons/react/24/outline'

interface DocumentUploadCardProps {
  onDocumentUploaded?: (document: any) => void
}

interface UploadedDocument {
  filename: string
  content_length: number
  upload_time: string
  file_type: 'pdf' | 'txt'
}

export function DocumentUploadCard({ onDocumentUploaded }: DocumentUploadCardProps) {
  const [isUploading, setIsUploading] = useState(false)
  const [uploadedDocuments, setUploadedDocuments] = useState<UploadedDocument[]>([])
  const [message, setMessage] = useState<{ type: 'success' | 'error' | 'info', text: string } | null>(null)
  const [stats, setStats] = useState<any>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const showMessage = (type: 'success' | 'error' | 'info', text: string) => {
    setMessage({ type, text })
    setTimeout(() => setMessage(null), 5000)
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
        // 首先尝试初始化RAG服务（使用默认配置）
        const initFormData = new FormData()
        initFormData.append('api_key', 'default-key') // 使用默认密钥
        initFormData.append('base_url', 'https://openrouter.ai/api/v1')

        await fetch('/api/rag/initialize', {
          method: 'POST',
          body: initFormData
        })

        // 上传文档
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
          onDocumentUploaded?.(data.data)
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
  }

  const loadStats = async () => {
    try {
      const response = await fetch('/api/rag/stats')
      const data = await response.json()
      
      if (data.success) {
        setStats(data.data)
      }
    } catch (error) {
      console.error('Failed to load stats:', error)
    }
  }

  const handleClearDocuments = async () => {
    if (!confirm('确定要清空所有文档吗？此操作不可撤销。')) {
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
        showMessage('success', '文档已清空')
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

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    
    const files = e.dataTransfer.files
    if (files.length > 0) {
      handleFileUpload(files)
    }
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 shadow-sm">
      {/* 消息提示 */}
      <AnimatePresence>
        {message && (
          <motion.div
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className={`p-3 text-sm flex items-center space-x-2 rounded-t-lg ${
              message.type === 'success' ? 'bg-green-50 text-green-700 border-green-200' :
              message.type === 'error' ? 'bg-red-50 text-red-700 border-red-200' :
              'bg-blue-50 text-blue-700 border-blue-200'
            } border-b`}
          >
            {message.type === 'success' && <CheckCircleIcon className="w-4 h-4" />}
            {message.type === 'error' && <XCircleIcon className="w-4 h-4" />}
            <span>{message.text}</span>
          </motion.div>
        )}
      </AnimatePresence>

      {/* 卡片头部 */}
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-500 rounded-full flex items-center justify-center">
              <DocumentPlusIcon className="w-4 h-4 text-white" />
            </div>
            <div>
              <h4 className="font-medium">文档上传</h4>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                上传PDF和TXT文件构建知识库
              </p>
            </div>
          </div>
          {stats && (
            <div className="text-right">
              <div className="text-sm font-medium">{stats.total_documents || 0} 个文档</div>
              <div className="text-xs text-gray-500">{stats.storage_size || '0 KB'}</div>
            </div>
          )}
        </div>
      </div>

      {/* 文件上传区域 */}
      <div className="p-4">
        <div 
          className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-6 text-center transition-colors hover:border-blue-400 dark:hover:border-blue-500"
          onDragOver={handleDragOver}
          onDrop={handleDrop}
        >
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
          <div className="mt-4 space-y-2">
            <div className="flex items-center justify-between">
              <h5 className="text-sm font-medium">已上传的文档</h5>
              <button
                onClick={handleClearDocuments}
                className="flex items-center space-x-1 px-2 py-1 text-xs text-red-600 hover:text-red-700 transition-colors"
              >
                <TrashIcon className="w-3 h-3" />
                <span>清空</span>
              </button>
            </div>
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
        <div className="mt-4 flex space-x-2">
          <button
            onClick={loadStats}
            className="flex-1 flex items-center justify-center space-x-2 px-4 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          >
            <ChartBarIcon className="w-4 h-4" />
            <span>刷新统计</span>
          </button>
        </div>
      </div>
    </div>
  )
}