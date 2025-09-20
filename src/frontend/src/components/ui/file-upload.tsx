'use client'

import { useState, useCallback } from 'react'
import { motion } from 'framer-motion'
import { 
  CloudArrowUpIcon, 
  DocumentIcon,
  XMarkIcon,
  CheckCircleIcon
} from '@heroicons/react/24/outline'

interface FileUploadProps {
  onFileUpload: (file: File) => void
  acceptedFormats?: string[]
  maxSize?: number // in MB
  className?: string
}

export function FileUpload({ 
  onFileUpload, 
  acceptedFormats = ['.xyz'],
  maxSize = 100,
  className = ''
}: FileUploadProps) {
  const [isDragOver, setIsDragOver] = useState(false)
  const [uploadedFile, setUploadedFile] = useState<File | null>(null)
  const [error, setError] = useState<string | null>(null)

  const validateFile = (file: File): string | null => {
    // Check file size
    if (file.size > maxSize * 1024 * 1024) {
      return `File size cannot exceed ${maxSize}MB`
    }

    // Check file extension
    const fileExtension = '.' + file.name.split('.').pop()?.toLowerCase()
    if (!acceptedFormats.includes(fileExtension)) {
      return `Unsupported file format. Supported formats: ${acceptedFormats.join(', ')}`
    }

    return null
  }

  const handleFileSelect = useCallback((file: File) => {
    const validationError = validateFile(file)
    if (validationError) {
      setError(validationError)
      return
    }

    setError(null)
    setUploadedFile(file)
    onFileUpload(file)
  }, [onFileUpload, maxSize, acceptedFormats])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)

    const files = Array.from(e.dataTransfer.files)
    if (files.length > 0) {
      handleFileSelect(files[0])
    }
  }, [handleFileSelect])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragOver(false)
  }, [])

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files
    if (files && files.length > 0) {
      handleFileSelect(files[0])
    }
  }

  const removeFile = () => {
    setUploadedFile(null)
    setError(null)
  }

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div className={`w-full ${className}`}>
      {!uploadedFile ? (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
          className={`
            relative border-2 border-dashed rounded-lg p-8 text-center transition-all duration-300
            ${isDragOver 
              ? 'border-blue-400 bg-blue-50 dark:bg-blue-900/20' 
              : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
            }
            ${error ? 'border-red-400 bg-red-50 dark:bg-red-900/20' : ''}
          `}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
        >
          <input
            type="file"
            accept={acceptedFormats.join(',')}
            onChange={handleFileInputChange}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
          />
          
          <div className="flex flex-col items-center space-y-4">
            <div className={`
              w-16 h-16 rounded-full flex items-center justify-center transition-colors
              ${isDragOver 
                ? 'bg-blue-100 dark:bg-blue-800' 
                : 'bg-gray-100 dark:bg-gray-800'
              }
            `}>
              <CloudArrowUpIcon className={`
                w-8 h-8 transition-colors
                ${isDragOver 
                  ? 'text-blue-600 dark:text-blue-400' 
                  : 'text-gray-400 dark:text-gray-500'
                }
              `} />
            </div>
            
            <div>
              <p className="text-lg font-medium text-gray-900 dark:text-white mb-2">
                {isDragOver ? 'Drop file to upload' : 'Drag file here or click to select'}
              </p>
              <p className="text-sm text-gray-500 dark:text-gray-400">
                Supported formats: {acceptedFormats.join(', ')} (Max {maxSize}MB)
              </p>
            </div>
          </div>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0, scale: 0.95 }}
          animate={{ opacity: 1, scale: 1 }}
          transition={{ duration: 0.3 }}
          className="border border-gray-200 dark:border-gray-700 rounded-lg p-4 bg-green-50 dark:bg-green-900/20"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-green-100 dark:bg-green-800 rounded-lg flex items-center justify-center">
                <DocumentIcon className="w-5 h-5 text-green-600 dark:text-green-400" />
              </div>
              <div>
                <p className="font-medium text-gray-900 dark:text-white">
                  {uploadedFile.name}
                </p>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {formatFileSize(uploadedFile.size)}
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-2">
              <CheckCircleIcon className="w-5 h-5 text-green-600 dark:text-green-400" />
              <button
                onClick={removeFile}
                className="p-1 hover:bg-gray-200 dark:hover:bg-gray-700 rounded transition-colors"
                aria-label="Remove file"
              >
                <XMarkIcon className="w-4 h-4 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300" />
              </button>
            </div>
          </div>
        </motion.div>
      )}

      {error && (
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          className="mt-3 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg"
        >
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        </motion.div>
      )}
    </div>
  )
}