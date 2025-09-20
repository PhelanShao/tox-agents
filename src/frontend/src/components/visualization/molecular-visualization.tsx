'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { 
  EyeIcon, 
  ArrowPathIcon,
  PhotoIcon,
  AdjustmentsHorizontalIcon
} from '@heroicons/react/24/outline'
import { apiClient } from '@/lib/api'

interface MolecularVisualizationProps {
  file: File | null
  onVisualizationChange?: (data: any) => void
  currentMoleculeIndex?: number
  onMoleculeIndexChange?: (index: number) => void
}

export function MolecularVisualization({ file, onVisualizationChange, currentMoleculeIndex = 0, onMoleculeIndexChange }: MolecularVisualizationProps) {
  const [isLoading, setIsLoading] = useState(false)
  const [visualization, setVisualization] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const [settings, setSettings] = useState({
    frameIndex: 0,
    representation: 'sticks',
    rotationX: 0,
    rotationY: 0,
    rotationZ: 0,
    zoom: 1.0
  })
  const [showControls, setShowControls] = useState(false)
  const [visualizationCache, setVisualizationCache] = useState<Map<string, any>>(new Map())

  // 只在文件变化时重新可视化，设置变化时使用缓存或重新生成
  useEffect(() => {
    if (file) {
      handleVisualization()
    }
  }, [file])

  // 当currentMoleculeIndex变化时，同步frameIndex（避免无限循环）
  useEffect(() => {
    if (currentMoleculeIndex !== settings.frameIndex) {
      console.log(`同步帧索引: ${settings.frameIndex} -> ${currentMoleculeIndex}`)
      setSettings(prev => ({ ...prev, frameIndex: currentMoleculeIndex }))
    }
  }, [currentMoleculeIndex, settings.frameIndex])

  // 当设置或分子索引变化时，检查缓存或重新生成
  useEffect(() => {
    if (file && (settings.frameIndex !== 0 || settings.representation !== 'sticks' ||
        settings.rotationX !== 0 || settings.rotationY !== 0 || settings.rotationZ !== 0 ||
        settings.zoom !== 1.0 || currentMoleculeIndex !== 0)) {
      handleVisualization()
    }
  }, [settings, currentMoleculeIndex])

  const getVisualizationCacheKey = (file: File, settings: any, moleculeIndex: number) => {
    return `${file.name}_${file.size}_${moleculeIndex}_${settings.frameIndex}_${settings.representation}_${settings.rotationX}_${settings.rotationY}_${settings.rotationZ}_${settings.zoom}`
  }

  const handleVisualization = async () => {
    if (!file) return

    // 检查缓存
    const cacheKey = getVisualizationCacheKey(file, settings, currentMoleculeIndex)
    const cachedVisualization = visualizationCache.get(cacheKey)
    if (cachedVisualization) {
      console.log('Using cached visualization result:', cacheKey)
      setVisualization(cachedVisualization)
      onVisualizationChange?.(cachedVisualization)
      return
    }

    setIsLoading(true)
    setError(null)

    try {
      const visualizationSettings = {
        ...settings,
        moleculeIndex: currentMoleculeIndex
      }
      const response = await apiClient.visualizeMolecule(file, visualizationSettings)
      
      if (response.success) {
        // 缓存可视化结果
        setVisualizationCache(prev => new Map(prev.set(cacheKey, response.data)))
        console.log('Caching new visualization result:', cacheKey)
        setVisualization(response.data)
        onVisualizationChange?.(response.data)
      } else {
        setError(response.message || 'Visualization failed')
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Error occurred during visualization')
    } finally {
      setIsLoading(false)
    }
  }

  const updateSetting = (key: string, value: any) => {
    setSettings(prev => ({ ...prev, [key]: value }))

    // 如果是frameIndex变化，同时通知父组件更新分子索引
    if (key === 'frameIndex' && onMoleculeIndexChange) {
      console.log(`Frame Selection滑块变化: ${value}`)
      onMoleculeIndexChange(value)
    }
  }

  const resetView = () => {
    setSettings({
      frameIndex: 0,
      representation: 'sticks',
      rotationX: 0,
      rotationY: 0,
      rotationZ: 0,
      zoom: 1.0
    })
  }

  const downloadImage = () => {
    if (!visualization?.image_base64) return

    const link = document.createElement('a')
    link.href = `data:image/png;base64,${visualization.image_base64}`
    link.download = `molecule_visualization_${Date.now()}.png`
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  if (!file) {
    return (
      <div className="glass-card p-8 text-center">
        <EyeIcon className="w-16 h-16 mx-auto text-gray-400 mb-4" />
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
          Molecular Visualization
        </h3>
        <p className="text-gray-500 dark:text-gray-400">
          Upload molecular file to view 3D structure
        </p>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      {/* 可视化显示区域 */}
      <div className="glass-card p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-medium">Molecular Structure</h3>
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowControls(!showControls)}
              className="btn-glass flex items-center space-x-2"
            >
              <AdjustmentsHorizontalIcon className="w-4 h-4" />
              <span>Controls</span>
            </button>
            <button
              onClick={resetView}
              className="btn-glass flex items-center space-x-2"
            >
              <ArrowPathIcon className="w-4 h-4" />
              <span>Reset</span>
            </button>
            {visualization && (
              <button
                onClick={downloadImage}
                className="btn-glass flex items-center space-x-2"
              >
                <PhotoIcon className="w-4 h-4" />
                <span>Save</span>
              </button>
            )}
          </div>
        </div>

        {/* 可视化内容 */}
        <div className="relative bg-gray-50 dark:bg-gray-800 rounded-lg overflow-hidden" style={{ minHeight: '400px' }}>
          {isLoading && (
            <div className="absolute inset-0 flex items-center justify-center bg-white/80 dark:bg-gray-900/80">
              <div className="flex items-center space-x-3">
                <ArrowPathIcon className="w-6 h-6 text-blue-500 animate-spin" />
                <span className="text-lg font-medium">Generating 3D visualization...</span>
              </div>
            </div>
          )}

          {error && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <div className="text-red-500 mb-2">Visualization Failed</div>
                <p className="text-sm text-gray-500 mb-4">{error}</p>
                <button onClick={handleVisualization} className="btn-primary">
                  Retry
                </button>
              </div>
            </div>
          )}

          {visualization?.image_base64 && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.5 }}
              className="w-full h-full flex items-center justify-center"
            >
              <img
                src={`data:image/png;base64,${visualization.image_base64}`}
                alt="Molecular structure"
                className="max-w-full max-h-full object-contain"
              />
            </motion.div>
          )}

          {!isLoading && !error && !visualization && (
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-center">
                <EyeIcon className="w-12 h-12 mx-auto text-gray-400 mb-4" />
                <p className="text-gray-500 dark:text-gray-400">
                  Click the button above to start visualization
                </p>
              </div>
            </div>
          )}
        </div>

        {/* 图例 */}
        {visualization?.legend && (
          <div className="mt-4 p-3 bg-gray-50 dark:bg-gray-800 rounded-lg">
            <div className="text-sm font-medium mb-2">Legend</div>
            <div className="text-xs text-gray-600 dark:text-gray-400" 
                 dangerouslySetInnerHTML={{ __html: visualization.legend }} />
          </div>
        )}
      </div>

      {/* 控制面板 */}
      {showControls && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="glass-card p-6"
        >
          <h4 className="text-lg font-medium mb-4">Visualization Controls</h4>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* 帧选择 */}
            {visualization?.total_frames > 1 && (
              <div>
                <label className="block text-sm font-medium mb-2">
                  Frame Selection ({settings.frameIndex + 1} / {visualization.total_frames})
                </label>
                <input
                  type="range"
                  min="0"
                  max={visualization.total_frames - 1}
                  value={settings.frameIndex}
                  onChange={(e) => updateSetting('frameIndex', parseInt(e.target.value))}
                  className="w-full"
                />
              </div>
            )}

            {/* 表示方式 */}
            <div>
              <label className="block text-sm font-medium mb-2">Representation</label>
              <select
                value={settings.representation}
                onChange={(e) => updateSetting('representation', e.target.value)}
                className="w-full p-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800"
              >
                <option value="sticks">Stick Model</option>
                <option value="ball_and_stick">Ball and Stick</option>
                <option value="spacefill">Space Fill</option>
                <option value="wireframe">Wireframe</option>
                <option value="surface">Surface</option>
              </select>
            </div>

            {/* 旋转控制 */}
            <div>
              <label className="block text-sm font-medium mb-2">X-axis Rotation</label>
              <input
                type="range"
                min="-180"
                max="180"
                value={settings.rotationX}
                onChange={(e) => updateSetting('rotationX', parseInt(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-500 mt-1">{settings.rotationX}°</div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Y-axis Rotation</label>
              <input
                type="range"
                min="-180"
                max="180"
                value={settings.rotationY}
                onChange={(e) => updateSetting('rotationY', parseInt(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-500 mt-1">{settings.rotationY}°</div>
            </div>

            <div>
              <label className="block text-sm font-medium mb-2">Z-axis Rotation</label>
              <input
                type="range"
                min="-180"
                max="180"
                value={settings.rotationZ}
                onChange={(e) => updateSetting('rotationZ', parseInt(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-500 mt-1">{settings.rotationZ}°</div>
            </div>

            {/* 缩放控制 */}
            <div>
              <label className="block text-sm font-medium mb-2">Zoom</label>
              <input
                type="range"
                min="0.1"
                max="5.0"
                step="0.1"
                value={settings.zoom}
                onChange={(e) => updateSetting('zoom', parseFloat(e.target.value))}
                className="w-full"
              />
              <div className="text-xs text-gray-500 mt-1">{settings.zoom}x</div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  )
}