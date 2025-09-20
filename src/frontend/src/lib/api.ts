/**
 * API客户端 - 与FastAPI后端通信
 * 保持原有功能逻辑，只是通过现代化的API调用
 */

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || ''
const DEFAULT_BINARY_MODEL_PATH = process.env.NEXT_PUBLIC_BINARY_MODEL_PATH || 'models/ToxPred_modelmini'
const DEFAULT_PROPERTY_MODEL_PATH = process.env.NEXT_PUBLIC_PROPERTY_MODEL_PATH || 'models/MD_model'
const DEFAULT_REFERENCE_PATH = process.env.NEXT_PUBLIC_REFERENCE_PATH || 'models/refscale.npz'

export interface ApiResponse<T = any> {
  success: boolean
  message?: string
  data?: T
  [key: string]: any
}

class ApiClient {
  private baseURL: string

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {},
    timeout: number = 30000  // 默认30秒超时
  ): Promise<ApiResponse<T>> {
    try {
      // 创建超时控制器
      const controller = new AbortController()
      const timeoutId = setTimeout(() => controller.abort(), timeout)
      
      const response = await fetch(`${this.baseURL}${endpoint}`, {
        ...options,
        signal: controller.signal,
        headers: {
          ...options.headers,
        },
      })

      clearTimeout(timeoutId)

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      return data
    } catch (error) {
      console.error('API request failed:', error)
      if (error instanceof Error && error.name === 'AbortError') {
        return {
          success: false,
          message: '请求超时，请稍后重试',
        }
      }
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Unknown error',
      }
    }
  }

  // 文件转换API
  async convertXyzToNpz(file: File): Promise<ApiResponse> {
    const formData = new FormData()
    formData.append('file', file)

    return this.request('/api/convert/xyz-to-npz', {
      method: 'POST',
      body: formData,
    })
  }

  async convertNpzToXyz(file: File): Promise<ApiResponse> {
    const formData = new FormData()
    formData.append('file', file)

    return this.request('/api/convert/npz-to-xyz', {
      method: 'POST',
      body: formData,
    })
  }

  // 预测API
  async predictBinary(
    file: File,
    modelPath: string = DEFAULT_BINARY_MODEL_PATH
  ): Promise<ApiResponse> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('model_path', modelPath)

    return this.request('/api/predict/binary', {
      method: 'POST',
      body: formData,
    })
  }

  async predictProperty(
    file: File,
    modelPath: string = DEFAULT_PROPERTY_MODEL_PATH,
    referencePath: string = DEFAULT_REFERENCE_PATH
  ): Promise<ApiResponse> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('model_path', modelPath)
    formData.append('reference_path', referencePath)

    // 使用自定义API路由避免代理超时问题
    return this.request('/api/predict/property', {
      method: 'POST',
      body: formData,
    }, 120000)  // 120秒超时
  }

  // 可视化API
  async visualizeMolecule(
    file: File,
    options: {
      frameIndex?: number
      representation?: string
      rotationX?: number
      rotationY?: number
      rotationZ?: number
      zoom?: number
      moleculeIndex?: number
    } = {}
  ): Promise<ApiResponse> {
    const formData = new FormData()
    formData.append('file', file)
    formData.append('frame_index', String(options.frameIndex || 0))
    formData.append('representation', options.representation || 'sticks')
    formData.append('rotation_x', String(options.rotationX || 0))
    formData.append('rotation_y', String(options.rotationY || 0))
    formData.append('rotation_z', String(options.rotationZ || 0))
    formData.append('zoom', String(options.zoom || 1.0))
    formData.append('molecule_index', String(options.moleculeIndex || 0))

    return this.request('/api/visualize/molecule', {
      method: 'POST',
      body: formData,
    })
  }

  // 聊天API
  async configureChat(baseUrl: string, apiKey: string): Promise<ApiResponse> {
    const formData = new FormData()
    formData.append('base_url', baseUrl)
    formData.append('api_key', apiKey)

    return this.request('/api/chat/configure', {
      method: 'POST',
      body: formData,
    })
  }

  async sendChatMessage(
    message: string,
    modelName: string = 'google/gemini-2.0-flash-thinking-exp:free',
    imagePath?: string
  ): Promise<ApiResponse> {
    const formData = new FormData()
    formData.append('message', message)
    formData.append('model_name', modelName)
    if (imagePath) {
      formData.append('image_path', imagePath)
    }

    return this.request('/api/chat/message', {
      method: 'POST',
      body: formData,
    })
  }

  // 获取可用模型列表
  async getAvailableModels(): Promise<ApiResponse> {
    return this.request('/api/chat/models')
  }

  // 导出API
  async exportFrame(
    frameIndex: number,
    exportFormat: string = 'PNG',
    binaryPredFile?: string,
    propertyPredFile?: string,
    currentImagePath?: string
  ): Promise<ApiResponse> {
    const formData = new FormData()
    formData.append('frame_index', String(frameIndex))
    formData.append('export_format', exportFormat)
    if (binaryPredFile) formData.append('binary_pred_file', binaryPredFile)
    if (propertyPredFile) formData.append('property_pred_file', propertyPredFile)
    if (currentImagePath) formData.append('current_image_path', currentImagePath)

    return this.request('/api/export/frame', {
      method: 'POST',
      body: formData,
    })
  }

  // 获取概率图表
  async getProbabilityPlot(csvPath: string): Promise<ApiResponse> {
    return this.request(`/api/plot/probability/${encodeURIComponent(csvPath)}`)
  }

  // 下载文件
  getDownloadUrl(filePath: string): string {
    return `${this.baseURL}/api/download/${encodeURIComponent(filePath)}`
  }

  // 获取图像URL
  getImageUrl(imagePath: string): string {
    return `${this.baseURL}/api/download/${encodeURIComponent(imagePath)}`
  }

  // ToxD4C API
  async getToxD4CInfo(): Promise<ApiResponse> {
    return this.request('/api/toxd4c/info')
  }

  async predictToxD4CSmiles(smilesInput: string): Promise<ApiResponse> {
    const formData = new FormData()
    formData.append('smiles_input', smilesInput)

    return this.request('/api/toxd4c/predict/smiles', {
      method: 'POST',
      body: formData,
    })
  }

  async predictToxD4CFile(file: File): Promise<ApiResponse> {
    const formData = new FormData()
    formData.append('file', file)

    return this.request('/api/toxd4c/predict/file', {
      method: 'POST',
      body: formData,
    })
  }

  // 通用文件转换API
  async convertFile(file: File, targetFormat?: string): Promise<ApiResponse> {
    const formData = new FormData()
    formData.append('file', file)
    if (targetFormat) {
      formData.append('target_format', targetFormat)
    }

    return this.request('/api/convert/file', {
      method: 'POST',
      body: formData,
    })
  }
}

// 导出单例实例
export const apiClient = new ApiClient()
// 导出默认类
export default ApiClient
