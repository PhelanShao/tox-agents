import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    // 获取表单数据
    const formData = await request.formData()
    
    // 转发到后端API
    const backendResponse = await fetch('http://localhost:8000/api/rag/upload', {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(60000), // 60秒，文件上传可能需要更长时间
    })

    if (!backendResponse.ok) {
      throw new Error(`Backend API error: ${backendResponse.status}`)
    }

    const data = await backendResponse.json()
    
    return NextResponse.json(data)
  } catch (error) {
    console.error('RAG upload API error:', error)
    
    if (error instanceof Error && error.name === 'TimeoutError') {
      return NextResponse.json(
        { 
          success: false, 
          message: '文档上传请求超时，请稍后重试' 
        },
        { status: 408 }
      )
    }
    
    return NextResponse.json(
      { 
        success: false, 
        message: error instanceof Error ? error.message : '文档上传失败' 
      },
      { status: 500 }
    )
  }
}