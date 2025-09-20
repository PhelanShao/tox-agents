import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    // 获取表单数据
    const formData = await request.formData()
    
    // 转发到后端API
    const backendResponse = await fetch('http://localhost:8000/api/chat/message-with-rag', {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(120000), // 120秒，聊天可能需要较长时间
    })

    if (!backendResponse.ok) {
      throw new Error(`Backend API error: ${backendResponse.status}`)
    }

    const data = await backendResponse.json()
    
    return NextResponse.json(data)
  } catch (error) {
    console.error('RAG enhanced chat API error:', error)
    
    if (error instanceof Error && error.name === 'TimeoutError') {
      return NextResponse.json(
        { 
          success: false, 
          message: 'RAG增强聊天请求超时，请稍后重试' 
        },
        { status: 408 }
      )
    }
    
    return NextResponse.json(
      { 
        success: false, 
        message: error instanceof Error ? error.message : 'RAG增强聊天失败' 
      },
      { status: 500 }
    )
  }
}