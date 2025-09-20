import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    // 转发到后端API
    const backendResponse = await fetch('http://localhost:8000/api/rag/clear', {
      method: 'POST',
      signal: AbortSignal.timeout(30000), // 30秒
    })

    if (!backendResponse.ok) {
      throw new Error(`Backend API error: ${backendResponse.status}`)
    }

    const data = await backendResponse.json()
    
    return NextResponse.json(data)
  } catch (error) {
    console.error('RAG clear API error:', error)
    
    if (error instanceof Error && error.name === 'TimeoutError') {
      return NextResponse.json(
        { 
          success: false, 
          message: '清空知识库请求超时，请稍后重试' 
        },
        { status: 408 }
      )
    }
    
    return NextResponse.json(
      { 
        success: false, 
        message: error instanceof Error ? error.message : '清空知识库失败' 
      },
      { status: 500 }
    )
  }
}