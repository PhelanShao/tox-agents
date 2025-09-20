import { NextRequest, NextResponse } from 'next/server'

export async function GET(request: NextRequest) {
  try {
    // 转发到后端API
    const backendResponse = await fetch('http://localhost:8000/api/rag/stats', {
      method: 'GET',
      signal: AbortSignal.timeout(30000), // 30秒
    })

    if (!backendResponse.ok) {
      throw new Error(`Backend API error: ${backendResponse.status}`)
    }

    const data = await backendResponse.json()
    
    return NextResponse.json(data)
  } catch (error) {
    console.error('RAG stats API error:', error)
    
    if (error instanceof Error && error.name === 'TimeoutError') {
      return NextResponse.json(
        { 
          success: false, 
          message: '获取统计信息请求超时，请稍后重试' 
        },
        { status: 408 }
      )
    }
    
    return NextResponse.json(
      { 
        success: false, 
        message: error instanceof Error ? error.message : '获取统计信息失败' 
      },
      { status: 500 }
    )
  }
}