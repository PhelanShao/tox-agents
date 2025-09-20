import { NextRequest, NextResponse } from 'next/server'

export async function POST(request: NextRequest) {
  try {
    // 获取表单数据
    const formData = await request.formData()
    
    // 转发到后端API
    const backendResponse = await fetch('http://localhost:8000/api/rag/query', {
      method: 'POST',
      body: formData,
      signal: AbortSignal.timeout(60000), // 60秒，查询可能需要较长时间
    })

    if (!backendResponse.ok) {
      throw new Error(`Backend API error: ${backendResponse.status}`)
    }

    const data = await backendResponse.json()
    
    return NextResponse.json(data)
  } catch (error) {
    console.error('RAG query API error:', error)
    
    if (error instanceof Error && error.name === 'TimeoutError') {
      return NextResponse.json(
        { 
          success: false, 
          message: '知识库查询请求超时，请稍后重试' 
        },
        { status: 408 }
      )
    }
    
    return NextResponse.json(
      { 
        success: false, 
        message: error instanceof Error ? error.message : '知识库查询失败' 
      },
      { status: 500 }
    )
  }
}