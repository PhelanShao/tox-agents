import { NextRequest, NextResponse } from 'next/server'
import { readFile } from 'fs/promises'
import { join } from 'path'

export async function GET(request: NextRequest) {
  try {
    // 示例文件的路径
    const exampleFilePath = join(process.cwd(), '..', 'examples.xyz')
    
    // 读取文件
    const fileBuffer = await readFile(exampleFilePath)
    
    // 返回文件
    return new NextResponse(fileBuffer, {
      status: 200,
      headers: {
        'Content-Type': 'application/octet-stream',
        'Content-Disposition': 'attachment; filename="examples.xyz"',
        'Content-Length': fileBuffer.length.toString(),
      },
    })
  } catch (error) {
    console.error('Error reading example file:', error)
    return NextResponse.json(
      { error: 'Failed to read example file' },
      { status: 500 }
    )
  }
}
