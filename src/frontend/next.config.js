/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    typedRoutes: true,
  },
  async headers() {
    return [
      {
        source: '/:path*{/}?',
        headers: [
          {
            key: 'X-Accel-Buffering',
            value: 'no'
          }
        ]
      }
    ]
  },
  async rewrites() {
    return [
      // 排除属性预测路由，让它使用自定义API路由
      // 其他API仍然使用代理
      {
        source: '/api/predict/binary',
        destination: 'http://localhost:8000/api/predict/binary',
      },
      {
        source: '/api/toxd4c/:path*',
        destination: 'http://localhost:8000/api/toxd4c/:path*',
      },
      {
        source: '/api/convert/:path*',
        destination: 'http://localhost:8000/api/convert/:path*',
      },
      {
        source: '/api/visualize/:path*',
        destination: 'http://localhost:8000/api/visualize/:path*',
      },
      {
        source: '/api/chat/:path*',
        destination: 'http://localhost:8000/api/chat/:path*',
      },
      {
        source: '/api/health',
        destination: 'http://localhost:8000/health',
      },
      {
        source: '/api/download/:path*',
        destination: 'http://localhost:8000/api/download/:path*',
      },
      // 通用API代理（排除已定义的路由）
      {
        source: '/api/:path((?!predict/property).*)',
        destination: 'http://localhost:8000/api/:path*',
      },
    ]
  },
  serverRuntimeConfig: {
    proxyTimeout: 120000, // 2分钟
  },
}

module.exports = nextConfig