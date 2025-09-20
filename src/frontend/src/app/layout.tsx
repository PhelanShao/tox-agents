import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { ThemeProvider } from '@/components/providers/theme-provider'
import { QueryProvider } from '@/components/providers/query-provider'
import { Toaster } from 'react-hot-toast'

const inter = Inter({
  subsets: ['latin'],
  display: 'swap',
  variable: '--font-inter',
})

export const metadata: Metadata = {
  title: {
    default: 'Molecular Toxicity Predictor',
    template: '%s | Molecular Toxicity Predictor',
  },
  description: 'Advanced AI-powered molecular toxicity prediction platform with cutting-edge visualization and analysis capabilities.',
  keywords: [
    'molecular toxicity',
    'AI prediction',
    'chemical analysis',
    'drug discovery',
    'computational chemistry',
    'machine learning',
    'molecular visualization',
  ],
  authors: [{ name: 'Molecular AI Lab' }],
  creator: 'Molecular AI Lab',
  publisher: 'Molecular AI Lab',
  formatDetection: {
    email: false,
    address: false,
    telephone: false,
  },
  openGraph: {
    type: 'website',
    locale: 'en_US',
    title: 'Molecular Toxicity Predictor',
    description: 'Advanced AI-powered molecular toxicity prediction platform',
    siteName: 'Molecular Toxicity Predictor',
  },
  robots: {
    index: true,
    follow: true,
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className={inter.variable} suppressHydrationWarning>
      <head>
        <meta name="theme-color" content="#0ea5e9" />
        <meta name="color-scheme" content="light dark" />
        <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=5" />
      </head>
      <body className={`${inter.className} antialiased`}>
        <QueryProvider>
          <ThemeProvider
            attribute="class"
            defaultTheme="system"
            enableSystem
            disableTransitionOnChange
          >
            <div className="relative min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 dark:from-slate-900 dark:via-slate-800 dark:to-indigo-900">
              {/* 背景装饰 */}
              <div className="fixed inset-0 cyber-grid opacity-30 pointer-events-none" />
              <div className="fixed inset-0 bg-gradient-to-br from-transparent via-primary-500/5 to-accent-500/5 pointer-events-none" />
              
              {/* 浮动装饰元素 */}
              <div className="fixed top-20 left-20 w-72 h-72 bg-primary-500/10 rounded-full blur-3xl animate-float pointer-events-none" />
              <div className="fixed bottom-20 right-20 w-96 h-96 bg-accent-500/10 rounded-full blur-3xl animate-float pointer-events-none" style={{ animationDelay: '2s' }} />
              <div className="fixed top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 w-[800px] h-[800px] bg-gradient-radial from-primary-500/5 to-transparent rounded-full blur-3xl pointer-events-none" />
              
              {/* 主要内容 */}
              <div className="relative z-10">
                {children}
              </div>
              
              {/* Toast 通知 */}
              <Toaster
                position="top-right"
                toastOptions={{
                  duration: 4000,
                  style: {
                    background: 'rgba(255, 255, 255, 0.1)',
                    backdropFilter: 'blur(10px)',
                    border: '1px solid rgba(255, 255, 255, 0.2)',
                    color: '#1f2937',
                  },
                  success: {
                    iconTheme: {
                      primary: '#10b981',
                      secondary: '#ffffff',
                    },
                  },
                  error: {
                    iconTheme: {
                      primary: '#ef4444',
                      secondary: '#ffffff',
                    },
                  },
                }}
              />
            </div>
          </ThemeProvider>
        </QueryProvider>
      </body>
    </html>
  )
}