'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'

interface ChatBubble {
  id: number
  text: string
  isAction?: boolean
  link?: string
}

const chatMessages: ChatBubble[] = [
  {
    id: 1,
    text: "Are microplastic degradation products toxic?"
  },
  {
    id: 2,
    text: "What toxic substances are produced when vinyl chloride burns?"
  },
  {
    id: 3,
    text: "What are the pathways for the formation of these toxic substances?"
  },
  {
    id: 4,
    text: "Try the nanoreactor-based molecular dynamics solution",
    isAction: true,
    link: "http://molreac.lwy-ai4water-lab.com"
  }
]

export function AnimatedChatBubbles() {
  const [visibleBubbles, setVisibleBubbles] = useState<ChatBubble[]>([])

  useEffect(() => {
    // 立即显示第一个气泡用于测试
    setVisibleBubbles([chatMessages[0]])

    // 然后开始正常的动画循环
    let currentIndex = 0
    const interval = setInterval(() => {
      currentIndex = (currentIndex + 1) % chatMessages.length
      if (currentIndex === 0) {
        setVisibleBubbles([])
        setTimeout(() => {
          setVisibleBubbles([chatMessages[0]])
        }, 500)
      } else {
        setVisibleBubbles(prev => [...prev, chatMessages[currentIndex]])
      }
    }, 2500)

    return () => clearInterval(interval)
  }, [])

  const handleActionClick = (link: string) => {
    window.open(link, '_blank')
  }

  return (
    <div className="flex flex-col items-center space-y-6 min-h-[300px] justify-center w-full">

      <AnimatePresence mode="wait">
        {visibleBubbles.map((bubble, index) => (
          <motion.div
            key={bubble.id}
            initial={{ opacity: 0, y: 30, scale: 0.9 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -30, scale: 0.9 }}
            transition={{
              duration: 0.8,
              type: "spring",
              stiffness: 120,
              damping: 20
            }}
            className={`relative w-full max-w-lg ${
              bubble.isAction ? 'self-center' : index % 2 === 0 ? 'self-start' : 'self-end'
            }`}
          >
            <div
              className={`
                px-6 py-4 rounded-2xl shadow-lg backdrop-blur-sm border
                ${bubble.isAction
                  ? 'bg-gradient-to-r from-blue-500 to-purple-600 text-white cursor-pointer hover:from-blue-600 hover:to-purple-700 transition-all duration-300 transform hover:scale-105 border-blue-400/50 shadow-blue-500/25'
                  : index % 2 === 0
                    ? 'bg-white/90 dark:bg-gray-800/90 text-gray-800 dark:text-gray-200 border-gray-200 dark:border-gray-600 shadow-gray-200/50 dark:shadow-gray-800/50'
                    : 'bg-gradient-to-r from-blue-500 to-blue-600 text-white border-blue-400/50 shadow-blue-500/25'
                }
              `}
              onClick={bubble.isAction && bubble.link ? () => handleActionClick(bubble.link) : undefined}
            >
              <p className="text-sm md:text-base font-medium">
                {bubble.text}
              </p>
              
              {/* 气泡尾巴 */}
              {!bubble.isAction && (
                <div
                  className={`
                    absolute top-1/2 transform -translate-y-1/2 w-0 h-0
                    ${index % 2 === 0
                      ? '-left-2 border-t-8 border-b-8 border-r-8 border-transparent border-r-white/90 dark:border-r-gray-800/90'
                      : '-right-2 border-t-8 border-b-8 border-l-8 border-transparent border-l-blue-500'
                    }
                  `}
                />
              )}
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
      
      {/* 提示文字 */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: visibleBubbles.length > 0 ? 1 : 0 }}
        className="text-center mt-8"
      >
        <p className="text-sm text-gray-500 dark:text-gray-400">
          Enhanced Sampling-Based Molecular Dynamics Nanoreactor
        </p>
      </motion.div>
    </div>
  )
}
