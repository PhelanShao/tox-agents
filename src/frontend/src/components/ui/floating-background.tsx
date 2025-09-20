'use client'

import { motion } from 'framer-motion'
import Image from 'next/image'
import { useEffect, useState } from 'react'

interface FloatingElement {
  id: number
  src: string
  alt: string
  initialX: number
  initialY: number
  scale: number
  duration: number
  delay: number
}

export function FloatingBackground() {
  const [elements, setElements] = useState<FloatingElement[]>([])
  const [particles, setParticles] = useState<Array<{id: number, initialX: number, initialY: number, targetX: number, targetY: number, duration: number}>>([])

  useEffect(() => {
    const images = [
      { src: '/分子.png', alt: 'Molecule' },
      { src: '/基因.png', alt: 'Gene' },
      { src: '/小鼠.png', alt: 'Mouse' },
      { src: '/药品.png', alt: 'Drug' }
    ]

    const generateElements = () => {
      const newElements: FloatingElement[] = []
      
      // 为每种图片生成多个实例
      images.forEach((image, imageIndex) => {
        for (let i = 0; i < 3; i++) {
          newElements.push({
            id: imageIndex * 3 + i,
            src: image.src,
            alt: image.alt,
            initialX: Math.random() * 100,
            initialY: Math.random() * 100,
            scale: 0.3 + Math.random() * 0.4, // 0.3 到 0.7 的缩放
            duration: 20 + Math.random() * 30, // 20-50秒的动画时长
            delay: Math.random() * 10 // 0-10秒的延迟
          })
        }
      })
      
      setElements(newElements)
    }

    const generateParticles = () => {
      if (typeof window !== 'undefined') {
        const newParticles = Array.from({ length: 20 }).map((_, i) => ({
          id: i,
          initialX: Math.random() * window.innerWidth,
          initialY: Math.random() * window.innerHeight,
          targetX: Math.random() * window.innerWidth,
          targetY: Math.random() * window.innerHeight,
          duration: 15 + Math.random() * 20
        }))
        setParticles(newParticles)
      }
    }

    generateElements()
    generateParticles()
  }, [])

  return (
    <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
      {elements.map((element) => (
        <motion.div
          key={element.id}
          className="absolute opacity-10 dark:opacity-10"
          initial={{
            x: `${element.initialX}vw`,
            y: `${element.initialY}vh`,
            scale: element.scale,
            rotate: 0
          }}
          animate={{
            x: [`${element.initialX}vw`, `${(element.initialX + 30) % 120 - 10}vw`, `${element.initialX}vw`],
            y: [`${element.initialY}vh`, `${(element.initialY + 40) % 120 - 10}vh`, `${element.initialY}vh`],
            rotate: [0, 360, 0],
            scale: [element.scale, element.scale * 1.2, element.scale]
          }}
          transition={{
            duration: element.duration,
            delay: element.delay,
            repeat: Infinity,
            ease: "easeInOut"
          }}
          style={{
            filter: 'blur(1px)',
            mixBlendMode: 'multiply'
          }}
        >
          <div className="relative w-32 h-32 md:w-48 md:h-48">
            <Image
              src={element.src}
              alt={element.alt}
              fill
              className="object-contain"
              priority={false}
            />
          </div>
        </motion.div>
      ))}
      
      {/* 额外的装饰性粒子 */}
      {particles.map((particle) => (
        <motion.div
          key={`particle-${particle.id}`}
          className="absolute w-1 h-1 bg-primary-200 dark:bg-primary-800 rounded-full opacity-20"
          initial={{
            x: particle.initialX,
            y: particle.initialY,
          }}
          animate={{
            x: particle.targetX,
            y: particle.targetY,
          }}
          transition={{
            duration: particle.duration,
            repeat: Infinity,
            repeatType: "reverse",
            ease: "linear"
          }}
        />
      ))}
    </div>
  )
}