'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import Image from 'next/image'
import Link from 'next/link'
import {
  BeakerIcon,
  Bars3Icon,
  XMarkIcon,
  MoonIcon,
  SunIcon
} from '@heroicons/react/24/outline'

export function Header() {
  const [isMenuOpen, setIsMenuOpen] = useState(false)
  const [isDarkMode, setIsDarkMode] = useState(false)

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode)
    // 这里可以添加实际的暗色模式切换逻辑
  }

  const handleNavClick = (href: string) => {
    window.location.href = href
  }

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-white/80 dark:bg-gray-900/80 backdrop-blur-md border-b border-gray-200 dark:border-gray-700">
      <div className="container-wide">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.5 }}
            className="flex items-center space-x-3"
          >
            <Link href="/" className="flex items-center space-x-3 hover:opacity-80 transition-opacity">
              <div className="w-8 h-8 bg-gradient-to-r from-blue-500 to-cyan-500 rounded-lg flex items-center justify-center">
                <BeakerIcon className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold text-gray-900 dark:text-white">
                ToxAgents
              </span>
              <div className="text-gray-400 dark:text-gray-500 flex items-center">|</div>
              <div className="relative w-16 h-16 flex items-center justify-center">
                <Image
                  src="/logo.png"
                  alt="Logo"
                  width={64}
                  height={64}
                  className="object-contain"
                />
              </div>
            </Link>
          </motion.div>

          {/* Desktop Navigation */}
          <nav className="hidden md:flex items-center space-x-8 ml-8 relative z-50">
            <Link
              href="/docs"
              className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors cursor-pointer relative z-50 px-3 py-2 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800"
              style={{ pointerEvents: 'auto', display: 'block' }}
              onClick={(e) => {
                e.preventDefault()
                handleNavClick('/docs')
              }}
            >
              Documentation
            </Link>
            <Link
              href="/about"
              className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors cursor-pointer relative z-50 px-3 py-2 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800"
              style={{ pointerEvents: 'auto', display: 'block' }}
              onClick={(e) => {
                e.preventDefault()
                handleNavClick('/about')
              }}
            >
              About
            </Link>
          </nav>

          {/* Right side controls */}
          <div className="flex items-center space-x-4">
            {/* Dark mode toggle */}
            <button
              onClick={toggleDarkMode}
              className="p-2 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              aria-label="Toggle dark mode"
            >
              {isDarkMode ? (
                <SunIcon className="w-5 h-5 text-gray-600 dark:text-gray-300" />
              ) : (
                <MoonIcon className="w-5 h-5 text-gray-600 dark:text-gray-300" />
              )}
            </button>

            {/* Mobile menu button */}
            <button
              onClick={() => setIsMenuOpen(!isMenuOpen)}
              className="md:hidden p-2 rounded-lg bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
              aria-label="Toggle menu"
            >
              {isMenuOpen ? (
                <XMarkIcon className="w-5 h-5 text-gray-600 dark:text-gray-300" />
              ) : (
                <Bars3Icon className="w-5 h-5 text-gray-600 dark:text-gray-300" />
              )}
            </button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {isMenuOpen && (
          <motion.nav
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            className="md:hidden py-4 border-t border-gray-200 dark:border-gray-700 relative z-50"
          >
            <div className="flex flex-col space-y-4">
              <Link
                href="/docs"
                className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors cursor-pointer relative z-50 px-3 py-2 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800"
                style={{ pointerEvents: 'auto', display: 'block' }}
                onClick={(e) => {
                  e.preventDefault()
                  setIsMenuOpen(false)
                  handleNavClick('/docs')
                }}
              >
                Documentation
              </Link>
              <Link
                href="/about"
                className="text-gray-600 dark:text-gray-300 hover:text-blue-600 dark:hover:text-blue-400 transition-colors cursor-pointer relative z-50 px-3 py-2 rounded-md hover:bg-gray-100 dark:hover:bg-gray-800"
                style={{ pointerEvents: 'auto', display: 'block' }}
                onClick={(e) => {
                  e.preventDefault()
                  setIsMenuOpen(false)
                  handleNavClick('/about')
                }}
              >
                About
              </Link>
            </div>
          </motion.nav>
        )}
      </div>
    </header>
  )
}