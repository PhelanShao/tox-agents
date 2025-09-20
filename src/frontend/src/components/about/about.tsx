'use client'

import { motion } from 'framer-motion'
import Link from 'next/link'
import {
  AcademicCapIcon,
  BeakerIcon,
  ChartBarIcon,
  CpuChipIcon,
  GlobeAltIcon,
  HeartIcon,
  LightBulbIcon,
  ShieldCheckIcon,
  SparklesIcon,
  UserGroupIcon,
  HomeIcon
} from '@heroicons/react/24/outline'
import { FloatingBackground } from '@/components/ui/floating-background'

export function About() {
  const features = [
    {
      icon: BeakerIcon,
      title: 'Advanced Toxicity Prediction',
      description: 'State-of-the-art deep learning framework for predicting 31 different molecular toxicity endpoints with high accuracy and reliability.'
    },
    {
      icon: CpuChipIcon,
      title: 'Multi-Modal Architecture',
      description: 'Innovative integration of Graph Neural Networks, Transformer architecture, and geometric information processing for comprehensive molecular analysis.'
    },
    {
      icon: ShieldCheckIcon,
      title: 'Uncertainty Quantification',
      description: 'Built-in uncertainty estimation provides confidence intervals for each prediction, enabling informed decision-making in critical applications.'
    },
    {
      icon: SparklesIcon,
      title: 'Hierarchical Learning',
      description: 'Four-level hierarchical representation learning from atoms to molecules, mimicking how chemists understand molecular structures.'
    },
    {
      icon: ChartBarIcon,
      title: 'High Performance',
      description: 'Optimized for both accuracy and efficiency, with classification AUC scores of 0.85-0.92 and regression R² scores of 0.75-0.88.'
    },
    {
      icon: GlobeAltIcon,
      title: 'Wide Applications',
      description: 'Supports drug discovery, chemical safety assessment, regulatory compliance, and environmental risk evaluation.'
    }
  ]

  const team = [
    {
      name: 'Research Team',
      role: 'AI & Computational Chemistry',
      description: 'Leading experts in machine learning, computational chemistry, and toxicology research.'
    },
    {
      name: 'Development Team',
      role: 'Software Engineering',
      description: 'Experienced developers specializing in AI frameworks, web technologies, and scientific computing.'
    },
    {
      name: 'Domain Experts',
      role: 'Toxicology & Drug Discovery',
      description: 'Industry professionals providing domain knowledge and validation for toxicity prediction models.'
    }
  ]

  return (
    <div className="min-h-screen relative">
      <FloatingBackground />
      <div className="max-w-6xl mx-auto p-6 space-y-16 relative z-10">
        {/* Back to Home Button */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          className="mb-8"
        >
          <Link
            href="/"
            className="inline-flex items-center gap-2 px-4 py-2 rounded-lg glass-card hover:bg-primary-50 dark:hover:bg-primary-900/20 transition-colors"
          >
            <HomeIcon className="w-4 h-4" />
            Back to Home
          </Link>
        </motion.div>
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card text-sm font-medium text-primary-600 dark:text-primary-400 mb-6">
          <HeartIcon className="w-4 h-4" />
          About ToxD4C
        </div>
        
        <h1 className="text-4xl md:text-5xl font-bold mb-6">
          <span className="text-gradient-primary">Revolutionizing</span>
          <span className="text-gradient-cyber"> Molecular Toxicity</span>
        </h1>
        
        <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
          ToxD4C is a cutting-edge deep learning framework that transforms how we predict and understand molecular toxicity, 
          making drug discovery safer and more efficient.
        </p>
      </motion.div>

      {/* Mission Statement */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="glass-card p-8 text-center"
      >
        <LightBulbIcon className="w-12 h-12 text-primary-500 mx-auto mb-4" />
        <h2 className="text-3xl font-bold mb-4">Our Mission</h2>
        <p className="text-lg text-gray-600 dark:text-gray-300 max-w-4xl mx-auto">
          To advance computational toxicology through innovative AI technologies, providing researchers and industry professionals 
          with powerful tools for accurate toxicity prediction, ultimately contributing to safer drug development and chemical assessment.
        </p>
      </motion.div>

      {/* Key Features */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
      >
        <h2 className="text-3xl font-bold text-center mb-12">Key Features</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, index) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * index }}
              className="glass-card p-6 hover:scale-105 transition-transform duration-300"
            >
              <div className="w-12 h-12 rounded-xl bg-gradient-to-r from-primary-500 to-primary-600 p-3 mb-4">
                <feature.icon className="w-full h-full text-white" />
              </div>
              <h3 className="text-xl font-semibold mb-3">{feature.title}</h3>
              <p className="text-gray-600 dark:text-gray-300">{feature.description}</p>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Technology Stack */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="glass-card p-8"
      >
        <h2 className="text-3xl font-bold text-center mb-8">Technology Stack</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div>
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <CpuChipIcon className="w-6 h-6 text-primary-500" />
              Core AI Technologies
            </h3>
            <ul className="space-y-2 text-gray-600 dark:text-gray-300">
              <li>• Graph Neural Networks (GNN)</li>
              <li>• Transformer Architecture</li>
              <li>• SE(3) Equivariant Networks</li>
              <li>• Bayesian Deep Learning</li>
              <li>• Multi-task Learning</li>
              <li>• Contrastive Learning</li>
            </ul>
          </div>
          <div>
            <h3 className="text-xl font-semibold mb-4 flex items-center gap-2">
              <BeakerIcon className="w-6 h-6 text-primary-500" />
              Chemical Informatics
            </h3>
            <ul className="space-y-2 text-gray-600 dark:text-gray-300">
              <li>• RDKit Molecular Processing</li>
              <li>• 3D Conformer Generation</li>
              <li>• Molecular Fingerprints</li>
              <li>• Chemical Descriptors</li>
              <li>• SMILES Processing</li>
              <li>• Molecular Visualization</li>
            </ul>
          </div>
        </div>
      </motion.div>

      {/* Team */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
      >
        <h2 className="text-3xl font-bold text-center mb-12">Our Team</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {team.map((member, index) => (
            <motion.div
              key={member.name}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 * index }}
              className="glass-card p-6 text-center"
            >
              <UserGroupIcon className="w-12 h-12 text-primary-500 mx-auto mb-4" />
              <h3 className="text-xl font-semibold mb-2">{member.name}</h3>
              <p className="text-primary-600 dark:text-primary-400 font-medium mb-3">{member.role}</p>
              <p className="text-gray-600 dark:text-gray-300">{member.description}</p>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Impact */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="glass-card p-8"
      >
        <h2 className="text-3xl font-bold text-center mb-8">Impact & Applications</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
          <div>
            <h3 className="text-xl font-semibold mb-4">Drug Discovery</h3>
            <p className="text-gray-600 dark:text-gray-300 mb-4">
              Accelerate pharmaceutical research by providing early toxicity screening, reducing the need for 
              expensive animal testing and improving the success rate of drug candidates.
            </p>
          </div>
          <div>
            <h3 className="text-xl font-semibold mb-4">Chemical Safety</h3>
            <p className="text-gray-600 dark:text-gray-300 mb-4">
              Support regulatory compliance and environmental protection by enabling rapid assessment of 
              chemical toxicity for new substances and industrial applications.
            </p>
          </div>
          <div>
            <h3 className="text-xl font-semibold mb-4">Research Advancement</h3>
            <p className="text-gray-600 dark:text-gray-300 mb-4">
              Contribute to the scientific understanding of molecular toxicity mechanisms through 
              interpretable AI models and comprehensive toxicity endpoint coverage.
            </p>
          </div>
          <div>
            <h3 className="text-xl font-semibold mb-4">Global Health</h3>
            <p className="text-gray-600 dark:text-gray-300 mb-4">
              Promote safer chemical development and usage worldwide, contributing to public health 
              protection and environmental sustainability.
            </p>
          </div>
        </div>
      </motion.div>

      {/* Call to Action */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.6 }}
        className="glass-card p-8 text-center"
      >
        <h2 className="text-3xl font-bold mb-4">Join the Revolution</h2>
        <p className="text-lg text-gray-600 dark:text-gray-300 mb-8 max-w-3xl mx-auto">
          Be part of the future of computational toxicology. Explore our platform, contribute to research, 
          or collaborate with our team to advance molecular safety assessment.
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <a
            href="https://github.com/PhelanShao/tox-agents"
            target="_blank"
            rel="noopener noreferrer"
            className="btn-primary"
          >
            Explore Source Code
          </a>
          <a
            href="https://bohrium.dp.tech/apps/tox-agents"
            target="_blank"
            rel="noopener noreferrer"
            className="btn-glass"
          >
            Try Demo
          </a>
          <a
            href="/docs"
            className="btn-glass"
          >
            Read Documentation
          </a>
        </div>
      </motion.div>
      </div>
    </div>
  )
}