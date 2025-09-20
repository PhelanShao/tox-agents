'use client'

import { motion } from 'framer-motion'
import Link from 'next/link'
import {
  BookOpenIcon,
  CpuChipIcon,
  BeakerIcon,
  ChartBarIcon,
  AcademicCapIcon,
  SparklesIcon,
  ShieldCheckIcon,
  ArrowRightIcon,
  HomeIcon
} from '@heroicons/react/24/outline'
import { FloatingBackground } from '@/components/ui/floating-background'

export function Documentation() {
  const sections = [
    {
      id: 'overview',
      title: 'Project Overview',
      icon: BookOpenIcon,
      content: `
        ToxD4C is an advanced deep learning framework for molecular toxicity prediction. 
        This framework innovatively integrates Graph Neural Networks, Transformer architecture, 
        geometric information processing, and chemical prior knowledge to provide accurate and 
        reliable toxicity prediction capabilities for drug discovery and chemical safety assessment.
      `
    },
    {
      id: 'core-goals',
      title: 'Core Objectives',
      icon: SparklesIcon,
      content: `
        • Multi-task toxicity prediction: Simultaneously predict 31 different toxicity endpoints (26 classification + 5 regression tasks)
        • Multi-modal information fusion: Integrate 2D graph structure, 3D geometric information, molecular fingerprints and chemical descriptors
        • Uncertainty quantification: Provide confidence estimates for each prediction
        • Enhanced interpretability: Provide model interpretability through attention mechanisms and hierarchical representation learning
      `
    },
    {
      id: 'architecture',
      title: 'Technical Architecture',
      icon: CpuChipIcon,
      content: `
        The ToxD4C framework consists of four core components:
        
        1. Multi-Modal Encoder Core
           - GNN-Transformer hybrid architecture
           - Dynamic fusion module with adaptive weight learning
           - Cross-attention mechanism for feature enhancement
        
        2. Geometric Information Processing
           - SE(3) equivariant layers for 3D molecular structure
           - Distance-aware message passing
           - Geometric-topological dual encoder
        
        3. Hierarchical Representation Learning
           - Four-level hierarchy: Atom → Functional Group → Scaffold → Molecule
           - Multi-scale GCN architecture with different receptive fields
           - Chemical feature encoding at multiple levels
        
        4. Multi-task Prediction Architecture
           - Task-specific heads for 31 toxicity endpoints
           - Uncertainty quantification with Bayesian inference
           - Contrastive learning for enhanced representation quality
      `
    },
    {
      id: 'features',
      title: 'Key Features',
      icon: ShieldCheckIcon,
      content: `
        Multi-Modal Deep Fusion:
        • Integrates four complementary molecular representation modalities
        • Dynamic weight generation based on molecular features
        • Cross-attention mechanism for deep information exchange
        
        Hierarchical Representation Learning:
        • Four-level hierarchical architecture mimicking chemist cognition
        • Multi-scale receptive fields (2/4/8-layer GCN)
        • Complete modeling from microscopic to macroscopic
        
        Intelligent Uncertainty Quantification:
        • Bayesian deep learning integration
        • Aleatoric and epistemic uncertainty modeling
        • Calibrated confidence intervals for risk assessment
        
        End-to-End Multi-Task Learning:
        • Simultaneous prediction of 31 toxicity endpoints
        • Shared representation learning and task knowledge transfer
        • Unified toxicity prediction platform
      `
    },
    {
      id: 'toxicity-tasks',
      title: 'Toxicity Prediction Tasks',
      icon: BeakerIcon,
      content: `
        Classification Tasks (26):
        • Carcinogenicity, Ames Mutagenicity, Cardiotoxicity
        • CYP Inhibition, Hepatotoxicity, Nephrotoxicity
        • Neurotoxicity, Skin Sensitization, Eye Irritation
        • Respiratory Toxicity, Reproductive Toxicity, Developmental Toxicity
        • Endocrine Disruption, Immunotoxicity, Genotoxicity
        • Hematotoxicity, Plasma Protein Binding, BBB Penetration
        • P-gp Substrate, hERG Blocking, Nuclear Receptor Activation
        • Stress Response Pathway, DNA Damage, Cell Cycle Toxicity
        • Mitochondrial Toxicity, Oxidative Stress
        
        Regression Tasks (5):
        • Acute Oral Toxicity LD50
        • Aquatic Toxicity LC50
        • Bioconcentration Factor BCF
        • Soil Adsorption Coefficient Koc
        • Octanol-Water Partition Coefficient LogP
      `
    },
    {
      id: 'performance',
      title: 'Model Performance',
      icon: ChartBarIcon,
      content: `
        Computational Efficiency:
        • Total parameters: ~50M
        • Hidden dimension: 512
        • Attention heads: 8
        • Maximum sequence length: 512
        • Parallel computation design with GPU optimization
        
        Prediction Accuracy:
        • Classification tasks average AUC: 0.85-0.92
        • Regression tasks average R²: 0.75-0.88
        • Calibration error (ECE): < 0.05
        • Uncertainty correlation coefficient: > 0.80
        
        Interpretability Analysis:
        • GAT attention weights show important chemical bonds and atoms
        • Transformer attention reveals long-range molecular interactions
        • Cross-attention reveals information fusion between modalities
        • Feature importance analysis across hierarchical levels
      `
    },
    {
      id: 'applications',
      title: 'Application Scenarios',
      icon: AcademicCapIcon,
      content: `
        Drug Discovery:
        • Early toxicity screening before synthesis
        • Lead compound optimization guidance
        • ADMET prediction integration
        
        Chemical Safety Assessment:
        • New chemical registration support (REACH compliance)
        • Environmental risk assessment
        • Occupational health protection
        
        Regulatory Science:
        • Computational toxicology advancement
        • Risk assessment modernization
        • International chemical safety standardization
        
        Technical Innovation Value:
        • First systematic multi-modal molecular toxicity prediction framework
        • Original combination of SE(3) equivariant processing + hierarchical learning
        • First deep application of contrastive learning in molecular toxicology
      `
    }
  ]

  return (
    <div className="min-h-screen relative">
      <FloatingBackground />
      <div className="max-w-6xl mx-auto p-6 space-y-8 relative z-10">
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
        className="text-center mb-12"
      >
        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card text-sm font-medium text-primary-600 dark:text-primary-400 mb-6">
          <BookOpenIcon className="w-4 h-4" />
          Technical Documentation
        </div>
        
        <h1 className="text-4xl md:text-5xl font-bold mb-6">
          <span className="text-gradient-primary">ToxD4C</span>
          <span className="text-gradient-cyber"> Documentation</span>
        </h1>
        
        <p className="text-xl text-gray-600 dark:text-gray-300 max-w-3xl mx-auto">
          Comprehensive technical documentation for the ToxD4C multi-modal molecular toxicity prediction framework
        </p>
      </motion.div>

      {/* Table of Contents */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="glass-card p-6"
      >
        <h2 className="text-2xl font-semibold mb-4 flex items-center gap-2">
          <BookOpenIcon className="w-6 h-6 text-primary-500" />
          Table of Contents
        </h2>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          {sections.map((section, index) => (
            <a
              key={section.id}
              href={`#${section.id}`}
              className="flex items-center gap-3 p-3 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors group"
            >
              <section.icon className="w-5 h-5 text-primary-500 group-hover:scale-110 transition-transform" />
              <span className="font-medium">{section.title}</span>
              <ArrowRightIcon className="w-4 h-4 ml-auto opacity-0 group-hover:opacity-100 transition-opacity" />
            </a>
          ))}
        </div>
      </motion.div>

      {/* Documentation Sections */}
      {sections.map((section, index) => (
        <motion.div
          key={section.id}
          id={section.id}
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 * (index + 2) }}
          className="glass-card p-8"
        >
          <h2 className="text-3xl font-bold mb-6 flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-r from-primary-500 to-primary-600 p-2">
              <section.icon className="w-full h-full text-white" />
            </div>
            {section.title}
          </h2>
          
          <div className="prose prose-lg dark:prose-invert max-w-none">
            <div className="whitespace-pre-line text-gray-700 dark:text-gray-300 leading-relaxed">
              {section.content}
            </div>
          </div>
        </motion.div>
      ))}

      {/* Footer */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.8 }}
        className="glass-card p-8 text-center"
      >
        <h3 className="text-2xl font-bold mb-4">Get Started</h3>
        <p className="text-gray-600 dark:text-gray-300 mb-6">
          Ready to explore molecular toxicity prediction with ToxD4C?
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <a
            href="https://github.com/PhelanShao/tox-agents"
            target="_blank"
            rel="noopener noreferrer"
            className="btn-primary"
          >
            View Source Code
          </a>
          <a
            href="https://bohrium.dp.tech/apps/tox-agents"
            target="_blank"
            rel="noopener noreferrer"
            className="btn-glass"
          >
            Try Demo
          </a>
        </div>
      </motion.div>
      </div>
    </div>
  )
}