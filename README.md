# 📘 SeDA (Semantic-Driven Adaptive Framework) – Complete Implementation

This repository provides the **complete implementation** of the *"A Semantic Driven Adaptive Framework for Few-Shot Knowledge Graph Completion"* paper, including all previously missing modules and the full training pipeline.

---

## 📌 Complete Availability

### ✅ Previously Missing (Now Implemented):
- **Relation Encoding Modules**: Complete BERT-based relation representation learning with Graph Attention Networks
- **Semantic-driven Negative Sampling**: Advanced negative sampling strategies based on semantic similarity and entity types
- **Contrastive Learning Framework**: Full implementation of Sections 4.3 & 4.4 with adaptive loss functions
- **Complete Training Pipeline**: End-to-end training and evaluation system with comprehensive metrics
- **BERT + GAT Integration**: Seamless combination of pre-trained BERT with Graph Attention Networks
- **LoRA Support**: Efficient fine-tuning with Low-Rank Adaptation for large language models

### ✅ Previously Available (Enhanced):
- **Neighborhood-based Semantic Extracting**: Enhanced semantic neighbor selection based on relevance and diversity
- **LLM-based Entity Description Generation**: Support for multiple LLMs (`LLAMA2`, `ChatGLM2`, `GPT`, `DeepSeek`)
- **Multi-dataset Support**: Complete support for `Nell-One` and `FB15k237-One` datasets
- **Advanced Scoring Paradigms**: Both TransE-based and FKGC-specific scoring methods

---

## 🏗️ Complete SeDA Framework

This implementation includes all four key modules from the paper:

1. **Neighborhood-based Semantic Extracting Module** (Section 4.1)
   - Semantic neighbor selection based on relevance and diversity
   - LLM-based entity description generation

2. **Relation Encoding Module** (Section 4.2) ✅ **NEW**
   - BERT-based relation representation learning
   - Graph Attention Network integration for structural information

3. **Semantic-driven Negative Sampling** (Section 4.3) ✅ **NEW**
   - Type-aware negative sample generation
   - Semantic similarity-based candidate selection
   - Multiple negative sampling strategies

4. **Contrastive Learning Framework** (Section 4.4) ✅ **NEW**
   - Complete contrastive learning implementation
   - Adaptive loss functions and training strategies
   - Margin ranking loss with regularization

---

## 🎯 Key Features

- **Complete SeDA Framework**: Full implementation of the semantic-driven adaptive approach
- **BERT + GAT Architecture**: Leverages pre-trained BERT with Graph Attention Networks
- **Advanced Negative Sampling**: Semantic-aware negative sample generation with multiple strategies
- **Contrastive Learning**: Implements the complete contrastive learning framework
- **LoRA Integration**: Efficient fine-tuning for large language models
- **Flexible Configuration**: Highly configurable for different datasets and settings
- **Comprehensive Evaluation**: Full evaluation pipeline with standard KGC metrics

---

```

---

## 📚 Citation

**Note**: Citation information will be updated after the paper is officially published.
```

---

## 📫 Contact

For questions, suggestions, or collaboration inquiries, please open a GitHub Issue or contact the authors.

---

**Note**: This is the complete implementation of the SeDA framework, providing all modules that were previously missing from the partial release. The code is ready for research and experimentation purposes.
