# AttenSum

> **A from-scratch Transformer text summarizer and interactive learning platform.**

AttenSum is a full-stack GenAI/NLP research and learning project built around a small **Transformer encoder-decoder implemented from scratch** using PyTorch.

The project has two connected goals:

1. **Summarize** long-form text into concise abstractive summaries.
2. **Understand** how a Transformer actually processes text, generates summaries, and can be evaluated and experimented with.

The planned application combines a React frontend, FastAPI backend, PyTorch Transformer, CNN/DailyMail dataset, ROUGE evaluation, experiment comparison, and an interactive Transformer execution visualization.

---

## Project Status

🚧 **Day 1 — Planning / Documentation Phase**

The product requirements and complete technical documentation have been prepared, but implementation has not started yet.

| Component | Status |
|---|---|
| PRD | ✅ Approved |
| Complete Documentation | ✅ Prepared |
| Transformer | ⏳ Planned |
| Dataset Pipeline | ⏳ Planned |
| Training | ⏳ Planned |
| Inference | ⏳ Planned |
| FastAPI Backend | ⏳ Planned |
| React Frontend | ⏳ Planned |
| ROUGE Evaluation | ⏳ Planned |
| Experiment Tracking | ⏳ Planned |
| Transformer Visualization | ⏳ Planned |
| Attention Visualization | ⏳ Planned |
| Authentication | ⏳ Planned |
| Testing | ⏳ Planned |
| Deployment | ⏳ Planned |

> This README describes the approved project direction. It does not claim that planned components are already implemented.

---

## Why AttenSum?

Most summarization tools focus on one question:

> **"What is the summary?"**

AttenSum asks additional questions:

- How did the Transformer process the input?
- How does self-attention work in the actual implementation?
- What happens inside the encoder?
- Why is decoder self-attention masked?
- How does the decoder use encoder information?
- How is the summary generated token by token?
- How good is the generated summary?
- Did a model or training change improve the result?

The goal is to connect:

```text
THEORY
   ↓
TRANSFORMER IMPLEMENTATION
   ↓
ACTUAL MODEL EXECUTION
   ↓
VISUALIZATION
   ↓
GENERATED SUMMARY
   ↓
ROUGE EVALUATION
   ↓
EXPERIMENTATION
   ↓
MODEL IMPROVEMENT
```

---

## Core Features

### Summarization

- Text input
- Input validation
- Abstractive summary generation
- Summary length control
  - Short
  - Medium
  - Detailed
- Original vs generated text comparison
- Copy/export

### Evaluation

- ROUGE-1
- ROUGE-2
- ROUGE-L
- Experiment comparison
- Evaluation dashboard

### Transformer Learning

- Tokenization
- Token embeddings
- Positional encoding
- Encoder
- Self-attention
- Multi-head attention
- Feed-forward networks
- Decoder
- Masked self-attention
- Encoder-decoder attention
- Output projection
- Autoregressive generation

### Interactive Model Exploration

The planned visualization follows the actual inference pipeline:

```text
Input Text
   ↓
Tokenization
   ↓
Token Embeddings
   ↓
Positional Encoding
   ↓
Encoder
   ↓
Self-Attention
   ↓
Decoder
   ↓
Masked Self-Attention
   ↓
Encoder-Decoder Attention
   ↓
Output Projection
   ↓
Autoregressive Generation
   ↓
Final Summary
```

A core design principle is that the visualization should represent **actual model execution**, rather than a fake animation layered over the UI. Conceptual simplifications will be identified as such.

---

## Architecture

The planned architecture is:

```text
┌──────────────────────────────┐
│        React Frontend        │
│                              │
│ Text Input                   │
│ Summary Display              │
│ Evaluation Dashboard        │
│ Experiment Comparison       │
│ Transformer Explorer        │
│ Attention Visualization     │
│ Execution Visualization    │
└──────────────┬───────────────┘
               │
            REST API
               │
┌──────────────▼───────────────┐
│        FastAPI Backend       │
│                              │
│ Validation                   │
│ Inference                    │
│ Evaluation                  │
│ Experiment Data             │
│ Visualization Data          │
│ Authentication / Sessions   │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│   From-Scratch Transformer  │
│                              │
│ Embeddings                  │
│ Positional Encoding         │
│ Encoder                     │
│ Self-Attention              │
│ Multi-Head Attention        │
│ Feed-Forward                │
│ Decoder                     │
│ Masked Self-Attention       │
│ Cross-Attention             │
│ Output Projection           │
│ Autoregressive Generation   │
└──────────────┬───────────────┘
               │
┌──────────────▼───────────────┐
│       CNN/DailyMail          │
│                              │
│ Train                        │
│ Validation                   │
│ Test                         │
└──────────────────────────────┘
```

---

## Technology Stack

| Area | Technology |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Backend | FastAPI |
| Frontend | React |
| Styling | Tailwind CSS |
| HTTP Client | Axios |
| NLP | NLTK |
| Evaluation | rouge-score |
| Dataset | CNN/DailyMail |
| Database | TBD |
| Authentication | TBD |
| Testing | TBD |
| Deployment | TBD |

---

## Dataset

The project uses the **CNN/DailyMail summarization dataset**.

The required split structure is:

```text
CNN/DailyMail
├── Train
├── Validation
└── Test
```

- **Train:** model training
- **Validation:** monitoring and configuration/model selection
- **Test:** final evaluation

The test set is reserved for final evaluation and should not be used for iterative model tuning.

---

## Development Roadmap

### Phase 1 — Foundation

- Study Transformer fundamentals
- Set up repository
- Set up FastAPI
- Set up React
- Acquire/explore CNN/DailyMail
- Build preprocessing pipeline
- Establish baseline pipeline

### Phase 2 — Transformer

- Implement embeddings
- Implement positional encoding
- Implement self-attention
- Implement multi-head attention
- Implement encoder
- Implement decoder
- Implement masking
- Implement output projection
- Implement autoregressive generation
- Train model
- Implement inference

### Phase 3 — Evaluation

- Implement ROUGE-1/2/L
- Evaluate model
- Track experiments
- Analyze failure cases

### Phase 4 — Product

- Build summarization UI
- Connect React to FastAPI
- Add validation and error handling
- Add source/summary comparison
- Add length control
- Add copy/export
- Add evaluation dashboard
- Add architecture information

### Phase 5 — Interactive Learning

- Build Transformer architecture explorer
- Build attention visualization
- Build internal execution visualization
- Connect visualization to actual inference
- Visualize tokenization, embeddings and positional encoding
- Visualize encoder/decoder stages
- Visualize attention and generation

### Phase 6 — Production Readiness

- Automated tests
- UI/UX refinement
- Authentication/session handling
- Documentation
- Reproduction instructions
- Deployment
- Demo and presentation

---

## Repository Structure

```text
attensum/
│
├── backend/
│   ├── app/
│   └── tests/
│
├── frontend/
│   ├── src/
│   └── tests/
│
├── model/
│   ├── transformer/
│   ├── training/
│   ├── inference/
│   └── configs/
│
├── data/
│
├── experiments/
│   ├── configs/
│   └── results/
│
├── scripts/
│
├── docs/
│   └── complete_documentation.md
│
├── tests/
│
├── README.md
└── .gitignore
```

The structure is intentionally a starting point and may evolve as implementation begins.

---

## Documentation

The complete project documentation contains the detailed:

- Business requirements
- Product requirements
- User journeys
- UX requirements
- Functional requirements
- Non-functional requirements
- Technical architecture
- High-level design
- Low-level design
- Transformer architecture
- Dataset and training design
- API design
- Visualization architecture
- Testing strategy
- Security
- Deployment
- Roadmap
- Architecture decisions
- Requirements traceability
- Viva/project-defense preparation

See:

`docs/complete_documentation.md`

---

## Project Principles

### 1. Build the Transformer from scratch

The primary model should not depend on a pretrained summarization model.

### 2. Keep the visualization truthful

If a visualization represents model execution, it should be connected to the actual model execution.

### 3. Evaluate, don't just demonstrate

Generated summaries should be evaluated using ROUGE-1, ROUGE-2 and ROUGE-L.

### 4. Preserve the test set

The test split should remain reserved for final evaluation.

### 5. Progressive disclosure

Beginners should be able to understand the high-level pipeline before being exposed to advanced model internals.

### 6. Document reality

Documentation should distinguish between:

- Implemented
- In progress
- Planned
- TBD
- Future

---

## Non-Goals

The initial project is not intended to:

- Compete with large commercial LLMs.
- Build a massive production-scale Transformer.
- Guarantee factual correctness.
- Support every language.
- Summarize arbitrary multimodal content.
- Replace professional editorial review.
- Become a complete tensor debugger.

---

## Future Scope

Potential future versions include:

- Larger from-scratch Transformer
- Pretrained baseline comparison
- Better long-document handling
- More configurable decoding
- Advanced experiment tracking
- Document upload
- PDF/text extraction
- Saved documents
- Team workspaces
- API access
- Query-focused summarization
- Multi-document summarization
- Domain-specific summarization
- Multilingual summarization
- Factuality checking

---

## Current Open Decisions

Some implementation details are intentionally not fixed yet:

- Transformer dimensions
- Number of layers
- Number of attention heads
- Vocabulary/tokenization strategy
- Sequence lengths
- Optimizer
- Learning rate
- Batch size
- Training schedule
- Decoding strategy
- Database
- Authentication mechanism
- Visualization library
- Testing framework
- Deployment platform
- CI/CD platform
- Exact API schemas

These will be decided during implementation and recorded in the documentation.

---

## License

TBD

---

## Project Status

**AttenSum is currently at Day 1: architecture and implementation planning.**

The next engineering milestone is to establish the repository foundation and begin the data/model implementation.
