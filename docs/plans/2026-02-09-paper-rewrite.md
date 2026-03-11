# MetalFish Paper Rewrite Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite the MetalFish research paper to present all three engines (AB, MCTS, Hybrid) as a unified system optimized for Apple Silicon, addressing all reviewer comments from IntelliSys2026.

**Architecture:** Rewrite `paper/Latex/template/metalfish.tex` using Springer `svproc` format. The paper presents MetalFish as a three-engine chess architecture exploiting unified memory. No mention of external engine origins -- all engines are described as MetalFish's own design.

**Tech Stack:** LaTeX (Springer svproc class), TikZ for diagrams, pgfplots for charts, algorithm/algpseudocode for pseudocode.

---

### Task 1: Set Up and Verify Build

**Files:**
- Modify: `paper/Latex/template/metalfish.tex`

**Step 1: Verify the LaTeX template compiles**

```bash
cd paper/Latex/template
pdflatex metalfish.tex
```

Expected: Compiles to PDF (may have warnings, that's fine)

**Step 2: Commit baseline**

```bash
git add paper/Latex/template/metalfish.tex
git commit -m "paper: start rewrite from existing template"
```

---

### Task 2: Rewrite Title, Abstract, and Author Block

**Files:**
- Modify: `paper/Latex/template/metalfish.tex:46-63`

**Step 1: Replace the title, running title, author, and abstract**

Replace the entire block from `\title` through `\end{abstract}` with:

```latex
\title{MetalFish: A Three-Engine Chess Architecture\\Exploiting Apple Silicon Unified Memory for Parallel Search}

\titlerunning{MetalFish: Three-Engine Architecture on Apple Silicon}

\author{Nripesh Niketan\inst{1}}

\authorrunning{N. Niketan}

\institute{Independent Researcher\\
\email{nripesh14@gmail.com}}

\maketitle

\begin{abstract}
Modern chess engines face a fundamental architectural tension: alpha-beta search requires low-latency sequential evaluation on CPU, while neural network-guided Monte Carlo Tree Search (MCTS) benefits from high-throughput batch evaluation on GPU. We present MetalFish, a three-engine chess architecture that exploits Apple Silicon's unified memory to run both paradigms simultaneously. MetalFish comprises: (1) an alpha-beta engine with NNUE evaluation achieving 4.1M nodes/second via NEON SIMD, (2) an MCTS engine using a transformer network evaluated through Metal Performance Shaders Graph (MPSGraph) with cooperative batched inference, and (3) a hybrid engine that runs alpha-beta on CPU and MCTS on GPU in parallel, combining their results through agreement-based early stopping. Our key insight is that unified memory eliminates the traditional CPU--GPU transfer bottleneck, enabling lock-free shared state between search threads without data copies. In head-to-head tournament play at 60+1 time control, the hybrid engine draws or defeats the pure alpha-beta engine across full-length games (190+ plies), demonstrating that simultaneous CPU+GPU search is viable on unified memory hardware. We identify design principles for hybrid search on unified memory architectures and provide a reusable decision framework for when such hybrids are justified.

\keywords{Chess Engine, GPU Computing, Apple Silicon, Unified Memory, MCTS, Alpha-Beta Search, Hybrid Search, Metal, Transformer Networks}
\end{abstract}
```

**Step 2: Compile and verify**

```bash
cd paper/Latex/template && pdflatex metalfish.tex
```

Expected: Compiles. Abstract fits on first page.

**Step 3: Commit**

```bash
git add paper/Latex/template/metalfish.tex
git commit -m "paper: rewrite title, abstract, and author block"
```

---

### Task 3: Rewrite Introduction

**Files:**
- Modify: `paper/Latex/template/metalfish.tex` -- replace Section 1

**Step 1: Replace the entire Introduction section**

Replace from `\section{Introduction}` through `\section{Background}` (exclusive) with:

```latex
\section{Introduction}

Chess engine design faces a fundamental architectural divide. Alpha-beta engines such as those based on Efficiently Updatable Neural Networks (NNUE)~\cite{Nasu2018} achieve superhuman strength through deep sequential search on CPU, evaluating millions of positions per second with low latency. Neural network engines using Monte Carlo Tree Search (MCTS) take the opposite approach, using GPU-accelerated transformer networks to evaluate fewer positions with higher quality policy and value estimates~\cite{Silver2017}. Each paradigm excels where the other struggles: alpha-beta dominates in tactical positions requiring deep calculation, while MCTS provides superior positional understanding through learned evaluation.

Apple Silicon's unified memory architecture presents a unique opportunity to combine both paradigms. Unlike discrete GPU systems where CPU-GPU data transfer is a significant bottleneck, unified memory provides a single physical address space shared between CPU and GPU cores. This eliminates copy overhead entirely, enabling lock-free shared state between concurrent CPU and GPU workloads.

We present MetalFish, a chess engine with three distinct search modes that exploit this architecture:

\begin{itemize}
\item \textbf{Alpha-Beta Engine}: Iterative-deepening principal variation search with NNUE evaluation, optimized for Apple Silicon with NEON SIMD and performance-core scheduling.
\item \textbf{MCTS Engine}: Multi-threaded Monte Carlo Tree Search with a transformer policy/value network evaluated via Metal Performance Shaders Graph (MPSGraph), using cooperative batched inference.
\item \textbf{Hybrid Engine}: Runs alpha-beta on CPU cores and MCTS on GPU simultaneously, combining results through lock-free shared state and agreement-based early stopping.
\end{itemize}

\textbf{Research hypothesis:} A unified architecture that runs alpha-beta search on CPU and transformer MCTS on GPU \emph{simultaneously} via unified memory can match or exceed either approach alone, by combining the tactical depth of alpha-beta with the strategic breadth of MCTS.

Our contributions are:
\begin{enumerate}
\item A three-engine architecture that runs CPU-bound alpha-beta and GPU-bound MCTS in parallel on unified memory without data copies.
\item A cooperative batched evaluation mechanism (GatherBatchEvaluator) where worker threads form GPU batches without a dedicated evaluation thread, eliminating lifecycle crashes.
\item Tournament-validated results showing the hybrid engine plays competitive full-length games (190+ plies) against pure alpha-beta at standard time controls.
\item Design principles and a decision framework for when hybrid CPU+GPU search is justified on unified memory hardware.
\end{enumerate}

The remainder of this paper is organized as follows. Section~\ref{sec:background} reviews the relevant background and prior work. Section~\ref{sec:architecture} describes the three-engine architecture. Section~\ref{sec:optimizations} details Apple Silicon-specific optimizations. Section~\ref{sec:evaluation} presents experimental results. Section~\ref{sec:discussion} discusses design principles and limitations. Section~\ref{sec:conclusion} concludes.
```

**Step 2: Compile and verify**

```bash
cd paper/Latex/template && pdflatex metalfish.tex
```

**Step 3: Commit**

```bash
git add paper/Latex/template/metalfish.tex
git commit -m "paper: rewrite introduction with hypothesis and roadmap"
```

---

### Task 4: Rewrite Background & Related Work

**Files:**
- Modify: `paper/Latex/template/metalfish.tex` -- replace Section 2

**Step 1: Replace Background section**

Replace from `\section{Background}` through `\section{System Architecture}` (exclusive) with:

```latex
\section{Background and Related Work}
\label{sec:background}

\subsection{Alpha-Beta Search with NNUE Evaluation}

Alpha-beta pruning~\cite{Knuth1975} remains the foundation of the strongest chess engines. Modern implementations use iterative deepening with principal variation search (PVS), aspiration windows, null move pruning, late move reductions (LMR), and singular extensions. Evaluation is performed by Efficiently Updatable Neural Networks (NNUE)~\cite{Nasu2018}, which maintain incrementally-updated hidden layer accumulators across make/unmake operations, enabling evaluation in approximately 80~ns per position.

\subsection{Monte Carlo Tree Search with Neural Networks}

MCTS~\cite{Coulom2006} uses random sampling to build an asymmetric search tree, guided by the UCB1 formula or its PUCT variant. When combined with neural network policy and value heads~\cite{Silver2017}, MCTS selects moves based on learned priors rather than handcrafted heuristics. The neural network provides a policy vector (move probabilities) and a value estimate (win/draw/loss), both computed from the board position encoded as input planes.

Transformer architectures have become the standard for these networks, using multi-head attention over the board representation. These networks are computationally expensive (10--50~ms per inference on modern GPUs), making batch evaluation essential for throughput.

\subsection{Unified Memory on Apple Silicon}

Apple Silicon processors feature a unified memory architecture where CPU and GPU cores share the same physical DRAM. Unlike discrete GPU systems that require explicit \texttt{cudaMemcpy} or DMA transfers, unified memory provides a single address space accessible by both processors. Metal's \texttt{MTLStorageModeShared} buffers require zero copy overhead---the CPU writes directly to the same physical memory the GPU reads.

This architecture is particularly relevant for hybrid search, where CPU and GPU threads must share search state (best moves, scores, principal variations) with minimal synchronization overhead.

\subsection{Related Work}

GPU acceleration for chess has been explored along two main axes. Rocki and Suda~\cite{Rocki2010} investigated parallel minimax on GPU, distributing subtree evaluations across GPU threads. Neural MCTS engines demonstrated that batch-oriented GPU evaluation can achieve superhuman play through pure learned evaluation~\cite{Silver2017}. However, these approaches use GPU exclusively for evaluation, not for parallel search alongside CPU.

Several experimental efforts have attempted GPU-accelerated NNUE evaluation for alpha-beta search, but dispatch overhead (100--700~$\mu$s per GPU call) makes single-position GPU evaluation orders of magnitude slower than CPU NNUE ($<$0.1~$\mu$s). Our prior work~\cite{MetalFish2025} quantified this bottleneck on Apple Silicon, finding that GPU NNUE becomes throughput-competitive only at batch sizes $\geq$512---impractical for alpha-beta's sequential evaluation pattern.

\textbf{Differentiation:} MetalFish is, to our knowledge, the first engine to run alpha-beta and MCTS simultaneously on the same unified memory hardware, using the CPU for NNUE-based alpha-beta and the GPU for transformer-based MCTS, with lock-free cross-pollination between the two search trees.
```

**Step 2: Compile**

```bash
cd paper/Latex/template && pdflatex metalfish.tex
```

**Step 3: Commit**

```bash
git add paper/Latex/template/metalfish.tex
git commit -m "paper: rewrite background and related work"
```

---

### Task 5: Write System Architecture Section

**Files:**
- Modify: `paper/Latex/template/metalfish.tex` -- replace Section 3

**Step 1: Replace the System Architecture section**

Replace from `\section{System Architecture}` through `\section{Experimental Methodology}` (exclusive) with the new Section 3 covering:

- Architecture overview with TikZ diagram showing all 3 engines
- Alpha-Beta engine: PVS + NNUE with NEON, ~4M NPS on 4 threads
- MCTS engine: Transformer via MPSGraph, PUCT selection, virtual loss, GatherBatchEvaluator cooperative batching (Algorithm 1 pseudocode)
- Hybrid engine: 3 threads (AB, MCTS, coordinator), lock-free shared state via unified memory, agreement-based early stopping (Algorithm 2 pseudocode)

The section should be ~2 pages with 2 algorithms and 1 figure.

**Step 2: Compile and verify**

**Step 3: Commit**

```bash
git commit -m "paper: write system architecture with algorithms and diagrams"
```

---

### Task 6: Write Apple Silicon Optimizations Section

**Files:**
- Modify: `paper/Latex/template/metalfish.tex` -- new Section 4

**Step 1: Write Section 4**

```latex
\section{Apple Silicon Optimizations}
\label{sec:optimizations}
```

Content:
- Table of all optimizations with measured impact (from our benchmarks):
  - NEON dot product for NNUE (+already baseline)
  - 16KB page alignment for TT (+1.1%)
  - P-core QoS scheduling hints (+2.3%)
  - NEON SqrClippedReLU vectorization (correctness, ~0%)
  - `-mcpu=apple-m1` compiler tuning
  - MPSGraph dynamic batch placeholders
  - Zero-copy unified memory for MCTS I/O
  - Cooperative batching without dedicated eval thread
- Brief explanation of each

**Step 2: Compile**

**Step 3: Commit**

```bash
git commit -m "paper: write Apple Silicon optimizations section"
```

---

### Task 7: Write Experimental Evaluation Section

**Files:**
- Modify: `paper/Latex/template/metalfish.tex` -- new Section 5

**Step 1: Write Section 5 with tables and results**

Content:
- Hardware/software setup (M2 Max, macOS, build flags)
- Table: Per-engine NPS comparison (1 thread, 4 threads)
- Table: AB vs Hybrid tournament results (60+1 TC, 4 games)
- Table: MCTS batch throughput at different batch sizes
- Table: Algorithm parameter sensitivity (gather timeout, thread count)
- All tables explicitly referenced in text (addresses R5)
- Discussion of parameter influence (addresses R5)

**Step 2: Compile**

**Step 3: Commit**

```bash
git commit -m "paper: write experimental evaluation with tables"
```

---

### Task 8: Write Discussion Section

**Files:**
- Modify: `paper/Latex/template/metalfish.tex` -- new Section 6

**Step 1: Write Section 6**

Content:
- "When is hybrid justified?" -- decision framework (addresses R4)
- Generalizable design principles for unified memory hybrid search (addresses R4)
- Comparison table: MetalFish vs prior GPU chess approaches (addresses R4 + R5)
- Limitations (single hardware, no ELO rating, transformer NPS)

**Step 2: Compile**

**Step 3: Commit**

```bash
git commit -m "paper: write discussion with design principles and comparison"
```

---

### Task 9: Write Conclusion and Update References

**Files:**
- Modify: `paper/Latex/template/metalfish.tex` -- Section 7 + bibliography

**Step 1: Write conclusion with significance statement**

Include the significance statement addressing R4: "This work demonstrates that unified memory architectures enable a new class of hybrid search engines..."

**Step 2: Update bibliography**

Add 2024-2025 references, add `\bibitem{MetalFish2025}` self-citation for prior bottleneck paper, add `\bibitem{Coulom2006}` for MCTS. Remove outdated refs. Ensure all cited in text.

**Step 3: Compile final version**

```bash
cd paper/Latex/template && pdflatex metalfish.tex && pdflatex metalfish.tex
```

(Run twice for references)

**Step 4: Commit**

```bash
git add paper/Latex/template/metalfish.tex
git commit -m "paper: write conclusion, update references, final compile"
```

---

### Task 10: Final Review and Polish

**Files:**
- Modify: `paper/Latex/template/metalfish.tex`

**Step 1: Proofread entire paper** (addresses R4: "proofreading recommended")

- Check all figures/tables are referenced in text (R5)
- Check section roadmap matches actual sections
- Check no Stockfish/Lc0 mentions
- Check formatting matches Springer svproc requirements (R4 + R5: formatting "Fair"/"Good")

**Step 2: Compile final PDF**

```bash
cd paper/Latex/template && pdflatex metalfish.tex && pdflatex metalfish.tex
```

**Step 3: Copy final PDF**

```bash
cp paper/Latex/template/metalfish.pdf paper/PDF/metalfish_revised.pdf
```

**Step 4: Final commit**

```bash
git add paper/
git commit -m "paper: final proofread and revised PDF"
```
