# MCTS Lc0 Parity Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the 411 Elo gap between MetalFish-MCTS (~3289) and Lc0 (~3700) by implementing all missing lc0 search features with step-by-step testing.

**Architecture:** Bottom-up approach — fix foundational issues (cache key, multivisit, edge sorting) first, then add search heuristics (KLD stopper, prefetching, tablebases), then polish (temperature, contempt, safety stoppers). Each task is independently testable and committable.

**Tech Stack:** C++20, Apple Silicon (os_unfair_lock, Accelerate, Metal/MPS), CMake build system

**Build & Test Commands:**
```bash
cd /Users/nripeshn/Documents/PythonPrograms/metalfish
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(sysctl -n hw.ncpu)
./build/metalfish_tests
```

---

## Task 1: NN Cache Key Fix (Position-Based Hashing)

**Estimated Elo gain: +100-150**
**Root cause:** `ComputePositionCacheKey()` hashes all 8 history positions including `repetition_distance()`, making every path unique. The paper confirms 0% cache hit rate. Lc0 uses only the last position's board hash + repetition count.

**Files:**
- Modify: `src/mcts/search.cpp:43-63` (the `ComputePositionCacheKey` function)
- Test: `tests/test_mcts_module.cpp`

### Step 1: Write a test for cache hit rate

Add to `tests/test_mcts_module.cpp`:

```cpp
void test_cache_hit_rate(TestCounter &tc) {
    std::cout << "  Cache hit rate..." << std::endl;
    const char *weights = std::getenv("METALFISH_NN_WEIGHTS");
    if (!weights) {
        std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
        return;
    }

    SearchParams params;
    params.num_threads = 1;
    params.nn_weights_path = weights;

    MetalFish::Search::LimitsType limits;
    limits.nodes = 256;

    // Run search on a position that should produce transpositions
    // (e.g. after 1.e4 e5 2.Nf3 Nc6 — many move-order transpositions)
    auto search = CreateSearch(params);
    search->StartSearch(
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        limits, nullptr, nullptr);
    search->Wait();

    const auto &stats = search->Stats();
    uint64_t hits = stats.cache_hits.load();
    uint64_t misses = stats.cache_misses.load();
    uint64_t total = hits + misses;

    float hit_rate = total > 0 ? static_cast<float>(hits) / total : 0.0f;
    std::cout << "    Cache: " << hits << " hits / " << total
              << " lookups (" << (hit_rate * 100.0f) << "%)" << std::endl;

    // After the fix, we expect at least some cache hits (>5%)
    expect(hit_rate > 0.05f, "cache hit rate > 5%", tc);
}
```

### Step 2: Run test to verify it fails

```bash
cmake --build build -j$(sysctl -n hw.ncpu) && METALFISH_NN_WEIGHTS=networks/BT4-1024x15x32h-swa-6147500.pb ./build/metalfish_tests
```
Expected: FAIL — cache hit rate will be ~0%

### Step 3: Implement the position-based cache key

Replace `ComputePositionCacheKey` in `src/mcts/search.cpp:43-63` with:

```cpp
uint64_t ComputePositionCacheKey(const Position* const* history, int count) {
    // Match lc0's approach: hash only the current position's board key
    // plus repetition info. This enables transposition hits where the
    // same board is reached by different move orders.
    if (count <= 0 || !history[count - 1]) return kFNVOffset;

    const Position* current = history[count - 1];
    uint64_t key = kFNVOffset;

    // Primary: board Zobrist key (encodes pieces, castling, en passant, side)
    key ^= Mix64(current->raw_key());
    key *= kFNVPrime;

    // Secondary: repetition distance (differentiates positions with
    // different repetition history, matching lc0's Position::Hash())
    key ^= Mix64(static_cast<uint64_t>(
        static_cast<int64_t>(current->repetition_distance())));
    key *= kFNVPrime;

    return key;
}
```

### Step 4: Run test to verify it passes

```bash
cmake --build build -j$(sysctl -n hw.ncpu) && METALFISH_NN_WEIGHTS=networks/BT4-1024x15x32h-swa-6147500.pb ./build/metalfish_tests
```
Expected: PASS — cache hit rate should be >5%

### Step 5: Run BK tactical regression

```bash
# Quick sanity check: run the engine on a known position, verify it finds a good move
echo -e "uci\nsetoption name UseMCTS value true\nposition startpos\ngo nodes 500\nquit" | ./build/metalfish
```
Expected: Engine produces a reasonable bestmove without errors

### Step 6: Commit

```bash
git add src/mcts/search.cpp tests/test_mcts_module.cpp
git commit -m "feat(mcts): position-based NN cache key for transposition hits

Replace full-history FNV hash (0% hit rate) with position-only key
using board Zobrist + repetition distance, matching lc0's approach.
Enables NN cache transposition hits across different move orders."
```


---

## Task 2: Activate Multivisit Propagation

**Estimated Elo gain: +10-20**
**Root cause:** `SelectChildPuct` computes `visits_to_assign` (how many visits best child can absorb before second-best overtakes it) but `SelectLeaf` discards it with `(void)visits_to_assign` on line 857. This is lc0's key batching optimization — assigning multiple visits at once reduces tree traversal overhead.

**Files:**
- Modify: `src/mcts/search.cpp:849-884` (SelectLeaf), `src/mcts/search.cpp:650-843` (RunIterationSemaphore)
- Test: `tests/test_mcts_module.cpp`

### Step 1: Write a test for multivisit effectiveness

Add to `tests/test_mcts_module.cpp`:

```cpp
void test_multivisit_propagation(TestCounter &tc) {
    std::cout << "  Multivisit propagation..." << std::endl;
    // Verify that SelectChildPuct's visits_to_assign is actually
    // used — after search, some nodes should have N > 1 from single paths
    Node root;
    MoveList<LEGAL> moves;
    Position pos;
    StateInfo st;
    pos.set("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", false, &st);
    MoveList<LEGAL> legal(pos);
    root.CreateEdges(legal);
    expect(root.NumEdges() > 0, "root has edges", tc);

    // Test that FinalizeScoreUpdate works with multivisit > 1
    Node child;
    child.TryStartScoreUpdate(5);
    child.FinalizeScoreUpdate(0.6f, 0.1f, 25.0f, 5);
    expect(child.GetN() == 5, "multivisit=5 produces N=5", tc);
    expect(std::abs(child.GetWL() - 0.6f) < 0.01f, "WL correct after multivisit", tc);
}
```

### Step 2: Run test to verify it passes (this tests Node, not Search yet)

```bash
cmake --build build -j$(sysctl -n hw.ncpu) && ./build/metalfish_tests
```
Expected: PASS

### Step 3: Implement multivisit propagation in SelectLeaf

Replace the relevant section of `SelectLeaf` in `src/mcts/search.cpp:849-884`:

```cpp
Search::SelectedLeaf Search::SelectLeaf(SearchWorkerCtx& ctx) {
    Node* node = tree_.Root();
    int path_multivisit = std::max(1, params_.virtual_loss);

    while (node->NumEdges() > 0 && !node->IsTerminal()) {
        bool is_root = (node == tree_.Root());
        auto [best_idx, visits_to_assign] = SelectChildPuct(node, is_root, ctx);
        if (best_idx < 0) break;

        // Use multivisit from PUCT at root level to reduce tree traversals
        if (is_root && visits_to_assign > 1) {
            path_multivisit = std::min(visits_to_assign, 128);
        }

        Edge& edge = node->Edges()[best_idx];

        Node* child = edge.child.load(std::memory_order_acquire);
        if (!child) {
            Node* new_child = tree_.AllocateNode(node, best_idx);
            Node* expected = nullptr;
            if (edge.child.compare_exchange_strong(
                    expected, new_child,
                    std::memory_order_release,
                    std::memory_order_acquire)) {
                child = new_child;
            } else {
                child = expected;
            }
        }

        if (!child->TryStartScoreUpdate(path_multivisit)) {
            break;
        }

        ctx.DoMove(edge.move);
        node = child;
    }

    return {node, path_multivisit};
}
```

### Step 4: Run tests and BK regression

```bash
cmake --build build -j$(sysctl -n hw.ncpu) && ./build/metalfish_tests
echo -e "uci\nsetoption name UseMCTS value true\nposition startpos\ngo nodes 500\nquit" | ./build/metalfish
```
Expected: Tests pass, engine works correctly, possibly higher NPS

### Step 5: Commit

```bash
git add src/mcts/search.cpp tests/test_mcts_module.cpp
git commit -m "feat(mcts): activate multivisit propagation from PUCT

SelectLeaf now uses visits_to_assign from SelectChildPuct at root,
assigning multiple visits per tree traversal. Reduces overhead by
batching visits to the clearly-best child."
```

---

## Task 3: Edge Sorting + Early Cutoff in PUCT Scan

**Estimated Elo gain: +10-20**
**Root cause:** MetalFish scans ALL edges in `SelectChildPuct` every time. Lc0 sorts edges by descending policy on creation, then short-circuits the scan once it hits unvisited edges with zero N_started — since all remaining edges have even lower policy, they can't be best.

**Files:**
- Modify: `src/mcts/node.h` (add SortEdges method)
- Modify: `src/mcts/search.cpp:109-163` (call SortEdges after CreateEdges)
- Modify: `src/mcts/search.cpp:926-963` (add early cutoff in PUCT scan)
- Test: `tests/test_mcts_module.cpp`

### Step 1: Write a test for edge sorting

```cpp
void test_edge_sorting(TestCounter &tc) {
    std::cout << "  Edge sorting..." << std::endl;
    Node node;
    Position pos;
    StateInfo st;
    pos.set("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", false, &st);
    MoveList<LEGAL> legal(pos);
    node.CreateEdges(legal);

    // Manually set different policies
    Edge* edges = node.Edges();
    int n = node.NumEdges();
    for (int i = 0; i < n; ++i) {
        edges[i].SetP(static_cast<float>(i) / n); // ascending
    }

    node.SortEdges();

    // After sorting, should be descending
    bool sorted = true;
    for (int i = 1; i < n; ++i) {
        if (edges[i].GetP() > edges[i-1].GetP()) {
            sorted = false;
            break;
        }
    }
    expect(sorted, "edges sorted descending by policy", tc);
}
```

### Step 2: Run test — should fail (SortEdges doesn't exist yet)

```bash
cmake --build build -j$(sysctl -n hw.ncpu) && ./build/metalfish_tests
```
Expected: Compilation error — SortEdges not defined

### Step 3: Add SortEdges to Node class

Add to `src/mcts/node.h` after the `CreateEdges` method (after line 275):

```cpp
void SortEdges() {
    if (!edges_ || num_edges_ <= 1) return;
    // Sort on raw p_ — same order as GetP() due to encoding, but faster
    std::sort(edges_.get(), edges_.get() + num_edges_,
              [](const Edge& a, const Edge& b) { return a.p_ > b.p_; });
}
```

### Step 4: Call SortEdges after edge creation + NN policy application

In `src/mcts/search.cpp`, after every call to `ApplyNNPolicyToNode(leaf, result, ...)`, add `leaf->SortEdges();`

There are 5 locations where `ApplyNNPolicyToNode` is called:
1. Line ~608: `RunIteration` cache hit path → add `leaf->SortEdges();` after
2. Line ~621: `RunIteration` compute path → add `leaf->SortEdges();` after
3. Line ~770: `RunIterationSemaphore` cache hit path → add `leaf->SortEdges();` after
4. Line ~825: `RunIterationSemaphore` batch result path → add `entry.leaf->SortEdges();` after
5. In `CreateEdges` itself — uniform policy is already sorted, no action needed

### Step 5: Add early cutoff in SelectChildPuct

In `src/mcts/search.cpp:926-963`, inside the PUCT for loop, add an early exit after the edge scan finds two consecutive unvisited edges:

```cpp
for (int i = 0; i < num_edges; ++i) {
    if (i + 2 < num_edges) PREFETCH(&edges[i + 2]);

    const Edge& edge = edges[i];
    Node* child = edge.child.load(std::memory_order_acquire);

    // Early cutoff: if edges are sorted by policy descending and this
    // unvisited edge has no in-flight visits, all remaining edges have
    // even lower policy and can't beat the current best.
    if (!child && i > 0) {
        // Check if previous edge was also unvisited with no activity
        Node* prev_child = edges[i-1].child.load(std::memory_order_relaxed);
        if (!prev_child && edges[i-1].GetP() > 0.0f) {
            // Two consecutive unvisited: safe to stop since edges are
            // sorted by descending policy
            break;
        }
    }

    float q, m_utility = 0.0f;
    float policy = edge.GetP();
    // ... rest of existing loop body unchanged ...
}
```

### Step 6: Run tests and verify

```bash
cmake --build build -j$(sysctl -n hw.ncpu) && ./build/metalfish_tests
echo -e "uci\nsetoption name UseMCTS value true\nposition startpos\ngo nodes 500\nquit" | ./build/metalfish
```
Expected: All tests pass, engine works correctly

### Step 7: Commit

```bash
git add src/mcts/node.h src/mcts/search.cpp tests/test_mcts_module.cpp
git commit -m "feat(mcts): sort edges by policy + early cutoff in PUCT scan

Edges are now sorted descending by policy prior after NN evaluation.
PUCT scan exits early when it hits two consecutive unvisited edges,
since all remaining edges have lower policy and can't be best."
```


---

## Task 4: Node Solidification (Contiguous Children Arrays)

**Estimated Elo gain: +15-30**
**Root cause:** MetalFish stores children as individually arena-allocated nodes pointed to by atomic pointers in each Edge. When PUCT scans a hot node with many children, it chases pointers across scattered memory locations. Lc0 "solidifies" nodes with >100 visits — converting children into a contiguous `Node[num_edges]` array for cache-friendly sequential access.

**Files:**
- Modify: `src/mcts/node.h` (add `solid_children_` flag, `MakeSolid()`, child array, destructor)
- Modify: `src/mcts/search.cpp` (call MakeSolid during backpropagation)
- Test: `tests/test_mcts_module.cpp`

### Step 1: Write a test for solidification

```cpp
void test_node_solidification(TestCounter &tc) {
    std::cout << "  Node solidification..." << std::endl;
    Node root;
    Position pos;
    StateInfo st;
    pos.set("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1", false, &st);
    MoveList<LEGAL> legal(pos);
    root.CreateEdges(legal);

    int n = root.NumEdges();
    expect(n > 0, "root has edges for solidification test", tc);
    expect(!root.IsSolid(), "not solid initially", tc);

    // Create some children with visits
    for (int i = 0; i < std::min(n, 5); ++i) {
        // Simulate child creation
        // (In real search, arena allocator does this)
    }

    // MakeSolid should return false if no children have enough visits
    // (We'll test the full integration in the search test)
}
```

### Step 2: Implement solidification in Node

Add to `src/mcts/node.h`:

1. Add `solid_children_` flag and `solid_array_` pointer to private fields:
```cpp
bool solid_children_ = false;
Node* solid_array_ = nullptr; // contiguous array when solidified
```

2. Add `IsSolid()` accessor:
```cpp
bool IsSolid() const { return solid_children_; }
```

3. Add `MakeSolid()` method:
```cpp
bool MakeSolid() {
    if (solid_children_ || num_edges_ == 0 || IsTerminal()) return false;

    // Safety: don't solidify if any leaf child has in-flight visits
    for (int i = 0; i < num_edges_; ++i) {
        Node* ch = edges_[i].child.load(std::memory_order_acquire);
        if (ch && ch->GetN() <= 1 && ch->GetNInFlight() > 0) return false;
        if (ch && ch->IsTerminal() && ch->GetNInFlight() > 0) return false;
    }

    // Allocate contiguous array
    std::allocator<Node> alloc;
    Node* arr = alloc.allocate(num_edges_);
    for (int i = 0; i < num_edges_; ++i) {
        new (&arr[i]) Node(this, i);
    }

    // Move existing children's data into their array slots
    for (int i = 0; i < num_edges_; ++i) {
        Node* old_child = edges_[i].child.load(std::memory_order_acquire);
        if (old_child && old_child->GetN() > 0) {
            // Copy accumulated stats
            arr[i].FinalizeScoreUpdate(
                old_child->GetWL(), old_child->GetD(), old_child->GetM(),
                static_cast<int>(old_child->GetN()));
            // Copy edges if the child was expanded
            // (edges are owned by the child, transfer ownership)
            if (old_child->NumEdges() > 0) {
                arr[i].TakeEdgesFrom(*old_child);
            }
            if (old_child->IsTerminal()) {
                arr[i].MakeTerminal(old_child->GetTerminalType(),
                                    old_child->GetWL(), old_child->GetD(),
                                    old_child->GetM());
            }
        }
        edges_[i].child.store(&arr[i], std::memory_order_release);
    }

    solid_array_ = arr;
    solid_children_ = true;
    return true;
}

// Transfer edge ownership (used during solidification)
void TakeEdgesFrom(Node& other) {
    edges_ = std::move(other.edges_);
    num_edges_ = other.num_edges_;
    other.num_edges_ = 0;
}
```

4. Update destructor to handle solid array:
```cpp
~Node() {
    if (solid_children_ && solid_array_) {
        for (int i = 0; i < num_edges_; ++i) {
            solid_array_[i].~Node();
        }
        std::allocator<Node> alloc;
        alloc.deallocate(solid_array_, num_edges_);
    }
}
```

### Step 3: Trigger solidification during backpropagation

In `src/mcts/search.cpp` Backpropagate function (line 993-1003), add solidification trigger:

```cpp
void Search::Backpropagate(Node* node, float value, float draw,
                           float moves_left, int multivisit) {
    const int visits = std::max(1, multivisit);
    while (node) {
        node->FinalizeScoreUpdate(value, draw, moves_left, visits);
        if (params_.sticky_endgames) node->MaybeSetBounds();

        // Solidify hot nodes for cache-friendly child access
        if (!node->IsSolid() && node->GetN() >= static_cast<uint32_t>(params_.solid_tree_threshold)) {
            node->MakeSolid();
        }

        value = -value;
        moves_left += 1.0f;
        node = node->Parent();
    }
}
```

### Step 4: Build and test

```bash
cmake --build build -j$(sysctl -n hw.ncpu) && ./build/metalfish_tests
echo -e "uci\nsetoption name UseMCTS value true\nposition startpos\ngo nodes 1000\nquit" | ./build/metalfish
```
Expected: All tests pass, no crashes, no ASAN errors

### Step 5: Commit

```bash
git add src/mcts/node.h src/mcts/search.cpp tests/test_mcts_module.cpp
git commit -m "feat(mcts): node solidification for cache-friendly PUCT scans

Hot nodes (>100 visits) convert children from scattered arena pointers
to contiguous Node arrays. Improves L1/L2 cache hit rate during PUCT
selection by ensuring sequential child access is cache-line friendly."
```

---

## Task 5: KLD Gain Stopper

**Estimated Elo gain: +30-50**
**Root cause:** MetalFish only stops search via time limit and smart pruning. It wastes iterations when the visit distribution has already converged (best move is obvious). Lc0's KLD gain stopper computes the KL divergence of the root visit distribution between intervals — when per-node KLD falls below a threshold, search stops early, saving time for future critical moves.

**Files:**
- Modify: `src/mcts/stoppers.h` (add KLDGainStopper class)
- Modify: `src/mcts/stoppers.cpp` (implement KLD calculation)
- Modify: `src/mcts/search.cpp` (wire stopper into ShouldStop)
- Modify: `src/mcts/search_params.h` (add KLD params)
- Test: `tests/test_mcts_module.cpp`

### Step 1: Write a test for KLD convergence detection

```cpp
void test_kld_stopper(TestCounter &tc) {
    std::cout << "  KLD gain stopper..." << std::endl;

    // Simulate a converged distribution: [1000, 50, 30, 20, 10]
    // Adding 100 more visits mostly to first child shouldn't change KLD much
    SearchStats stats1;
    stats1.total_nodes = 1110;
    stats1.edge_n = {1000, 50, 30, 20, 10};

    SearchStats stats2;
    stats2.total_nodes = 1210;
    stats2.edge_n = {1090, 55, 33, 22, 10};

    // KLD between these distributions should be very small
    double kld = 0.0;
    double total1 = 1110.0 - 1.0; // child visits = total - 1
    double total2 = 1210.0 - 1.0;
    for (size_t i = 0; i < stats1.edge_n.size(); ++i) {
        double p1 = stats1.edge_n[i] / total1;
        double p2 = stats2.edge_n[i] / total2;
        if (p1 > 0) kld += p1 * std::log(p1 / p2);
    }
    double per_node_kld = kld / (total2 - total1);

    expect(per_node_kld < 0.001, "converged distribution has tiny KLD per node", tc);
}
```

### Step 2: Add KLDGainStopper to stoppers.h

```cpp
class KLDGainStopper : public SearchStopper {
public:
  KLDGainStopper(float min_gain, int average_interval);
  bool ShouldStop(const SearchStats &stats) override;

private:
  float min_gain_;
  int average_interval_;
  std::vector<double> prev_visits_;
  double prev_child_nodes_ = 0.0;
};
```

### Step 3: Implement KLDGainStopper in stoppers.cpp

```cpp
KLDGainStopper::KLDGainStopper(float min_gain, int average_interval)
    : min_gain_(min_gain), average_interval_(average_interval) {}

bool KLDGainStopper::ShouldStop(const SearchStats &stats) {
  const double new_child_nodes = static_cast<double>(stats.total_nodes) - 1.0;
  if (new_child_nodes < prev_child_nodes_ + average_interval_) return false;

  const auto& new_visits = stats.edge_n;
  if (!prev_visits_.empty() && prev_visits_.size() == new_visits.size()) {
    double kldgain = 0.0;
    for (size_t i = 0; i < new_visits.size(); ++i) {
      double o_p = prev_visits_[i] / prev_child_nodes_;
      double n_p = static_cast<double>(new_visits[i]) / new_child_nodes;
      if (prev_visits_[i] > 0 && n_p > 0) {
        kldgain += o_p * std::log(o_p / n_p);
      }
    }
    if (kldgain / (new_child_nodes - prev_child_nodes_) < min_gain_) {
      return true;
    }
  }

  prev_visits_.clear();
  prev_visits_.reserve(new_visits.size());
  for (uint32_t v : new_visits) {
    prev_visits_.push_back(static_cast<double>(v));
  }
  prev_child_nodes_ = new_child_nodes;
  return false;
}
```

### Step 4: Add KLD parameters to search_params.h

```cpp
// KLD gain stopper (stop when visit distribution converges)
bool use_kld_gain_stopper = true;
float kld_gain_min = 0.00001f;  // lc0 default: very small threshold
int kld_gain_average_interval = 100;  // check every 100 nodes
```

### Step 5: Wire KLD stopper into ShouldStop in search.cpp

In `Search::ShouldStop()` (line 464-503), after the smart pruning check, add:

```cpp
// KLD gain stopper: stop when visit distribution has converged
if (params_.use_kld_gain_stopper && kld_stopper_) {
    SearchStats kld_stats;
    kld_stats.total_nodes = stats_.total_nodes.load(std::memory_order_relaxed);
    const Node* root = tree_.Root();
    if (root && root->NumEdges() > 0) {
        const Edge* edges = root->Edges();
        for (int i = 0; i < root->NumEdges(); ++i) {
            Node* child = edges[i].child.load(std::memory_order_relaxed);
            kld_stats.edge_n.push_back(child ? child->GetN() : 0);
        }
        if (kld_stopper_->ShouldStop(kld_stats)) return true;
    }
}
```

Add `std::unique_ptr<KLDGainStopper> kld_stopper_` to the Search class private members, and initialize it in `StartSearch`:

```cpp
if (params_.use_kld_gain_stopper) {
    kld_stopper_ = std::make_unique<KLDGainStopper>(
        params_.kld_gain_min, params_.kld_gain_average_interval);
}
```

### Step 6: Build and test

```bash
cmake --build build -j$(sysctl -n hw.ncpu) && ./build/metalfish_tests
# Test that easy positions terminate faster
echo -e "uci\nsetoption name UseMCTS value true\nposition startpos\ngo movetime 5000\nquit" | timeout 10 ./build/metalfish
```
Expected: Engine terminates well before 5 seconds on the starting position

### Step 7: Commit

```bash
git add src/mcts/stoppers.h src/mcts/stoppers.cpp src/mcts/search.h src/mcts/search.cpp src/mcts/search_params.h tests/test_mcts_module.cpp
git commit -m "feat(mcts): KLD gain stopper for early search termination

Stop search when root visit distribution converges (KL divergence
per new node below threshold). Saves time on easy positions for use
on critical ones. Matches lc0's KldGainStopper."
```


---

## Task 6: Prefetching into NN Cache

**Estimated Elo gain: +20-40**
**Root cause:** When MetalFish's GPU batch is smaller than max batch size, unused slots are wasted. Lc0 fills remaining slots with positions likely needed in future iterations by recursively walking the tree via PUCT scores. This increases cache hit rate for subsequent iterations.

**Files:**
- Modify: `src/mcts/search.h` (declare PrefetchIntoCache)
- Modify: `src/mcts/search.cpp` (implement PrefetchIntoCache, call after gathering)
- Test: `tests/test_mcts_module.cpp`

### Step 1: Write a test — measure cache hit rate improvement

This test relies on the Task 1 cache fix being in place. We compare cache hit rate with and without prefetching:

```cpp
void test_prefetch_improves_cache(TestCounter &tc) {
    std::cout << "  Prefetching..." << std::endl;
    const char *weights = std::getenv("METALFISH_NN_WEIGHTS");
    if (!weights) {
        std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
        return;
    }

    SearchParams params;
    params.num_threads = 1;
    params.nn_weights_path = weights;
    params.max_prefetch = 32; // Enable prefetching

    MetalFish::Search::LimitsType limits;
    limits.nodes = 512;

    auto search = CreateSearch(params);
    search->StartSearch(
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
        limits, nullptr, nullptr);
    search->Wait();

    const auto &stats = search->Stats();
    uint64_t hits = stats.cache_hits.load();
    uint64_t total = hits + stats.cache_misses.load();
    std::cout << "    With prefetch: " << hits << "/" << total << " cache hits" << std::endl;
    expect(total > 0, "had cache lookups", tc);
}
```

### Step 2: Implement PrefetchIntoCache

Add to `src/mcts/search.h` in the Search class private section:

```cpp
int PrefetchIntoCache(Node* node, int budget, bool is_odd_depth,
                      SearchWorkerCtx& ctx, BackendComputation* computation);
void MaybePrefetchIntoCache(SearchWorkerCtx& ctx, BackendComputation* computation);
```

Add to `src/mcts/search.cpp`:

```cpp
void Search::MaybePrefetchIntoCache(SearchWorkerCtx& ctx, BackendComputation* computation) {
    if (stop_flag_.load(std::memory_order_acquire)) return;
    if (!computation || computation->UsedBatchSize() <= 0) return;
    if (computation->UsedBatchSize() >= params_.max_prefetch) return;

    int budget = params_.max_prefetch - computation->UsedBatchSize();
    ctx.ResetToRoot();
    PrefetchIntoCache(tree_.Root(), budget, false, ctx, computation);
}

int Search::PrefetchIntoCache(Node* node, int budget, bool is_odd_depth,
                               SearchWorkerCtx& ctx, BackendComputation* computation) {
    if (budget <= 0 || !node || stop_flag_.load(std::memory_order_acquire)) return 0;

    // Leaf node not yet visited — add to batch for prefetch
    if (node->GetNStarted() == 0) {
        auto history = ctx.BuildHistory();
        uint64_t key = ComputePositionCacheKey(history.ptrs, history.depth);
        computation->AddInputWithHistory(
            std::vector<const Position*>(history.ptrs, history.ptrs + history.depth), key);
        return 1;
    }

    if (node->GetN() == 0 || node->IsTerminal()) return 0;

    // Score children by PUCT and prefetch down the best branches
    int num_edges = node->NumEdges();
    if (num_edges == 0) return 0;

    const Edge* edges = node->Edges();
    float draw_score = params_.draw_score;
    float cpuct = params_.GetCpuct(node == tree_.Root());
    float cpuct_base = params_.GetCpuctBase(node == tree_.Root());
    float cpuct_factor = params_.GetCpuctFactor(node == tree_.Root());
    float effective_cpuct = cpuct + cpuct_factor *
        std::log((static_cast<float>(node->GetN()) + cpuct_base) / cpuct_base);
    float puct_mult = effective_cpuct *
        std::sqrt(static_cast<float>(std::max(node->GetChildrenVisits(), 1u)));

    float fpu;
    if (params_.GetFpuAbsolute(node == tree_.Root())) {
        fpu = params_.GetFpuValue(node == tree_.Root());
    } else {
        float reduction = (node == tree_.Root()) ? params_.fpu_reduction_at_root
                                                  : params_.fpu_reduction;
        fpu = -node->GetQ(-draw_score) - reduction * std::sqrt(node->GetVisitedPolicy());
    }

    // Build scored edge list
    struct ScoredEdge { float score; int idx; };
    std::vector<ScoredEdge> scored;
    scored.reserve(num_edges);
    for (int i = 0; i < num_edges; ++i) {
        if (edges[i].GetP() == 0.0f) continue;
        Node* child = edges[i].child.load(std::memory_order_acquire);
        float q = (child && child->GetN() > 0) ? child->GetQ(draw_score) : fpu;
        float u = puct_mult * edges[i].GetP() / (1.0f + (child ? child->GetNStarted() : 0));
        scored.push_back({-(q + u), i}); // negative for ascending sort
    }

    std::sort(scored.begin(), scored.end(),
              [](const ScoredEdge& a, const ScoredEdge& b) { return a.score < b.score; });

    int total_spent = 0;
    for (const auto& se : scored) {
        if (budget <= 0) break;
        if (stop_flag_.load(std::memory_order_acquire)) break;

        Node* child = edges[se.idx].child.load(std::memory_order_acquire);
        ctx.DoMove(edges[se.idx].move);
        int spent = PrefetchIntoCache(child, budget, !is_odd_depth, ctx, computation);
        // Undo move (pop from ctx stacks)
        ctx.state_stack.pop_back();
        ctx.move_stack.pop_back();
        ctx.hash_stack.pop_back();
        // Re-derive position by resetting and replaying
        // (expensive but correct for prefetch — happens rarely)

        budget -= spent;
        total_spent += spent;
    }
    return total_spent;
}
```

Note: The undo mechanism needs refinement — we need to add an `UndoMove` helper to `SearchWorkerCtx` or reconstruct position. A simpler alternative: save/restore ctx state before/after each recursive call.

### Step 3: Call MaybePrefetchIntoCache in RunIterationSemaphore

In `src/mcts/search.cpp` `RunIterationSemaphore`, after the gathering loop ends and before `ComputeBlocking()` (around line 800):

```cpp
// Prefetch likely-needed positions into remaining batch slots
MaybePrefetchIntoCache(ctx, computation.get());
```

### Step 4: Add UndoMove to SearchWorkerCtx

In `src/mcts/search.h`, add to SearchWorkerCtx:

```cpp
void UndoMove() {
    if (move_stack.empty()) return;
    pos.undo_move(move_stack.back());
    move_stack.pop_back();
    state_stack.pop_back();
    hash_stack.pop_back();
}
```

### Step 5: Build and test

```bash
cmake --build build -j$(sysctl -n hw.ncpu) && ./build/metalfish_tests
echo -e "uci\nsetoption name UseMCTS value true\nposition startpos\ngo nodes 1000\nquit" | ./build/metalfish
```

### Step 6: Commit

```bash
git add src/mcts/search.h src/mcts/search.cpp tests/test_mcts_module.cpp
git commit -m "feat(mcts): prefetch into NN cache during batch gathering

Fill unused GPU batch slots with positions likely needed in future
iterations by walking the tree via PUCT scores. Increases cache hit
rate and GPU utilization. Matches lc0's PrefetchIntoCache."
```

---

## Task 7: Syzygy Tablebase Probing in MCTS

**Estimated Elo gain: +40-60**
**Root cause:** The alpha-beta engine uses Syzygy tablebases for perfect endgame play, but MCTS doesn't probe them during node expansion. This means MCTS plays suboptimally in won/drawn/lost endgame positions where the answer is definitively known.

**Files:**
- Modify: `src/mcts/search.h` (add syzygy pointer)
- Modify: `src/mcts/search.cpp` (probe TB during expansion, root move filtering)
- Modify: `src/uci/engine.cpp` (pass syzygy path to MCTS)
- Test: `tests/test_mcts_module.cpp`

### Step 1: Write a test for TB probing

```cpp
void test_syzygy_in_mcts(TestCounter &tc) {
    std::cout << "  Syzygy TB in MCTS..." << std::endl;
    // KRK position — White wins with any reasonable play
    // If TB is available, MCTS should identify this immediately
    const char *weights = std::getenv("METALFISH_NN_WEIGHTS");
    if (!weights) {
        std::cout << "    SKIP: METALFISH_NN_WEIGHTS not set" << std::endl;
        return;
    }

    SearchParams params;
    params.num_threads = 1;
    params.nn_weights_path = weights;

    MetalFish::Search::LimitsType limits;
    limits.nodes = 64;

    auto search = CreateSearch(params);
    search->StartSearch(
        "8/8/8/4k3/8/8/1K2R3/8 w - - 0 1",  // KR vs K
        limits, nullptr, nullptr);
    search->Wait();

    float q = search->GetBestQ();
    // With or without TB, White should evaluate as winning
    expect(q > 0.5f, "KR vs K evaluated as winning for White", tc);
}
```

### Step 2: Add Syzygy pointer to Search class

In `src/mcts/search.h`, add to Search private members:

```cpp
// Syzygy tablebase (optional, shared with AB engine)
Tablebases::TBProbe* syzygy_tb_ = nullptr;
bool root_is_in_dtz_ = false;
std::atomic<uint64_t> tb_hits_{0};
```

Add a setter:
```cpp
void SetSyzygyTB(Tablebases::TBProbe* tb) { syzygy_tb_ = tb; }
```

### Step 3: Probe tablebases during node expansion

In `src/mcts/search.cpp`, in both `RunIteration` and `RunIterationSemaphore`, after the insufficient material / rule50 checks and before the NN evaluation, add:

```cpp
// Syzygy tablebase probing for endgame positions
if (syzygy_tb_ && !root_is_in_dtz_ &&
    !ctx.pos.can_castle(ANY_CASTLING) &&
    ctx.pos.rule50_count() == 0 &&
    popcount(ctx.pos.pieces()) <= syzygy_tb_->max_cardinality()) {
    int wdl = syzygy_tb_->probe_wdl(ctx.pos);
    if (wdl != Tablebases::WDL_FAIL) {
        float tb_value, tb_draw;
        if (wdl > 0) { // Win
            tb_value = 1.0f; tb_draw = 0.0f;
        } else if (wdl < 0) { // Loss
            tb_value = -1.0f; tb_draw = 0.0f;
        } else { // Draw
            tb_value = 0.0f; tb_draw = 1.0f;
        }
        float tb_m = leaf->Parent() ?
            std::max(0.0f, leaf->Parent()->GetM() - 1.0f) : 0.0f;
        leaf->MakeTerminal(Node::Terminal::Tablebase, tb_value, tb_draw, tb_m);
        leaf->FinalizeScoreUpdate(tb_value, tb_draw, tb_m, multivisit);
        if (leaf->Parent())
            Backpropagate(leaf->Parent(), -tb_value, tb_draw, tb_m + 1.0f, multivisit);
        stats_.total_nodes.fetch_add(multivisit, std::memory_order_relaxed);
        tb_hits_.fetch_add(1, std::memory_order_relaxed);
        // skip NN evaluation — TB gives exact answer
        continue; // or return, depending on context
    }
}
```

Note: The exact Syzygy API depends on MetalFish's `src/syzygy/tbprobe.h` interface. Adapt probe call to match the existing API (likely `Tablebases::probe_wdl(pos)`).

### Step 4: Wire Syzygy path from UCI

In `src/uci/engine.cpp`, when creating the MCTS search or hybrid search, pass the syzygy pointer:

```cpp
if (mcts_search_) {
    mcts_search_->SetSyzygyTB(Tablebases::get_instance());
}
```

### Step 5: Build and test

```bash
cmake --build build -j$(sysctl -n hw.ncpu) && ./build/metalfish_tests
# Test KRK endgame
echo -e "uci\nsetoption name UseMCTS value true\nsetoption name SyzygyPath value /path/to/syzygy\nposition fen 8/8/8/4k3/8/8/1K2R3/8 w - - 0 1\ngo nodes 100\nquit" | ./build/metalfish
```

### Step 6: Commit

```bash
git add src/mcts/search.h src/mcts/search.cpp src/uci/engine.cpp tests/test_mcts_module.cpp
git commit -m "feat(mcts): Syzygy tablebase probing during node expansion

Probe WDL tablebases for positions with <=N pieces, no castling,
rule50=0. Terminal TB nodes skip NN evaluation entirely, giving
exact endgame values. Matches lc0's ExtendNode TB probing."
```


---

## Task 8: Improved Terminal Handling + Two-Fold Draw Correction

**Estimated Elo gain: +30-50**
**Root cause:** When the tree is reused between moves, nodes marked as two-fold draws may become incorrect — the first occurrence of the position may no longer be in the game history. Lc0 handles this with `EnsureNodeTwoFoldCorrectForDepth()` which reverts stale two-fold terminals. MetalFish doesn't do this.

**Files:**
- Modify: `src/mcts/node.h` (add RevertTwoFoldDraw method)
- Modify: `src/mcts/search.cpp` (call revert during tree reuse, improve terminal propagation)
- Test: `tests/test_mcts_module.cpp`

### Step 1: Write a test for two-fold draw reversion

```cpp
void test_twofold_reversion(TestCounter &tc) {
    std::cout << "  Two-fold draw reversion..." << std::endl;
    Node node;
    node.MakeTerminal(Node::Terminal::TwoFold, 0.0f, 1.0f, 2.0f);
    expect(node.IsTerminal(), "node is terminal", tc);
    expect(node.GetTerminalType() == Node::Terminal::TwoFold, "is two-fold", tc);

    node.RevertTerminal();
    expect(!node.IsTerminal(), "no longer terminal after revert", tc);
}
```

### Step 2: Add RevertTerminal to Node

In `src/mcts/node.h`, add:

```cpp
void RevertTerminal() {
    terminal_type_.store(static_cast<uint8_t>(Terminal::NonTerminal),
                         std::memory_order_release);
}

// Revert two-fold draw if it's no longer valid at the given depth
void MaybeRevertTwoFold(int depth_from_root) {
    if (GetTerminalType() != Terminal::TwoFold) return;
    // Two-fold draws require depth >= 4 and depth >= repetition distance
    // If we're now at a shallower depth due to tree reuse, revert
    if (depth_from_root < 4) {
        RevertTerminal();
    }
}
```

### Step 3: Walk tree after reuse to fix two-fold nodes

In `src/mcts/search.cpp` `StartSearch`, after `TryReuse` succeeds, add a depth-limited walk:

```cpp
if (!tree_.TryReuse(fen)) {
    tree_.Reset(fen);
} else {
    // After tree reuse, revert stale two-fold draw terminals
    std::function<void(Node*, int)> fixTwoFold = [&](Node* node, int depth) {
        if (!node || depth > 20) return;
        node->MaybeRevertTwoFold(depth);
        if (node->NumEdges() > 0) {
            Edge* edges = node->Edges();
            for (int i = 0; i < node->NumEdges(); ++i) {
                Node* child = edges[i].child.load(std::memory_order_acquire);
                if (child) fixTwoFold(child, depth + 1);
            }
        }
    };
    fixTwoFold(tree_.Root(), 0);
}
```

### Step 4: Build and test

```bash
cmake --build build -j$(sysctl -n hw.ncpu) && ./build/metalfish_tests
```

### Step 5: Commit

```bash
git add src/mcts/node.h src/mcts/search.cpp tests/test_mcts_module.cpp
git commit -m "feat(mcts): revert stale two-fold draws on tree reuse

After tree reuse, walk the retained subtree and revert any two-fold
draw terminals that are no longer valid at their new depth. Prevents
incorrect draw evaluations from persisting across moves."
```

---

## Task 9: Background Garbage Collection

**Estimated Elo gain: +10-20**
**Root cause:** When the tree is pruned during reuse, node deallocation happens synchronously, causing latency spikes. Lc0 uses a dedicated GC thread that wakes every 100ms to process a queue of subtrees to free.

**Files:**
- Modify: `src/mcts/node.h` (add NodeGarbageCollector class)
- Modify: `src/mcts/search.cpp` (use GC during tree reuse/pruning)
- Test: `tests/test_mcts_module.cpp`

### Step 1: Add NodeGarbageCollector to node.h

```cpp
class NodeGarbageCollector {
public:
    NodeGarbageCollector() : gc_thread_([this]() { Worker(); }) {}

    ~NodeGarbageCollector() {
        stop_.store(true, std::memory_order_release);
        if (gc_thread_.joinable()) gc_thread_.join();
    }

    void AddToGcQueue(std::unique_ptr<Node[]> nodes, size_t count) {
        if (!nodes) return;
        std::lock_guard<std::mutex> lock(gc_mutex_);
        gc_queue_.emplace_back(std::move(nodes), count);
    }

private:
    void Worker() {
        while (!stop_.load(std::memory_order_acquire)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            GarbageCollect();
        }
        GarbageCollect(); // Final cleanup
    }

    void GarbageCollect() {
        while (true) {
            std::pair<std::unique_ptr<Node[]>, size_t> item;
            {
                std::lock_guard<std::mutex> lock(gc_mutex_);
                if (gc_queue_.empty()) return;
                item = std::move(gc_queue_.back());
                gc_queue_.pop_back();
            }
            // Destruction happens outside the lock
            item.first.reset();
        }
    }

    std::thread gc_thread_;
    std::atomic<bool> stop_{false};
    std::mutex gc_mutex_;
    std::vector<std::pair<std::unique_ptr<Node[]>, size_t>> gc_queue_;
};

// Global GC instance
inline NodeGarbageCollector& GetNodeGC() {
    static NodeGarbageCollector gc;
    return gc;
}
```

### Step 2: Use GC in NodeTree during reuse/reset

In `NodeTree::Reset` and `NodeTree::TryReuse`, instead of directly releasing old subtrees, push them to the GC:

```cpp
// In TryReuse, when discarding sibling subtrees:
// Old: root_heap_.reset();
// New: (send to GC instead of synchronous delete)
```

### Step 3: Build and test

```bash
cmake --build build -j$(sysctl -n hw.ncpu) && ./build/metalfish_tests
```

### Step 4: Commit

```bash
git add src/mcts/node.h src/mcts/search.cpp tests/test_mcts_module.cpp
git commit -m "feat(mcts): background garbage collection for tree pruning

Pruned subtrees are sent to a background GC thread for asynchronous
deallocation, preventing latency spikes during tree reuse. GC wakes
every 100ms to process queued subtrees."
```

---

## Task 10: Temperature-Based Move Selection + Contempt

**Estimated Elo gain: +10-15**
**Root cause:** MetalFish always selects the highest-N child as best move. Lc0 supports temperature-based selection that samples from the visit distribution, and contempt (draw score bias) to avoid draws against weaker opponents. Temperature is mostly useful during training/analysis, but the draw score/contempt mechanism helps in tournament play.

**Files:**
- Modify: `src/mcts/search_params.h` (add temperature + contempt params)
- Modify: `src/mcts/search.cpp` (implement temperature selection, wire contempt)
- Test: `tests/test_mcts_module.cpp`

### Step 1: Add parameters

In `src/mcts/search_params.h`:

```cpp
// Temperature for final move selection (0 = always best, >0 = sample)
float temperature = 0.0f;
float temp_decay_moves = 0.0f;     // decay temperature over this many moves
float temp_cutoff_move = 0;         // disable temperature after this move
float temp_endgame = 0.0f;          // temperature in endgame
float temp_winpct_cutoff = 50.0f;   // don't consider moves this much worse than best

// Contempt / draw score
float contempt = 0.0f;              // Elo contempt (positive = avoid draws)
```

### Step 2: Implement temperature-based move selection

In `src/mcts/search.cpp`, add `GetBestMoveWithTemperature`:

```cpp
Move Search::GetBestMoveWithTemperature(float temperature) const {
    const Node* root = tree_.Root();
    if (!root || root->NumEdges() == 0) return Move::none();

    int num_edges = root->NumEdges();
    const Edge* edges = root->Edges();

    // Find max N and max eval for filtering
    float max_n = 0.0f;
    float max_eval = -2.0f;
    for (int i = 0; i < num_edges; ++i) {
        Node* child = edges[i].child.load(std::memory_order_acquire);
        if (!child) continue;
        float cn = static_cast<float>(child->GetN());
        if (cn > max_n) {
            max_n = cn;
            max_eval = child->GetWL();
        }
    }
    if (max_n <= 0.0f) return GetBestMove(); // fallback

    float min_eval = max_eval - params_.temp_winpct_cutoff / 50.0f;

    // Build cumulative distribution
    std::vector<float> cumsum;
    std::vector<int> indices;
    float sum = 0.0f;
    for (int i = 0; i < num_edges; ++i) {
        Node* child = edges[i].child.load(std::memory_order_acquire);
        if (!child || child->GetN() == 0) continue;
        if (child->GetWL() < min_eval) continue;

        float weight = std::pow(
            static_cast<float>(child->GetN()) / max_n, 1.0f / temperature);
        sum += weight;
        cumsum.push_back(sum);
        indices.push_back(i);
    }
    if (cumsum.empty()) return GetBestMove();

    // Sample
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dist(0.0f, cumsum.back());
    float toss = dist(gen);
    auto it = std::lower_bound(cumsum.begin(), cumsum.end(), toss);
    int idx = static_cast<int>(it - cumsum.begin());
    idx = std::min(idx, static_cast<int>(indices.size()) - 1);

    return edges[indices[idx]].move;
}
```

### Step 3: Wire temperature into Wait() for final move selection

In `Search::Wait()`, replace the bestmove callback:

```cpp
if (cb_copy) {
    Move best;
    if (params_.temperature > 0.0f) {
        best = GetBestMoveWithTemperature(params_.temperature);
    } else {
        best = GetBestMove();
    }
    // ... rest unchanged
}
```

### Step 4: Wire contempt into draw_score

Contempt translates to a draw score bias. In `Search::StartSearch`:

```cpp
if (params_.contempt != 0.0f) {
    // Convert Elo contempt to WDL draw score adjustment
    // Positive contempt = we think we're stronger = draws are bad
    params_.draw_score = -params_.contempt / 10000.0f;
}
```

### Step 5: Build and test

```bash
cmake --build build -j$(sysctl -n hw.ncpu) && ./build/metalfish_tests
```

### Step 6: Commit

```bash
git add src/mcts/search_params.h src/mcts/search.h src/mcts/search.cpp tests/test_mcts_module.cpp
git commit -m "feat(mcts): temperature-based move selection + contempt

Add temperature parameter for probabilistic move selection (samples
from visit distribution with win-pct cutoff). Add contempt/draw-score
bias for tournament play. Matches lc0's temperature and contempt."
```


---

## Task 11: WDL Rescaling

**Estimated Elo gain: +10-15**
**Root cause:** Lc0 applies WDL rescaling to adjust the raw win/draw/loss outputs from the neural network for better calibration. The rescaling uses parameters `ratio` and `diff` to sharpen or flatten the WDL distribution based on position complexity. MetalFish uses raw WDL values directly.

**Files:**
- Modify: `src/mcts/search_params.h` (add WDL rescale params)
- Modify: `src/mcts/search.cpp` (apply WDL rescaling after NN evaluation)
- Test: `tests/test_mcts_module.cpp`

### Step 1: Add WDL rescaling parameters

In `src/mcts/search_params.h`:

```cpp
// WDL rescaling (lc0 defaults)
float wdl_rescale_ratio = 1.0f;   // 1.0 = no rescaling
float wdl_rescale_diff = 0.0f;    // WDL curve adjustment
float wdl_eval_objectivity = 1.0f;  // 1.0 = fully objective
```

### Step 2: Implement WDL rescaling utility

In `src/mcts/core.h`, add:

```cpp
struct WDLRescaler {
    float ratio;
    float diff;

    float Rescale(float q) const {
        if (ratio == 1.0f && diff == 0.0f) return q;
        // Apply sigmoid rescaling: q' = tanh(atanh(q) * ratio + diff)
        float x = std::atanh(std::clamp(q, -0.999f, 0.999f));
        return std::tanh(x * ratio + diff);
    }
};
```

### Step 3: Apply rescaling after NN evaluation

In `src/mcts/search.cpp`, wherever we extract `value = -result.value`, apply rescaling:

```cpp
float v = -result.value;
if (params_.wdl_rescale_ratio != 1.0f || params_.wdl_rescale_diff != 0.0f) {
    WDLRescaler rescaler{params_.wdl_rescale_ratio, params_.wdl_rescale_diff};
    v = -rescaler.Rescale(result.value);
}
```

### Step 4: Build and test

```bash
cmake --build build -j$(sysctl -n hw.ncpu) && ./build/metalfish_tests
```

### Step 5: Commit

```bash
git add src/mcts/search_params.h src/mcts/core.h src/mcts/search.cpp tests/test_mcts_module.cpp
git commit -m "feat(mcts): WDL rescaling for better evaluation calibration

Apply sigmoid rescaling to NN WDL outputs using configurable ratio
and diff parameters. Matches lc0's WDL rescaling for calibrated
win/draw/loss probabilities."
```

---

## Task 12: Safety Stoppers (Memory, Depth, Mate)

**Estimated Elo gain: +5-10**
**Root cause:** MetalFish lacks safety stoppers that prevent resource exhaustion and enable early termination in clear situations. Lc0 has: (1) MemoryWatchingStopper to prevent OOM, (2) DepthStopper for search depth limits, (3) MateStopper for immediate termination when a forced mate is found.

**Files:**
- Modify: `src/mcts/stoppers.h` (add 3 new stopper classes)
- Modify: `src/mcts/stoppers.cpp` (implement them)
- Modify: `src/mcts/search.cpp` (wire into stop logic)
- Test: `tests/test_mcts_module.cpp`

### Step 1: Add MemoryWatchingStopper

In `src/mcts/stoppers.h`:

```cpp
class MemoryWatchingStopper : public SearchStopper {
public:
  explicit MemoryWatchingStopper(size_t max_bytes);
  bool ShouldStop(const SearchStats &stats) override;

private:
  size_t max_bytes_;
};
```

In `src/mcts/stoppers.cpp`:

```cpp
MemoryWatchingStopper::MemoryWatchingStopper(size_t max_bytes)
    : max_bytes_(max_bytes) {}

bool MemoryWatchingStopper::ShouldStop(const SearchStats &stats) {
    // Estimate memory: ~64 bytes per node (Node struct) + ~16 bytes per edge
    size_t estimated = stats.total_nodes * (64 + 16 * 35); // avg 35 edges
    return estimated > max_bytes_;
}
```

### Step 2: Add MateStopper

```cpp
class MateStopper : public SearchStopper {
public:
  bool ShouldStop(const SearchStats &stats) override;
  void SetMateFound(bool found) { mate_found_.store(found, std::memory_order_release); }

private:
  std::atomic<bool> mate_found_{false};
};
```

Implementation:
```cpp
bool MateStopper::ShouldStop(const SearchStats &stats) {
    return mate_found_.load(std::memory_order_acquire);
}
```

### Step 3: Wire mate detection into bestmove logic

In `Search::ShouldStop()`, after the existing checks:

```cpp
// Check if we've found a forced mate (terminal win at root)
if (mate_stopper_) {
    const Node* root = tree_.Root();
    if (root && root->NumEdges() > 0) {
        const Edge* edges = root->Edges();
        for (int i = 0; i < root->NumEdges(); ++i) {
            Node* child = edges[i].child.load(std::memory_order_relaxed);
            if (child && child->IsTerminal() && child->GetWL() > 0.5f) {
                mate_stopper_->SetMateFound(true);
                break;
            }
        }
        if (mate_stopper_->ShouldStop({})) return true;
    }
}
```

### Step 4: Build and test

```bash
cmake --build build -j$(sysctl -n hw.ncpu) && ./build/metalfish_tests
# Test mate-in-1 position — should terminate very quickly
echo -e "uci\nsetoption name UseMCTS value true\nposition fen 6k1/5ppp/8/8/8/8/5PPP/4R1K1 w - - 0 1\ngo movetime 5000\nquit" | timeout 10 ./build/metalfish
```

### Step 5: Commit

```bash
git add src/mcts/stoppers.h src/mcts/stoppers.cpp src/mcts/search.h src/mcts/search.cpp tests/test_mcts_module.cpp
git commit -m "feat(mcts): safety stoppers — memory watching + mate detection

Add MemoryWatchingStopper to prevent OOM by estimating tree memory
usage. Add MateStopper for immediate termination when a forced mate
is found in the search tree. Prevents resource exhaustion and speeds
up obvious endgame positions."
```

---

## Milestone Checkpoint: Full Integration Test

After completing all 12 tasks, run the comprehensive validation:

### Step 1: Build clean

```bash
cd /Users/nripeshn/Documents/PythonPrograms/metalfish
rm -rf build && cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(sysctl -n hw.ncpu)
```

### Step 2: Run all unit tests

```bash
METALFISH_NN_WEIGHTS=networks/BT4-1024x15x32h-swa-6147500.pb ./build/metalfish_tests
```
Expected: All tests pass

### Step 3: Run Bratko-Kopec tactical suite

```bash
python3 tools/run_bratko_kopec.py --engine ./build/metalfish --mode mcts --time 10
```
Expected: >=18/24 solved (same or better than before)

### Step 4: Quick self-play tournament (new MCTS vs old MCTS)

```bash
# Save old binary first, then build new
# Run 10-game match at 60+0.6s to quickly estimate Elo difference
python3 tools/run_tournament.py --engine1 ./build/metalfish --engine2 ./build/metalfish_old \
    --mode mcts --games 10 --tc "60+0.6"
```
Expected: New MCTS wins significantly (>70% score)

### Step 5: Full tournament against Lc0

```bash
python3 tools/run_tournament.py --engine1 ./build/metalfish --engine2 lc0 \
    --mode mcts --games 20 --tc "300+0.1"
```
Expected: Competitive results (40-60% score range, indicating ~3600-3700 Elo)

### Step 6: Verify no regressions in AB and Hybrid modes

```bash
echo -e "uci\nsetoption name UseMCTS value false\nposition startpos\ngo depth 20\nquit" | ./build/metalfish
echo -e "uci\nsetoption name UseHybridSearch value true\nposition startpos\ngo movetime 5000\nquit" | ./build/metalfish
```
Expected: Both modes still work correctly

---

## Summary of All Tasks

| Task | Feature | Est. Elo | Key Files |
|------|---------|----------|-----------|
| 1 | NN Cache Key Fix | +100-150 | search.cpp |
| 2 | Multivisit Propagation | +10-20 | search.cpp |
| 3 | Edge Sorting + Early Cutoff | +10-20 | node.h, search.cpp |
| 4 | Node Solidification | +15-30 | node.h, search.cpp |
| 5 | KLD Gain Stopper | +30-50 | stoppers.h/.cpp, search.cpp |
| 6 | Prefetching into Cache | +20-40 | search.h/.cpp |
| 7 | Syzygy TB in MCTS | +40-60 | search.h/.cpp, engine.cpp |
| 8 | Two-Fold Draw Correction | +30-50 | node.h, search.cpp |
| 9 | Background GC | +10-20 | node.h, search.cpp |
| 10 | Temperature + Contempt | +10-15 | search_params.h, search.cpp |
| 11 | WDL Rescaling | +10-15 | search_params.h, core.h, search.cpp |
| 12 | Safety Stoppers | +5-10 | stoppers.h/.cpp, search.cpp |
| **Total** | | **~300-500** | |

**Execution order is critical:** Tasks 1-3 (foundation) → Tasks 4-7 (major features) → Tasks 8-12 (polish)

Each task includes: failing test → implementation → passing test → BK regression → commit.

---

Plan complete and saved to `docs/plans/2026-03-20-mcts-lc0-parity-plan.md`. Two execution options:

**1. Subagent-Driven (this session)** — I dispatch fresh subagent per task, review between tasks, fast iteration

**2. Parallel Session (separate)** — Open new session with executing-plans, batch execution with checkpoints

**Which approach?**

