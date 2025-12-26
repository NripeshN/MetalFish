/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Transposition Table implementation using unified memory for
  zero-copy CPU-GPU access.
*/

#include "core/types.h"
#include "metal/allocator.h"
#include <cstring>
#include <atomic>

namespace MetalFish {

/**
 * Transposition Table Entry
 * Packed into 16 bytes for cache efficiency and atomic access
 */
struct TTEntry {
    Key key16;        // 16 bits of the position key
    Move move;        // Best move
    int16_t value;    // Search value
    int16_t eval;     // Static evaluation
    uint8_t genBound; // Generation (6 bits) + Bound type (2 bits)
    int8_t depth;     // Search depth
    
    void save(Key k, Value v, bool pv, Bound b, Depth d, Move m, Value ev, uint8_t gen) {
        key16 = (uint16_t)k;
        move = m;
        value = (int16_t)v;
        eval = (int16_t)ev;
        genBound = uint8_t(gen << 2) | uint8_t(b) | (pv ? 4 : 0);
        depth = (int8_t)d;
    }
    
    Move get_move() const { return move; }
    Value get_value() const { return value; }
    Value get_eval() const { return eval; }
    Depth get_depth() const { return depth; }
    Bound get_bound() const { return Bound(genBound & 3); }
    bool is_pv() const { return genBound & 4; }
};

/**
 * Transposition Table Cluster
 * Each cluster contains 3 entries for better cache utilization
 */
struct TTCluster {
    TTEntry entry[3];
    char padding[64 - 3 * sizeof(TTEntry)]; // Align to 64 bytes (cache line)
};

static_assert(sizeof(TTCluster) == 64, "TTCluster must be 64 bytes");

/**
 * Transposition Table
 * Uses unified memory for shared CPU-GPU access
 */
class TranspositionTable {
public:
    TranspositionTable() = default;
    ~TranspositionTable() { free(); }
    
    /**
     * Resize the table to the specified size in MB
     */
    void resize(size_t mb_size) {
        free();
        
        cluster_count_ = mb_size * 1024 * 1024 / sizeof(TTCluster);
        
        // Allocate using Metal unified memory
        buffer_ = Metal::MetalAllocator::instance().allocate(
            cluster_count_ * sizeof(TTCluster));
        
        table_ = static_cast<TTCluster*>(buffer_.contents());
        clear();
    }
    
    /**
     * Clear all entries
     */
    void clear() {
        if (table_) {
            std::memset(table_, 0, cluster_count_ * sizeof(TTCluster));
        }
        generation_ = 0;
    }
    
    /**
     * Probe the table for a position
     */
    TTEntry* probe(Key key, bool& found) const {
        TTEntry* tte = first_entry(key);
        uint16_t key16 = (uint16_t)key;
        
        for (int i = 0; i < 3; ++i) {
            if (tte[i].key16 == key16 || !tte[i].get_move()) {
                found = tte[i].key16 == key16;
                return &tte[i];
            }
        }
        
        // Return entry with lowest depth for replacement
        TTEntry* replace = tte;
        for (int i = 1; i < 3; ++i) {
            if (replacement_score(replace) > replacement_score(&tte[i])) {
                replace = &tte[i];
            }
        }
        
        found = false;
        return replace;
    }
    
    /**
     * Increment generation counter (called each new search)
     */
    void new_search() {
        generation_ += 8; // High bits for generation
    }
    
    uint8_t generation() const { return generation_; }
    
    /**
     * Get hashfull (permill)
     */
    int hashfull() const {
        int count = 0;
        for (size_t i = 0; i < 1000 && i < cluster_count_; ++i) {
            for (int j = 0; j < 3; ++j) {
                if ((table_[i].entry[j].genBound >> 2) == (generation_ >> 2)) {
                    ++count;
                }
            }
        }
        return count / 3;
    }
    
    // Get pointer for GPU access
    void* gpu_ptr() const { return table_; }
    size_t size_bytes() const { return cluster_count_ * sizeof(TTCluster); }

private:
    TTCluster* table_ = nullptr;
    Metal::Buffer buffer_;
    size_t cluster_count_ = 0;
    uint8_t generation_ = 0;
    
    TTEntry* first_entry(Key key) const {
        return &table_[((size_t)key) % cluster_count_].entry[0];
    }
    
    int replacement_score(const TTEntry* tte) const {
        return tte->get_depth() - ((generation_ - tte->genBound) & 0xFC);
    }
    
    void free() {
        if (buffer_.valid()) {
            Metal::MetalAllocator::instance().free(buffer_);
            buffer_ = Metal::Buffer{};
            table_ = nullptr;
            cluster_count_ = 0;
        }
    }
};

// Global transposition table
static TranspositionTable TT;

void tt_resize(size_t mb) { TT.resize(mb); }
void tt_clear() { TT.clear(); }
void tt_new_search() { TT.new_search(); }
int tt_hashfull() { return TT.hashfull(); }

} // namespace MetalFish

