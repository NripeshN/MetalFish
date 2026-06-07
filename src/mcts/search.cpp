/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan
  MCTS Search Pipeline Implementation - Optimized for Apple Silicon
  Licensed under GPL-3.0
*/

#include "search.h"
#include "../hybrid/shared_tt.h"
#include "core.h"

#include "../core/movegen.h"
#include "../syzygy/tbprobe.h"
#include "../uci/uci.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>

#ifdef __APPLE__
#include <pthread/qos.h>
#include <vecLib/vDSP.h>
#include <vecLib/vForce.h>
#endif

namespace MetalFish {
namespace MCTS {

constexpr size_t kNodeSizeBudget =
#ifdef __APPLE__
    128;
#else
    3 * CACHE_LINE_SIZE;
#endif

static_assert(sizeof(Node) >= sizeof(double), "Node must contain WL");
static_assert(sizeof(Node) <= kNodeSizeBudget, "Node exceeds size budget");
static_assert(alignof(Node) == CACHE_LINE_SIZE, "Node alignment mismatch");

namespace {

constexpr int kMaxEdges = 256;
constexpr uint64_t kFNVPrime = 1099511628211ULL;

int64_t SteadyNowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool EnvFlagEnabled(const char *name) {
  const char *value = std::getenv(name);
  if (!value || !*value)
    return false;
  std::string s(value);
  std::transform(s.begin(), s.end(), s.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return s != "0" && s != "false" && s != "off" && s != "no";
}

int EnvInt(const char *name, int fallback, int min_value, int max_value) {
  const char *value = std::getenv(name);
  if (!value || !*value)
    return fallback;
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value)
    return fallback;
  return std::clamp(static_cast<int>(parsed), min_value, max_value);
}

bool HasMatingMaterial(const Position &pos) {
  if (pos.pieces(PAWN) || pos.pieces(ROOK) || pos.pieces(QUEEN))
    return true;

  const Bitboard knights = pos.pieces(KNIGHT);
  const Bitboard bishops = pos.pieces(BISHOP);
  if (popcount(knights | bishops) < 2)
    return false; // K vs K, K+B vs K, K+N vs K.

  if (knights)
    return true; // Any knight plus another minor can produce mate.

  // Only kings and bishops remain. Bishop-only material can mate only when
  // at least one bishop exists on each square colour.
  constexpr Bitboard LightSquares = 0x55AA55AA55AA55AAULL;
  constexpr Bitboard DarkSquares = 0xAA55AA55AA55AA55ULL;
  return (bishops & LightSquares) && (bishops & DarkSquares);
}

bool ShouldAdjudicateRepetitionDraw(const Position &pos, int plies_from_root,
                                    bool two_fold_draws, float *moves_left_out,
                                    Node::Terminal *terminal_out) {
  const int rep = pos.repetition_distance();
  if (rep == 0)
    return false;

  if (rep < 0) {
    if (moves_left_out)
      *moves_left_out = 0.0f;
    if (terminal_out)
      *terminal_out = Node::Terminal::EndOfGame;
    return true;
  }

  if (!two_fold_draws)
    return false;

  if (plies_from_root >= 4 && plies_from_root >= rep) {
    if (moves_left_out)
      *moves_left_out = static_cast<float>(rep);
    if (terminal_out)
      *terminal_out = Node::Terminal::TwoFold;
    return true;
  }

  return false;
}

void ApplyNNPolicyToNode(Node *node, const EvaluationResult &result,
                         float softmax_temp) {
  const int num_edges = node->NumEdges();
  if (num_edges == 0)
    return;

  const float inv_temp = 1.0f / softmax_temp;
  const int n = std::min(num_edges, kMaxEdges);

  float logits_buf[kMaxEdges];
  float priors_buf[kMaxEdges];

  Edge *edges = node->Edges();
  const bool policy_order_matches =
      result.policy_priors.size() == static_cast<size_t>(n) && [&]() {
        for (int i = 0; i < n; ++i) {
          if (result.policy_priors[i].first != edges[i].move)
            return false;
        }
        return true;
      }();

#ifdef __APPLE__
  for (int i = 0; i < n; ++i) {
    logits_buf[i] = policy_order_matches ? result.policy_priors[i].second
                                         : result.get_policy(edges[i].move);
  }

  float max_logit;
  vDSP_maxv(logits_buf, 1, &max_logit, static_cast<vDSP_Length>(n));
  float neg_max = -max_logit;
  vDSP_vsadd(logits_buf, 1, &neg_max, logits_buf, 1,
             static_cast<vDSP_Length>(n));
  vDSP_vsmul(logits_buf, 1, &inv_temp, logits_buf, 1,
             static_cast<vDSP_Length>(n));
  int vn = n;
  vvexpf(priors_buf, logits_buf, &vn);
  float sum;
  vDSP_sve(priors_buf, 1, &sum, static_cast<vDSP_Length>(n));
#else
  float max_logit = -std::numeric_limits<float>::infinity();
  for (int i = 0; i < n; ++i) {
    logits_buf[i] = policy_order_matches ? result.policy_priors[i].second
                                         : result.get_policy(edges[i].move);
    if (logits_buf[i] > max_logit)
      max_logit = logits_buf[i];
  }
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    priors_buf[i] = FastMath::FastExp((logits_buf[i] - max_logit) * inv_temp);
    sum += priors_buf[i];
  }
#endif

  if (sum <= 0.0f) {
    const float uniform = 1.0f / static_cast<float>(n);
    for (int i = 0; i < n; ++i)
      edges[i].SetP(uniform);
    return;
  }

  const float inv_sum = 1.0f / sum;
#ifdef __APPLE__
  vDSP_vsmul(priors_buf, 1, &inv_sum, priors_buf, 1,
             static_cast<vDSP_Length>(n));
#else
  for (int i = 0; i < n; ++i)
    priors_buf[i] *= inv_sum;
#endif

  for (int i = 0; i < n; ++i)
    edges[i].SetP(priors_buf[i]);
  node->SortEdges();
}

int ResolveAutoMinibatchSize(const SearchParams &params,
                             const NN::BackendCapabilities &capabilities) {
  if (capabilities.actual_backend == "cuda") {
    if (capabilities.stable_execution_batch_size > 0) {
      return std::clamp(capabilities.stable_execution_batch_size, 1, 256);
    }
    if (params.cuda_stable_execution_batch_size > 0)
      return std::clamp(params.cuda_stable_execution_batch_size, 1, 256);
    return 16;
  }

#ifdef __APPLE__
  return 1;
#else
  return params.GetNumThreads() >= 8 ? 64 : 32;
#endif
}

bool ShouldReplayWarmCudaGraph(const SearchParams &params,
                               const NN::BackendCapabilities &capabilities) {
#ifdef USE_CUDA
  if (!params.cuda_graph_execution)
    return false;
  return capabilities.actual_backend == "cuda";
#else
  (void)params;
  (void)capabilities;
  return false;
#endif
}

} // anonymous namespace

Search::Search(const SearchParams &params, std::unique_ptr<Backend> backend)
    : params_(params), backend_(std::move(backend)) {
  if (!backend_) {
    std::string path = params_.nn_weights_path;
    if (path.empty()) {
      const char *env = std::getenv("METALFISH_NN_WEIGHTS");
      if (env)
        path = env;
    }
    if (!path.empty()) {
      try {
        backend_ = std::make_unique<Backend>(
            path, static_cast<size_t>(std::max(1, params_.nn_cache_size)),
            params_.GetBackendConfig());
        std::cerr << "[MCTS] Loaded transformer weights: " << path << std::endl;
      } catch (const std::exception &e) {
        std::cerr << "[MCTS] Failed to load weights (" << path
                  << "): " << e.what() << std::endl;
      }
    } else {
      std::cerr << "[MCTS] WARNING: No transformer weights path set. "
                << "Set via UCI option NNWeights or env "
                << "METALFISH_NN_WEIGHTS." << std::endl;
    }
  }

  if (backend_) {
    const NN::BackendCapabilities backend_capabilities =
        backend_->GetBackendCapabilities();
    if (params_.minibatch_size_auto) {
      const int resolved_minibatch =
          ResolveAutoMinibatchSize(params_, backend_capabilities);
      if (resolved_minibatch != params_.minibatch_size) {
        std::cerr << "[MCTS] Resolved auto minibatch: initial="
                  << params_.minibatch_size << " actual=" << resolved_minibatch
                  << " backend=" << backend_capabilities.actual_backend
                  << std::endl;
      }
      params_.minibatch_size = resolved_minibatch;
    }
    Position warmup_pos;
    StateInfo warmup_st;
    warmup_pos.set("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
                   false, &warmup_st);
    const uint64_t warmup_base = warmup_pos.raw_key();
    const bool replay_warm_cuda_graph =
        ShouldReplayWarmCudaGraph(params_, backend_capabilities);
    auto warmup_batch = [&](int batch_size, int passes, uint64_t salt,
                            bool update_latency_margin) {
      for (int pass = 0; pass < passes; ++pass) {
        auto comp = backend_->CreateComputation();
        for (int i = 0; i < batch_size; ++i) {
          comp->AddInput(
              warmup_pos,
              warmup_base ^
                  (salt + static_cast<uint64_t>(pass) * 0x9e3779b97f4a7c15ULL +
                   static_cast<uint64_t>(i) * kFNVPrime));
        }
        const auto batch_start = std::chrono::steady_clock::now();
        comp->ComputeBlocking();
        if (update_latency_margin) {
          const auto batch_elapsed =
              std::chrono::duration_cast<std::chrono::milliseconds>(
                  std::chrono::steady_clock::now() - batch_start)
                  .count();
          UpdateBackendLatencyMargin(batch_elapsed);
        }
      }
    };

    warmup_batch(1, replay_warm_cuda_graph ? 2 : 1, 0x9e3779b97f4a7c15ULL,
                 false);
    if (params_.minibatch_size > 1) {
      const int warmup_batch_size =
          replay_warm_cuda_graph ? std::clamp(params_.minibatch_size, 1, 256)
                                 : std::clamp(params_.minibatch_size, 8, 256);
      const int warmup_passes = replay_warm_cuda_graph ? 3 : 1;
      warmup_batch(warmup_batch_size, warmup_passes, 0xd1b54a32d192ed03ULL,
                   true);
    }
    if (replay_warm_cuda_graph) {
      std::cerr << "info string MCTS backend warmup actual="
                << backend_->GetNetworkInfo() << std::endl;
    }
    backend_->Cache().Clear();
  }
}

Search::~Search() {
  Stop();
  Wait();
}

void Search::NewGame() {
  Stop();
  Wait();
  ClearCallbacks();

  if (backend_)
    backend_->Cache().Clear();

  stats_.reset();
  tmgr_ = TimeManagerState{};
  nodes_at_movestart_.store(0, std::memory_order_release);
  batches_at_movestart_.store(0, std::memory_order_release);
  ponder_mode_active_.store(false, std::memory_order_release);
  first_eval_time_ms_.store(-1, std::memory_order_release);
  latest_hints_.Reset();
  root_search_moves_.clear();
  active_root_search_moves_.clear();
  root_search_filter_active_ = false;
  active_root_search_filter_active_ = false;

  std::unique_lock<std::shared_mutex> lock(tree_structure_mutex_);
  root_visit_baseline_.clear();
  tree_.Reset("");
}

void Search::StartSearch(const std::string &fen,
                         const ::MetalFish::Search::LimitsType &limits,
                         BestMoveCallback best_cb, InfoCallback info_cb) {
  Stop();
  Wait();

  stats_.reset();
  nodes_at_movestart_.store(0, std::memory_order_release);
  batches_at_movestart_.store(0, std::memory_order_release);
  first_eval_time_ms_.store(-1, std::memory_order_release);
  stop_flag_.store(false, std::memory_order_release);
  running_.store(true, std::memory_order_release);
  limits_ = limits;
  ponder_mode_active_.store(limits.ponderMode, std::memory_order_release);
  best_move_cb_ = best_cb;
  info_cb_ = info_cb;
  search_start_ms_.store(SteadyNowMs(), std::memory_order_release);

  {
    Position root_pos;
    StateInfo root_st;
    root_pos.set(fen, false, &root_st);
    root_color_ = root_pos.side_to_move();
    BuildRootSearchMoves(root_pos);
  }

  const bool same_tree_root = tree_.RootFen() == fen;
  const bool root_filter_changed =
      root_search_filter_active_ != active_root_search_filter_active_ ||
      root_search_moves_ != active_root_search_moves_;
  const bool needs_filtered_root_reset =
      root_filter_changed && (same_tree_root || root_search_filter_active_);

  {
    std::unique_lock<std::shared_mutex> lock(tree_structure_mutex_);
    if (needs_filtered_root_reset || !tree_.TryReuse(fen)) {
      tree_.Reset(fen);
    } else {
      std::function<void(Node *, int)> fixTwoFold = [&](Node *node, int depth) {
        if (!node || depth > 50)
          return;
        node->MaybeRevertTwoFold(depth);
        if (node->NumEdges() > 0) {
          Edge *edges = node->Edges();
          for (int i = 0; i < node->NumEdges(); ++i) {
            Node *child = edges[i].child.load(std::memory_order_acquire);
            if (child)
              fixTwoFold(child, depth + 1);
          }
        }
      };
      fixTwoFold(tree_.Root(), 0);
    }
    CaptureRootVisitBaselineLocked();
  }
  active_root_search_filter_active_ = root_search_filter_active_;
  active_root_search_moves_ = root_search_moves_;

  if (params_.contempt != 0.0f) {
    params_.draw_score = -params_.contempt / 10000.0f;
  }

  time_budget_ms_.store(CalculateTimeBudget(), std::memory_order_release);

  gathering_permit_.store(1, std::memory_order_relaxed);
  backend_waiting_.store(0, std::memory_order_relaxed);

  ConfigureStopper();

  int num_threads = params_.GetNumThreads();
  worker_ctxs_.clear();
  workers_.clear();

  for (int i = 0; i < num_threads; ++i) {
    auto ctx = std::make_unique<SearchWorkerCtx>();
    ctx->worker_id = i;
    worker_ctxs_.push_back(std::move(ctx));
  }

  for (int i = 0; i < num_threads; ++i) {
    workers_.emplace_back(&Search::WorkerThreadMain, this, i);
  }
}

void Search::Stop() { stop_flag_.store(true, std::memory_order_release); }

void Search::PonderHit() {
  if (!running_.load(std::memory_order_acquire))
    return;

  if (!ponder_mode_active_.exchange(false, std::memory_order_acq_rel))
    return;

  search_start_ms_.store(SteadyNowMs(), std::memory_order_release);
  nodes_at_movestart_.store(stats_.total_nodes.load(std::memory_order_relaxed),
                            std::memory_order_release);
  batches_at_movestart_.store(
      stats_.total_batches.load(std::memory_order_relaxed),
      std::memory_order_release);
  first_eval_time_ms_.store(-1, std::memory_order_release);
  time_budget_ms_.store(CalculateTimeBudget(), std::memory_order_release);
  ConfigureStopper();
}

bool MCTSIsKingsidePawnLever(const Position &pos, Move move) {
  if (move == Move::none() || move.type_of() != NORMAL)
    return false;

  const Square from = move.from_sq();
  const Square to = move.to_sq();
  const Piece piece = pos.piece_on(from);
  if (piece == NO_PIECE || type_of(piece) != PAWN || !pos.empty(to))
    return false;
  if (file_of(from) < FILE_F || file_of(from) != file_of(to))
    return false;

  const Color us = color_of(piece);
  const Direction push = pawn_push(us);
  if (to != from + push && to != from + push + push)
    return false;
  if (relative_rank(us, to) < RANK_4)
    return false;
  return bool(attacks_bb<PAWN>(to, us) & pos.pieces(~us, PAWN));
}

bool MCTSIsCentralPawnBreak(const Position &pos, Move move) {
  if (move == Move::none() || move.type_of() != NORMAL)
    return false;

  const Square from = move.from_sq();
  const Square to = move.to_sq();
  const Piece piece = pos.piece_on(from);
  if (piece == NO_PIECE || type_of(piece) != PAWN || !pos.empty(to))
    return false;
  if (file_of(from) != file_of(to))
    return false;

  const Color us = color_of(piece);
  if (to != from + pawn_push(us))
    return false;

  const File target_file = file_of(to);
  if (target_file != FILE_D && target_file != FILE_E)
    return false;
  if (relative_rank(us, to) != RANK_5)
    return false;

  return bool(attacks_bb<PAWN>(to, us) & pos.pieces(~us, PAWN));
}

bool MCTSIsMinorCentralPawnCapture(const Position &pos, Move move) {
  if (move == Move::none() || move.type_of() != NORMAL || !pos.capture(move))
    return false;

  const Piece piece = pos.piece_on(move.from_sq());
  const Piece captured = pos.piece_on(move.to_sq());
  if (piece == NO_PIECE || captured == NO_PIECE)
    return false;

  const PieceType pt = type_of(piece);
  if (pt != KNIGHT && pt != BISHOP)
    return false;
  if (type_of(captured) != PAWN)
    return false;

  const File target_file = file_of(move.to_sq());
  return target_file == FILE_D || target_file == FILE_E;
}

bool MCTSIsMinorKingPawnCheckCapture(const Position &pos, Move move) {
  if (move == Move::none() || move.type_of() != NORMAL || !pos.capture(move))
    return false;

  const Piece piece = pos.piece_on(move.from_sq());
  const Piece captured = pos.piece_on(move.to_sq());
  if (piece == NO_PIECE || captured == NO_PIECE)
    return false;

  const PieceType pt = type_of(piece);
  if (pt != KNIGHT && pt != BISHOP)
    return false;
  if (type_of(captured) != PAWN)
    return false;
  if (file_of(move.to_sq()) != FILE_F)
    return false;
  if (relative_rank(color_of(piece), move.to_sq()) != RANK_7)
    return false;

  return pos.gives_check(move);
}

bool MCTSIsMinorHighValueCapture(const Position &pos, Move move) {
  if (move == Move::none() || move.type_of() != NORMAL || !pos.capture(move))
    return false;

  const Piece piece = pos.piece_on(move.from_sq());
  const Piece captured = pos.piece_on(move.to_sq());
  if (piece == NO_PIECE || captured == NO_PIECE)
    return false;

  const PieceType pt = type_of(piece);
  if (pt != KNIGHT && pt != BISHOP)
    return false;

  const PieceType captured_pt = type_of(captured);
  return captured_pt == ROOK || captured_pt == QUEEN;
}

bool MCTSIsMinorCentralQuietMove(const Position &pos, Move move) {
  if (move == Move::none() || move.type_of() != NORMAL || pos.capture(move))
    return false;

  const Square from = move.from_sq();
  const Square to = move.to_sq();
  const Piece piece = pos.piece_on(from);
  if (piece == NO_PIECE || !pos.empty(to))
    return false;

  const PieceType pt = type_of(piece);
  if (pt != KNIGHT && pt != BISHOP)
    return false;

  const File target_file = file_of(to);
  if (target_file != FILE_D && target_file != FILE_E)
    return false;

  const Rank target_rank = relative_rank(color_of(piece), to);
  return target_rank >= RANK_4 && target_rank <= RANK_5;
}

bool MCTSIsMinorQuietAttacksMajor(const Position &pos, Move move) {
  if (move == Move::none() || move.type_of() != NORMAL || pos.capture(move))
    return false;

  const Square from = move.from_sq();
  const Square to = move.to_sq();
  const Piece piece = pos.piece_on(from);
  if (piece == NO_PIECE || !pos.empty(to))
    return false;

  const PieceType pt = type_of(piece);
  if (pt != KNIGHT && pt != BISHOP)
    return false;

  const Bitboard occupied_after =
      (pos.pieces() ^ square_bb(from)) | square_bb(to);
  return bool(attacks_bb(pt, to, occupied_after) &
              pos.pieces(~color_of(piece), ROOK, QUEEN));
}

bool MCTSIsMinorQuietAttacksQueen(const Position &pos, Move move) {
  if (move == Move::none() || move.type_of() != NORMAL || pos.capture(move))
    return false;

  const Square from = move.from_sq();
  const Square to = move.to_sq();
  const Piece piece = pos.piece_on(from);
  if (piece == NO_PIECE || !pos.empty(to))
    return false;

  const PieceType pt = type_of(piece);
  if (pt != KNIGHT && pt != BISHOP)
    return false;

  const Bitboard occupied_after =
      (pos.pieces() ^ square_bb(from)) | square_bb(to);
  return bool(attacks_bb(pt, to, occupied_after) &
              pos.pieces(~color_of(piece), QUEEN));
}

bool MCTSIsMinorFifthRankQuietMove(const Position &pos, Move move) {
  if (move == Move::none() || move.type_of() != NORMAL || pos.capture(move))
    return false;

  const Square to = move.to_sq();
  const Piece piece = pos.piece_on(move.from_sq());
  if (piece == NO_PIECE || !pos.empty(to))
    return false;

  const PieceType pt = type_of(piece);
  if (pt != KNIGHT && pt != BISHOP)
    return false;

  const File target_file = file_of(to);
  if (target_file != FILE_C && target_file != FILE_F)
    return false;

  return relative_rank(color_of(piece), to) == RANK_5;
}

bool MCTSHasHeavyPieceOnSeventh(const Position &pos, Color us) {
  Bitboard heavy = pos.pieces(us, ROOK, QUEEN);
  while (heavy) {
    if (relative_rank(us, pop_lsb(heavy)) == RANK_7)
      return true;
  }
  return false;
}

bool MCTSIsAdvancedPromotionSupportQueenMove(const Position &pos, Move move) {
  if (move == Move::none() || move.type_of() != NORMAL || pos.capture(move))
    return false;

  const Piece piece = pos.piece_on(move.from_sq());
  if (piece == NO_PIECE || type_of(piece) != QUEEN || !pos.empty(move.to_sq()))
    return false;

  const Color us = color_of(piece);
  const File target_file = file_of(move.to_sq());
  const Rank target_rank = relative_rank(us, move.to_sq());
  if (target_rank > RANK_5)
    return false;

  Bitboard pawns = pos.pieces(us, PAWN);
  while (pawns) {
    const Square pawn_sq = pop_lsb(pawns);
    if (file_of(pawn_sq) == target_file &&
        relative_rank(us, pawn_sq) == RANK_7) {
      return true;
    }
  }
  return false;
}

bool MCTSIsQuietQueenCheck(const Position &pos, Move move) {
  if (move == Move::none() || move.type_of() != NORMAL || pos.capture(move))
    return false;

  const Piece piece = pos.piece_on(move.from_sq());
  if (piece == NO_PIECE || type_of(piece) != QUEEN || !pos.empty(move.to_sq()))
    return false;

  return pos.gives_check(move);
}

bool MCTSIsQuietQueenKingNetMove(const Position &pos, Move move) {
  if (move == Move::none() || move.type_of() != NORMAL || pos.capture(move))
    return false;

  const Piece piece = pos.piece_on(move.from_sq());
  if (piece == NO_PIECE || type_of(piece) != QUEEN || !pos.empty(move.to_sq()))
    return false;
  if (pos.gives_check(move))
    return false;

  const Color us = color_of(piece);
  const Square their_king = pos.square<KING>(~us);
  const Square to = move.to_sq();
  const int file_dist = std::abs(static_cast<int>(file_of(to)) -
                                 static_cast<int>(file_of(their_king)));
  const int rank_dist = std::abs(static_cast<int>(rank_of(to)) -
                                 static_cast<int>(rank_of(their_king)));
  if (std::max(file_dist, rank_dist) > 2)
    return false;

  const Bitboard occupied_after =
      (pos.pieces() ^ square_bb(move.from_sq())) | square_bb(move.to_sq());
  const Bitboard king_zone = attacks_bb<KING>(their_king);
  return bool(attacks_bb<QUEEN>(to, occupied_after) & king_zone);
}

bool MCTSRootHighPolicyLeverCandidate(uint32_t root_visits,
                                      uint32_t best_visits,
                                      uint32_t candidate_visits,
                                      float best_policy, float best_q,
                                      float candidate_policy,
                                      float candidate_q) {
  const uint32_t safe_best_visits = std::max<uint32_t>(1, best_visits);
  bool underexplored = false;
  if (root_visits <= 80) {
    underexplored = static_cast<uint64_t>(candidate_visits) * 5 <=
                    static_cast<uint64_t>(safe_best_visits) * 3;
  } else if (root_visits <= 120) {
    underexplored = static_cast<uint64_t>(candidate_visits) * 2 <=
                    static_cast<uint64_t>(safe_best_visits);
  } else {
    underexplored = static_cast<uint64_t>(candidate_visits) * 5 <=
                    static_cast<uint64_t>(safe_best_visits) * 2;
  }
  const float q_gap = best_q - candidate_q;
  const bool near_visit_policy_tie =
      root_visits >= 240 && root_visits <= 850 &&
      static_cast<uint64_t>(candidate_visits) * 100 >=
          static_cast<uint64_t>(safe_best_visits) * 84 &&
      candidate_policy >= 0.20f && candidate_policy >= best_policy * 1.80f &&
      q_gap <= 0.06f;
  if (best_q >= 0.0f && candidate_q < 0.0f)
    return false;
  if (root_visits >= 24 && root_visits < 40) {
    return candidate_visits >= 6 && underexplored &&
           candidate_policy >= 0.20f &&
           candidate_policy >= best_policy * 1.20f && q_gap <= 0.18f;
  }
  return root_visits >= 40 && candidate_visits >= 8 &&
         ((root_visits <= 600 && underexplored) || near_visit_policy_tie) &&
         candidate_policy >= 0.20f && candidate_policy >= best_policy * 1.15f &&
         q_gap <= 0.20f;
}

bool MCTSRootLowPolicyLeverCandidate(uint32_t root_visits, uint32_t best_visits,
                                     uint32_t candidate_visits,
                                     int candidate_rank, float best_policy,
                                     float best_q, float candidate_policy,
                                     float candidate_q) {
  const bool rank_ok = (candidate_rank >= 5 && candidate_rank <= 6) ||
                       (candidate_rank == 7 && candidate_visits >= 2 &&
                        candidate_policy >= 0.045f) ||
                       (candidate_rank == 4 && candidate_visits >= 2 &&
                        candidate_policy >= 0.035f);
  if (root_visits >= 24 && root_visits < 40) {
    const bool tiny_rank_ok =
        (candidate_rank >= 5 && candidate_rank <= 6) ||
        (candidate_rank == 7 && candidate_policy >= 0.045f) ||
        (candidate_rank == 4 && candidate_policy >= 0.035f);
    if (!tiny_rank_ok || candidate_visits == 0)
      return false;

    const float q_gap_limit = candidate_visits >= 2 ? 0.09f : 0.05f;
    return static_cast<uint64_t>(candidate_visits) * 2 <
               std::max<uint32_t>(1, best_visits) &&
           candidate_policy >= 0.035f && candidate_policy <= 0.08f &&
           candidate_policy <= best_policy * 0.35f &&
           best_q - candidate_q <= q_gap_limit;
  }
  return root_visits >= 40 && root_visits <= 600 && rank_ok &&
         candidate_visits >= 2 &&
         candidate_visits * 3 < std::max<uint32_t>(1, best_visits) &&
         candidate_policy >= 0.035f && candidate_policy <= 0.08f &&
         candidate_policy <= best_policy * 0.35f &&
         best_q - candidate_q <= 0.09f;
}

bool MCTSRootLowPolicyLeverProbeCandidate(uint32_t root_visits,
                                          int candidate_policy_rank,
                                          float candidate_policy) {
  const bool rank_ok =
      (candidate_policy_rank >= 5 && candidate_policy_rank <= 6) ||
      (candidate_policy_rank == 7 && candidate_policy >= 0.045f) ||
      (candidate_policy_rank == 4 && candidate_policy >= 0.035f);
  return root_visits >= 16 && root_visits <= 80 && rank_ok &&
         candidate_policy >= 0.035f && candidate_policy <= 0.08f;
}

bool MCTSRootCentralPawnBreakProbeCandidate(uint32_t root_visits,
                                            int candidate_policy_rank,
                                            float candidate_policy) {
  return root_visits >= 16 && root_visits <= 80 && candidate_policy_rank >= 4 &&
         candidate_policy_rank <= 8 && candidate_policy >= 0.035f &&
         candidate_policy <= 0.080f;
}

bool MCTSRootTinyLowVisitQOverrideCandidate(uint32_t root_visits,
                                            uint32_t best_visits,
                                            uint32_t candidate_visits,
                                            float best_policy, float best_q,
                                            float candidate_policy,
                                            float candidate_q) {
  return root_visits >= 24 && root_visits <= 32 && candidate_visits >= 4 &&
         candidate_visits * 2 >= std::max<uint32_t>(1, best_visits) &&
         best_q < 0.65f && candidate_policy >= 0.045f &&
         candidate_policy <= best_policy * 0.65f &&
         candidate_q > best_q + 0.020f;
}

bool MCTSRootTacticalCaptureProbeCandidate(uint32_t root_visits,
                                           int candidate_policy_rank,
                                           float candidate_policy) {
  return root_visits >= 16 && root_visits <= 600 &&
         candidate_policy_rank >= 10 && candidate_policy_rank <= 16 &&
         candidate_policy >= 0.006f && candidate_policy <= 0.020f;
}

bool MCTSRootHighValueCaptureProbeCandidate(uint32_t root_visits,
                                            int candidate_policy_rank,
                                            float candidate_policy) {
  return root_visits >= 24 && root_visits <= 180 &&
         candidate_policy_rank >= 2 && candidate_policy_rank <= 8 &&
         candidate_policy >= 0.045f && candidate_policy <= 0.200f;
}

bool MCTSRootTacticalQuietProbeCandidate(uint32_t root_visits,
                                         int candidate_policy_rank,
                                         float candidate_policy) {
  return root_visits >= 16 && root_visits <= 80 && candidate_policy_rank >= 5 &&
         candidate_policy_rank <= 10 && candidate_policy >= 0.025f &&
         candidate_policy <= 0.070f;
}

bool MCTSRootQuietMajorAttackProbeCandidate(uint32_t root_visits,
                                            int candidate_policy_rank,
                                            float candidate_policy) {
  if (root_visits < 32 || root_visits > 180 || candidate_policy_rank < 1 ||
      candidate_policy_rank > 10)
    return false;

  if (candidate_policy_rank <= 4)
    return candidate_policy >= 0.080f && candidate_policy <= 0.360f;

  return candidate_policy >= 0.018f && candidate_policy <= 0.080f;
}

bool MCTSRootDeepTacticalQuietProbeCandidate(uint32_t root_visits,
                                             int candidate_policy_rank,
                                             float candidate_policy) {
  if (root_visits < 32 || root_visits > 180 || candidate_policy_rank < 2 ||
      candidate_policy_rank > 10)
    return false;

  if (candidate_policy_rank <= 4)
    return candidate_policy >= 0.080f && candidate_policy <= 0.180f;

  return candidate_policy >= 0.018f && candidate_policy <= 0.080f;
}

bool MCTSRootFifthRankQuietProbeCandidate(uint32_t root_visits,
                                          int candidate_policy_rank,
                                          float candidate_policy) {
  if (root_visits < 32 || root_visits > 180 || candidate_policy_rank < 2 ||
      candidate_policy_rank > 12)
    return false;

  if (candidate_policy_rank <= 4)
    return candidate_policy >= 0.080f && candidate_policy <= 0.180f;

  return candidate_policy >= 0.018f && candidate_policy <= 0.080f;
}

bool MCTSRootFifthRankCurrentOverrideCandidate(
    uint32_t root_visits, uint32_t best_current_visits,
    uint32_t candidate_current_visits, float best_q, float candidate_q,
    float candidate_policy) {
  return root_visits >= 64 && root_visits <= 220 &&
         candidate_current_visits >= 32 &&
         candidate_current_visits >= best_current_visits + 12 &&
         candidate_policy >= 0.018f && candidate_policy <= 0.080f &&
         candidate_q > best_q + 0.020f;
}

bool MCTSRootQuietQueenCheckProbeCandidate(uint32_t root_visits,
                                           int candidate_policy_rank,
                                           float candidate_policy) {
  if (root_visits < 16 || root_visits > 600 || candidate_policy_rank < 2 ||
      candidate_policy_rank > 32)
    return false;

  if (candidate_policy_rank <= 8)
    return candidate_policy >= 0.020f && candidate_policy <= 0.220f;

  return candidate_policy >= 0.004f && candidate_policy <= 0.080f;
}

bool MCTSRootQuietQueenCheckProbeStillViable(uint32_t candidate_visits,
                                             float candidate_q) {
  return candidate_visits == 0 || candidate_q >= 0.0f;
}

bool MCTSRootQuietQueenKingNetProbeCandidate(uint32_t root_visits,
                                             int candidate_policy_rank,
                                             float candidate_policy) {
  return root_visits >= 16 && root_visits <= 180 &&
         candidate_policy_rank >= 2 && candidate_policy_rank <= 8 &&
         candidate_policy >= 0.080f && candidate_policy <= 0.220f;
}

bool MCTSRootAdvancedPromotionSupportCandidate(uint32_t root_visits,
                                               uint32_t best_visits,
                                               uint32_t candidate_visits,
                                               float best_policy, float best_q,
                                               float candidate_policy,
                                               float candidate_q) {
  return root_visits >= 48 && root_visits <= 220 && candidate_visits >= 16 &&
         candidate_visits * 2 >= std::max<uint32_t>(1, best_visits) &&
         candidate_policy >= 0.120f &&
         candidate_policy >= best_policy * 1.25f &&
         best_q - candidate_q <= 0.150f;
}

bool MCTSRootPawnEndgameEnPassantCandidate(uint32_t root_visits,
                                           uint32_t best_visits,
                                           uint32_t candidate_visits,
                                           bool best_is_capture,
                                           bool candidate_is_en_passant,
                                           float best_q, float candidate_q) {
  if (best_is_capture || !candidate_is_en_passant)
    return false;
  if (root_visits < 16 || root_visits > 384)
    return false;
  if (candidate_visits < 2)
    return false;
  if (static_cast<uint64_t>(candidate_visits) * 8 <
      static_cast<uint64_t>(std::max<uint32_t>(1, best_visits))) {
    return false;
  }
  return best_q - candidate_q <= 0.080f;
}

bool MCTSRootMinorPawnEndgameCaptureProtected(
    const Position &pos, Move best_move, Move candidate_move, float best_policy,
    float best_q, float candidate_policy, float candidate_q) {
  if (pos.count<QUEEN>() != 0 || pos.count<ROOK>() != 0 ||
      pos.count<KNIGHT>() != 0 || pos.count<BISHOP>() > 1) {
    return false;
  }
  if (!pos.capture(best_move) || pos.capture(candidate_move))
    return false;

  const Piece best_piece = pos.piece_on(best_move.from_sq());
  const Piece captured_piece = pos.piece_on(best_move.to_sq());
  if (best_piece == NO_PIECE || type_of(best_piece) != PAWN ||
      captured_piece == NO_PIECE || type_of(captured_piece) != PAWN) {
    return false;
  }

  return best_policy >= 0.55f && candidate_policy <= best_policy * 0.45f &&
         candidate_q <= best_q + 0.13f;
}

bool MCTSRootLowVisitQOverrideCandidate(uint32_t best_visits,
                                        uint32_t candidate_visits, float best_q,
                                        float candidate_q,
                                        float near_equal_required_gap,
                                        float candidate_policy,
                                        bool allow_strong_gap_candidate) {
  const bool near_visit_candidate =
      candidate_visits >= 6 && candidate_visits * 2 >= best_visits;
  const bool decisive_low_visit_candidate = candidate_visits >= 3 &&
                                            candidate_q >= 0.95f &&
                                            candidate_q > best_q + 0.50f;
  const bool strong_gap_candidate =
      allow_strong_gap_candidate && candidate_visits >= 16 &&
      candidate_visits * 3 >= std::max<uint32_t>(1, best_visits) &&
      candidate_policy >= 0.045f && candidate_q > best_q + 0.35f;
  if (!near_visit_candidate && !decisive_low_visit_candidate &&
      !strong_gap_candidate)
    return false;

  const float required_gap =
      decisive_low_visit_candidate
          ? 0.0f
          : (strong_gap_candidate
                 ? 0.0f
                 : (candidate_visits * 9 >= best_visits * 8
                        ? near_equal_required_gap
                        : (candidate_visits * 4 >= best_visits * 3 ? 0.05f
                                                                   : 0.07f)));
  return candidate_q > best_q + required_gap;
}

bool MCTSRootHighPolicyVisitLeaderProtected(uint32_t best_visits,
                                            uint32_t candidate_visits,
                                            float best_policy, float best_q,
                                            float candidate_policy,
                                            float candidate_q) {
  if (best_visits < 32 || candidate_visits < 8)
    return false;
  if (candidate_visits * 2 < std::max<uint32_t>(1, best_visits))
    return false;
  if (best_policy < 0.250f || candidate_policy > best_policy * 0.180f)
    return false;
  if (candidate_q >= 0.950f && candidate_q > best_q + 0.500f)
    return false;
  return candidate_q - best_q <= 0.250f;
}

bool MCTSRootClockLowVisitQOverrideCandidate(uint32_t root_current_visits,
                                             uint32_t best_current_visits,
                                             uint32_t candidate_current_visits,
                                             float best_q, float candidate_q,
                                             float candidate_policy) {
  if (root_current_visits < 24 || root_current_visits > 160)
    return false;
  if (best_current_visits == 0 && candidate_current_visits < 8)
    return false;

  return MCTSRootLowVisitQOverrideCandidate(
      std::max<uint32_t>(1, best_current_visits), candidate_current_visits,
      best_q, candidate_q, 0.02f, candidate_policy, false);
}

namespace {
float ExponentialDecay(float from, float to, float halflife_steps,
                       float steps) {
  return to - (to - from) * std::pow(0.5f, steps / halflife_steps);
}
} // namespace

void Search::Wait() {
  const bool had_active_search =
      running_.load(std::memory_order_acquire) || !workers_.empty();

  for (auto &t : workers_) {
    if (t.joinable())
      t.join();
  }
  workers_.clear();
  running_.store(false, std::memory_order_release);

  const int64_t elapsed_ms =
      SteadyNowMs() - search_start_ms_.load(std::memory_order_acquire);
  uint64_t total_nodes = stats_.total_nodes.load(std::memory_order_relaxed);
  const uint64_t nodes_offset =
      nodes_at_movestart_.load(std::memory_order_acquire);
  const uint64_t move_nodes =
      total_nodes > nodes_offset ? total_nodes - nodes_offset : total_nodes;

  if (elapsed_ms > 0 && move_nodes > 0) {
    float actual_nps = 1000.0f * move_nodes / elapsed_ms;
    if (tmgr_.nps_reliable) {
      tmgr_.nps = tmgr_.nps * 0.5f + actual_nps * 0.5f;
    } else {
      tmgr_.nps = actual_nps;
      tmgr_.nps_reliable = true;
    }
  }

  if (tmgr_.move_allocated_time_ms > 0 && elapsed_ms > 0) {
    float actual_timeuse =
        static_cast<float>(elapsed_ms) / tmgr_.move_allocated_time_ms;
    float update_rate =
        tmgr_.avg_ms_per_move > 0
            ? static_cast<float>(elapsed_ms) / tmgr_.avg_ms_per_move
            : 1.0f;
    tmgr_.timeuse =
        ExponentialDecay(tmgr_.timeuse, actual_timeuse, 5.51f, update_rate);
    tmgr_.timeuse = std::max(tmgr_.timeuse, 0.34f);
  }

  tmgr_.last_move_final_nodes = static_cast<int64_t>(total_nodes);

  if (had_active_search && stopper_) {
    SearchStats final_stats = CollectSearchStats();
    std::lock_guard<std::mutex> lock(stopper_mutex_);
    stopper_->OnSearchDone(final_stats);
  }

  if (had_active_search) {
    SendInfo();
  }

  auto cb_copy = best_move_cb_;
  best_move_cb_ = nullptr;

  if (cb_copy) {
    Move best;
    if (params_.temperature > 0.0f) {
      best = GetBestMoveWithTemperature(params_.temperature);
    } else {
      best = GetBestMove();
    }
    if (best == Move::none()) {
      best = FirstRootMoveOrLegal();
    }
    std::vector<Move> pv = GetPV();
    Move ponder = pv.size() > 1 ? pv[1] : Move::none();
    cb_copy(best, ponder);
  }
}

void Search::ClearCallbacks() {
  best_move_cb_ = nullptr;
  info_cb_ = nullptr;
}

bool Search::IsSearchActive() const {
  return running_.load(std::memory_order_acquire);
}

void Search::ConfigureStopper() {
  auto stopper = std::make_unique<ChainedStopper>();
  bool smart_pruning = false;

  if (!limits_.infinite &&
      !ponder_mode_active_.load(std::memory_order_acquire)) {
    const int64_t time_budget_ms =
        time_budget_ms_.load(std::memory_order_acquire);
    const bool node_only_limit =
        limits_.nodes > 0 && time_budget_ms <= 0 && limits_.movetime <= 0 &&
        limits_.time[WHITE] <= 0 && limits_.time[BLACK] <= 0;
    if (time_budget_ms > 0) {
      stopper->Add(std::make_unique<TimeLimitStopper>(time_budget_ms));
    }
    if (limits_.nodes > 0) {
      stopper->Add(std::make_unique<NodeLimitStopper>(limits_.nodes));
    }
    if (!node_only_limit && params_.kld_gain_min > 0.0f) {
      const int64_t kld_min_elapsed_ms =
          time_budget_ms > 0 ? std::min<int64_t>(500, (time_budget_ms * 2) / 3)
                             : 0;
      stopper->Add(std::make_unique<KLDGainStopper>(
          params_.kld_gain_min, params_.kld_gain_average_interval,
          kld_min_elapsed_ms));
    }
    if (!node_only_limit && params_.smart_pruning_factor > 0.0f) {
      stopper->Add(std::make_unique<SmartPruningStopper>(
          params_.smart_pruning_factor, params_.smart_pruning_minimum_batches));
      smart_pruning = true;
    }
  }

  {
    std::lock_guard<std::mutex> lock(stopper_mutex_);
    stopper_ = std::move(stopper);
    latest_hints_.Reset();
  }
  smart_pruning_enabled_.store(smart_pruning, std::memory_order_release);
}

namespace {

// Log-logistic remaining-moves estimate (Lc0 fitted parameters).
float EstimateMovesToGo(int ply, float midpoint = 45.2f,
                        float steepness = 5.93f) {
  float move = ply / 2.0f;
  return midpoint * std::pow(1.0f + 2.0f * std::pow(move / midpoint, steepness),
                             1.0f / steepness) -
         move;
}

} // namespace

int64_t Search::CalculateTimeBudget() {
  if (limits_.nodes > 0 && limits_.movetime <= 0 && !limits_.infinite &&
      limits_.time[WHITE] <= 0 && limits_.time[BLACK] <= 0) {
    return 0;
  }
  const int64_t overhead = static_cast<int64_t>(params_.move_overhead_ms);
  if (ponder_mode_active_.load(std::memory_order_acquire))
    return 0;
  if (limits_.movetime > 0)
    return std::max<int64_t>(1, limits_.movetime - overhead);
  if (limits_.infinite)
    return 0;

  Color us = root_color_;
  int64_t time_left = limits_.time[us];
  int64_t inc = limits_.inc[us];
  if (time_left <= 0)
    return std::max(int64_t(50), inc);

  if (time_left < 500) {
    return std::max(int64_t(50), std::min(time_left / 4, inc));
  }

  int ply = 2 * (limits_.movestogo > 0 ? 40 : 30);
  {
    Position pos;
    StateInfo st;
    std::string fen = tree_.RootFen();
    if (!fen.empty()) {
      pos.set(fen, false, &st);
      ply = pos.game_ply();
    }
  }

  if (params_.time_manager == "alphazero") {
    const float available =
        std::max(0.0f, static_cast<float>(time_left - overhead));
    return std::max<int64_t>(
        1, static_cast<int64_t>(std::min(
               available, available * params_.alphazero_time_pct / 100.0f)));
  }

  if (params_.time_manager == "simple") {
    const float available =
        std::max(0.0f, static_cast<float>(time_left - overhead));
    const float time_ratio =
        time_left > 0 ? static_cast<float>(inc) / static_cast<float>(time_left)
                      : 0.0f;
    float pct = (1.4f + static_cast<float>(ply) * 0.049f) * 0.01f;
    pct += time_ratio * 1.5f;
    return std::max<int64_t>(
        1, static_cast<int64_t>(std::min(available, available * pct)));
  }

  if (params_.time_manager == "legacy") {
    float moves_to_go = EstimateMovesToGo(ply, 51.5f, 7.0f);
    if (limits_.movestogo > 0 && limits_.movestogo < moves_to_go) {
      moves_to_go = static_cast<float>(limits_.movestogo);
    }
    moves_to_go = std::max(moves_to_go, 1.0f);

    const float total_moves_time =
        std::max(0.0f, static_cast<float>(time_left) +
                           static_cast<float>(std::max<int64_t>(0, inc)) *
                               (moves_to_go - 1.0f) -
                           static_cast<float>(overhead));
    float this_move_time = total_moves_time / moves_to_go;
    this_move_time *= params_.slowmover;
    return std::max<int64_t>(
        1, static_cast<int64_t>(std::min(
               this_move_time,
               std::max(0.0f, static_cast<float>(time_left - overhead)))));
  }

  // 9% of initial time (from Lc0)
  if (tmgr_.first_move) {
    tmgr_.piggybank_ms = static_cast<int64_t>(time_left * 0.09f);
    tmgr_.first_move = false;
  }

  uint32_t current_nodes = tree_.Root() ? tree_.Root()->GetN() : 0;
  if (tmgr_.last_move_final_nodes > 0 && current_nodes > 0) {
    float this_reuse = static_cast<float>(current_nodes) /
                       static_cast<float>(tmgr_.last_move_final_nodes);
    float update_rate =
        tmgr_.avg_ms_per_move > 0.0f
            ? tmgr_.move_allocated_time_ms / tmgr_.avg_ms_per_move
            : 1.0f;
    tmgr_.tree_reuse =
        ExponentialDecay(tmgr_.tree_reuse, this_reuse, 3.39f, update_rate);
    tmgr_.tree_reuse = std::min(tmgr_.tree_reuse, 0.73f);
  }

  float remaining_moves = EstimateMovesToGo(ply);
  if (limits_.movestogo > 0 && limits_.movestogo < remaining_moves) {
    remaining_moves = static_cast<float>(limits_.movestogo);
  }
  remaining_moves = std::max(remaining_moves, 1.0f);

  float max_piggybank =
      36.5f *
      std::max(0.0f, static_cast<float>(time_left) +
                         static_cast<float>(inc) * (remaining_moves - 1.0f) -
                         overhead) /
      remaining_moves;
  tmgr_.piggybank_ms =
      std::min(tmgr_.piggybank_ms, static_cast<int64_t>(max_piggybank));

  float total_remaining =
      std::max(0.0f, static_cast<float>(time_left) -
                         static_cast<float>(tmgr_.piggybank_ms) +
                         static_cast<float>(inc) * (remaining_moves - 1.0f) -
                         static_cast<float>(overhead));

  float remaining_game_nodes = total_remaining * tmgr_.nps / 1000.0f;
  float avg_nodes_per_move = remaining_game_nodes / remaining_moves;
  tmgr_.avg_ms_per_move = total_remaining / remaining_moves;

  float nodes_with_reuse = avg_nodes_per_move / (1.0f - tmgr_.tree_reuse);
  float new_nodes_needed = std::max(0.0f, nodes_with_reuse - current_nodes);

  float expected_ms = new_nodes_needed / tmgr_.nps * 1000.0f;

  // 12% to piggybank (from Lc0)
  float to_piggybank =
      std::min(max_piggybank - tmgr_.piggybank_ms, expected_ms * 0.12f);
  expected_ms -= to_piggybank;

  tmgr_.move_allocated_time_ms = expected_ms / tmgr_.timeuse;

  float max_move_time = static_cast<float>(time_left) * 0.42f;
  if (tmgr_.move_allocated_time_ms > max_move_time) {
    tmgr_.move_allocated_time_ms = max_move_time;
  }

  float hard_cap = static_cast<float>(time_left - overhead);
  if (tmgr_.move_allocated_time_ms > hard_cap) {
    tmgr_.move_allocated_time_ms = std::max(0.0f, hard_cap);
  }

  tmgr_.piggybank_ms += static_cast<int64_t>(to_piggybank);

  float min_time = std::min(50.0f, hard_cap / 3.0f);
  tmgr_.move_allocated_time_ms =
      std::max(tmgr_.move_allocated_time_ms, min_time);

  return std::max<int64_t>(1,
                           static_cast<int64_t>(tmgr_.move_allocated_time_ms));
}

SearchStats Search::CollectSearchStats() const {
  SearchStats stats;
  stats.time_since_movestart_ms =
      SteadyNowMs() - search_start_ms_.load(std::memory_order_acquire);

  int64_t latency_margin = 0;
  if (time_budget_ms_.load(std::memory_order_acquire) > 0 &&
      params_.GetNumThreads() > 1 && params_.minibatch_size > 1 &&
      stats_.nn_evaluations.load(std::memory_order_relaxed) > 0) {
    latency_margin = backend_latency_margin_ms_.load(std::memory_order_relaxed);
  }
  stats.time_since_movestart_ms += latency_margin;

  stats.total_nodes = stats_.total_nodes.load(std::memory_order_relaxed);
  const uint64_t nodes_offset =
      nodes_at_movestart_.load(std::memory_order_acquire);
  stats.nodes_since_movestart =
      stats.total_nodes > nodes_offset ? stats.total_nodes - nodes_offset : 0;
  const uint64_t total_batches =
      stats_.total_batches.load(std::memory_order_relaxed);
  const uint64_t batches_offset =
      batches_at_movestart_.load(std::memory_order_acquire);
  stats.batches_since_movestart =
      total_batches > batches_offset ? total_batches - batches_offset : 0;
  if (stats.batches_since_movestart == 0)
    stats.batches_since_movestart = stats.nodes_since_movestart;

  int64_t first_eval_time_ms =
      first_eval_time_ms_.load(std::memory_order_acquire);
  if (stats.nodes_since_movestart > 0 && first_eval_time_ms < 0) {
    int64_t expected = -1;
    first_eval_time_ms_.compare_exchange_strong(
        expected, stats.time_since_movestart_ms, std::memory_order_acq_rel);
    first_eval_time_ms = first_eval_time_ms_.load(std::memory_order_acquire);
  }
  stats.time_since_first_batch_ms =
      first_eval_time_ms >= 0
          ? std::max<int64_t>(0, stats.time_since_movestart_ms -
                                     first_eval_time_ms)
          : 0;

  std::shared_lock<std::shared_mutex> lock(tree_structure_mutex_);
  const Node *root = tree_.Root();
  if (!root || root->NumEdges() == 0)
    return stats;

  const int num_edges = root->NumEdges();
  const Edge *edges = root->Edges();
  stats.edge_n.reserve(num_edges);

  uint32_t max_n = 0;
  bool max_n_has_max_q_plus_m = true;
  float max_q_plus_m = -1000.0f;
  const float root_q =
      root->GetN() > 0 ? -root->GetQ(-params_.draw_score) : 0.0f;
  MovesLeftEvaluator m_eval(params_, root->GetM(), root_q);

  for (int i = 0; i < num_edges; ++i) {
    Node *child = edges[i].child.load(std::memory_order_relaxed);
    const uint32_t n = child ? child->GetN() : 0;
    stats.edge_n.push_back(n);

    float q = root_q;
    float m_utility = m_eval.GetDefaultMUtility();
    if (child && n > 0) {
      q = child->GetQ(params_.draw_score);
      if (m_eval.IsEnabled())
        m_utility = m_eval.GetMUtility(child->GetM(), q);

      if (child->IsTerminal() && child->GetWL() > 0.0f)
        stats.win_found = true;
      if (child->IsTerminal() && child->GetWL() < 0.0f)
        ++stats.num_losing_edges;
      if (q > -0.98f)
        stats.may_resign = false;
    }

    const float q_plus_m = q + m_utility;
    if (n > max_n) {
      max_n = n;
      max_n_has_max_q_plus_m = false;
    }
    if (q_plus_m >= max_q_plus_m) {
      max_n_has_max_q_plus_m = (max_n == n);
      max_q_plus_m = q_plus_m;
    }
  }

  if (!max_n_has_max_q_plus_m)
    stats.time_usage_hint = SearchStats::TimeUsageHint::NeedMoreTime;

  return stats;
}

bool Search::ShouldStop() const {
  if (stop_flag_.load(std::memory_order_acquire))
    return true;

  SearchStats stats = CollectSearchStats();
  StoppersHints hints;
  hints.Reset();

  bool stop = false;
  {
    std::lock_guard<std::mutex> lock(stopper_mutex_);
    if (stopper_ && stats.total_nodes > 0)
      stop = stopper_->ShouldStop(stats, &hints);
    latest_hints_ = hints;
  }
  if (stop)
    return true;

  if (stats.win_found)
    return true;

  return false;
}

void Search::WorkerThreadMain(int thread_id) {
#ifdef __APPLE__
  pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#endif

  SearchWorkerCtx &ctx = *worker_ctxs_[thread_id];
  ctx.SetRootFen(tree_.RootFen());

  auto last_info = std::chrono::steady_clock::now();
  bool use_semaphore =
      limits_.nodes == 0 && backend_ && params_.minibatch_size > 1;

  do {
    if (use_semaphore)
      RunIterationSemaphore(ctx);
    else
      RunIteration(ctx);

    if (thread_id == 0) {
      auto now = std::chrono::steady_clock::now();
      auto since =
          std::chrono::duration_cast<std::chrono::milliseconds>(now - last_info)
              .count();
      if (since >= 1000) {
        SendInfo();
        last_info = now;
      }
    }
  } while (!ShouldStop());

  stats_.cache_hits.fetch_add(ctx.local_cache_hits, std::memory_order_relaxed);
  stats_.cache_misses.fetch_add(ctx.local_cache_misses,
                                std::memory_order_relaxed);
}

void Search::RunIteration(SearchWorkerCtx &ctx) {
  ctx.ResetToRoot();

  auto selected = SelectLeaf(ctx);
  Node *leaf = selected.node;
  int multivisit = std::max(1, selected.multivisit);
  if (!leaf)
    return;

  MoveList<LEGAL> moves(ctx.pos);
  if (moves.size() == 0) {
    bool in_check = ctx.pos.checkers() != 0;
    float value = in_check ? 1.0f : 0.0f;
    float draw = in_check ? 0.0f : 1.0f;
    leaf->MakeTerminal(Node::Terminal::EndOfGame, value, draw, 0.0f);
    leaf->FinalizeScoreUpdate(value, draw, 0.0f, multivisit);
    if (leaf->Parent())
      Backpropagate(leaf->Parent(), -value, draw, 1.0f, multivisit);
    stats_.total_nodes.fetch_add(multivisit, std::memory_order_relaxed);
    return;
  }

  const bool is_root_leaf = (leaf == tree_.Root());
  if (!is_root_leaf) {
    if (!HasMatingMaterial(ctx.pos) || ctx.pos.rule50_count() > 99) {
      leaf->MakeTerminal(Node::Terminal::EndOfGame, 0.0f, 1.0f, 0.0f);
      leaf->FinalizeScoreUpdate(0.0f, 1.0f, 0.0f, multivisit);
      if (leaf->Parent())
        Backpropagate(leaf->Parent(), 0.0f, 1.0f, 1.0f, multivisit);
      stats_.total_nodes.fetch_add(multivisit, std::memory_order_relaxed);
      return;
    }

    if (Tablebases::MaxCardinality > 0 && !ctx.pos.can_castle(ANY_CASTLING) &&
        ctx.pos.rule50_count() == 0 &&
        popcount(ctx.pos.pieces()) <= Tablebases::MaxCardinality) {
      Tablebases::ProbeState state;
      Tablebases::WDLScore wdl = Tablebases::probe_wdl(ctx.pos, &state);
      if (state != Tablebases::FAIL) {
        const int tb_wdl = static_cast<int>(wdl);
        float tb_value = TablebaseWDLToParentWL(tb_wdl);
        float tb_draw = TablebaseWDLToDraw(tb_wdl);
        float tb_m = 0.0f;
        if (leaf->Parent()) {
          tb_m = std::max(0.0f, leaf->Parent()->GetM() - 1.0f);
        }
        leaf->MakeTerminal(Node::Terminal::Tablebase, tb_value, tb_draw, tb_m);
        leaf->FinalizeScoreUpdate(tb_value, tb_draw, tb_m, multivisit);
        if (leaf->Parent())
          Backpropagate(leaf->Parent(), -tb_value, tb_draw, tb_m + 1.0f,
                        multivisit);
        stats_.total_nodes.fetch_add(multivisit, std::memory_order_relaxed);
        return;
      }
    }

    float rep_moves_left = 0.0f;
    Node::Terminal rep_terminal = Node::Terminal::EndOfGame;
    const int plies_from_root = static_cast<int>(ctx.move_stack.size());
    if (ShouldAdjudicateRepetitionDraw(ctx.pos, plies_from_root,
                                       params_.two_fold_draws, &rep_moves_left,
                                       &rep_terminal)) {
      leaf->MakeTerminal(rep_terminal, 0.0f, 1.0f, rep_moves_left);
      leaf->FinalizeScoreUpdate(0.0f, 1.0f, rep_moves_left, multivisit);
      if (leaf->Parent())
        Backpropagate(leaf->Parent(), 0.0f, 1.0f, rep_moves_left + 1.0f,
                      multivisit);
      stats_.total_nodes.fetch_add(multivisit, std::memory_order_relaxed);
      return;
    }
  }

  float value = 0.0f, draw = 0.0f, moves_left_val = 30.0f;

  if (shared_tt_ && leaf->NumEdges() == 0) {
    auto tt_result = shared_tt_->Probe(ctx.pos, 8);
    if (tt_result.found) {
      {
        std::unique_lock<std::shared_mutex> lock(tree_structure_mutex_);
        if (leaf->NumEdges() == 0)
          CreateLeafEdges(leaf, moves);
      }
      float v = -tt_result.value;
      float d = tt_result.draw;
      Backpropagate(leaf, v, d, 30.0f, multivisit);
      stats_.total_nodes.fetch_add(multivisit, std::memory_order_relaxed);
      return;
    }
  }

  if (leaf->NumEdges() == 0 && backend_) {
    auto apply_nn_result = [&](const EvaluationResult &result) {
      {
        std::unique_lock<std::shared_mutex> lock(tree_structure_mutex_);
        if (leaf->NumEdges() == 0) {
          CreateLeafEdges(leaf, moves);
          ApplyNNPolicyToNode(leaf, result, PolicySoftmaxTempForNode(leaf));
        }
        if (params_.add_dirichlet_noise && leaf == tree_.Root())
          AddDirichletNoise(leaf);
      }
      {
        WDLRescaler rescaler{params_.wdl_rescale_ratio,
                             params_.wdl_rescale_diff};
        value = -rescaler.Rescale(result.value);
      }
      draw = result.has_wdl ? result.wdl[1] : 0.0f;
      moves_left_val = result.has_moves_left ? result.moves_left : 30.0f;
    };

    const uint64_t cache_key =
        ctx.CurrentNNCacheKey(params_.cache_history_length);
    EvaluationResult cached;
    if (backend_->Cache().Lookup(cache_key, static_cast<int>(moves.size()),
                                 cached)) {
      ctx.local_cache_hits++;
      apply_nn_result(cached);
    } else {
      SearchWorkerCtx::HistoryBuffer history;
      ctx.BuildHistory(history);
      auto computation = AcquireComputation();
      auto add_result = computation->AddInputWithHistoryKnownCacheMiss(
          history.ptrs, history.depth, cache_key, moves.begin(),
          static_cast<int>(moves.size()));

      if (add_result == BackendComputation::CACHE_HIT) {
        ctx.local_cache_hits++;
        apply_nn_result(computation->GetResult(0));
      } else {
        ctx.local_cache_misses++;
        computation->ComputeBlocking();
        apply_nn_result(computation->GetResult(0));
        stats_.nn_evaluations.fetch_add(1, std::memory_order_relaxed);
        stats_.total_batches.fetch_add(1, std::memory_order_relaxed);
      }
      ReleaseComputation(std::move(computation));
    }
  } else if (leaf->NumEdges() > 0 && leaf->GetN() > 0) {
    value = leaf->GetWL();
    draw = leaf->GetD();
    moves_left_val = leaf->GetM();
  } else if (leaf->NumEdges() > 0) {
    CancelPathScoreUpdate(leaf, multivisit);
    return;
  }

  Backpropagate(leaf, value, draw, moves_left_val, multivisit);
  stats_.total_nodes.fetch_add(multivisit, std::memory_order_relaxed);
}

void Search::RunIterationSemaphore(SearchWorkerCtx &ctx) {
  while (true) {
    if (ShouldStop())
      return;
    int expected = 1;
    if (gathering_permit_.compare_exchange_weak(expected, 0,
                                                std::memory_order_acquire))
      break;
    CPU_PAUSE();
  }

  struct BatchEntry {
    Node *leaf;
    SearchWorkerCtx::HistoryBuffer *history;
    int multivisit;
    int computation_idx;
  };

  std::vector<BatchEntry> local_batch;
  local_batch.reserve(params_.minibatch_size);
  ctx.batch_histories.clear();
  auto &batch_histories = ctx.batch_histories;
  uint64_t planned_visits = 0;

  std::unique_ptr<BackendComputation> computation;
  if (backend_)
    computation = AcquireComputation();

  int collision_events = 0;
  int collision_visits = 0;
  int out_of_order_count = 0;
  int max_out_of_order =
      (params_.out_of_order_eval &&
       params_.max_out_of_order_evals_factor > 0.0f)
          ? std::max(1, static_cast<int>(params_.minibatch_size *
                                         params_.max_out_of_order_evals_factor))
          : std::numeric_limits<int>::max();
  uint32_t tree_size = tree_.Root() ? tree_.Root()->GetN() : 0;

  CollisionStats coll_stats;
  coll_stats.max_collision_events = params_.max_collision_events;
  coll_stats.max_collision_visits = params_.max_collision_visits;
  coll_stats.scaling_start = params_.max_collision_visits_scaling_start;
  coll_stats.scaling_end = params_.max_collision_visits_scaling_end;
  coll_stats.scaling_power = params_.max_collision_visits_scaling_power;
  int max_coll_visits = coll_stats.GetMaxCollisionVisits(tree_size);

  int target_batch = params_.minibatch_size;
  if (limits_.nodes > 0) {
    const uint64_t searched =
        stats_.total_nodes.load(std::memory_order_relaxed);
    if (searched >= static_cast<uint64_t>(limits_.nodes)) {
      gathering_permit_.store(1, std::memory_order_release);
      return;
    }
    const uint64_t remaining = static_cast<uint64_t>(limits_.nodes) - searched;
    target_batch =
        std::max(1, std::min(target_batch, static_cast<int>(remaining)));
  }

  while (static_cast<int>(local_batch.size()) < target_batch && !ShouldStop()) {
    if (limits_.nodes > 0) {
      const uint64_t searched =
          stats_.total_nodes.load(std::memory_order_relaxed);
      if (searched + planned_visits >= static_cast<uint64_t>(limits_.nodes)) {
        break;
      }
    }

    if (collision_events >= params_.max_collision_events ||
        collision_visits >= max_coll_visits)
      break;

    ctx.ResetToRoot();

    auto selected = SelectLeaf(ctx);
    Node *leaf = selected.node;
    int multivisit = std::max(1, selected.multivisit);
    if (!leaf) {
      collision_events++;
      collision_visits += multivisit;
      continue;
    }

    MoveList<LEGAL> moves(ctx.pos);
    if (moves.size() == 0) {
      bool in_check = ctx.pos.checkers() != 0;
      float v = in_check ? 1.0f : 0.0f;
      float d = in_check ? 0.0f : 1.0f;
      leaf->MakeTerminal(Node::Terminal::EndOfGame, v, d, 0.0f);
      leaf->FinalizeScoreUpdate(v, d, 0.0f, multivisit);
      if (leaf->Parent())
        Backpropagate(leaf->Parent(), -v, d, 1.0f, multivisit);
      stats_.total_nodes.fetch_add(multivisit, std::memory_order_relaxed);
      out_of_order_count++;
      if (out_of_order_count >= max_out_of_order)
        break;
      continue;
    }

    const bool is_root_leaf = (leaf == tree_.Root());
    if (!is_root_leaf) {
      if (!HasMatingMaterial(ctx.pos) || ctx.pos.rule50_count() > 99) {
        leaf->MakeTerminal(Node::Terminal::EndOfGame, 0.0f, 1.0f, 0.0f);
        leaf->FinalizeScoreUpdate(0.0f, 1.0f, 0.0f, multivisit);
        if (leaf->Parent())
          Backpropagate(leaf->Parent(), 0.0f, 1.0f, 1.0f, multivisit);
        stats_.total_nodes.fetch_add(multivisit, std::memory_order_relaxed);
        out_of_order_count++;
        if (out_of_order_count >= max_out_of_order)
          break;
        continue;
      }

      if (Tablebases::MaxCardinality > 0 && !ctx.pos.can_castle(ANY_CASTLING) &&
          ctx.pos.rule50_count() == 0 &&
          popcount(ctx.pos.pieces()) <= Tablebases::MaxCardinality) {
        Tablebases::ProbeState state;
        Tablebases::WDLScore wdl = Tablebases::probe_wdl(ctx.pos, &state);
        if (state != Tablebases::FAIL) {
          const int tb_wdl = static_cast<int>(wdl);
          float tb_value = TablebaseWDLToParentWL(tb_wdl);
          float tb_draw = TablebaseWDLToDraw(tb_wdl);
          float tb_m = 0.0f;
          if (leaf->Parent()) {
            tb_m = std::max(0.0f, leaf->Parent()->GetM() - 1.0f);
          }
          leaf->MakeTerminal(Node::Terminal::Tablebase, tb_value, tb_draw,
                             tb_m);
          leaf->FinalizeScoreUpdate(tb_value, tb_draw, tb_m, multivisit);
          if (leaf->Parent())
            Backpropagate(leaf->Parent(), -tb_value, tb_draw, tb_m + 1.0f,
                          multivisit);
          stats_.total_nodes.fetch_add(multivisit, std::memory_order_relaxed);
          out_of_order_count++;
          if (out_of_order_count >= max_out_of_order)
            break;
          continue;
        }
      }

      float rep_moves_left = 0.0f;
      Node::Terminal rep_terminal = Node::Terminal::EndOfGame;
      const int plies_from_root = static_cast<int>(ctx.move_stack.size());
      if (ShouldAdjudicateRepetitionDraw(ctx.pos, plies_from_root,
                                         params_.two_fold_draws,
                                         &rep_moves_left, &rep_terminal)) {
        leaf->MakeTerminal(rep_terminal, 0.0f, 1.0f, rep_moves_left);
        leaf->FinalizeScoreUpdate(0.0f, 1.0f, rep_moves_left, multivisit);
        if (leaf->Parent())
          Backpropagate(leaf->Parent(), 0.0f, 1.0f, rep_moves_left + 1.0f,
                        multivisit);
        stats_.total_nodes.fetch_add(multivisit, std::memory_order_relaxed);
        out_of_order_count++;
        if (out_of_order_count >= max_out_of_order)
          break;
        continue;
      }
    }

    if (shared_tt_ && leaf->NumEdges() == 0) {
      auto tt_result = shared_tt_->Probe(ctx.pos, 8);
      if (tt_result.found) {
        {
          std::unique_lock<std::shared_mutex> lock(tree_structure_mutex_);
          if (leaf->NumEdges() == 0)
            CreateLeafEdges(leaf, moves);
        }
        float v = -tt_result.value;
        float d = tt_result.draw;
        Backpropagate(leaf, v, d, 30.0f, multivisit);
        stats_.total_nodes.fetch_add(multivisit, std::memory_order_relaxed);
        out_of_order_count++;
        if (out_of_order_count >= max_out_of_order)
          break;
        continue;
      }
    }

    if (leaf->NumEdges() > 0 || !backend_) {
      collision_events++;
      collision_visits += multivisit;
      CancelPathScoreUpdate(leaf, multivisit);
      continue;
    }

    const uint64_t cache_key =
        ctx.CurrentNNCacheKey(params_.cache_history_length);
    EvaluationResult cached;
    const bool direct_cache_hit = backend_->Cache().Lookup(
        cache_key, static_cast<int>(moves.size()), cached);
    BackendComputation::AddInputResult add_result =
        BackendComputation::CACHE_HIT;
    int computation_idx = -1;
    SearchWorkerCtx::HistoryBuffer *hist_buf_ptr = nullptr;

    if (!direct_cache_hit) {
      batch_histories.emplace_back();
      SearchWorkerCtx::HistoryBuffer &hist_buf_sem = batch_histories.back();
      ctx.BuildHistory(hist_buf_sem);
      add_result = computation->AddInputWithHistoryKnownCacheMiss(
          hist_buf_sem.ptrs, hist_buf_sem.depth, cache_key, moves.begin(),
          static_cast<int>(moves.size()));
      computation_idx = computation->TotalInputs() - 1;
      hist_buf_ptr = &hist_buf_sem;
    }

    if (direct_cache_hit || add_result == BackendComputation::CACHE_HIT) {
      if (!direct_cache_hit)
        batch_histories.pop_back();
      ctx.local_cache_hits++;
      const auto &result =
          direct_cache_hit ? cached : computation->GetResult(computation_idx);
      {
        std::unique_lock<std::shared_mutex> lock(tree_structure_mutex_);
        if (leaf->NumEdges() == 0) {
          CreateLeafEdges(leaf, moves);
          ApplyNNPolicyToNode(leaf, result, PolicySoftmaxTempForNode(leaf));
        }
        if (params_.add_dirichlet_noise && leaf == tree_.Root())
          AddDirichletNoise(leaf);
      }
      float v;
      {
        WDLRescaler rescaler{params_.wdl_rescale_ratio,
                             params_.wdl_rescale_diff};
        v = -rescaler.Rescale(result.value);
      }
      float d = result.has_wdl ? result.wdl[1] : 0.0f;
      float ml = result.has_moves_left ? result.moves_left : 30.0f;
      Backpropagate(leaf, v, d, ml, multivisit);
      stats_.total_nodes.fetch_add(multivisit, std::memory_order_relaxed);
      out_of_order_count++;
      if (out_of_order_count >= max_out_of_order)
        break;
      continue;
    }

    ctx.local_cache_misses++;
    BatchEntry entry;
    entry.leaf = leaf;
    entry.history = hist_buf_ptr;
    entry.multivisit = multivisit;
    entry.computation_idx = computation_idx;
    local_batch.push_back(std::move(entry));
    planned_visits += static_cast<uint64_t>(multivisit);
  }

  if (local_batch.empty()) {
    gathering_permit_.store(1, std::memory_order_release);
    if (computation)
      ReleaseComputation(std::move(computation));
    return;
  }

  auto cancel_local_batch = [&]() {
    for (const auto &entry : local_batch)
      CancelPathScoreUpdate(entry.leaf, entry.multivisit);
  };

  if (ShouldStop()) {
    gathering_permit_.store(1, std::memory_order_release);
    cancel_local_batch();
    if (computation)
      ReleaseComputation(std::move(computation));
    return;
  }

  backend_waiting_.fetch_add(1, std::memory_order_release);
  gathering_permit_.store(1, std::memory_order_release);

  ctx.prefetch_histories.clear();
  auto &prefetch_histories = ctx.prefetch_histories;
  MaybePrefetchIntoCache(ctx, computation.get(), prefetch_histories);
  const int backend_evaluations = computation->UsedBatchSize();

  if (ShouldStop()) {
    backend_waiting_.fetch_sub(1, std::memory_order_relaxed);
    cancel_local_batch();
    if (computation)
      ReleaseComputation(std::move(computation));
    return;
  }

  const auto batch_start = std::chrono::steady_clock::now();
  computation->ComputeBlocking();
  const auto batch_elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - batch_start)
          .count();
  UpdateBackendLatencyMargin(batch_elapsed);

  if (ShouldStop()) {
    cancel_local_batch();
    backend_waiting_.fetch_sub(1, std::memory_order_relaxed);
    if (computation)
      ReleaseComputation(std::move(computation));
    return;
  }

  backend_waiting_.fetch_sub(1, std::memory_order_relaxed);

  uint64_t total_visits = 0;
  for (auto &entry : local_batch) {
    const auto &result = computation->GetResult(entry.computation_idx);

    if (entry.leaf->NumEdges() == 0) {
      std::unique_lock<std::shared_mutex> lock(tree_structure_mutex_);
      if (entry.leaf->NumEdges() == 0) {
        if (entry.leaf == tree_.Root() && root_search_filter_active_) {
          const Position &leaf_pos =
              *entry.history->ptrs[entry.history->depth - 1];
          MoveList<LEGAL> leaf_moves(leaf_pos);
          CreateLeafEdges(entry.leaf, leaf_moves);
          ApplyNNPolicyToNode(entry.leaf, result,
                              PolicySoftmaxTempForNode(entry.leaf));
          if (params_.add_dirichlet_noise)
            AddDirichletNoise(entry.leaf);
        } else if (!result.policy_priors.empty()) {
          entry.leaf->CreateEdges(result.policy_priors);
          ApplyNNPolicyToNode(entry.leaf, result,
                              PolicySoftmaxTempForNode(entry.leaf));
          if (params_.add_dirichlet_noise && entry.leaf == tree_.Root())
            AddDirichletNoise(entry.leaf);
        } else {
          const Position &leaf_pos =
              *entry.history->ptrs[entry.history->depth - 1];
          MoveList<LEGAL> leaf_moves(leaf_pos);
          if (leaf_moves.size() > 0) {
            CreateLeafEdges(entry.leaf, leaf_moves);
            ApplyNNPolicyToNode(entry.leaf, result,
                                PolicySoftmaxTempForNode(entry.leaf));
            if (params_.add_dirichlet_noise && entry.leaf == tree_.Root())
              AddDirichletNoise(entry.leaf);
          }
        }
      }
    }

    float v;
    {
      WDLRescaler rescaler{params_.wdl_rescale_ratio, params_.wdl_rescale_diff};
      v = -rescaler.Rescale(result.value);
    }
    float d = result.has_wdl ? result.wdl[1] : 0.0f;
    float ml = result.has_moves_left ? result.moves_left : 30.0f;

    Backpropagate(entry.leaf, v, d, ml, entry.multivisit);
    total_visits += static_cast<uint64_t>(entry.multivisit);
  }

  stats_.total_nodes.fetch_add(total_visits, std::memory_order_relaxed);
  if (total_visits > 0)
    stats_.total_batches.fetch_add(1, std::memory_order_relaxed);
  stats_.nn_evaluations.fetch_add(static_cast<uint64_t>(backend_evaluations),
                                  std::memory_order_relaxed);

  if (computation)
    ReleaseComputation(std::move(computation));
}

Search::SelectedLeaf Search::SelectLeaf(SearchWorkerCtx &ctx) {
  std::shared_lock<std::shared_mutex> tree_lock(tree_structure_mutex_);
  Node *node = tree_.Root();
  const bool allow_multivisit = params_.GetNumThreads() > 1 &&
                                params_.minibatch_size > 1 &&
                                limits_.nodes <= 0;
  int path_multivisit =
      allow_multivisit ? std::max(1, params_.virtual_loss) : 1;

  std::array<Node *, MAX_PLY + 16> vl_path{};
  int vl_path_size = 0;
  bool root_marked = false;

  auto cancel_virtual_loss = [&]() {
    for (int i = 0; i < vl_path_size; ++i)
      vl_path[i]->CancelScoreUpdate(path_multivisit);
    vl_path_size = 0;
  };
  auto push_virtual_loss_node = [&](Node *n) {
    if (vl_path_size >= static_cast<int>(vl_path.size())) {
      n->CancelScoreUpdate(path_multivisit);
      cancel_virtual_loss();
      return false;
    }
    vl_path[vl_path_size++] = n;
    return true;
  };

  while (node->NumEdges() > 0 && !node->IsTerminal()) {
    bool is_root = (node == tree_.Root());
    auto [best_idx, visits_to_assign] = SelectChildPuct(node, is_root, ctx);
    if (best_idx < 0) {
      cancel_virtual_loss();
      return {nullptr, 0};
    }

    if (allow_multivisit && is_root && visits_to_assign > 1) {
      path_multivisit = std::min(visits_to_assign, 128);
    }

    if (is_root && !root_marked) {
      if (!node->TryStartScoreUpdate(path_multivisit)) {
        cancel_virtual_loss();
        return {nullptr, 0};
      }
      if (!push_virtual_loss_node(node))
        return {nullptr, 0};
      root_marked = true;
    }

    Edge &edge = node->Edges()[best_idx];

    Node *child = edge.child.load(std::memory_order_acquire);
    if (!child) {
      Node *new_child = tree_.AllocateNode(node, best_idx);
      Node *expected = nullptr;
      if (edge.child.compare_exchange_strong(expected, new_child,
                                             std::memory_order_release,
                                             std::memory_order_acquire)) {
        child = new_child;
      } else {
        child = expected;
      }
    }

    if (!child->TryStartScoreUpdate(path_multivisit)) {
      cancel_virtual_loss();
      return {nullptr, 0};
    }

    if (!push_virtual_loss_node(child))
      return {nullptr, 0};
    ctx.DoMove(edge.move);
    node = child;
  }

  if (!root_marked && node == tree_.Root()) {
    if (!node->TryStartScoreUpdate(path_multivisit)) {
      cancel_virtual_loss();
      return {nullptr, 0};
    }
    if (!push_virtual_loss_node(node))
      return {nullptr, 0};
    root_marked = true;
  }

  return {node, path_multivisit};
}

Search::PuctResult Search::SelectChildPuct(Node *node, bool is_root,
                                           SearchWorkerCtx &ctx) {
  int num_edges = node->NumEdges();
  if (num_edges == 0)
    return {-1, 1};

  uint32_t parent_n = node->GetN();
  float parent_q = node->GetN() > 0 ? -node->GetQ(-params_.draw_score) : 0.0f;

  float cpuct = params_.GetCpuct(is_root);
  float cpuct_base = params_.GetCpuctBase(is_root);
  float cpuct_factor = params_.GetCpuctFactor(is_root);
  float effective_cpuct =
      cpuct +
      cpuct_factor *
          std::log((static_cast<float>(parent_n) + cpuct_base) / cpuct_base);

  uint32_t children_visits = node->GetChildrenVisits();
  float cpuct_sqrt_n =
      effective_cpuct *
      std::sqrt(static_cast<float>(std::max(children_visits, 1u)));

  float visited_policy = node->GetVisitedPolicy();
  float fpu;
  if (params_.GetFpuAbsolute(is_root)) {
    fpu = params_.GetFpuValue(is_root);
  } else {
    float reduction =
        is_root ? params_.fpu_reduction_at_root : params_.fpu_reduction;
    fpu = parent_q - reduction * std::sqrt(visited_policy);
  }

  MovesLeftEvaluator m_eval(params_, node->GetM(), parent_q);

  const Edge *edges = node->Edges();
  int best_idx = -1;
  float best_score = -1e9f;
  float second_best_score = -1e9f;
  int best_n_started = 0;
  float best_policy = 0.0f;
  int root_best_idx = -1;
  uint32_t root_best_n = 0;
  int64_t root_remaining_playouts = std::numeric_limits<int64_t>::max();
  bool use_root_smart_pruning = false;

  if (is_root && smart_pruning_enabled_.load(std::memory_order_acquire)) {
    {
      std::lock_guard<std::mutex> lock(stopper_mutex_);
      if (latest_hints_.HasEstimatedRemainingPlayouts()) {
        root_remaining_playouts = latest_hints_.GetEstimatedRemainingPlayouts();
        use_root_smart_pruning = true;
      }
    }
    for (int i = 0; i < num_edges; ++i) {
      Node *child = edges[i].child.load(std::memory_order_acquire);
      const uint32_t cn = child ? child->GetN() : 0;
      if (cn > root_best_n) {
        root_best_n = cn;
        root_best_idx = i;
      }
    }
  }

  PREFETCH(&edges[0]);
  if (num_edges > 1)
    PREFETCH(&edges[1]);

  for (int i = 0; i < num_edges; ++i) {
    if (i + 2 < num_edges)
      PREFETCH(&edges[i + 2]);

    const Edge &edge = edges[i];
    Node *child = edge.child.load(std::memory_order_acquire);

    if (child && child->GetN() == 0 && child->GetNInFlight() > 0) {
      continue;
    }

    float q, m_utility = 0.0f;
    float policy = edge.GetP();
    int n_started = 0;

    if (child) {
      uint32_t cn = child->GetN();
      uint32_t cn_flight = child->GetNInFlight();
      n_started = static_cast<int>(cn + cn_flight);

      q = (cn > 0) ? child->GetQ(params_.draw_score) : fpu;

      if (cn > 0 && m_eval.IsEnabled()) {
        m_utility = m_eval.GetMUtility(child->GetM(), q);
      }
    } else {
      q = fpu;
      m_utility = m_eval.GetDefaultMUtility();
    }

    if (use_root_smart_pruning && root_best_idx >= 0 && i != root_best_idx) {
      const uint32_t child_n = child ? child->GetN() : 0;
      if (root_remaining_playouts <
          static_cast<int64_t>(root_best_n) - static_cast<int64_t>(child_n)) {
        continue;
      }
    }

    float u = cpuct_sqrt_n * policy / (1.0f + static_cast<float>(n_started));
    float score = q + u + m_utility;

    if (score > best_score) {
      second_best_score = best_score;
      best_score = score;
      best_idx = i;
      best_n_started = n_started;
      best_policy = policy;
    } else if (score > second_best_score) {
      second_best_score = score;
    }
  }

  if (params_.root_tactical_capture_probe && is_root && best_idx >= 0 &&
      children_visits >= 16 && children_visits <= 600 &&
      !ctx.pos.capture(edges[best_idx].move)) {
    int probe_idx = -1;
    float probe_policy = 0.0f;
    const int policy_rank_limit = std::min(num_edges, 16);
    for (int i = 0; i < policy_rank_limit; ++i) {
      const bool low_policy_capture =
          MCTSIsMinorCentralPawnCapture(ctx.pos, edges[i].move) ||
          MCTSIsMinorKingPawnCheckCapture(ctx.pos, edges[i].move);
      const bool high_value_capture =
          limits_.movetime > 0 &&
          MCTSIsMinorHighValueCapture(ctx.pos, edges[i].move);
      if (!low_policy_capture && !high_value_capture)
        continue;

      Node *child = edges[i].child.load(std::memory_order_acquire);
      const uint32_t target_visits = high_value_capture ? 32 : 16;
      if (child && child->GetN() >= target_visits)
        continue;
      if (child && child->GetNInFlight() > 0)
        continue;

      const float policy = edges[i].GetP();
      const bool candidate_ok = high_value_capture
                                    ? MCTSRootHighValueCaptureProbeCandidate(
                                          children_visits, i + 1, policy)
                                    : MCTSRootTacticalCaptureProbeCandidate(
                                          children_visits, i + 1, policy);
      if (!candidate_ok)
        continue;
      if (probe_idx < 0 || policy > probe_policy) {
        probe_idx = i;
        probe_policy = policy;
      }
    }

    if (probe_idx >= 0)
      return {probe_idx, 1};
  }

  if (params_.root_tactical_capture_probe && is_root && best_idx >= 0 &&
      children_visits >= 16 && children_visits <= 80 &&
      !ctx.pos.capture(edges[best_idx].move)) {
    bool already_probed = false;
    int probe_idx = -1;
    float probe_policy = 0.0f;
    const int policy_rank_limit = std::min(num_edges, 10);
    for (int i = 0; i < policy_rank_limit; ++i) {
      if (!MCTSIsMinorCentralQuietMove(ctx.pos, edges[i].move))
        continue;

      Node *child = edges[i].child.load(std::memory_order_acquire);
      if (child && child->GetN() >= 8) {
        already_probed = true;
        break;
      }
      if (child && child->GetNInFlight() > 0)
        continue;

      const float policy = edges[i].GetP();
      if (!MCTSRootTacticalQuietProbeCandidate(children_visits, i + 1,
                                               policy)) {
        continue;
      }
      if (probe_idx < 0 || policy > probe_policy) {
        probe_idx = i;
        probe_policy = policy;
      }
    }

    if (!already_probed && probe_idx >= 0)
      return {probe_idx, 1};
  }

  if (params_.low_policy_root_lever_selection && is_root && best_idx >= 0 &&
      children_visits >= 16 && children_visits <= 80) {
    Position root_pos;
    StateInfo root_state;
    root_pos.set(tree_.RootFen(), false, &root_state);
    if (!MCTSIsKingsidePawnLever(root_pos, edges[best_idx].move) &&
        !MCTSIsCentralPawnBreak(root_pos, edges[best_idx].move)) {
      int probe_idx = -1;
      float probe_policy = 0.0f;
      const int policy_rank_limit = std::min(num_edges, 8);
      for (int i = 0; i < policy_rank_limit; ++i) {
        const bool kingside_lever =
            MCTSIsKingsidePawnLever(root_pos, edges[i].move);
        const bool central_break =
            MCTSIsCentralPawnBreak(root_pos, edges[i].move);
        if (!kingside_lever && !central_break)
          continue;

        Node *child = edges[i].child.load(std::memory_order_acquire);
        if (child && child->GetN() >= 2)
          continue;
        if (child && child->GetNInFlight() > 0)
          continue;

        const float policy = edges[i].GetP();
        const bool candidate_ok = kingside_lever
                                      ? MCTSRootLowPolicyLeverProbeCandidate(
                                            children_visits, i + 1, policy)
                                      : MCTSRootCentralPawnBreakProbeCandidate(
                                            children_visits, i + 1, policy);
        if (!candidate_ok) {
          continue;
        }
        if (probe_idx < 0 || policy > probe_policy) {
          probe_idx = i;
          probe_policy = policy;
        }
      }

      if (probe_idx >= 0)
        return {probe_idx, 1};
    }
  }

  if (params_.root_tactical_capture_probe && is_root && best_idx >= 0 &&
      limits_.movetime > 0 && children_visits >= 32 && children_visits <= 180) {
    int probe_idx = -1;
    float probe_policy = 0.0f;
    const int policy_rank_limit = std::min(num_edges, 12);
    for (int i = 0; i < policy_rank_limit; ++i) {
      const bool attacks_major =
          MCTSIsMinorQuietAttacksMajor(ctx.pos, edges[i].move);
      const Piece moving_piece = ctx.pos.piece_on(edges[i].move.from_sq());
      const bool fifth_rank =
          moving_piece != NO_PIECE &&
          MCTSIsMinorFifthRankQuietMove(ctx.pos, edges[i].move) &&
          MCTSHasHeavyPieceOnSeventh(ctx.pos, color_of(moving_piece));
      if (!attacks_major && !fifth_rank)
        continue;

      const float policy = edges[i].GetP();
      Node *child = edges[i].child.load(std::memory_order_acquire);
      const bool attacks_queen =
          attacks_major && MCTSIsMinorQuietAttacksQueen(ctx.pos, edges[i].move);
      const uint32_t target_visits =
          (fifth_rank || (attacks_queen && policy >= 0.180f)) ? 48 : 32;
      if (child && child->GetN() >= target_visits)
        continue;
      if (child && child->GetNInFlight() > 0)
        continue;

      const bool candidate_ok =
          attacks_major ? MCTSRootQuietMajorAttackProbeCandidate(
                              children_visits, i + 1, policy)
          : fifth_rank  ? MCTSRootFifthRankQuietProbeCandidate(children_visits,
                                                               i + 1, policy)
                        : MCTSRootDeepTacticalQuietProbeCandidate(
                              children_visits, i + 1, policy);
      if (!candidate_ok) {
        continue;
      }
      if (probe_idx < 0 || policy > probe_policy) {
        probe_idx = i;
        probe_policy = policy;
      }
    }

    if (probe_idx >= 0)
      return {probe_idx, 1};
  }

  if (params_.root_tactical_capture_probe && is_root && best_idx >= 0 &&
      (limits_.movetime > 0 || limits_.nodes > 0) && children_visits >= 16 &&
      children_visits <= 600 && !ctx.pos.gives_check(edges[best_idx].move)) {
    int probe_idx = -1;
    float probe_policy = 0.0f;
    const int policy_rank_limit = std::min(num_edges, 32);
    for (int i = 0; i < policy_rank_limit; ++i) {
      if (!MCTSIsQuietQueenCheck(ctx.pos, edges[i].move))
        continue;

      Node *child = edges[i].child.load(std::memory_order_acquire);
      const float policy = edges[i].GetP();
      const uint32_t target_visits = policy >= 0.020f ? 32 : 16;
      if (child && child->GetN() >= target_visits)
        continue;
      if (child && !MCTSRootQuietQueenCheckProbeStillViable(child->GetN(),
                                                            child->GetWL()))
        continue;
      if (child && child->GetNInFlight() > 0)
        continue;

      if (!MCTSRootQuietQueenCheckProbeCandidate(children_visits, i + 1,
                                                 policy)) {
        continue;
      }
      if (probe_idx < 0 || policy > probe_policy) {
        probe_idx = i;
        probe_policy = policy;
      }
    }

    if (probe_idx >= 0)
      return {probe_idx, 1};
  }

  if (params_.root_tactical_capture_probe && is_root && best_idx >= 0 &&
      limits_.movetime > 0 && children_visits >= 16 && children_visits <= 180) {
    int probe_idx = -1;
    float probe_policy = 0.0f;
    const int policy_rank_limit = std::min(num_edges, 8);
    for (int i = 0; i < policy_rank_limit; ++i) {
      if (i == best_idx)
        continue;
      if (!MCTSIsQuietQueenKingNetMove(ctx.pos, edges[i].move))
        continue;

      Node *child = edges[i].child.load(std::memory_order_acquire);
      const float policy = edges[i].GetP();
      const uint32_t target_visits = policy >= 0.120f ? 48 : 24;
      if (child && child->GetN() >= target_visits)
        continue;
      if (child && child->GetNInFlight() > 0)
        continue;

      if (!MCTSRootQuietQueenKingNetProbeCandidate(children_visits, i + 1,
                                                   policy)) {
        continue;
      }
      if (probe_idx < 0 || policy > probe_policy) {
        probe_idx = i;
        probe_policy = policy;
      }
    }

    if (probe_idx >= 0)
      return {probe_idx, 1};
  }

  int visits_to_assign = 1;
  if (best_idx >= 0 && num_edges > 1 && second_best_score > -1e8f) {
    float best_q_component =
        best_score - cpuct_sqrt_n * best_policy /
                         (1.0f + static_cast<float>(best_n_started));
    float margin = second_best_score - best_q_component;
    if (margin > 0.0f) {
      float v = cpuct_sqrt_n * best_policy / margin -
                static_cast<float>(best_n_started) - 1.0f;
      visits_to_assign = std::max(1, static_cast<int>(v));
      visits_to_assign = std::min(visits_to_assign, 128);
    }
  }

  return {best_idx, visits_to_assign};
}

void Search::Backpropagate(Node *node, float value, float draw,
                           float moves_left, int multivisit) {
  const int visits = std::max(1, multivisit);
  while (node) {
    node->FinalizeScoreUpdate(value, draw, moves_left, visits);
    if (params_.sticky_endgames)
      node->MaybeSetBounds();
    if (params_.solid_tree_threshold > 0 &&
        node->GetN() >= static_cast<uint32_t>(params_.solid_tree_threshold) &&
        !node->IsSolid()) {
      std::unique_lock<std::shared_mutex> lock(tree_structure_mutex_);
      node->MakeSolid();
    }
    value = -value;
    moves_left += 1.0f;
    node = node->Parent();
  }
}

void Search::CancelPathScoreUpdate(Node *leaf, int multivisit) {
  const int visits = std::max(1, multivisit);
  for (Node *node = leaf; node; node = node->Parent())
    node->CancelScoreUpdate(visits);
}

void Search::MaybePrefetchIntoCache(
    SearchWorkerCtx &ctx, BackendComputation *computation,
    std::deque<SearchWorkerCtx::HistoryBuffer> &prefetch_histories) {
  if (stop_flag_.load(std::memory_order_acquire))
    return;
  if (!computation || computation->UsedBatchSize() <= 0)
    return;
  // Fixed-node searches benchmark completed visits; speculative evals waste
  // the node budget and distort NPS.
  if (limits_.nodes > 0)
    return;
  if (computation->UsedBatchSize() >= params_.max_prefetch)
    return;

  int budget = params_.max_prefetch - computation->UsedBatchSize();
  ctx.ResetToRoot();
  PrefetchIntoCache(tree_.Root(), budget, ctx, computation, prefetch_histories);
}

int Search::PrefetchIntoCache(
    Node *node, int budget, SearchWorkerCtx &ctx,
    BackendComputation *computation,
    std::deque<SearchWorkerCtx::HistoryBuffer> &prefetch_histories) {
  if (budget <= 0 || stop_flag_.load(std::memory_order_acquire))
    return 0;

  auto prefetch_current_position = [&]() {
    const uint64_t key = ctx.CurrentNNCacheKey(params_.cache_history_length);
    const Position &pos = ctx.pos;
    MoveList<LEGAL> moves(pos);
    EvaluationResult cached;
    if (backend_->Cache().Lookup(key, static_cast<int>(moves.size()), cached))
      return 0;

    prefetch_histories.emplace_back();
    SearchWorkerCtx::HistoryBuffer &history = prefetch_histories.back();
    ctx.BuildHistory(history);
    auto add_result = computation->AddInputWithHistoryKnownCacheMiss(
        history.ptrs, history.depth, key, moves.begin(),
        static_cast<int>(moves.size()));
    if (add_result != BackendComputation::QUEUED)
      prefetch_histories.pop_back();
    return add_result == BackendComputation::QUEUED ? 1 : 0;
  };

  if (!node)
    return prefetch_current_position();

  if (node->GetNStarted() == 0) {
    return prefetch_current_position();
  }

  if (node->GetN() == 0 || node->IsTerminal() || node->NumEdges() == 0)
    return 0;

  int num_edges = node->NumEdges();
  const Edge *edges = node->Edges();
  bool is_root = (node == tree_.Root());

  float cpuct = params_.GetCpuct(is_root);
  float cpuct_base = params_.GetCpuctBase(is_root);
  float cpuct_factor = params_.GetCpuctFactor(is_root);
  float effective_cpuct =
      cpuct +
      cpuct_factor * std::log((static_cast<float>(node->GetN()) + cpuct_base) /
                              cpuct_base);
  float puct_mult =
      effective_cpuct *
      std::sqrt(static_cast<float>(std::max(node->GetChildrenVisits(), 1u)));

  float fpu;
  if (params_.GetFpuAbsolute(is_root)) {
    fpu = params_.GetFpuValue(is_root);
  } else {
    float reduction =
        is_root ? params_.fpu_reduction_at_root : params_.fpu_reduction;
    fpu = -node->GetQ(-params_.draw_score) -
          reduction * std::sqrt(node->GetVisitedPolicy());
  }

  int best_idx = -1;
  float best_score = -1e9f;
  for (int i = 0; i < num_edges && i < 8; ++i) {
    Node *child = edges[i].child.load(std::memory_order_acquire);
    float policy = edges[i].GetP();
    if (policy <= 0.0f)
      continue;
    float q =
        (child && child->GetN() > 0) ? child->GetQ(params_.draw_score) : fpu;
    int n_started = child ? child->GetNStarted() : 0;
    float u = puct_mult * policy / (1.0f + static_cast<float>(n_started));
    float score = q + u;
    if (score > best_score) {
      best_score = score;
      best_idx = i;
    }
  }

  if (best_idx < 0)
    return 0;

  Node *child = edges[best_idx].child.load(std::memory_order_acquire);
  ctx.DoMove(edges[best_idx].move);
  int spent =
      PrefetchIntoCache(child, budget, ctx, computation, prefetch_histories);
  ctx.UndoMove();

  return spent;
}

void Search::AddDirichletNoise(Node *root) {
  int num_edges = root->NumEdges();
  if (num_edges == 0 || params_.noise_epsilon <= 0.0f)
    return;

  Edge *edges = root->Edges();
  std::random_device rd;
  std::mt19937 gen(rd());
  std::gamma_distribution<float> gamma(params_.noise_alpha, 1.0f);

  std::vector<float> noise(num_edges);
  float noise_sum = 0.0f;
  for (int i = 0; i < num_edges; ++i) {
    noise[i] = gamma(gen);
    noise_sum += noise[i];
  }

  if (noise_sum < std::numeric_limits<float>::min())
    return;

  for (int i = 0; i < num_edges; ++i) {
    float current = edges[i].GetP();
    float noisy = (1.0f - params_.noise_epsilon) * current +
                  params_.noise_epsilon * (noise[i] / noise_sum);
    edges[i].SetP(noisy);
  }
  root->SortEdges();
}

void Search::UpdateBackendLatencyMargin(int64_t elapsed_ms) {
  if (elapsed_ms <= 0)
    return;

  const int64_t sample = std::clamp<int64_t>(elapsed_ms + 10, 10, 2000);
  const int64_t previous =
      backend_latency_margin_ms_.load(std::memory_order_relaxed);
  const int64_t updated =
      previous > 0 ? (previous * 3 + sample + 2) / 4 : sample;
  backend_latency_margin_ms_.store(updated, std::memory_order_relaxed);
}

void Search::BuildRootSearchMoves(const Position &root_pos) {
  root_search_moves_.clear();
  root_search_filter_active_ = !limits_.searchmoves.empty();
  if (!root_search_filter_active_)
    return;

  root_search_moves_.reserve(limits_.searchmoves.size());
  for (const auto &uci_move : limits_.searchmoves) {
    Move move = UCIEngine::to_move(root_pos, uci_move);
    if (move == Move::none())
      continue;
    if (std::find(root_search_moves_.begin(), root_search_moves_.end(), move) ==
        root_search_moves_.end()) {
      root_search_moves_.push_back(move);
    }
  }
}

void Search::CreateLeafEdges(Node *leaf, const MoveList<LEGAL> &moves) {
  if (leaf == tree_.Root() && root_search_filter_active_) {
    leaf->CreateEdges(root_search_moves_.data(),
                      static_cast<int>(root_search_moves_.size()));
    return;
  }
  leaf->CreateEdges(moves);
}

Move Search::FirstRootMoveOrLegal() const {
  if (root_search_filter_active_)
    return root_search_moves_.empty() ? Move::none()
                                      : root_search_moves_.front();

  Position pos;
  StateInfo st;
  pos.set(tree_.RootFen(), false, &st);
  MoveList<LEGAL> moves(pos);
  return moves.size() > 0 ? *moves.begin() : Move::none();
}

void Search::CaptureRootVisitBaselineLocked() {
  root_visit_baseline_.clear();

  const Node *root = tree_.Root();
  if (!root || root->NumEdges() == 0)
    return;

  const int num_edges = root->NumEdges();
  const Edge *edges = root->Edges();
  root_visit_baseline_.reserve(static_cast<size_t>(num_edges));
  for (int i = 0; i < num_edges; ++i) {
    const Node *child = edges[i].child.load(std::memory_order_acquire);
    root_visit_baseline_.push_back(
        {edges[i].move.raw(), child ? child->GetN() : 0});
  }
}

uint32_t Search::RootVisitBaselineLocked(Move move) const {
  const uint32_t raw = move.raw();
  for (const auto &[candidate, visits] : root_visit_baseline_) {
    if (candidate == raw)
      return visits;
  }
  return 0;
}

void Search::ApplyNNPolicy(Node *node, const EvaluationResult &result,
                           float softmax_temp) {
  ApplyNNPolicyToNode(node, result, softmax_temp);
}

float Search::PolicySoftmaxTempForNode(const Node *node) const {
  return node == tree_.Root() ? params_.root_policy_softmax_temp
                              : params_.policy_softmax_temp;
}

Search::RootMoveStats Search::GetBestMoveStatsLocked() const {
  const Node *root = tree_.Root();
  if (!root || root->NumEdges() == 0)
    return {};

  RootMoveStats tablebase_best;
  if (TryGetRootTablebaseMoveStatsLocked(&tablebase_best))
    return tablebase_best;

  int num_edges = root->NumEdges();
  const Edge *edges = root->Edges();

  int best_idx = -1;
  uint32_t best_n = 0;
  uint32_t total_child_visits = 0;
  float best_q = -2.0f;
  float best_m = 999.0f;
  bool best_is_terminal_win = false;
  uint32_t total_current_child_visits = 0;

  for (int i = 0; i < num_edges; ++i) {
    Node *child = edges[i].child.load(std::memory_order_acquire);
    if (!child)
      continue;

    uint32_t cn = child->GetN();
    if (cn == 0)
      continue;
    total_child_visits += cn;
    const uint32_t baseline = RootVisitBaselineLocked(edges[i].move);
    total_current_child_visits += cn >= baseline ? cn - baseline : 0;

    float cq = child->GetWL();
    bool is_terminal = child->IsTerminal();
    bool is_win = is_terminal && cq > 0.5f;
    bool is_loss = is_terminal && cq < -0.5f;

    bool prefer = false;
    if (best_idx < 0) {
      prefer = true;
    } else if (is_win && !best_is_terminal_win) {
      prefer = true;
    } else if (is_win && best_is_terminal_win) {
      prefer = child->GetM() < best_m;
    } else if (!is_win && best_is_terminal_win) {
      prefer = false;
    } else if (is_loss) {
      if (best_idx >= 0) {
        Node *best_child =
            edges[best_idx].child.load(std::memory_order_acquire);
        bool best_is_loss = best_child && best_child->IsTerminal() &&
                            best_child->GetWL() < -0.5f;
        if (best_is_loss) {
          prefer = child->GetM() > best_m;
        }
      }
    } else {
      if (cn != best_n)
        prefer = cn > best_n;
      else if (std::abs(cq - best_q) > 0.001f)
        prefer = cq > best_q;
      else
        prefer = edges[i].GetP() > edges[best_idx].GetP();
    }

    if (prefer) {
      best_idx = i;
      best_n = cn;
      best_q = cq;
      best_m = child->GetM();
      best_is_terminal_win = is_win;
    }
  }

  const bool fixed_node_limited_search =
      limits_.nodes > 0 && limits_.movetime <= 0 && !limits_.infinite &&
      limits_.time[WHITE] <= 0 && limits_.time[BLACK] <= 0;
  const bool fixed_movetime_search =
      limits_.movetime > 0 && !limits_.infinite && limits_.time[WHITE] <= 0 &&
      limits_.time[BLACK] <= 0;
  const bool fixed_low_root_visit_search =
      fixed_node_limited_search || fixed_movetime_search;
  const bool clock_managed_search =
      limits_.movetime <= 0 && limits_.nodes <= 0 && limits_.depth <= 0 &&
      limits_.mate <= 0 && !limits_.infinite &&
      (limits_.time[WHITE] > 0 || limits_.time[BLACK] > 0);
  const uint32_t q_override_visit_cap =
      fixed_movetime_search ? static_cast<uint32_t>(std::max(
                                  0, params_.fixed_movetime_q_override_cap))
                            : 96;
  const float near_equal_required_gap = fixed_movetime_search ? 0.02f : 0.05f;
  if (fixed_low_root_visit_search && best_idx >= 0 && !best_is_terminal_win &&
      total_child_visits <= q_override_visit_cap) {
    Position root_pos;
    StateInfo root_state;
    root_pos.set(tree_.RootFen(), false, &root_state);
    int q_idx = best_idx;
    float q_best = best_q;
    uint32_t q_best_visits = best_n;
    const int q_pass_limit =
        params_.low_visit_q_override_rescan ? num_edges : 1;
    for (int q_pass = 0; q_pass < q_pass_limit; ++q_pass) {
      bool q_changed = false;
      for (int i = 0; i < num_edges; ++i) {
        if (i == q_idx)
          continue;
        Node *child = edges[i].child.load(std::memory_order_acquire);
        if (!child)
          continue;
        const uint32_t cn = child->GetN();
        const float cq = child->GetWL();
        const bool protects_high_policy_capture =
            root_pos.capture(edges[q_idx].move) &&
            !root_pos.capture(edges[i].move) && edges[q_idx].GetP() >= 0.50f &&
            edges[i].GetP() <= edges[q_idx].GetP() * 0.60f &&
            cq < q_best + 0.10f;
        const bool protects_minor_pawn_endgame_capture =
            MCTSRootMinorPawnEndgameCaptureProtected(
                root_pos, edges[q_idx].move, edges[i].move, edges[q_idx].GetP(),
                q_best, edges[i].GetP(), cq);
        const bool protects_high_policy_visit_leader =
            MCTSRootHighPolicyVisitLeaderProtected(q_best_visits, cn,
                                                   edges[q_idx].GetP(), q_best,
                                                   edges[i].GetP(), cq);
        if (protects_high_policy_capture ||
            protects_minor_pawn_endgame_capture ||
            protects_high_policy_visit_leader)
          continue;
        if (MCTSRootTinyLowVisitQOverrideCandidate(
                total_child_visits,
                params_.low_visit_q_override_rescan ? q_best_visits : best_n,
                cn, edges[q_idx].GetP(), q_best, edges[i].GetP(), cq)) {
          q_idx = i;
          q_best = cq;
          q_best_visits = cn;
          q_changed = true;
          continue;
        }
        if (MCTSRootLowVisitQOverrideCandidate(
                params_.low_visit_q_override_rescan ? q_best_visits : best_n,
                cn, q_best, cq, near_equal_required_gap, edges[i].GetP(),
                fixed_movetime_search)) {
          q_idx = i;
          q_best = cq;
          q_best_visits = cn;
          q_changed = true;
        }
      }
      if (!q_changed)
        break;
    }
    if (q_idx != best_idx) {
      best_idx = q_idx;
      best_n = edges[best_idx].child.load(std::memory_order_acquire)->GetN();
      best_q = q_best;
    }
  }

  if (params_.low_policy_root_lever_selection && fixed_movetime_search &&
      best_idx >= 0 && !best_is_terminal_win &&
      total_child_visits <= q_override_visit_cap) {
    Position root_pos;
    StateInfo root_state;
    root_pos.set(tree_.RootFen(), false, &root_state);
    const bool best_is_capture = root_pos.capture(edges[best_idx].move);
    if (root_pos.non_pawn_material() == VALUE_ZERO && !best_is_capture) {
      int ep_idx = -1;
      float ep_q = -2.0f;
      uint32_t ep_visits = 0;
      for (int i = 0; i < num_edges; ++i) {
        if (i == best_idx || edges[i].move.type_of() != EN_PASSANT)
          continue;
        Node *child = edges[i].child.load(std::memory_order_acquire);
        if (!child)
          continue;
        const uint32_t cn = child->GetN();
        const float cq = child->GetWL();
        if (!MCTSRootPawnEndgameEnPassantCandidate(total_child_visits, best_n,
                                                   cn, best_is_capture, true,
                                                   best_q, cq)) {
          continue;
        }
        if (ep_idx < 0 || cq > ep_q ||
            (std::abs(cq - ep_q) <= 0.000001f && cn > ep_visits)) {
          ep_idx = i;
          ep_q = cq;
          ep_visits = cn;
        }
      }
      if (ep_idx >= 0) {
        best_idx = ep_idx;
        best_n = ep_visits;
        best_q = ep_q;
      }
    }
  }

  if (clock_managed_search && best_idx >= 0 && !best_is_terminal_win &&
      total_current_child_visits > 0) {
    int q_idx = best_idx;
    float q_best = best_q;
    const uint32_t best_baseline =
        RootVisitBaselineLocked(edges[best_idx].move);
    uint32_t q_best_current =
        best_n >= best_baseline ? best_n - best_baseline : 0;

    for (int i = 0; i < num_edges; ++i) {
      Node *child = edges[i].child.load(std::memory_order_acquire);
      if (!child)
        continue;
      const uint32_t cn = child->GetN();
      const uint32_t baseline = RootVisitBaselineLocked(edges[i].move);
      const uint32_t current_visits = cn >= baseline ? cn - baseline : 0;
      const float cq = child->GetWL();
      if (MCTSRootClockLowVisitQOverrideCandidate(
              total_current_child_visits, q_best_current, current_visits,
              q_best, cq, edges[i].GetP())) {
        q_idx = i;
        q_best = cq;
        q_best_current = current_visits;
      }
    }
    if (q_idx != best_idx) {
      best_idx = q_idx;
      best_n = edges[best_idx].child.load(std::memory_order_acquire)->GetN();
      best_q = q_best;
    }
  }

  const bool reused_root_current_search =
      total_current_child_visits > 0 &&
      total_current_child_visits < total_child_visits;
  if (reused_root_current_search && best_idx >= 0 && !best_is_terminal_win) {
    Position root_pos;
    StateInfo root_state;
    root_pos.set(tree_.RootFen(), false, &root_state);

    if (MCTSIsAdvancedPromotionSupportQueenMove(root_pos,
                                                edges[best_idx].move)) {
      int q_idx = best_idx;
      float q_best = best_q;
      const uint32_t best_baseline =
          RootVisitBaselineLocked(edges[best_idx].move);
      uint32_t q_best_current =
          best_n >= best_baseline ? best_n - best_baseline : 0;

      for (int i = 0; i < num_edges; ++i) {
        if (i == best_idx)
          continue;
        Node *child = edges[i].child.load(std::memory_order_acquire);
        if (!child)
          continue;
        if (!MCTSIsAdvancedPromotionSupportQueenMove(root_pos, edges[i].move))
          continue;
        const uint32_t cn = child->GetN();
        const uint32_t baseline = RootVisitBaselineLocked(edges[i].move);
        const uint32_t current_visits = cn >= baseline ? cn - baseline : 0;
        const float cq = child->GetWL();
        if (MCTSRootClockLowVisitQOverrideCandidate(
                total_current_child_visits, q_best_current, current_visits,
                q_best, cq, edges[i].GetP())) {
          q_idx = i;
          q_best = cq;
          q_best_current = current_visits;
        }
      }
      if (q_idx != best_idx) {
        best_idx = q_idx;
        best_n = edges[best_idx].child.load(std::memory_order_acquire)->GetN();
        best_q = q_best;
      }
    }
  }

  if (fixed_movetime_search && best_idx >= 0 && !best_is_terminal_win &&
      total_child_visits <= 220) {
    Position root_pos;
    StateInfo root_state;
    root_pos.set(tree_.RootFen(), false, &root_state);

    const uint32_t best_baseline =
        RootVisitBaselineLocked(edges[best_idx].move);
    const uint32_t best_current =
        best_n >= best_baseline ? best_n - best_baseline : 0;

    int fifth_rank_idx = -1;
    float fifth_rank_q = -2.0f;
    for (int i = 0; i < num_edges; ++i) {
      if (i == best_idx)
        continue;
      Node *child = edges[i].child.load(std::memory_order_acquire);
      if (!child)
        continue;
      const Move move = edges[i].move;
      const Piece moving_piece = root_pos.piece_on(move.from_sq());
      if (moving_piece == NO_PIECE ||
          !MCTSIsMinorFifthRankQuietMove(root_pos, move) ||
          !MCTSHasHeavyPieceOnSeventh(root_pos, color_of(moving_piece))) {
        continue;
      }
      const uint32_t visits = child->GetN();
      const uint32_t baseline = RootVisitBaselineLocked(move);
      const uint32_t current_visits =
          visits >= baseline ? visits - baseline : 0;
      const float cq = child->GetWL();
      const float policy = edges[i].GetP();
      if (!MCTSRootFifthRankCurrentOverrideCandidate(
              total_child_visits, best_current, current_visits, best_q, cq,
              policy)) {
        continue;
      }
      if (fifth_rank_idx < 0 || cq > fifth_rank_q) {
        fifth_rank_idx = i;
        fifth_rank_q = cq;
      }
    }
    if (fifth_rank_idx >= 0) {
      best_idx = fifth_rank_idx;
      best_n = edges[best_idx].child.load(std::memory_order_acquire)->GetN();
      best_q = fifth_rank_q;
    }
  }

  if (fixed_movetime_search && best_idx >= 0 && !best_is_terminal_win &&
      total_child_visits <= 220) {
    Position root_pos;
    StateInfo root_state;
    root_pos.set(tree_.RootFen(), false, &root_state);

    const float best_policy = edges[best_idx].GetP();
    const bool best_is_promotion_support =
        MCTSIsAdvancedPromotionSupportQueenMove(root_pos, edges[best_idx].move);
    int support_idx = -1;
    float support_policy = 0.0f;
    float support_q = -2.0f;
    if (!best_is_promotion_support) {
      for (int i = 0; i < num_edges; ++i) {
        if (i == best_idx)
          continue;
        Node *child = edges[i].child.load(std::memory_order_acquire);
        if (!child)
          continue;
        if (!MCTSIsAdvancedPromotionSupportQueenMove(root_pos, edges[i].move))
          continue;
        const uint32_t cn = child->GetN();
        const float cq = child->GetWL();
        const float policy = edges[i].GetP();
        if (!MCTSRootAdvancedPromotionSupportCandidate(total_child_visits,
                                                       best_n, cn, best_policy,
                                                       best_q, policy, cq)) {
          continue;
        }
        if (support_idx < 0 || policy > support_policy ||
            (std::abs(policy - support_policy) <= 0.000001f &&
             cq > support_q)) {
          support_idx = i;
          support_policy = policy;
          support_q = cq;
        }
      }
    }
    if (support_idx >= 0) {
      best_idx = support_idx;
      best_n = edges[best_idx].child.load(std::memory_order_acquire)->GetN();
      best_q = support_q;
    }
  }

  if (params_.high_policy_root_lever_selection && best_idx >= 0 &&
      !best_is_terminal_win) {
    Position root_pos;
    StateInfo root_state;
    root_pos.set(tree_.RootFen(), false, &root_state);
    const float best_policy = edges[best_idx].GetP();
    int lever_idx = -1;
    float lever_policy = 0.0f;
    float lever_q = -2.0f;
    for (int i = 0; i < num_edges; ++i) {
      if (i == best_idx)
        continue;
      Node *child = edges[i].child.load(std::memory_order_acquire);
      if (!child)
        continue;
      const uint32_t cn = child->GetN();
      const float cq = child->GetWL();
      const float policy = edges[i].GetP();
      if (!MCTSIsKingsidePawnLever(root_pos, edges[i].move))
        continue;
      if (!MCTSRootHighPolicyLeverCandidate(total_child_visits, best_n, cn,
                                            best_policy, best_q, policy, cq))
        continue;
      if (lever_idx < 0 || policy > lever_policy ||
          (std::abs(policy - lever_policy) <= 0.000001f && cq > lever_q)) {
        lever_idx = i;
        lever_policy = policy;
        lever_q = cq;
      }
    }
    if (lever_idx >= 0) {
      best_idx = lever_idx;
      best_n = edges[best_idx].child.load(std::memory_order_acquire)->GetN();
      best_q = lever_q;
    }
  }

  if (params_.low_policy_root_lever_selection && best_idx >= 0 &&
      !best_is_terminal_win) {
    Position root_pos;
    StateInfo root_state;
    root_pos.set(tree_.RootFen(), false, &root_state);
    if (!MCTSIsKingsidePawnLever(root_pos, edges[best_idx].move)) {
      struct RankedRootMove {
        int idx;
        uint32_t visits;
        float q;
        float policy;
      };
      std::vector<RankedRootMove> ranked;
      ranked.reserve(static_cast<size_t>(num_edges));
      for (int i = 0; i < num_edges; ++i) {
        Node *child = edges[i].child.load(std::memory_order_acquire);
        if (!child)
          continue;
        ranked.push_back({i, child->GetN(), child->GetWL(), edges[i].GetP()});
      }
      std::sort(ranked.begin(), ranked.end(),
                [](const RankedRootMove &a, const RankedRootMove &b) {
                  if (a.visits != b.visits)
                    return a.visits > b.visits;
                  if (std::abs(a.q - b.q) > 0.001f)
                    return a.q > b.q;
                  return a.policy > b.policy;
                });

      const float best_policy = edges[best_idx].GetP();
      int lever_idx = -1;
      float lever_policy = 0.0f;
      float lever_q = -2.0f;
      for (size_t rank_pos = 0; rank_pos < ranked.size(); ++rank_pos) {
        const int i = ranked[rank_pos].idx;
        if (i == best_idx)
          continue;
        if (!MCTSIsKingsidePawnLever(root_pos, edges[i].move))
          continue;
        if (!MCTSRootLowPolicyLeverCandidate(
                total_child_visits, best_n, ranked[rank_pos].visits,
                static_cast<int>(rank_pos + 1), best_policy, best_q,
                ranked[rank_pos].policy, ranked[rank_pos].q)) {
          continue;
        }
        if (lever_idx < 0 || ranked[rank_pos].q > lever_q ||
            (std::abs(ranked[rank_pos].q - lever_q) <= 0.000001f &&
             ranked[rank_pos].policy > lever_policy)) {
          lever_idx = i;
          lever_policy = ranked[rank_pos].policy;
          lever_q = ranked[rank_pos].q;
        }
      }
      if (lever_idx >= 0) {
        best_idx = lever_idx;
        best_n = edges[best_idx].child.load(std::memory_order_acquire)->GetN();
        best_q = lever_q;
      }
    }
  }

  if (best_idx < 0) {
    int best_policy_idx = 0;
    float best_policy = edges[0].GetP();
    for (int i = 1; i < num_edges; ++i) {
      float p = edges[i].GetP();
      if (p > best_policy) {
        best_policy = p;
        best_policy_idx = i;
      }
    }
    return {edges[best_policy_idx].move, root->GetWL(), 0,
            edges[best_policy_idx].GetP(), 0};
  }

  const uint32_t baseline = RootVisitBaselineLocked(edges[best_idx].move);
  const uint32_t current_n = best_n >= baseline ? best_n - baseline : 0;
  return {edges[best_idx].move, best_q, best_n, edges[best_idx].GetP(),
          current_n};
}

bool Search::TryGetRootTablebaseMoveStatsLocked(RootMoveStats *out) const {
  if (!out || Tablebases::MaxCardinality <= 0)
    return false;

  const Node *root = tree_.Root();
  if (!root || root->NumEdges() == 0)
    return false;

  Position pos;
  StateInfo st;
  pos.set(tree_.RootFen(), false, &st);
  if (pos.can_castle(ANY_CASTLING) ||
      popcount(pos.pieces()) > Tablebases::MaxCardinality) {
    return false;
  }

  const Edge *edges = root->Edges();
  const int num_edges = root->NumEdges();
  int best_idx = -1;
  int best_wdl = -3;
  int best_dtz = 0;
  bool best_has_dtz = false;

  for (int i = 0; i < num_edges; ++i) {
    StateInfo child_st;
    pos.do_move(edges[i].move, child_st);

    Tablebases::ProbeState wdl_state;
    int root_wdl = 0;
    if (pos.is_draw(1)) {
      root_wdl = static_cast<int>(Tablebases::WDLDraw);
      wdl_state = Tablebases::OK;
    } else {
      root_wdl = -static_cast<int>(Tablebases::probe_wdl(pos, &wdl_state));
    }

    bool has_dtz = false;
    int root_dtz = 0;
    if (wdl_state != Tablebases::FAIL) {
      Tablebases::ProbeState dtz_state;
      root_dtz = -Tablebases::probe_dtz(pos, &dtz_state);
      has_dtz = dtz_state != Tablebases::FAIL;
    }

    pos.undo_move(edges[i].move);

    if (wdl_state == Tablebases::FAIL)
      return false;

    bool prefer = best_idx < 0 || root_wdl > best_wdl;
    if (!prefer && root_wdl == best_wdl) {
      if (root_wdl > 0 && has_dtz &&
          (!best_has_dtz || std::abs(root_dtz) < std::abs(best_dtz))) {
        prefer = true;
      } else if (root_wdl < 0 && has_dtz &&
                 (!best_has_dtz || std::abs(root_dtz) > std::abs(best_dtz))) {
        prefer = true;
      } else if (root_wdl == 0 && best_idx >= 0) {
        Node *child = edges[i].child.load(std::memory_order_acquire);
        Node *best_child =
            edges[best_idx].child.load(std::memory_order_acquire);
        const uint32_t cn = child ? child->GetN() : 0;
        const uint32_t best_n = best_child ? best_child->GetN() : 0;
        prefer = cn > best_n ||
                 (cn == best_n && edges[i].GetP() > edges[best_idx].GetP());
      }
    }

    if (prefer) {
      best_idx = i;
      best_wdl = root_wdl;
      best_dtz = root_dtz;
      best_has_dtz = has_dtz;
    }
  }

  if (best_idx < 0)
    return false;

  Node *child = edges[best_idx].child.load(std::memory_order_acquire);
  const uint32_t visits = child ? child->GetN() : 0;
  const uint32_t baseline = RootVisitBaselineLocked(edges[best_idx].move);
  const uint32_t current_visits = visits >= baseline ? visits - baseline : 0;
  float q = 0.0f;
  if (best_wdl >= static_cast<int>(Tablebases::WDLWin))
    q = 1.0f;
  else if (best_wdl <= static_cast<int>(Tablebases::WDLLoss))
    q = -1.0f;
  *out = {edges[best_idx].move, q, visits, edges[best_idx].GetP(),
          current_visits};
  return true;
}

Search::RootMoveStats Search::GetBestMoveStats() const {
  std::shared_lock<std::shared_mutex> lock(tree_structure_mutex_);
  return GetBestMoveStatsLocked();
}

std::vector<Search::RootMoveStats>
Search::GetRootMoveStats(int max_moves) const {
  std::shared_lock<std::shared_mutex> lock(tree_structure_mutex_);
  const Node *root = tree_.Root();
  if (!root || root->NumEdges() == 0)
    return {};

  std::vector<RootMoveStats> stats;
  const int num_edges = root->NumEdges();
  stats.reserve(static_cast<size_t>(num_edges));
  const Edge *edges = root->Edges();

  for (int i = 0; i < num_edges; ++i) {
    Node *child = edges[i].child.load(std::memory_order_acquire);
    const uint32_t visits = child ? child->GetN() : 0;
    const uint32_t baseline = RootVisitBaselineLocked(edges[i].move);
    const uint32_t current_visits = visits >= baseline ? visits - baseline : 0;
    const float q = child ? child->GetWL() : root->GetWL();
    stats.push_back(
        {edges[i].move, q, visits, edges[i].GetP(), current_visits});
  }

  std::sort(stats.begin(), stats.end(),
            [](const RootMoveStats &a, const RootMoveStats &b) {
              if (a.visits != b.visits)
                return a.visits > b.visits;
              if (std::abs(a.q - b.q) > 0.001f)
                return a.q > b.q;
              return a.policy > b.policy;
            });

  if (max_moves > 0 && stats.size() > static_cast<size_t>(max_moves))
    stats.resize(static_cast<size_t>(max_moves));
  return stats;
}

Move Search::GetBestMove() const { return GetBestMoveStats().move; }

Move Search::GetBestMoveWithTemperature(float temperature) const {
  std::shared_lock<std::shared_mutex> lock(tree_structure_mutex_);
  const Node *root = tree_.Root();
  if (!root || root->NumEdges() == 0)
    return Move::none();

  int num_edges = root->NumEdges();
  const Edge *edges = root->Edges();

  float max_n = 0.0f;
  float max_eval = -2.0f;
  for (int i = 0; i < num_edges; ++i) {
    Node *child = edges[i].child.load(std::memory_order_acquire);
    if (!child || child->GetN() == 0)
      continue;
    float cn = static_cast<float>(child->GetN());
    if (cn > max_n) {
      max_n = cn;
      max_eval = child->GetWL();
    }
  }
  if (max_n <= 0.0f)
    return GetBestMoveStatsLocked().move;

  float min_eval = max_eval - params_.temp_winpct_cutoff / 50.0f;

  std::vector<float> cumsum;
  std::vector<int> indices;
  float sum = 0.0f;
  for (int i = 0; i < num_edges; ++i) {
    Node *child = edges[i].child.load(std::memory_order_acquire);
    if (!child || child->GetN() == 0)
      continue;
    if (child->GetWL() < min_eval)
      continue;

    float weight =
        std::pow(static_cast<float>(child->GetN()) / max_n, 1.0f / temperature);
    sum += weight;
    cumsum.push_back(sum);
    indices.push_back(i);
  }
  if (cumsum.empty())
    return GetBestMoveStatsLocked().move;

  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dist(0.0f, cumsum.back());
  float toss = dist(gen);
  auto it = std::lower_bound(cumsum.begin(), cumsum.end(), toss);
  int idx = static_cast<int>(it - cumsum.begin());
  idx = std::min(idx, static_cast<int>(indices.size()) - 1);

  return edges[indices[idx]].move;
}

std::vector<Move> Search::GetPV() const {
  std::shared_lock<std::shared_mutex> lock(tree_structure_mutex_);
  std::vector<Move> pv;
  const Node *node = tree_.Root();
  const Move root_best = GetBestMoveStatsLocked().move;

  while (node && node->NumEdges() > 0) {
    int num_edges = node->NumEdges();
    const Edge *edges = node->Edges();

    int best_idx = -1;
    uint32_t best_n = 0;
    if (pv.empty() && root_best != Move::none()) {
      for (int i = 0; i < num_edges; ++i) {
        if (edges[i].move == root_best) {
          Node *child = edges[i].child.load(std::memory_order_acquire);
          if (child) {
            best_n = child->GetN();
            best_idx = i;
          }
          break;
        }
      }
    }
    if (best_idx < 0) {
      for (int i = 0; i < num_edges; ++i) {
        Node *child = edges[i].child.load(std::memory_order_acquire);
        if (child && child->GetN() > best_n) {
          best_n = child->GetN();
          best_idx = i;
        }
      }
    }

    if (best_idx < 0) {
      if (pv.empty() && num_edges > 0) {
        int best_policy_idx = 0;
        float best_policy = edges[0].GetP();
        for (int i = 1; i < num_edges; ++i) {
          float policy = edges[i].GetP();
          if (policy > best_policy) {
            best_policy = policy;
            best_policy_idx = i;
          }
        }
        pv.push_back(edges[best_policy_idx].move);
      }
      break;
    }
    pv.push_back(edges[best_idx].move);
    node = edges[best_idx].child.load(std::memory_order_acquire);
  }

  return pv;
}

float Search::GetBestQ() const { return GetBestMoveStats().q; }

void Search::SendInfo() {
  if (!info_cb_)
    return;

  const int64_t elapsed_ms =
      SteadyNowMs() - search_start_ms_.load(std::memory_order_acquire);

  uint64_t nn_evals = stats_.nn_evaluations.load(std::memory_order_relaxed);
  uint64_t nodes = stats_.total_nodes.load(std::memory_order_relaxed);
  uint64_t cache_hits = stats_.cache_hits.load(std::memory_order_relaxed);
  uint64_t cache_misses = stats_.cache_misses.load(std::memory_order_relaxed);
  uint64_t nps = elapsed_ms > 0 ? (nodes * 1000) / elapsed_ms : 0;

  float q = GetBestQ();
  int cp;
  if (std::abs(q) > 0.99f) {
    cp = q > 0 ? 10000 : -10000;
  } else {
    cp = static_cast<int>(111.714 * std::tan(1.5621 * q));
  }

  std::vector<Move> pv = GetPV();
  int depth = static_cast<int>(pv.size());
  if (depth < 1)
    depth = 1;

  std::ostringstream ss;
  ss << "info depth " << depth << " seldepth " << depth << " nodes " << nodes
     << " nps " << nps << " time " << elapsed_ms << " score cp " << cp
     << " string nn_evals " << nn_evals << " cache_hits " << cache_hits
     << " cache_misses " << cache_misses;

  if (!pv.empty()) {
    ss << " pv";
    for (const Move &m : pv) {
      ss << " " << UCIEngine::move(m, false);
    }
  }

  if (EnvFlagEnabled("METALFISH_MCTS_ROOT_TRACE")) {
    const int root_trace_moves =
        EnvInt("METALFISH_MCTS_ROOT_TRACE_MOVES", 8, 1, 64);
    const auto root_stats = GetRootMoveStats(root_trace_moves);
    if (!root_stats.empty()) {
      ss << " root";
      ss << std::fixed << std::setprecision(3);
      for (const auto &entry : root_stats) {
        ss << " " << UCIEngine::move(entry.move, false) << ":n=" << entry.visits
           << ":cn=" << entry.current_visits << ":q=" << entry.q
           << ":p=" << entry.policy;
      }
    }
  }

  info_cb_(ss.str());
}

void Search::InjectPVBoost(const Move *pv, int pv_len, int ab_depth,
                           float weight) {
  if (pv_len <= 0)
    return;
  weight = std::clamp(weight, 0.0f, 1.0f);
  if (weight <= 0.0f)
    return;

  std::unique_lock<std::shared_mutex> lock(tree_structure_mutex_);
  float boost = weight * std::min(1.0f, static_cast<float>(ab_depth) / 20.0f);
  Node *node = tree_.Root();

  for (int i = 0; i < pv_len && node && node->NumEdges() > 0; ++i) {
    Edge *edges = node->Edges();
    int num = node->NumEdges();
    bool found = false;

    float total = 0.0f;
    for (int e = 0; e < num; ++e)
      total += edges[e].GetP();
    if (total <= 0.0f)
      break;

    for (int e = 0; e < num; ++e) {
      if (edges[e].move == pv[i]) {
        float depth_boost = boost * (1.0f / (1.0f + 0.5f * i));
        float current = edges[e].GetP();
        float boosted = current * (1.0f + depth_boost);

        float new_total = total - current + boosted;
        if (new_total > 0.0f) {
          float scale = total / new_total;
          for (int j = 0; j < num; ++j) {
            if (j != e) {
              edges[j].SetP(edges[j].GetP() * scale);
            } else {
              edges[j].SetP(boosted * scale);
            }
          }
        }

        node = edges[e].child.load(std::memory_order_relaxed);
        found = true;
        break;
      }
    }
    if (!found)
      break;
  }
}

std::unique_ptr<Search> CreateSearch(const SearchParams &config) {
  std::unique_ptr<Backend> backend;
  if (!config.nn_weights_path.empty()) {
    try {
      backend = std::make_unique<Backend>(
          config.nn_weights_path,
          static_cast<size_t>(std::max(1, config.nn_cache_size)),
          config.GetBackendConfig());
    } catch (const std::exception &e) {
      std::cerr << "[MCTS] CreateSearch: backend creation failed: " << e.what()
                << std::endl;
      return nullptr;
    }
  }
  return std::make_unique<Search>(config, std::move(backend));
}

} // namespace MCTS
} // namespace MetalFish
