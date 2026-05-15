/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "proto/net.pb.h"

namespace MetalFish {
namespace NN {

struct BaseWeights {
  explicit BaseWeights(const MetalFishNN::Weights &weights);

  using Vec = std::vector<float>;
  struct ConvBlock {
    explicit ConvBlock(const MetalFishNN::Weights::ConvBlock &block);

    Vec weights;
    Vec biases;
    Vec bn_gammas;
    Vec bn_betas;
    Vec bn_means;
    Vec bn_stddivs;
  };

  struct SEunit {
    explicit SEunit(const MetalFishNN::Weights::SEunit &se);
    Vec w1;
    Vec b1;
    Vec w2;
    Vec b2;
  };

  struct Residual {
    explicit Residual(const MetalFishNN::Weights::Residual &residual);
    ConvBlock conv1;
    ConvBlock conv2;
    SEunit se;
    bool has_se;
  };

  struct Smolgen {
    explicit Smolgen(const MetalFishNN::Weights::Smolgen &smolgen);
    Vec compress;
    Vec dense1_w;
    Vec dense1_b;
    Vec ln1_gammas;
    Vec ln1_betas;
    Vec dense2_w;
    Vec dense2_b;
    Vec ln2_gammas;
    Vec ln2_betas;
  };

  struct MHA {
    explicit MHA(const MetalFishNN::Weights::MHA &mha);
    Vec q_w;
    Vec q_b;
    Vec k_w;
    Vec k_b;
    Vec v_w;
    Vec v_b;
    Vec dense_w;
    Vec dense_b;
    Smolgen smolgen;
    bool has_smolgen;
  };

  struct FFN {
    explicit FFN(const MetalFishNN::Weights::FFN &mha);
    Vec dense1_w;
    Vec dense1_b;
    Vec dense2_w;
    Vec dense2_b;
  };

  struct EncoderLayer {
    explicit EncoderLayer(const MetalFishNN::Weights::EncoderLayer &encoder);
    MHA mha;
    Vec ln1_gammas;
    Vec ln1_betas;
    FFN ffn;
    Vec ln2_gammas;
    Vec ln2_betas;
  };

  ConvBlock input;

  Vec ip_emb_preproc_w;
  Vec ip_emb_preproc_b;

  Vec ip_emb_w;
  Vec ip_emb_b;

  Vec ip_emb_ln_gammas;
  Vec ip_emb_ln_betas;

  Vec ip_mult_gate;
  Vec ip_add_gate;

  FFN ip_emb_ffn;
  Vec ip_emb_ffn_ln_gammas;
  Vec ip_emb_ffn_ln_betas;

  std::vector<EncoderLayer> encoder;
  int encoder_head_count;

  std::vector<Residual> residual;

  ConvBlock moves_left;
  Vec ip_mov_w;
  Vec ip_mov_b;
  Vec ip1_mov_w;
  Vec ip1_mov_b;
  Vec ip2_mov_w;
  Vec ip2_mov_b;

  Vec smolgen_w;
  bool has_smolgen;
};

struct LegacyWeights : public BaseWeights {
  explicit LegacyWeights(const MetalFishNN::Weights &weights);

  ConvBlock policy1;
  ConvBlock policy;
  Vec ip_pol_w;
  Vec ip_pol_b;
  Vec ip2_pol_w;
  Vec ip2_pol_b;
  Vec ip3_pol_w;
  Vec ip3_pol_b;
  Vec ip4_pol_w;
  int pol_encoder_head_count;
  std::vector<EncoderLayer> pol_encoder;

  ConvBlock value;
  Vec ip_val_w;
  Vec ip_val_b;
  Vec ip1_val_w;
  Vec ip1_val_b;
  Vec ip2_val_w;
  Vec ip2_val_b;
};

struct MultiHeadWeights : public BaseWeights {
  explicit MultiHeadWeights(const MetalFishNN::Weights &weights);

  struct PolicyHead {
    explicit PolicyHead(const MetalFishNN::Weights::PolicyHead &policyhead,
                        Vec &w, Vec &b);

  private:
    Vec _ip_pol_w;
    Vec _ip_pol_b;

  public:
    Vec &ip_pol_w;
    Vec &ip_pol_b;
    ConvBlock policy1;
    ConvBlock policy;
    Vec ip2_pol_w;
    Vec ip2_pol_b;
    Vec ip3_pol_w;
    Vec ip3_pol_b;
    Vec ip4_pol_w;
    int pol_encoder_head_count;
    std::vector<EncoderLayer> pol_encoder;
  };

  struct ValueHead {
    explicit ValueHead(const MetalFishNN::Weights::ValueHead &valuehead);
    ConvBlock value;
    Vec ip_val_w;
    Vec ip_val_b;
    Vec ip1_val_w;
    Vec ip1_val_b;
    Vec ip2_val_w;
    Vec ip2_val_b;
    Vec ip_val_err_w;
    Vec ip_val_err_b;
  };

private:
  Vec ip_pol_w;
  Vec ip_pol_b;

public:
  std::unordered_map<std::string, ValueHead> value_heads;
  std::unordered_map<std::string, PolicyHead> policy_heads;
};

enum InputEmbedding {
  INPUT_EMBEDDING_NONE = 0,
  INPUT_EMBEDDING_PE_MAP = 1,
  INPUT_EMBEDDING_PE_DENSE = 2,
};

} // namespace NN
} // namespace MetalFish
