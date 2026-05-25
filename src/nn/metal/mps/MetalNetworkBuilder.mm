/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#import "MetalNetworkBuilder.h"
#import "../../tables/attention_policy_map.h"
#import "../../weights.h"
#import "NetworkGraph.h"

#include <atomic>
#include <stdexcept>

namespace MetalFish {
namespace NN {
namespace Metal {

namespace {

int NextGraphId() {
  static std::atomic<int> next_graph_id{0};
  return next_graph_id.fetch_add(1, std::memory_order_relaxed);
}

MetalNetworkGraph *GraphOrThrow(int graph_id, const char *operation) {
  if (graph_id < 0) {
    throw std::runtime_error(std::string(operation) +
                             " called before Metal graph initialization");
  }

  MetalNetworkGraph *graph =
      [MetalNetworkGraph getGraphAt:[NSNumber numberWithInt:graph_id]];
  if (graph == nil) {
    throw std::runtime_error(std::string("Metal graph missing during ") +
                             operation);
  }
  return graph;
}

} // namespace

MetalNetworkBuilder::MetalNetworkBuilder(void) {}
MetalNetworkBuilder::~MetalNetworkBuilder(void) {
  if (graph_id >= 0) {
    [MetalNetworkGraph removeGraphAt:[NSNumber numberWithInt:graph_id]];
  }
}

std::string MetalNetworkBuilder::init(int gpu_id) {
  NSArray<id<MTLDevice>> *devices = MTLCopyAllDevices();

  if ((NSUInteger)gpu_id >= [devices count]) {
    [NSException
         raise:@"Could not find device"
        format:@"Could not find a GPU or CPU compute device with specified id"];
    return "";
  }

  if (this->graph_id >= 0) {
    [MetalNetworkGraph removeGraphAt:[NSNumber numberWithInt:this->graph_id]];
  }
  this->graph_id = NextGraphId();

  [MetalNetworkGraph graphWithDevice:devices[gpu_id]
                               index:[NSNumber numberWithInt:this->graph_id]];

  this->gpu_id = gpu_id;

  return std::string([devices[gpu_id].name UTF8String]);
}

void MetalNetworkBuilder::build(int kInputPlanes, MultiHeadWeights &weights,
                                InputEmbedding embedding, bool attn_body,
                                bool attn_policy, bool conv_policy, bool wdl,
                                bool moves_left, Activations &activations,
                                std::string &policy_head,
                                std::string &value_head,
                                const std::vector<NetworkOutputTarget>
                                    &decoded_output_targets) {
  MetalNetworkGraph *graph = GraphOrThrow(this->graph_id, "Metal graph build");
  NSString *defaultActivation =
      [NSString stringWithUTF8String:activations.default_activation.c_str()];
  NSString *smolgenActivation =
      [NSString stringWithUTF8String:activations.smolgen_activation.c_str()];
  NSString *ffnActivation =
      [NSString stringWithUTF8String:activations.ffn_activation.c_str()];
  NSString *policyHead = [NSString stringWithUTF8String:policy_head.c_str()];
  NSString *valueHead = [NSString stringWithUTF8String:value_head.c_str()];

  MPSGraphTensor *layer = [graph inputPlaceholderWithInputChannels:kInputPlanes
                                                             label:@"inputs"];

  MPSGraphTensor *maskTensor =
      [graph maskPlaceholderWithInputChannels:kInputPlanes
                                        label:@"inputs/mask"];

  layer = [graph expandInputTensorWithMask:maskTensor
                                     input:layer
                                     label:@"inputs/expand"];

  const NSUInteger kernelSize = 3;
  const bool isPeDenseEmbedding =
      embedding == InputEmbedding::INPUT_EMBEDDING_PE_DENSE;

  if (weights.has_smolgen) {
    [graph setGlobalSmolgenWeights:&weights.smolgen_w[0]];
  }

  if (weights.residual.size() > 0) {

    const NSUInteger channelSize =
        weights.input.weights.size() / (kInputPlanes * kernelSize * kernelSize);

    layer = [graph addConvolutionBlockWithParent:layer
                                  outputChannels:channelSize
                                      kernelSize:kernelSize
                                         weights:&weights.input.weights[0]
                                          biases:&weights.input.biases[0]
                                      activation:defaultActivation
                                           label:@"input/conv"];

    for (size_t i = 0; i < weights.residual.size(); i++) {
      const bool hasSe = weights.residual[i].has_se;
      float *seWeights1 = hasSe ? weights.residual[i].se.w1.data() : nullptr;
      float *seBiases1 = hasSe ? weights.residual[i].se.b1.data() : nullptr;
      float *seWeights2 = hasSe ? weights.residual[i].se.w2.data() : nullptr;
      float *seBiases2 = hasSe ? weights.residual[i].se.b2.data() : nullptr;
      layer = [graph
          addResidualBlockWithParent:layer
                      outputChannels:channelSize
                          kernelSize:kernelSize
                            weights1:&weights.residual[i].conv1.weights[0]
                             biases1:&weights.residual[i].conv1.biases[0]
                            weights2:&weights.residual[i].conv2.weights[0]
                             biases2:&weights.residual[i].conv2.biases[0]
                               label:[NSString stringWithFormat:@"block_%zu", i]
                               hasSe:hasSe ? YES : NO
                          seWeights1:seWeights1
                           seBiases1:seBiases1
                          seWeights2:seWeights2
                           seBiases2:seBiases2
                         seFcOutputs:weights.residual[i].se.b1.size()
                          activation:defaultActivation];
    }
  }

  if (attn_body) {
    assert(weights.ip_emb_b.size() > 0);

    layer = [graph transposeChannelsWithTensor:layer
                                     withShape:@[ @(-1), @64, layer.shape[1] ]
                                         label:@"input/nchw_nhwc"];

    if (weights.residual.size() == 0) {
      // No residual means pure transformer, so process input position encoding.
      if (isPeDenseEmbedding) {
        layer = [graph
            dynamicPositionEncodingWithTensor:layer
                                        width:weights.ip_emb_preproc_b.size() /
                                              64
                                      weights:&weights.ip_emb_preproc_w[0]
                                       biases:&weights.ip_emb_preproc_b[0]
                                        label:@"input/position_encoding"];
      } else {
        layer = [graph positionEncodingWithTensor:layer
                                        withShape:@[ @64, @64 ]
                                          weights:&Tables::kPosEncoding[0][0]
                                             type:nil
                                            label:@"input/position_encoding"];
      }
    }

    layer = [graph addFullyConnectedLayerWithParent:layer
                                     outputChannels:weights.ip_emb_b.size()
                                            weights:&weights.ip_emb_w[0]
                                             biases:&weights.ip_emb_b[0]
                                         activation:defaultActivation
                                              label:@"input/embedding"];

    if (isPeDenseEmbedding) {
      layer =
          [graph addLayerNormalizationWithParent:layer
                           scaledSecondaryTensor:nil
                                          gammas:&weights.ip_emb_ln_gammas[0]
                                           betas:&weights.ip_emb_ln_betas[0]
                                           alpha:1.0
                                         epsilon:1e-3
                                           label:@"input/embedding/ln"];
    }

    // # !!! input gate
    // flow = ma_gating(flow, name=name+'embedding')
    // def ma_gating(inputs, name):
    //     out = Gating(name=name+'/mult_gate', additive=False)(inputs)
    //     out = Gating(name=name+'/add_gate', additive=True)(out)
    if (weights.ip_mult_gate.size() > 0) {
      layer = [graph addGatingLayerWithParent:layer
                                      weights:&weights.ip_mult_gate[0]
                                withOperation:@"mult"
                                        label:@"input/mult_gate"];
    }
    if (weights.ip_add_gate.size() > 0) {
      layer = [graph addGatingLayerWithParent:layer
                                      weights:&weights.ip_add_gate[0]
                                withOperation:@"add"
                                        label:@"input/add_gate"];
    }

    float alpha = (float)pow(2.0 * weights.encoder.size(), -0.25);
    if (isPeDenseEmbedding) {
      MPSGraphTensor *ffn = [graph
          addFullyConnectedLayerWithParent:layer
                            outputChannels:weights.ip_emb_ffn.dense1_b.size()
                                   weights:&weights.ip_emb_ffn.dense1_w[0]
                                    biases:&weights.ip_emb_ffn.dense1_b[0]
                                activation:ffnActivation
                                     label:@"input/embedding/ffn/dense1"];

      ffn = [graph
          addFullyConnectedLayerWithParent:ffn
                            outputChannels:weights.ip_emb_ffn.dense2_b.size()
                                   weights:&weights.ip_emb_ffn.dense2_w[0]
                                    biases:&weights.ip_emb_ffn.dense2_b[0]
                                activation:nil
                                     label:@"input/embedding/ffn/dense2"];

      layer = [graph
          addLayerNormalizationWithParent:layer
                    scaledSecondaryTensor:ffn
                                   gammas:&weights.ip_emb_ffn_ln_gammas[0]
                                    betas:&weights.ip_emb_ffn_ln_betas[0]
                                    alpha:alpha
                                  epsilon:1e-3
                                    label:@"input/embedding/ffn_ln"];
    }

    for (size_t i = 0; i < weights.encoder.size(); i++) {
      layer = [graph
          addEncoderLayerWithParent:layer
                      legacyWeights:weights.encoder[i]
                              heads:weights.encoder_head_count
                      embeddingSize:weights.ip_emb_b.size()
                  smolgenActivation:smolgenActivation
                      ffnActivation:ffnActivation
                              alpha:alpha
                            epsilon:isPeDenseEmbedding ? 1e-3 : 1e-6
                           normtype:@"layernorm"
                              label:[NSString
                                        stringWithFormat:@"encoder_%zu", i]];
    }
  }

  MPSGraphTensor *policy;
  if (attn_policy && !attn_body) {
    policy = [graph transposeChannelsWithTensor:layer
                                      withShape:@[ @(-1), @64, layer.shape[1] ]
                                          label:@"policy/nchw_nhwc"];
  } else {
    policy = layer;
  }

  policy =
      [graph makePolicyHeadWithTensor:policy
                      attentionPolicy:attn_policy
                    convolutionPolicy:conv_policy
                        attentionBody:attn_body
                    defaultActivation:defaultActivation
                    smolgenActivation:smolgenActivation
                        ffnActivation:ffnActivation
                           policyHead:weights.policy_heads.at(policy_head)
                                label:[NSString stringWithFormat:@"policy/%@",
                                                                 policyHead]];

  MPSGraphTensor *value =
      [graph makeValueHeadWithTensor:layer
                       attentionBody:attn_body
                             wdlHead:wdl
                   defaultActivation:defaultActivation
                           valueHead:weights.value_heads.at(value_head)
                               label:[NSString stringWithFormat:@"value/%@",
                                                                valueHead]];

  MPSGraphTensor *mlh = nil;
  if (moves_left) {
    if (attn_body) {
      mlh = [graph addFullyConnectedLayerWithParent:layer
                                     outputChannels:weights.ip_mov_b.size()
                                            weights:&weights.ip_mov_w[0]
                                             biases:&weights.ip_mov_b[0]
                                         activation:defaultActivation
                                              label:@"moves_left/embedding"];
    } else {
      mlh =
          [graph addConvolutionBlockWithParent:layer
                                outputChannels:weights.moves_left.biases.size()
                                    kernelSize:1
                                       weights:&weights.moves_left.weights[0]
                                        biases:&weights.moves_left.biases[0]
                                    activation:defaultActivation
                                         label:@"moves_left/conv"];
    }

    mlh = [graph flatten2DTensor:mlh axis:1 name:@"moves_left/flatten"];

    mlh = [graph addFullyConnectedLayerWithParent:mlh
                                   outputChannels:weights.ip1_mov_b.size()
                                          weights:&weights.ip1_mov_w[0]
                                           biases:&weights.ip1_mov_b[0]
                                       activation:defaultActivation
                                            label:@"moves_left/fc1"];

    mlh = [graph addFullyConnectedLayerWithParent:mlh
                                   outputChannels:weights.ip2_mov_b.size()
                                          weights:&weights.ip2_mov_w[0]
                                           biases:&weights.ip2_mov_b[0]
                                       activation:@"relu"
                                            label:@"moves_left/fc2"];
  }

  NSMutableArray<MPSGraphTensor *> *resultTensors =
      [NSMutableArray arrayWithCapacity:decoded_output_targets.size()];
  for (NetworkOutputTarget target : decoded_output_targets) {
    switch (target) {
    case NetworkOutputTarget::Policy:
      [resultTensors addObject:policy];
      break;
    case NetworkOutputTarget::Value:
      [resultTensors addObject:value];
      break;
    case NetworkOutputTarget::MovesLeft:
      if (mlh == nil) {
        throw std::runtime_error(
            "Metal graph requested moves-left output without a moves-left head");
      }
      [resultTensors addObject:mlh];
      break;
    case NetworkOutputTarget::RawPolicy:
      throw std::runtime_error(
          "Metal graph does not expose raw policy as a decoded output");
    }
  }
  [graph setResultTensors:resultTensors];
}

void MetalNetworkBuilder::forwardEval(float *inputs, uint64_t *masks,
                                      int batchSize,
                                      std::vector<float *> output_mems) {
  @autoreleasepool {
    MetalNetworkGraph *graph =
        GraphOrThrow(this->graph_id, "Metal graph inference");
    [graph runInferenceWithBatchSize:batchSize
                              inputs:inputs
                               masks:masks
                             outputs:&output_mems[0]];
  }
}

} // namespace Metal
} // namespace NN
} // namespace MetalFish
