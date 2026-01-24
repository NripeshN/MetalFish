/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/

#include "loader.h"

#include <zlib.h>

#include <algorithm>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace MetalFish {
namespace NN {

namespace {

const std::uint32_t kWeightMagic = 0x1c0;
const int kStartingSize = 8 * 1024 * 1024;  // 8M

std::string DecompressGzip(const std::string& filename) {
  std::string buffer;
  buffer.resize(kStartingSize);
  int bytes_read = 0;

  FILE* fp = fopen(filename.c_str(), "rb");
  if (!fp) {
    throw std::runtime_error("Cannot read weights from " + filename);
  }

  fflush(fp);
  gzFile file = gzdopen(dup(fileno(fp)), "rb");
  fclose(fp);
  
  if (!file) {
    throw std::runtime_error("Cannot process file " + filename);
  }

  while (true) {
    const int sz = gzread(file, &buffer[bytes_read], buffer.size() - bytes_read);
    if (sz < 0) {
      int errnum;
      gzclose(file);
      throw std::runtime_error("gzip error reading file");
    }
    if (sz == static_cast<int>(buffer.size()) - bytes_read) {
      bytes_read = buffer.size();
      buffer.resize(buffer.size() * 2);
    } else {
      bytes_read += sz;
      buffer.resize(bytes_read);
      break;
    }
  }
  gzclose(file);

  return buffer;
}

void FixOlderWeightsFile(WeightsFile* file) {
  using nf = MetalFishNN::NetworkFormat;
  
  auto* net = file->mutable_format()->mutable_network_format();
  const auto has_network_format = file->format().has_network_format();
  
  if (!has_network_format) {
    net->set_input(nf::INPUT_CLASSICAL_112_PLANE);
    net->set_output(nf::OUTPUT_CLASSICAL);
    net->set_network(nf::NETWORK_CLASSICAL_WITH_HEADFORMAT);
    net->set_value(nf::VALUE_CLASSICAL);
    net->set_policy(nf::POLICY_CLASSICAL);
  }
  
  auto network_format = file->format().network_format().network();
  
  if (network_format == nf::NETWORK_CLASSICAL) {
    net->set_network(nf::NETWORK_CLASSICAL_WITH_HEADFORMAT);
    net->set_value(nf::VALUE_CLASSICAL);
    net->set_policy(nf::POLICY_CLASSICAL);
  } else if (network_format == nf::NETWORK_SE) {
    net->set_network(nf::NETWORK_SE_WITH_HEADFORMAT);
    net->set_value(nf::VALUE_CLASSICAL);
    net->set_policy(nf::POLICY_CLASSICAL);
  } else if (network_format == nf::NETWORK_SE_WITH_HEADFORMAT &&
             file->weights().encoder().size() > 0) {
    net->set_network(nf::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT);
    if (file->weights().has_smolgen_w()) {
      net->set_ffn_activation(nf::ACTIVATION_RELU_2);
      net->set_smolgen_activation(nf::ACTIVATION_SWISH);
    }
  } else if (network_format == nf::NETWORK_AB_LEGACY_WITH_MULTIHEADFORMAT) {
    net->set_network(nf::NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT);
  }
  
  if (file->format().network_format().network() == 
      nf::NETWORK_ATTENTIONBODY_WITH_HEADFORMAT) {
    auto weights = file->weights();
    if (weights.has_policy_heads() && weights.has_value_heads()) {
      net->set_network(nf::NETWORK_ATTENTIONBODY_WITH_MULTIHEADFORMAT);
      net->set_input_embedding(nf::INPUT_EMBEDDING_PE_DENSE);
    }
    if (!file->format().network_format().has_input_embedding()) {
      net->set_input_embedding(nf::INPUT_EMBEDDING_PE_MAP);
    }
  }
}

WeightsFile ParseWeightsProto(const std::string& buffer) {
  WeightsFile net;
  if (!net.ParseFromString(buffer)) {
    throw std::runtime_error("Failed to parse protobuf weights file");
  }

  if (net.magic() != kWeightMagic) {
    throw std::runtime_error("Invalid weight file: bad magic number");
  }

  FixOlderWeightsFile(&net);
  return net;
}

}  // namespace

WeightsFile LoadWeightsFromFile(const std::string& filename) {
  auto buffer = DecompressGzip(filename);
  
  if (buffer.size() < 2) {
    throw std::runtime_error("Invalid weight file: too small");
  }

  return ParseWeightsProto(buffer);
}

std::optional<WeightsFile> LoadWeights(std::string_view location) {
  std::string loc(location);
  
  if (loc == "<autodiscover>") {
    auto discovered = DiscoverWeightsFile();
    if (discovered.empty()) {
      return std::nullopt;
    }
    loc = discovered;
  }
  
  return LoadWeightsFromFile(loc);
}

std::string DiscoverWeightsFile() {
  // Check common locations for weights files
  const std::vector<std::string> locations = {
    "networks/",
    "./",
    "../networks/",
  };
  
  const std::vector<std::string> extensions = {
    ".pb.gz",
    ".pb",
  };
  
  for (const auto& dir : locations) {
    for (const auto& ext : extensions) {
      // Look for common network file patterns
      std::string pattern = dir + "*" + ext;
      // Simple check - in real implementation would scan directory
      // For now, just return empty to indicate no autodiscovery
    }
  }
  
  return "";
}

FloatVector DecodeLayer(const MetalFishNN::Weights::Layer& layer) {
  FloatVector result;
  
  const auto& params = layer.params();
  const auto encoding = layer.encoding();
  
  if (encoding == MetalFishNN::Weights::Layer::FLOAT32) {
    // Directcopy float32 data
    result.resize(params.size() / sizeof(float));
    std::memcpy(result.data(), params.data(), params.size());
  } else if (encoding == MetalFishNN::Weights::Layer::FLOAT16 ||
             encoding == MetalFishNN::Weights::Layer::BFLOAT16 ||
             encoding == MetalFishNN::Weights::Layer::LINEAR16) {
    // Decode 16-bit formats
    const size_t count = params.size() / 2;
    result.resize(count);
    
    const float min_val = layer.min_val();
    const float max_val = layer.max_val();
    const float range = max_val - min_val;
    
    for (size_t i = 0; i < count; ++i) {
      uint16_t raw;
      std::memcpy(&raw, params.data() + i * 2, 2);
      
      if (encoding == MetalFishNN::Weights::Layer::LINEAR16) {
        // Linear dequantization
        result[i] = min_val + (raw / 65535.0f) * range;
      } else if (encoding == MetalFishNN::Weights::Layer::FLOAT16) {
        // IEEE 754 half precision
        uint32_t sign = (raw & 0x8000) << 16;
        uint32_t exponent = (raw & 0x7C00) >> 10;
        uint32_t mantissa = (raw & 0x03FF);
        
        uint32_t f32;
        if (exponent == 0) {
          if (mantissa == 0) {
            f32 = sign;
          } else {
            // Denormalized
            f32 = sign | ((exponent + 112) << 23) | (mantissa << 13);
          }
        } else if (exponent == 31) {
          f32 = sign | 0x7F800000 | (mantissa << 13);
        } else {
          f32 = sign | ((exponent + 112) << 23) | (mantissa << 13);
        }
        
        std::memcpy(&result[i], &f32, 4);
      } else {
        // BFLOAT16
        uint32_t f32 = raw << 16;
        std::memcpy(&result[i], &f32, 4);
      }
    }
  } else {
    throw std::runtime_error("Unsupported weight encoding");
  }
  
  return result;
}

}  // namespace NN
}  // namespace MetalFish
