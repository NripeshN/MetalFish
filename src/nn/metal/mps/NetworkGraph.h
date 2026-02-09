/*
  MetalFish - A GPU-accelerated UCI chess engine
  Copyright (C) 2025 Nripesh Niketan

  Licensed under GPL-3.0
*/
#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#import "../../weights.h"

@interface MPSGraphTensor(MetalExtensions)

-(NSUInteger) size;

-(NSUInteger) sizeOfDimensions:(NSArray<NSNumber *> * __nonnull)dimensions;

@end

static MPSImageFeatureChannelFormat fcFormat = MPSImageFeatureChannelFormatFloat16;

@interface MetalNetworkGraph : MPSGraph {
@public
    // Keep the device and command queue objects around for ease of use.
    MPSGraphDevice * _device;
    id<MTLCommandQueue> _queue;
    NSUInteger _maxBatchSize;

    // Input tensor and tensor data placeholders.
    MPSGraphTensor * _inputTensor;
    MPSGraphTensor * _maskTensor;

    // Variables to track results of graph inference.
    NSArray<MPSGraphTensor *> * _resultTensors;
    NSArray<MPSGraphTensor *> * _targetTensors;
    NSMutableDictionary<NSNumber *, MPSGraphTensorDataDictionary *> * _resultDataDicts;
    NSMutableDictionary<NSString *, MPSGraphTensor *> * _readVariables;

    // Variables for triple buffering
    dispatch_semaphore_t _doubleBufferingSemaphore;

    // Global smolgen weights.
    float * __nullable _globalSmolgenWeights;
}

+(MetalNetworkGraph * _Nonnull) getGraphAt:(NSNumber * _Nonnull)index;

+(void) graphWithDevice:(id<MTLDevice> __nonnull)device
                  index:(NSNumber * _Nonnull)index
               maxBatch:(NSUInteger)maxBatch;

-(nonnull instancetype) initWithDevice:(id<MTLDevice> __nonnull)device
                              maxBatch:(NSUInteger)maxBatch;

-(nonnull MPSGraphTensor *) inputPlaceholderWithInputChannels:(NSUInteger)channels
                                                        label:(NSString * __nullable)label;

-(nonnull MPSGraphTensor *) maskPlaceholderWithInputChannels:(NSUInteger)channels
                                                       label:(NSString * __nullable)label;

-(nonnull MPSGraphTensor *) expandInputTensorWithMask:(MPSGraphTensor * __nonnull)maskTensor
                                                input:(MPSGraphTensor * __nonnull)inputTensor
                                                label:(NSString * __nonnull)label;

- (nonnull MPSGraphTensor *) broadcastByStackingTensor:(MPSGraphTensor * __nonnull)input
                                                  axis:(NSInteger)axis
                                                 times:(NSUInteger)times
                                                  name:(NSString * __nonnull)name;

-(nonnull MPSGraphTensor *) addConvolutionBlockWithParent:(MPSGraphTensor * __nonnull)parent
                                           outputChannels:(NSUInteger)outputChannels
                                               kernelSize:(NSUInteger)kernelSize
                                                  weights:(float * __nonnull)weights
                                                   biases:(float * __nonnull)biases
                                               activation:(NSString * __nullable)activation
                                                    label:(NSString * __nonnull)label;

-(nonnull MPSGraphTensor *) addResidualBlockWithParent:(MPSGraphTensor * __nonnull)parent
                                        outputChannels:(NSUInteger)outputChannels
                                            kernelSize:(NSUInteger)kernelSize
                                              weights1:(float * __nonnull)weights1
                                               biases1:(float * __nonnull)biases1
                                              weights2:(float * __nonnull)weights2
                                               biases2:(float * __nonnull)biases2
                                                 label:(NSString * __nonnull)label
                                                 hasSe:(BOOL)hasSe
                                            seWeights1:(float * __nullable)seWeights1
                                             seBiases1:(float * __nullable)seBiases1
                                            seWeights2:(float * __nullable)seWeights2
                                             seBiases2:(float * __nullable)seBiases2
                                           seFcOutputs:(NSUInteger)seFcOutputs
                                            activation:(NSString * __nullable)activation;

-(nonnull MPSGraphTensor *) addFullyConnectedLayerWithParent:(MPSGraphTensor * __nonnull)parent
                                              outputChannels:(NSUInteger)outputChannels
                                                     weights:(float * __nonnull)weights
                                                      biases:(float * __nullable)biases
                                                  activation:(NSString * __nullable)activation
                                                       label:(NSString * __nonnull)label;

-(nonnull MPSGraphTensor *) addEncoderLayerWithParent:(MPSGraphTensor * __nonnull)parent
                                        legacyWeights:(MetalFish::NN::MultiHeadWeights::EncoderLayer &)weights
                                                heads:(NSUInteger)heads
                                        embeddingSize:(NSUInteger)embeddingSize
                                    smolgenActivation:(NSString * __nullable)smolgenActivation
                                        ffnActivation:(NSString * __nonnull)ffnActivation
                                                alpha:(float)alpha
                                              epsilon:(float)epsilon
                                             normtype:(NSString * __nonnull)normtype
                                                label:(NSString * __nonnull)label;

-(nonnull MPSGraphTensor *) addLayerNormalizationWithParent:(MPSGraphTensor * __nonnull)parent
                                      scaledSecondaryTensor:(MPSGraphTensor * __nullable)secondary
                                                     gammas:(float * __nonnull)gammas
                                                      betas:(float * __nonnull)betas
                                                      alpha:(float)alpha
                                                    epsilon:(float)epsilon
                                                      label:(NSString * __nonnull)label;

-(nonnull MPSGraphTensor *) addRmsNormalizationWithParent:(MPSGraphTensor * __nonnull)parent
                                    scaledSecondaryTensor:(MPSGraphTensor * __nullable)secondary
                                                   gammas:(float * __nonnull)gammas
                                                    alpha:(float)alpha
                                                    label:(NSString * __nonnull)label;

-(nonnull MPSGraphTensor *) scaledMHAMatmulWithQueries:(MPSGraphTensor * __nonnull)queries
                                              withKeys:(MPSGraphTensor * __nonnull)keys
                                            withValues:(MPSGraphTensor * __nonnull)values
                                                 heads:(NSUInteger)heads
                                                parent:(MPSGraphTensor * __nonnull)parent
                                               smolgen:(MetalFish::NN::MultiHeadWeights::Smolgen * __nullable)smolgen
                                     smolgenActivation:(NSString * __nullable)smolgenActivation
                                                 label:(NSString * __nonnull)label;

-(nonnull MPSGraphTensor *) scaledQKMatmulWithQueries:(MPSGraphTensor * __nonnull)queries
                                             withKeys:(MPSGraphTensor * __nonnull)keys
                                                scale:(float)scale
                                                label:(NSString * __nonnull)label;

-(nonnull MPSGraphTensor *) attentionPolicyPromoMatmulConcatWithParent:(MPSGraphTensor * __nonnull)parent
                                                              withKeys:(MPSGraphTensor * __nonnull)keys
                                                               weights:(float * __nonnull)weights
                                                             inputSize:(NSUInteger)inputSize
                                                            outputSize:(NSUInteger)outputSize
                                                             sliceFrom:(NSUInteger)sliceFrom
                                                           channelSize:(NSUInteger)channelSize
                                                                 label:(NSString * __nonnull)label;

-(nonnull MPSGraphTensor *) transposeChannelsWithTensor:(MPSGraphTensor * __nonnull)tensor
                                              withShape:(MPSShape * __nonnull)withShape
                                                  label:(NSString * __nonnull)label;

-(nonnull MPSGraphTensor *) positionEncodingWithTensor:(MPSGraphTensor * __nonnull)tensor
                                             withShape:(MPSShape * __nonnull)shape
                                               weights:(const float * __nonnull)encodings
                                                  type:(NSString * __nullable)type
                                                 label:(NSString * __nonnull)label;

-(nonnull MPSGraphTensor *) dynamicPositionEncodingWithTensor:(MPSGraphTensor * __nonnull)tensor
                                                        width:(const NSUInteger)width
                                                      weights:(float * __nonnull)weights
                                                       biases:(float * __nonnull)biases
                                                        label:(NSString * __nonnull)label;

-(nonnull MPSGraphTensor *) addGatingLayerWithParent:(MPSGraphTensor * __nonnull)parent
                                             weights:(const float * __nonnull)weights
                                       withOperation:(NSString * __nonnull)op
                                               label:(NSString * __nonnull)label;

-(nonnull MPSGraphTensor *) makePolicyHeadWithTensor:(MPSGraphTensor * __nonnull)policy
                                     attentionPolicy:(BOOL)attentionPolicy
                                   convolutionPolicy:(BOOL)convolutionPolicy
                                       attentionBody:(BOOL)attentionBody
                                   defaultActivation:(NSString * __nullable)defaultActivation
                                   smolgenActivation:(NSString * __nullable)smolgenActivation
                                       ffnActivation:(NSString * __nullable)ffnActivation
                                          policyHead:(MetalFish::NN::MultiHeadWeights::PolicyHead &)head
                                               label:(NSString * __nonnull)label;

-(nonnull MPSGraphTensor *) makeValueHeadWithTensor:(MPSGraphTensor * __nonnull)value
                                      attentionBody:(BOOL)attentionBody
                                            wdlHead:(BOOL)wdl
                                  defaultActivation:(NSString * __nullable)defaultActivation
                                          valueHead:(MetalFish::NN::MultiHeadWeights::ValueHead &)head
                                              label:(NSString * __nonnull)label;

-(void) setGlobalSmolgenWeights:(float * __nonnull)weights;

-(void) setResultTensors:(NSArray<MPSGraphTensor *> * __nonnull)results;

-(nonnull NSArray<MPSGraphTensor *> *) runInferenceWithBatchSize:(NSUInteger)batchSize
                                                          inputs:(float * __nonnull)inputs
                                                           masks:(uint64_t * __nonnull)masks
                                                         outputs:(float * __nonnull * __nonnull)outputBuffers;

-(nonnull MPSCommandBuffer *) runCommandSubBatchWithInputs:(float * __nonnull)inputs
                                                     masks:(uint64_t * __nonnull)masks
                                                  subBatch:(NSUInteger)subBatch
                                              subBatchSize:(NSUInteger)subBatchSize;

-(void) copyResultsToBuffers:(float * __nonnull * __nonnull)outputBuffers
                subBatchSize:(NSUInteger)subBatchSize;

@end
