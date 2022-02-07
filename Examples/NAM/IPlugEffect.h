#pragma once

//#define TORCHIN

#if IPLUG_DSP
#include <vector>
#ifdef TORCHIN
#include <torch/script.h>
#endif
#endif
#include "IPlug_include_in_plug_hdr.h"

#if IPLUG_DSP
//Clean this up...
#define RECEPTIVE_FIELD 8192;
#endif


const int kNumPresets = 1;

enum EParams
{
  kGain = 0,
  kNumParams
};

using namespace iplug;
using namespace igraphics;

class NeuralAmpModeler final : public Plugin
{
public:
  NeuralAmpModeler(const InstanceInfo& info);

#if IPLUG_DSP // http://bit.ly/2S64BDd
  void ProcessBlock(sample** inputs, sample** outputs, int nFrames) override;

private:
#ifdef TORCHIN
  torch::jit::script::Module module;
#endif
  bool have_module;
  // Hard-coded to model architecture TODO better
  const int receptive_field = RECEPTIVE_FIELD;

  // Input buffer
  const int input_buffer_size = 16384;  // Hard-coded to twice the model's receptive field
  const int input_buffer_channels = 1;  // Mono
  // First location where we add new samples from the input
  int input_buffer_offset;
  std::vector<float> input_buffer;

  // Load in the model and set have_module
  void LoadModel();

  // The main processing logic

  // A fallback in case the model didn't load
  void ProcessFallback(sample** inputs, sample** outputs, const int nFrames, const int nChans);

  // Prepare the input buffer
  void ProcessInputBufferUpdate(sample** inputs, const int nFrames);
  // The actual LibTorch call
  void ProcessTorch(sample** outputs, const int nFrames, const int nChans);
  // Finally, the volume knob
  void ProcessGain(sample** outputs, const int nFrames, const int nChans);
#endif
};
