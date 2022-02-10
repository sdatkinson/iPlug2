#pragma once

//#define TORCHIN

#if IPLUG_DSP
#include <memory>
#include <vector>
#include "dsp.h"
#endif

#include "IPlug_include_in_plug_hdr.h"

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
  std::unique_ptr<DSP> dsp;

  void ProcessFallback(sample** inputs, sample** outputs, const int nChans, const int nFrames);
#endif
};
