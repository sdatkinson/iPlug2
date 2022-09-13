#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#if IPLUG_DSP
#include <memory>
#include "dsp.h"
#endif

#include "IPlug_include_in_plug_hdr.h"

const int kNumPresets = 1;

enum EParams
{
  kInputGain = 0,
  kOutputGain,
  // OD
  //kParametricDrive,
  //kParametricLevel,
  //kParametricTone,
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

  // Collect all of the parameters from the display to provide them to to the DSP module
  std::unordered_map<std::string, double> _get_params();
  // If something is wrong, then this implements a fallback so that we still ensure the
  // Required output
  void ProcessFallback(sample** inputs, sample** outputs, const int nChans, const int nFrames);

  // Param names in the order they appear on the GUI
  // OD
  // const std::vector<std::string> _param_names{ "Input", "Output", "Drive", "Level", "Tone"};
  // Snapshot
   const std::vector<std::string> _param_names{ "Input", "Output" };
#endif
};
