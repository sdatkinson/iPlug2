#pragma once

#include <string>
#include <vector>
#include <unordered_map>

#if IPLUG_DSP
#include <memory>
#include "dsp.h"
#endif

#include "IPlug_include_in_plug_hdr.h"

const int kNumPresets = 1;  // ?

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

enum EStatusIDs
{
  kStatusIDGeneral = 1000,
  kStatusIDNumStatusIDs
};

using namespace iplug;
using namespace igraphics;

class NeuralAmpModeler final : public Plugin
{
public:
  NeuralAmpModeler(const InstanceInfo& info);

  bool SerializeState(IByteChunk& chunk) const override;
  int UnserializeState(const IByteChunk& chunk, int startPos) override;
  void OnUIOpen() override;

#if IPLUG_DSP // http://bit.ly/2S64BDd
  void ProcessBlock(sample** inputs, sample** outputs, int nFrames) override;
#endif

private:
  // Try to load the dsp and update the status message accordingly
  void _get_dsp_and_update_status(WDL_String& dsp_path, IGraphics* pGraphics);

  // Path to where the model was saved.
  std::string _model_path = "";
#ifdef IPLUG_DSP
  // The DSP actually being used:
  std::unique_ptr<DSP> _dsp = NULL;
  // Manages switching what DSP is being used.
  std::unique_ptr<DSP> _staged_dsp = NULL;

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

