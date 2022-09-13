#include <filesystem>
#include <iostream>
#include <utility>

#include <cmath>  // pow

#include "IPlugEffect.h"
#include "IPlug_include_in_plug_src.h"
#include "IControls.h"

NeuralAmpModeler::NeuralAmpModeler(const InstanceInfo& info)
: Plugin(info, MakeConfig(kNumParams, kNumPresets))
{
  // Input & output levels
  GetParam(kInputGain)->InitGain(this->_param_names[kInputGain].c_str(), 0.0, -20.0, 20.0, 0.1);
  GetParam(kOutputGain)->InitGain(this->_param_names[kOutputGain].c_str(), 0.0, -20.0, 20.0, 0.1);
  // The other knobs
  for (int i = 2; i<kNumParams; i++)
    // FIXME use default values somehow
    GetParam(i)->InitDouble(this->_param_names[i].c_str(), 0.5, 0.0, 1.0, 0.01);
  

#if IPLUG_DSP
  try {
     //this->dsp = get_dsp(std::filesystem::path("C:\\path\\to\\your\\exported\\model"));
     this->dsp = get_hard_dsp();  // See get_dsp.cpp, HardCodedModel.h
  }
  catch (std::exception& e) {
    std::cerr << "Failed to read DSP module" << std::endl;
    std::cerr << e.what() << std::endl;
  }
#endif

#if IPLUG_EDITOR // http://bit.ly/2S64BDd
  mMakeGraphicsFunc = [&]() {
    return MakeGraphics(*this, PLUG_WIDTH, PLUG_HEIGHT, PLUG_FPS, GetScaleForScreen(PLUG_WIDTH, PLUG_HEIGHT));
  };
  
  mLayoutFunc = [&](IGraphics* pGraphics) {
    pGraphics->AttachCornerResizer(EUIResizerMode::Scale, false);
    pGraphics->AttachPanelBackground(COLOR_GRAY);
    pGraphics->LoadFont("Roboto-Regular", ROBOTO_FN);
    const IRECT b = pGraphics->GetBounds();
    const char* name = "NAM (Deluxe Reverb)";
    pGraphics->AttachControl(new ITextControl(b.GetMidVPadded(50), name, IText(50)));
    // Knobs
    {
      const float y_above = -100.0;
      const float y_below = 100.0;
      const float dx = 100.0;
      // Input & output strength on top:
      assert(kNumParams >= 2);
      {
        const float h_offset = -float(0.5) * float(2-1) * dx;
        for (int i = 0; i < 2; i++)
        {
          const float h_shift = dx * float(i);
          pGraphics->AttachControl(
            new IVKnobControl(
              b.GetCentredInside(100.0).GetVShifted(y_above).GetHShifted(h_offset + h_shift),
              i
            )
          );
        }
      }

      // Model parameters on bottom:
      {
        const int num_model_parameters = kNumParams - 2;
        const float h_offset = -float(0.5) * float(num_model_parameters - 1) * dx;
        for (int i = 2; i < kNumParams; i++)
        {
          const float h_shift = dx * float(i-2);
          pGraphics->AttachControl(
            new IVKnobControl(
              b.GetCentredInside(100.0).GetVShifted(y_below).GetHShifted(h_offset + h_shift),
              i
            )
          );
        }
      }
    }
  };
#endif
}

//=============================================================================
#if IPLUG_DSP

void NeuralAmpModeler::ProcessBlock(sample** inputs, sample** outputs, int nFrames)
{
  if (nFrames == 0)
    return;

  const int nChans = this->NOutChansConnected();
  const double input_gain = pow(10.0, this->GetParam(kInputGain)->Value() / 10.0);
  const double output_gain = pow(10.0, this->GetParam(kOutputGain)->Value() / 10.0);
  const std::unordered_map<std::string, double> params = this->_get_params();
  if (this->dsp != nullptr) {
    this->dsp->process(inputs, outputs, nChans, nFrames, input_gain, output_gain, params);
    this->dsp->finalize_(nFrames);
  }
  else
    this->ProcessFallback(inputs, outputs, nChans, nFrames);
}

std::unordered_map<std::string, double> NeuralAmpModeler::_get_params()
{
  std::unordered_map<std::string, double> params;
  // Input and output gain are params 0 and 1; the rest is here.
  // Other params
  for (int i = 2; i < kNumParams; i++)
     params[this->_param_names[i]] = this->GetParam(i)->Value();
    //params.insert(std::make_pair<std::string, double>(this->_param_names[i], this->GetParam(i)->Value()));
  return params;
}

void NeuralAmpModeler::ProcessFallback(sample** inputs, sample** outputs, const int nChans, const int nFrames)
{
  for (int c = 0; c < nChans; c++)
    for (int s = 0; s < nFrames; s++)
      outputs[c][s] = inputs[c][s];
}

#endif
