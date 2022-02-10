#include <filesystem>
#include <iostream>
#include <memory>

#include "IPlugEffect.h"
#include "IPlug_include_in_plug_src.h"
#include "IControls.h"

NeuralAmpModeler::NeuralAmpModeler(const InstanceInfo& info)
: Plugin(info, MakeConfig(kNumParams, kNumPresets))
{
  GetParam(kGain)->InitDouble("Gain", 50., 0., 100.0, 0.01, "%");

#if IPLUG_DSP
  this->dsp = get_dsp("C:\\Users\\steve\\src\\neural-amp-modeler-2\\exported_models\\model_linear_test");
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
#ifdef TORCHIN
    const char* name = "NAM2 (Torch)";
#else
    const char* name = "NAM2";
#endif
    pGraphics->AttachControl(new ITextControl(b.GetMidVPadded(50), name, IText(50)));
    pGraphics->AttachControl(new IVKnobControl(b.GetCentredInside(100).GetVShifted(-100), kGain));
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
  const double gain = GetParam(kGain)->Value() / 100.0;

  if (this->dsp != nullptr) {
    this->dsp->process(inputs, outputs, nChans, nFrames);
    this->dsp->finalize(nFrames);
    this->dsp->process_gain(outputs, nChans, nFrames, gain);
  }
  else
    this->ProcessFallback(inputs, outputs, nChans, nFrames);
}

void NeuralAmpModeler::ProcessFallback(sample** inputs, sample** outputs, const int nChans, const int nFrames)
{
  for (int c = 0; c < nChans; c++)
    for (int s = 0; s < nFrames; s++)
      outputs[c][s] = inputs[c][s];
}

#endif
