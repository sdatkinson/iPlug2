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
  //try {
  //   get_dsp(std::filesystem::path("C:\\path\\to\\your\\exported\\model"));
  //  this->dsp = get_hard_dsp();  // See get_dsp.cpp, HardCodedModel.h
  //}
  //catch (std::exception& e) {
  //  //std::cerr << "Failed to read DSP module" << std::endl;
  //  //std::cerr << e.what() << std::endl;
  //}
#endif

  // I'm not interested in making this a WAM so removing IPLUG_EDITOR guards so
  // that we can access the get_dsp() inside the button.
// #if IPLUG_EDITOR // http://bit.ly/2S64BDd
  mMakeGraphicsFunc = [&]() {
    return MakeGraphics(*this, PLUG_WIDTH, PLUG_HEIGHT, PLUG_FPS, GetScaleForScreen(PLUG_WIDTH, PLUG_HEIGHT));
  };
  
  mLayoutFunc = [&](IGraphics* pGraphics) {
    pGraphics->AttachCornerResizer(EUIResizerMode::Scale, false);
    pGraphics->AttachPanelBackground(COLOR_GRAY);
    pGraphics->LoadFont("Roboto-Regular", ROBOTO_FN);
    const IRECT b = pGraphics->GetBounds();
    const char* name = "Neural Amp Modeler (Snapshot)";
    pGraphics->AttachControl(new ITextControl(b.GetMidVPadded(50), name, IText(50)));

    // General plugin status message
    pGraphics->AttachControl(
      new ITextControl(b.GetMidVPadded(50).GetVShifted(100), "", IText(14)),
      kStatusIDGeneral
    );

    // Knobs
    {
      const float y_above = -80.0;
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

      // Model parameters on bottom
      // Not for snpashots, TODO remove
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

    // Model loader button
    auto loadModel = [pGraphics, this](IControl* pCaller) {
      WDL_String dir;
      pGraphics->PromptForDirectory(dir);
      if (dir.GetLength())
        this->_get_dsp_and_update_status(dir, pGraphics);
    };
    {
      IVStyle buttonStyle;
      buttonStyle.labelText = IText(16);
      pGraphics->AttachControl(
        new IVButtonControl(
          b.GetCentredInside(200.0, 40.0).GetVShifted(70.0),
          SplashClickActionFunc,
          "Choose model...",
          buttonStyle
        )
      )->SetAnimationEndActionFunction(loadModel);
    }
  };
// #endif
}

bool NeuralAmpModeler::SerializeState(IByteChunk& chunk) const
{
  // Model directory (don't serialize the model itself; we'll just load it again when we unserialize)
  chunk.PutStr(this->_model_path.c_str());
  return SerializeParams(chunk);
}

int NeuralAmpModeler::UnserializeState(const IByteChunk& chunk, int startPos)
{
  WDL_String dir;
  startPos = chunk.GetStr(dir, startPos);
  this->_model_path = std::string(dir.Get());
  this->_dsp = NULL;
  return UnserializeParams(chunk, startPos);
}

void NeuralAmpModeler::OnUIOpen()
{
  Plugin::OnUIOpen();
  // cf loadModel() above
  WDL_String dir(this->_model_path.c_str());
  if (dir.GetLength())
    this->_get_dsp_and_update_status(dir, this->GetUI());
  this->GetUI()->SetAllControlsDirty();
}

void NeuralAmpModeler::_get_dsp_and_update_status(WDL_String& dsp_path, IGraphics* pGraphics)
{
  std::string previous_model;
  try {
    //if (this->_dsp != nullptr) {
    //  this->_dsp.reset();
    //  this->_dsp = NULL;
    //}
    previous_model = this->_model_path;
    this->_staged_dsp = get_dsp(std::filesystem::path(dsp_path.Get()));
    this->_model_path = std::string(dsp_path.Get());
    std::stringstream ss;
    ss << "Loaded model at " << this->_model_path;
    ((ITextControl*)pGraphics->GetControlWithTag(kStatusIDGeneral))->SetStr(ss.str().c_str());
  }
  catch (std::exception& e) {
    std::stringstream ss;
    ss << "FAILED to load model at " << dsp_path.Get();
    ((ITextControl*)pGraphics->GetControlWithTag(kStatusIDGeneral))->SetStr(ss.str().c_str());
    if (this->_staged_dsp != nullptr) {
      this->_staged_dsp = NULL;
    }
    this->_model_path = previous_model;
    std::cerr << "Failed to read DSP module" << std::endl;
    std::cerr << e.what() << std::endl;
  }
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
  
  if (this->_staged_dsp != nullptr) {
    // Move from staged to active DSP
    this->_dsp = std::move(this->_staged_dsp);
    this->_staged_dsp = NULL;
  }
  if (this->_dsp != nullptr) {
    this->_dsp->process(inputs, outputs, nChans, nFrames, input_gain, output_gain, params);
    this->_dsp->finalize_(nFrames);
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
