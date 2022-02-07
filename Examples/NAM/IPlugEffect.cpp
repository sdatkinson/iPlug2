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
  this->LoadModel();
  if (this->have_module) {
    this->input_buffer.resize(this->input_buffer_size);
    this->input_buffer_offset = this->receptive_field - 1;
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
#ifdef TORCHIN
    const char* name = "NAM2 (Torch)";
#else
    const char* name = "NAM2 (No Torch)";
#endif
    pGraphics->AttachControl(new ITextControl(b.GetMidVPadded(50), name, IText(50)));
    pGraphics->AttachControl(new IVKnobControl(b.GetCentredInside(100).GetVShifted(-100), kGain));
  };
#endif
}

//================================================================================================================
#if IPLUG_DSP
void NeuralAmpModeler::LoadModel()
{
  // Letsgooooo
  // https://pytorch.org/tutorials/advanced/cpp_export.html
  // Config thank you based TDS
  // https://towardsdatascience.com/setting-up-a-c-project-in-visual-studio-2019-with-libtorch-1-6-ad8a0e49e82c
  // Accessed 2022-02-06

  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
#ifdef TORCHIN  // IPlugEffect.h
    const char* filename = "C:\\Users\\steve\\src\\neural-amp-modeler-2\\bin\\train\\outputs\\2022-02-06-13-06-26\\model_epoch=28.pt";
    if (std::filesystem::exists(filename)) {
      this->module = torch::jit::load(filename);
      this->have_module = true;
    }
    else
      this->have_module = false;
#else
    this->have_module = true;
#endif
  }
#ifdef TORCHIN
  catch (const c10::Error& e) {
#else
  catch (const std::exception& e) {
#endif
    std::cerr << "error loading the model\n";
    std::cerr << e.what() << std::flush;
    this->have_module = false;
  }

  std::cout << "ok\n";
}

void NeuralAmpModeler::ProcessFallback(sample** inputs, sample** outputs, const int nFrames, const int nChans)
{
  for (int c = 0; c < nChans; c++)
    for (int s = 0; s < nFrames; s++)
      outputs[c][s] = inputs[c][s];
}

void NeuralAmpModeler::ProcessInputBufferUpdate(sample** inputs, const int nFrames)
{
  // If we'd run off the end of the input buffer, then we need to move the data back to the start of the
  // buffer and start again.
  if (this->input_buffer_offset + nFrames >= this->input_buffer.size()) {
    // Copy the input buffer back
    // RF-1 samples because we've got at least one new one inbound.
    for (int i = 0, j=this->input_buffer_offset - this->receptive_field + 1; i < this->receptive_field - 1; i++, j++)
      // Example:
      // * Currently at input_buffer_offset=1234
      // * Receptive field is 32
      // * So I want the 32-1 = 31 last samples.
      // * 1234-31 = 1203
      // * [1203, 1234) into positions [0,31)
      //   * Check: 1234-1203 = 31 ok good. Just like [1,2) 2-1 = 1.
      // * Then put the index at 31.
      // 
      // Off-by-ones amirite jeepers
      this->input_buffer[i] = this->input_buffer[j];
    // And reset the offset.
    // Since there is guaranteed to be at least nFrames>=1 new samples, we can overwrite the
    // last sample of the first receptive field, i.e. [0,RF) so RF-1.
    this->input_buffer_offset = this->receptive_field - 1;
  }

  // Put the new samples into the input buffer
  {
    const int c = 0;  // MONO
    for (int i = this->input_buffer_offset, j=0; j < nFrames; i++, j++)
      this->input_buffer[i] = inputs[c][j];
  }
}

void NeuralAmpModeler::ProcessTorch(sample** outputs, const int nFrames, const int nChans)
{
#ifdef TORCHIN
  // Hoo boy probably way inefficient.
  const int i0 = this->input_buffer_offset - this->receptive_field + 1;
  const int n = this->receptive_field + nFrames - 1;
  std::vector<torch::jit::IValue> inputs;
  for (int i=i0; i < i0 + n; i++)
    inputs.push_back(this->input_buffer[i]);
  at::Tensor output;
  output = this->module.forward(inputs).toTensor();
  // Transfer to output buffer:
  sample* output_data = output.data<sample>();
  for (int c = 0; c < nChans; c++)
    for (int i = 0; i < nFrames; i++)
      outputs[c][i] = output_data[i];
#else
  for (int c = 0; c < nChans; c++)
    for (int i = 0, j = this->input_buffer_offset; i < nFrames; i++, j++)
      // So you can be sure there's some computation happening!
      outputs[c][i] = 0.5 * this->input_buffer[j];
#endif
}

void NeuralAmpModeler::ProcessGain(sample** outputs, const int nFrames, const int nChans)
{
  const double gain = GetParam(kGain)->Value() / 100.0;
  for (int c = 0; c < nChans; c++)
    for (int s = 0; s < nFrames; s++)
      outputs[c][s] *= gain;
}

void NeuralAmpModeler::ProcessBlock(sample** inputs, sample** outputs, int nFrames)
{
  if (nFrames == 0)
    return;

  const int nChans = NOutChansConnected();
  if (this->have_module) {
    this->ProcessInputBufferUpdate(inputs, nFrames);
    this->ProcessTorch(outputs, nFrames, nChans);
    this->input_buffer_offset += nFrames;
  }
  else
    this->ProcessFallback(inputs, outputs, nFrames, nChans);
  this->ProcessGain(outputs, nFrames, nChans);
}
#endif
