#pragma once

#if IPLUG_DSP

#include <filesystem>
#include <memory>
#include <vector>
#include "IPlugConstants.h"

// HACK
using sample = double;

class DSP
{
public:
  // Basic null DSP: copy the inputs to the outputs
  virtual void process(sample** inputs, sample** outputs, const int num_channels, const int num_frames);
  // Anything to take care of before next buffer comes in.
  virtual void finalize(const int num_frames);
  // Finally, the volume knob :)
  void process_gain(sample** outputs, const int num_channels, const int num_frames, const double gain);
};

// Class where an input buffer is kept so that long-time effects can be captured.
class Buffer : public DSP
{
public:
  Buffer(const int receptive_field);
  void finalize(const int num_frames);
  void set_receptive_field(const int new_receptive_field);
protected:
  // Input buffer
  const int input_buffer_channels = 1;  // Mono
  int receptive_field;
  // First location where we add new samples from the input
  int input_buffer_offset;
  std::vector<float> input_buffer;
  std::vector<float> output_buffer;

  void update_buffers(sample** inputs, const int num_frames);
  void reset_input_buffer();
};

// Basic linear model (an IR!)
class Linear : public Buffer
{
public:
  Linear(const int receptive_field, const bool bias, const std::vector<float> &params);

  void process(sample** inputs, sample** outputs, const int num_channels, const int num_frames) override;
protected:
  std::vector<float> weight;
  float bias;
};

std::unique_ptr<DSP> get_dsp(const std::filesystem::path dirname);

#endif  // IPLUG_DSP