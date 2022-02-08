#include "dsp.h"

void DSP::process(sample** inputs, sample** outputs, const int num_channels, const int num_frames)
{
  for (int c = 0; c < num_channels; c++)
    for (int s = 0; s < num_frames; s++)
      outputs[c][s] = inputs[c][s];
}

void DSP::finalize(const int num_frames)
{}

void DSP::process_gain(sample** outputs, const int num_channels, const int num_frames, const double gain)
{
  for (int c = 0; c < num_channels; c++)
    for (int s = 0; s < num_frames; s++)
    outputs[c][s] *= gain;
}

//=============================================================================

Buffer::Buffer()
{
  this->set_receptive_field(512);
}

void Buffer::set_receptive_field(const int new_receptive_field)
{
  this->receptive_field = new_receptive_field;
  this->input_buffer.resize(2 * new_receptive_field);
  this->reset_input_buffer();
}

void Buffer::update_buffers(sample** inputs, const int num_frames)
{
  // If we'd run off the end of the input buffer, then we need to move the data back to the start of the
  // buffer and start again.
  if (this->input_buffer_offset + num_frames > this->input_buffer.size()) {
    // Copy the input buffer back
    // RF-1 samples because we've got at least one new one inbound.
    for (int i = 0, j = this->input_buffer_offset - this->receptive_field; i < this->receptive_field; i++, j++)
      this->input_buffer[i] = this->input_buffer[j];
    // And reset the offset.
    // Even though we could be stingy about that one sample that we won't be using
    // (because a new set is incoming) it's probably not worth the hyper-optimization
    // and liable for bugs.
    // And the code looks way tidier this way.
    this->input_buffer_offset = this->receptive_field;
  }
  // Put the new samples into the input buffer
  {
    const int c = 0;  // MONO
    for (int i = this->input_buffer_offset, j = 0; j < num_frames; i++, j++)
      this->input_buffer[i] = (float) inputs[c][j];
  }
  // And resize the output buffer:
  this->output_buffer.resize(num_frames);
}

void Buffer::reset_input_buffer()
{
  this->input_buffer_offset = this->receptive_field;
}

void Buffer::finalize(const int num_frames)
{
  this->input_buffer_offset += num_frames;
}

//=============================================================================

Linear::Linear()
{
  this->weight.resize(this->receptive_field);
  // I dunno
  for (int i = 0; i < this->weight.size(); i++)
    this->weight[i] = (float) 0.5 / (this->weight.size() - i);
}

void Linear::process(
  sample** inputs,
  sample** outputs,
  const int num_channels,
  const int num_frames
)
{
  this->Buffer::update_buffers(inputs, num_frames);

  // Main computation!
  for (int i = 0; i < num_frames; i++) {
    this->output_buffer[i] = 0.0;
    const int offset = this->input_buffer_offset - this->weight.size() + i + 1;
    for (int j = 0, k = offset; j < this->weight.size(); j++, k++)
      this->output_buffer[i] += this->weight[j] * this->input_buffer[k];
  }
  // Copy to external output arrays:
  for (int c = 0; c < num_channels; c++)
    for (int s = 0; s < num_frames; s++)
      outputs[c][s] = (double) this->output_buffer[s];
}