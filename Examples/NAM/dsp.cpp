#include <cmath>  // pow, tanh
#include <filesystem>
#include <fstream>

#include "json.hpp"
#include "numpy_util.h"
#include "dsp.h"

constexpr auto _INPUT_BUFFER_SAFETY_FACTOR = 8;

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

//void DSP::_test(
//  const Eigen::VectorXf input,
//  const Eigen::VectorXf expected_output
//) const
//{
//  Eigen::VectorXf actual_output = this->process(input);
//  if (!actual_output.isApprox(expected_output))
//    throw std::exception("Actual output does not match expected output");
//}

// Buffer =====================================================================

Buffer::Buffer(const int receptive_field)
{
  this->set_receptive_field(receptive_field);
}

void Buffer::set_receptive_field(const int new_receptive_field)
{
  this->set_receptive_field(new_receptive_field, _INPUT_BUFFER_SAFETY_FACTOR * new_receptive_field);
};

void Buffer::set_receptive_field(const int new_receptive_field, const int input_buffer_size)
{
  this->receptive_field = new_receptive_field;
  this->input_buffer.resize(input_buffer_size);
  this->reset_input_buffer();
}

void Buffer::update_buffers(sample** inputs, const int num_frames)
{
  // Make sure that the buffer is big enough for the receptive field and the frames needed!
  {
    const int minimum_input_buffer_size = this->receptive_field + _INPUT_BUFFER_SAFETY_FACTOR * num_frames;
      if (this->input_buffer.size() < minimum_input_buffer_size)
        this->input_buffer.resize(minimum_input_buffer_size);
  }

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

// Linear =====================================================================

Linear::Linear(const int receptive_field, const bool bias, const std::vector<float> &params) : Buffer(receptive_field)
{
  if (params.size() != (receptive_field + (bias ? 1 : 0)))
    throw std::exception("Params vector does not match expected size based on architecture parameters");

  this->weight.resize(this->receptive_field);
  // Pass in in reverse order so that dot products work out of the box.
  for (int i = 0; i < this->receptive_field; i++)
    this->weight(i) = params[receptive_field - 1 - i];
  this->bias = bias ? params[receptive_field] : (float) 0.0;
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
    const int offset = this->input_buffer_offset - this->weight.size() + i + 1;
    auto input = Eigen::Map<const Eigen::VectorXf>(&this->input_buffer[offset], this->receptive_field);
    this->output_buffer[i] = this->bias + this->weight.dot(input);
  }
  // Copy to external output arrays:
  for (int c = 0; c < num_channels; c++)
    for (int s = 0; s < num_frames; s++)
      outputs[c][s] = (double) this->output_buffer[s];
}


// WaveNet ====================================================================

void wavenet::Conv1D::set_params(
  const int in_channels,
  const int out_channels,
  const int dilation,
  const bool do_bias,
  std::vector<float>::iterator& params
)
{
  this->weight.resize(WAVENET_KERNEL_SIZE);
  for (int i = 0; i<this->weight.size(); i++)
    this->weight[i].resize(out_channels, in_channels);  // y = Ax, input array (C,L)
  for (int i = 0; i < out_channels; i++)
    for (int j = 0; j < in_channels; j++)
      for (int k = 0; k < this->weight.size(); k++)
        this->weight[k](i, j) = *(params++);
  if (do_bias) {
    this->bias.resize(out_channels);
    for (int i = 0; i < out_channels; i++)
      this->bias(i) = *(params++);
  }
  else
    this->bias.resize(0);
  this->dilation = dilation;
}

void wavenet::Conv1D::process_(const Eigen::MatrixXf &input, Eigen::MatrixXf &output) const
{
  // #correct
  // #speed
  if (this->bias.size() == 0)
    for (int j = 0; j < output.cols(); j++)
      // FIXME hot-code width 2 ;)
      // FIXME verify correct...probably not!
      output.col(j) = this->weight[0] * input.col(j) + this->weight[1] * input.col(j + this->dilation);
  else
    for (int j = 0; j < output.cols(); j++)
      // FIXME hot-code width 2 ;)
      // FIXME verify correct...probably not!
      output.col(j) = this->bias + this->weight[0] * input.col(j) + this->weight[1] * input.col(j + this->dilation);
}

int wavenet::Conv1D::get_num_params() const
{
  int num_params = this->bias.size();
  for (int i = 0; i < this->weight.size(); i++)
    num_params += this->weight[i].size();
  return num_params;
}

int wavenet::Conv1D::get_out_channels() const
{
  return this->weight[0].rows();
}

wavenet::BatchNorm::BatchNorm(const int dim, std::vector<float>::iterator& params)
{
  // Extract from param buffer
  Eigen::VectorXf running_mean(dim);
  Eigen::VectorXf running_var(dim);
  Eigen::VectorXf weight(dim);
  Eigen::VectorXf bias(dim);
  for (int i = 0; i < dim; i++)
    running_mean(i) = *(params++);
  for (int i = 0; i < dim; i++)
    running_var(i) = *(params++);
  for (int i = 0; i < dim; i++)
    weight(i) = *(params++);
  for (int i = 0; i < dim; i++)
    bias(i) = *(params++);
  float eps = *(params++);

  // Convert to scale & loc
  this->scale.resize(dim);
  this->loc.resize(dim);
  for (int i = 0; i < dim; i++)
    this->scale(i) = weight(i) / sqrt(eps + running_var(i));
  this->loc = bias - this->scale.cwiseProduct(running_mean);
}

void wavenet::BatchNorm::process_(Eigen::MatrixXf& x) const
{
  // todo using colwise?
  // #speed but conv probably dominates
  for (int i = 0; i < x.cols(); i++) {
    x.col(i) = x.col(i).cwiseProduct(this->scale);
    x.col(i) += this->loc;
  }
}

void wavenet::WaveNetBlock::set_params(
  const int in_channels,
  const int out_channels,
  const int dilation,
  const bool batchnorm,
  std::vector<float>::iterator &params
)
{
  this->_batchnorm = batchnorm;
  this->conv.set_params(in_channels, out_channels, dilation, !batchnorm, params);
  if (this->_batchnorm)
    this->batchnorm = BatchNorm(out_channels, params);
}

void wavenet::WaveNetBlock::process_(const Eigen::MatrixXf& input, Eigen::MatrixXf &output) const
{
  this->conv.process_(input, output);
  if (this->_batchnorm)
    this->batchnorm.process_(output);
  this->tanh_(output);
}

void wavenet::WaveNetBlock::tanh_(Eigen::MatrixXf &x) const
{
  for (int j = 0; j < x.cols(); j++)
    for (int i = 0; i < x.rows(); i++)
      x(i, j) = tanh(x(i, j));
}

int wavenet::WaveNetBlock::get_out_channels() const
{
  return this->conv.get_out_channels();
}

wavenet::Head::Head(const int channels, std::vector<float>::iterator& params)
{
  this->weight.resize(channels);
  for (int i = 0; i < channels; i++)
    this->weight[i] = *(params++);
  this->bias = *(params++);
}

Eigen::VectorXf wavenet::Head::process(const Eigen::MatrixXf &input) const
{
  Eigen::VectorXf output(input.cols());
  for (int i = 0; i < input.cols(); i++)
    output(i) = this->bias + input.col(i).dot(this->weight);
  return output;
}

wavenet::WaveNet::WaveNet(
  const int channels,
  const int num_layers,
  const bool batchnorm,
  std::vector<float> &params
) :
  Buffer(mypow(2, num_layers))
{
  this->_verify_params(channels, num_layers, batchnorm, params.size());
  this->blocks.resize(num_layers);
  std::vector<float>::iterator it = params.begin();
  int in_channels = 1;
  for (int i = 0; i < num_layers; i++) 
    this->blocks[i].set_params(i == 0 ? 1 : channels, channels, mypow(2, i), batchnorm, it);
  this->block_vals.resize(num_layers + 1);
  this->head = Head(channels, it);
  if (it != params.end())
    throw std::exception("Didn't touch all the params when initializing wavenet");
  //this->_test(test_input, test_output);
}

void wavenet::WaveNet::process(
  sample** inputs,
  sample** outputs,
  const int num_channels,
  const int num_frames
)
{
  this->update_buffers(inputs, num_frames);
  // Main computation!
  const int receptive_field = this->_get_receptive_field();
  const int offset = this->input_buffer_offset - receptive_field + 1;
  const int input_length = receptive_field + num_frames - 1;
  for (int i = 0; i < input_length; i++)
    this->block_vals[0](0, i) = this->input_buffer[i + offset];
  for (int i = 0; i < this->blocks.size(); i++)
    this->blocks[i].process_(this->block_vals[i], this->block_vals[i + 1]);
  Eigen::VectorXf output = this->head.process(this->block_vals[this->blocks.size()]);
  // Copy to external output arrays:
  for (int c = 0; c < num_channels; c++)
    for (int s = 0; s < num_frames; s++)
      outputs[c][s] = (double)output(s);
}

void wavenet::WaveNet::_verify_params(
  const int channels,
  const int num_layers,
  const bool batchnorm,
  const int actual_params
)
{
  // TODO
}

int wavenet::WaveNet::_get_receptive_field() const
{
  return mypow(2, this->blocks.size());
}

void wavenet::WaveNet::update_buffers(sample** inputs, const int num_frames)
{
  this->Buffer::update_buffers(inputs, num_frames);

  int buffer_size = num_frames + this->_get_receptive_field() - 1;
  this->block_vals[0].resize(1, buffer_size);
  for (int i = 0; i < this->blocks.size(); i++) {
    const int dilation = mypow(2, i);
    // Assume kernel size 2
    buffer_size -= dilation;
    // Resize output buffer:
    this->block_vals[i + 1].resize(this->blocks[i].get_out_channels(), buffer_size);
  }
  if (buffer_size != num_frames)
    throw std::exception("Output of convolutions should match num_frames!");
}

//=============================================================================

std::unique_ptr<DSP> get_dsp(const std::filesystem::path dirname)
{
  const std::filesystem::path config_filename = dirname / std::filesystem::path("config.json");
  if (!std::filesystem::exists(config_filename))
    throw std::exception("Config JSON doesn't exist!\n");
  std::ifstream i(config_filename);
  nlohmann::json j;
  i >> j;
  if (j["version"] != "0.1.0")
    throw std::exception("Require version 0.1.0");

  auto architecture = j["architecture"];
  nlohmann::json config = j["config"];

  if (architecture == "Linear") {
    const int receptive_field = config["receptive_field"];
    const bool bias = config["bias"];
    std::vector<float> params = numpy_util::load_to_vector(dirname / std::filesystem::path("weights.npy"));
    return std::make_unique<Linear>(receptive_field, bias, params);
  }
  else if (architecture == "WaveNet") {
    const int channels = config["channels"];
    const int num_layers = config["num_layers"];
    const bool batchnorm = config["batchnorm"];
    std::vector<float> params = numpy_util::load_to_vector(dirname / std::filesystem::path("weights.npy"));
    return std::make_unique<wavenet::WaveNet>(channels, num_layers, batchnorm, params);
  }
  else {
    throw std::exception("Unrecognized architecture");
  }
}

int mypow(const int base, const int exponent)
{
  int result = 1;
  for (int i = 0; i < exponent; i++)
    result *= base;
  return result;
}