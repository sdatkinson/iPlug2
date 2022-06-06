#pragma once

#if IPLUG_DSP

#include <filesystem>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include "IPlugConstants.h"

// HACK
using sample = double;

// Class for providing params from the plugin to the DSP module
// For now, we'll work with doubles. Later, we'll add other types.
class DSPParam
{
public:
  const char* name;
  const double val;
};
// And the params shall be provided as a std::vector<DSPParam>.

class DSP
{
public:
  DSP();
  // Basic null DSP: copy the inputs to the outputs
  virtual void process(
    sample** inputs,
    sample** outputs,
    const int num_channels,
    const int num_frames,
    const double input_gain,
    const double output_gain,
    const std::unordered_map<std::string, double>& params
  );
  // Anything to take care of before next buffer comes in.
  // For example:
  // * Validate params as not stale
  virtual void finalize(const int num_frames);

protected:
  // Parameters
  std::unordered_map<std::string, double> _params;
  // If the params have changed since the last buffer was processed:
  bool _stale_params;
  // Where to store the samples after applying input gain
  std::vector<float> _input_post_gain;
  // Output of the core DSP algorithm
  std::vector<float> _core_dsp_output;

  // Methods

  // Copy the parameters to the DSP module.
  // If anything has changed, then set this->_stale_params.
  void _get_params_(const std::unordered_map<std::string, double>& input_params);

  // Apply the input gain
  // Result populates this->_input_post_gain
  void _apply_input_level_(sample** inputs, const int num_channels, const int num_frames, const double gain);

  // i.e. ensure the size is correct.
  void _ensure_core_dsp_output_ready_();

  // The core of your DSP algorithm.
  // Access the inputs in this->_input_post_gain
  // Place the outputs in this->_core_dsp_output
  virtual void _process_core_();

  // Copy this->_core_dsp_output to output and apply the output volume
  void _apply_output_level_(sample** outputs, const int num_channels, const int num_frames, const double gain);
};

// Class where an input buffer is kept so that long-time effects can be captured.
// (e.g. conv nets or impulse responses, where we need history that's longer than the
// sample buffer that's coming in.)
class Buffer : public DSP
{
public:
  Buffer(const int receptive_field);
  void finalize(const int num_frames);
protected:
  // Input buffer
  const int _input_buffer_channels = 1;  // Mono
  int _receptive_field;
  // First location where we add new samples from the input
  long _input_buffer_offset;
  std::vector<float> _input_buffer;
  std::vector<float> _output_buffer;

  void _set_receptive_field(const int new_receptive_field, const int input_buffer_size);
  void _set_receptive_field(const int new_receptive_field);
  void _reset_input_buffer();
  // Use this->_input_post_gain
  virtual void _update_buffers_();
  virtual void _rewind_buffers_();
};

// Basic linear model (an IR!)
class Linear : public Buffer
{
public:
  Linear(const int receptive_field, const bool bias, const std::vector<float> &params);
  void _process_core_() override;
protected:
  Eigen::VectorXf weight;
  float bias;
};

// WaveNet ====================================================================

#define WAVENET_KERNEL_SIZE 2

namespace wavenet {
  // Custom Conv that avoids re-computing on pieces of the input and trusts
  // that the corresponding outputs are where they need to be.
  // Beware: this is clever!
  class Conv1D
  {
  public:
    Conv1D() { this->dilation = 1; };
    void set_params(
      const int in_channels,
      const int out_channels,
      const int dilation,
      const bool do_bias,
      std::vector<float>::iterator& params
    );
    void process_(
      const Eigen::MatrixXf &input,
      Eigen::MatrixXf &output,
      const long i_start,
      const long i_end
    ) const;
    long get_num_params() const;
    long get_out_channels() const;
    int get_dilation() const { return this->dilation; };
  private:
    // Gonna wing this...
    // conv[kernel](cout, cin)
    std::vector<Eigen::MatrixXf> weight;
    Eigen::VectorXf bias;
    int dilation;
  };

  // Batch normalization
  // In prod mode, so really just an elementwise affine layer.
  class BatchNorm
  {
  public:
    BatchNorm() {};
    BatchNorm(const int dim, std::vector<float>::iterator& params);
    void process_(
      Eigen::MatrixXf& input,
      const long i_start,
      const long i_end
    ) const;

  private:
    // TODO simplify to just ax+b
    // y = (x-m)/sqrt(v+eps) * w + bias
    // y = ax+b
    // a = w / sqrt(v+eps)
    // b = a * m + bias
    Eigen::VectorXf scale;
    Eigen::VectorXf loc;
  };

  class WaveNetBlock
  {
  public:
    WaveNetBlock() { this->_batchnorm = false; };
    void set_params(
      const int in_channels,
      const int out_channels,
      const int dilation,
      const bool batchnorm,
      const std::string activation,
      std::vector<float>::iterator& params
    );
    void process_(
      const Eigen::MatrixXf& input,
      Eigen::MatrixXf &output,
      const long i_start,
      const long i_end
    ) const;
    int get_out_channels() const;
    Conv1D conv;
  private:
    BatchNorm batchnorm;
    bool _batchnorm;
    std::string activation;
    void _tanh_(Eigen::MatrixXf &x, const long i_start, const long i_end) const;
    void _relu_(Eigen::MatrixXf& x, const long i_start, const long i_end) const;
  };

  class Head
  {
  public:
    Head() { this->bias = (float)0.0; };
    Head(const int channels, std::vector<float>::iterator& params);
    void process_(
      const Eigen::MatrixXf &input,
      Eigen::VectorXf &output,
      const long i_start,
      const long i_end
    ) const;
  private:
    Eigen::VectorXf weight;
    float bias;
  };

  class WaveNet : public Buffer
  {
  public:
    WaveNet(
      const int channels,
      const std::vector<int>& dilations,
      const bool batchnorm,
      const std::string activation,
      std::vector<float> &params
    );
    void _process_core_() override;
  protected:
    std::vector<WaveNetBlock> _blocks;
    std::vector<Eigen::MatrixXf> _block_vals;
    Eigen::VectorXf _head_output;
    Head _head;
    void _verify_params(
      const int channels,
      const std::vector<int> &dilations,
      const bool batchnorm,
      const int actual_params
    );
    void _update_buffers_() override;
    void _rewind_buffers_() override;

    // The net starts with random parameters inside; we need to wait for a full
    // receptive field to pass through before we can count on the output being
    // ok. This implements a gentle "ramp-up" so that there's no "pop" at the
    // start.
    long _anti_pop_countdown;
    const long _anti_pop_ramp = 100;
    void _anti_pop_();
    void _reset_anti_pop_();
  };
};  // namespace wavenet

// Utilities ==================================================================

// Verify that the config that we are building our model from is supported by
// this plugin version.
void verify_config_version(const std::string version);

std::unique_ptr<DSP> get_dsp(const std::filesystem::path dirname);

// Hard-coded model:
std::unique_ptr<DSP> get_hard_dsp();

#endif  // IPLUG_DSP