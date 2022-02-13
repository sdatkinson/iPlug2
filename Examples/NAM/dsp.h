#pragma once

#if IPLUG_DSP

#include <filesystem>
#include <iterator>
#include <memory>
#include <vector>

#include <Eigen/Dense>
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
//protected:
//   Check the model against a test input/output pair
//  void _test(
//    const Eigen::VectorXf input,
//    const Eigen::VectorXf expected_output
//  ) const;
};

// Class where an input buffer is kept so that long-time effects can be captured.
class Buffer : public DSP
{
public:
  Buffer(const int receptive_field);
  void finalize(const int num_frames);
  void set_receptive_field(const int new_receptive_field, const int input_buffer_size);
  void set_receptive_field(const int new_receptive_field);
protected:
  // Input buffer
  const int input_buffer_channels = 1;  // Mono
  int receptive_field;
  // First location where we add new samples from the input
  long input_buffer_offset;
  std::vector<float> input_buffer;
  std::vector<float> output_buffer;

  virtual void update_buffers(sample** inputs, const int num_frames);
  virtual void rewind_buffers();
  void reset_input_buffer();
};

// Basic linear model (an IR!)
class Linear : public Buffer
{
public:
  Linear(const int receptive_field, const bool bias, const std::vector<float> &params);
  void process(sample** inputs, sample** outputs, const int num_channels, const int num_frames) override;
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
  private:
    // Gonna wing this...
    // conv[kernel](cout, cin)
    std::vector<Eigen::MatrixXf> weight;
    Eigen::VectorXf bias;
    int dilation;
  };

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
      std::vector<float>::iterator& params
    );
    void process_(
      const Eigen::MatrixXf& input,
      Eigen::MatrixXf &output,
      const long i_start,
      const long i_end
    ) const;
    int get_out_channels() const;
  private:
    Conv1D conv;
    BatchNorm batchnorm;
    bool _batchnorm;
    // And the Tanh is assumed for now.
    void tanh_(Eigen::MatrixXf &x, const long i_start, const long i_end) const;
  };

  class Head
  {
  public:
    Head() { this->bias = (float)0.0; };
    Head(const int channels, std::vector<float>::iterator& params);
    Eigen::VectorXf process(
      const Eigen::MatrixXf &input,
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
      const int num_layers,
      const bool batchnorm,
      std::vector<float> &params
    );
    void process(sample** inputs, sample** outputs, const int num_channels, const int num_frames) override;
  protected:
    std::vector<WaveNetBlock> blocks;
    std::vector<Eigen::MatrixXf> block_vals;
    Head head;
    void _verify_params(const int channels, const int num_layers, const bool batchnorm, const int actual_params);
    int _get_receptive_field() const;
    void update_buffers(sample** inputs, const int num_frames);
    void rewind_buffers();
  };
};  // namespace wavenet

// Utilities ==================================================================

std::unique_ptr<DSP> get_dsp(const std::filesystem::path dirname);

// Why am I doing this lol
int mypow(const int base, const int exponent);

#endif  // IPLUG_DSP