#include <fstream>
#include <unordered_set>

#include "dsp.h"
#include "json.hpp"
#include "lstm.h"
#include "numpy_util.h"
#include "HardCodedModel.h"


void verify_config_version(const std::string version)
{
  const std::unordered_set<std::string> supported_versions({"0.2.0", "0.2.1"});
  if (supported_versions.find(version) == supported_versions.end())
    throw std::exception("Unsupported config version");
}

std::unique_ptr<DSP> get_dsp(const std::filesystem::path dirname)
{
  const std::filesystem::path config_filename = dirname / std::filesystem::path("config.json");
  if (!std::filesystem::exists(config_filename))
    throw std::exception("Config JSON doesn't exist!\n");
  std::ifstream i(config_filename);
  nlohmann::json j;
  i >> j;
  verify_config_version(j["version"]);

  auto architecture = j["architecture"];
  nlohmann::json config = j["config"];

  if (architecture == "Linear")
  {
    const int receptive_field = config["receptive_field"];
    const bool bias = config["bias"];
    std::vector<float> params = numpy_util::load_to_vector(dirname / std::filesystem::path("weights.npy"));
    return std::make_unique<Linear>(receptive_field, bias, params);
  }
  else if (architecture == "WaveNet")
  {
    const int channels = config["channels"];
    const bool batchnorm = config["batchnorm"];
    std::vector<int> dilations;
    for (int i = 0; i < config["dilations"].size(); i++)
      dilations.push_back(config["dilations"][i]);
    const std::string activation = config["activation"];
    std::vector<float> params = numpy_util::load_to_vector(dirname / std::filesystem::path("weights.npy"));
    return std::make_unique<wavenet::WaveNet>(channels, dilations, batchnorm, activation, params);
  }
  else if (architecture == "CatLSTM")
  {
    const int num_layers = config["num_layers"];
    const int input_size = config["input_size"];
    const int hidden_size = config["hidden_size"];
    std::vector<float> params = numpy_util::load_to_vector(dirname / std::filesystem::path("weights.npy"));
    return std::make_unique<lstm::LSTM>(num_layers, input_size, hidden_size, params, config["parametric"]);
  }
  else
  {
    throw std::exception("Unrecognized architecture");
  }
}

std::unique_ptr<DSP> get_hard_dsp()
{
  // Values are defined in HardCodedModel.h
  verify_config_version(std::string(PYTHON_MODEL_VERSION));
#ifndef ARCHITECTURE
  const std::string ARCHITECTURE = "WaveNet";
#endif
  if (ARCHITECTURE == "WaveNet")
    return std::make_unique<wavenet::WaveNet>(CHANNELS, DILATIONS, BATCHNORM, ACTIVATION, PARAMS);
  else
    throw std::exception("Unrecognized architecture");
}
