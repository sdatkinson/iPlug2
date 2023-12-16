/*
 ==============================================================================
 
 This file is part of the iPlug 2 library. Copyright (C) the iPlug 2 developers.
 
 See LICENSE.txt for  more info.
 
 ==============================================================================
*/

#pragma once

#include <iostream>
#include <functional>
#include <cmath>

#include "IPlugPlatform.h"

#include "heapbuf.h"
#include "ptrlist.h"

#include "LanczosResampler.h"

BEGIN_IPLUG_NAMESPACE

enum ESRCMode
{
  kLinearInterpolation = 0,
  kCubicInterpolation,
  kLancsoz,
  kNumResamplingModes
};

/** A multichannel realtime-resampler that can be used to resample audio processing
 * to a specified sample rate for the situation where you have some arbitary DSP
 * code that requires a specific sample rate.

 * Three modes are supported:
 * - Linear interpolation: simple linear interpolation between samples
 * - Cubic interpolation: cubic interpolation between samples
 * - Lanczos: Lanczos resampling uses an approximation of the sinc function to
 *   interpolate between samples. This is the highest quality resampling mode.
 * 
 * The Lanczos resampler has a configurable filter size (A) that affects the 
 * latency of the resampler. It can also optionally use SIMD instructions to
 * when T==float.
 *
 *
 * @tparam T the sampletype
 * @tparam NCHANS the number of channels
 * @tparam A The Lanczos filter size for the LanczosResampler resampler mode
   A higher value makes the filter closer to an 
   ideal stop-band that rejects high-frequency content (anti-aliasing), 
   but at the expense of higher latency
 */
template<typename T = double, int NCHANS=2, size_t A=12>
class RealtimeResampler
{
public:
  using BlockProcessFunc = std::function<void(T**, T**, int)>;
  using LanczosResampler = LanczosResampler<T, NCHANS, A>;

  RealtimeResampler(double renderingSampleRate, ESRCMode mode = ESRCMode::kLinearInterpolation)
  : mResamplingMode(mode)
  , mRenderingSampleRate(renderingSampleRate)
  {
  }
  
  RealtimeResampler(const RealtimeResampler&) = delete;
  RealtimeResampler& operator=(const RealtimeResampler&) = delete;

  void SetResamplingMode(ESRCMode mode)
  {
    mResamplingMode = mode;
    Reset(mInputSampleRate);
  }
  
  void Reset(double inputSampleRate, int blockSize = DEFAULT_BLOCK_SIZE)
  {
    if (mInputSampleRate == inputSampleRate && mMaxBlockSize == blockSize)
    {
      ClearBuffers();
      return;
    }

    mMaxBlockSize = blockSize * 2;
    mInputSampleRate = inputSampleRate;
    mUpRatio = mInputSampleRate / mRenderingSampleRate;
    mDownRatio = mRenderingSampleRate / mInputSampleRate;
    mResampledData.Resize(mMaxBlockSize * NCHANS);  // Doesn't matter that this may contain junk.
    mScratchPtrs.Empty();
    
    for (auto chan=0; chan<NCHANS; chan++)
    {
      mScratchPtrs.Add(mResampledData.Get() + (chan * mMaxBlockSize));
    }

    if (mResamplingMode == ESRCMode::kLancsoz)
    {
      // This doesn't work for more than 1 channel.
      mResamplerUp = std::make_unique<LanczosResampler>(mInputSampleRate, mRenderingSampleRate);
      mResamplerDown = std::make_unique<LanczosResampler>(mRenderingSampleRate, mInputSampleRate);
      
      // Need clean inputs because we're going to get going with them!
      ClearBuffers();
        
      /* Prepopulate the upsampler with silence so it can run ahead */
      // Needs to be end-to-end!
      const auto midSamples = mResamplerDown->GetNumSamplesRequiredFor(1);
      mLatency = int(mResamplerUp->GetNumSamplesRequiredFor(midSamples));
      // Push some silence and process it so the *down* resampler is ready!
      mResamplerUp->PushBlock(mScratchPtrs.GetList(), mLatency);
      const size_t populated = mResamplerUp->PopBlock(mScratchPtrs.GetList(), midSamples);
      if (populated < midSamples) {
        throw std::runtime_error("Didn't get enough samples required for pre-population!");
      }
      // FIXME we should be using func() here, but we don't have it...
      // func(mScratchPtrs.GetList(), mScratchPtrs.GetList(), (int)midSamples);
      mResamplerDown->PushBlock(mScratchPtrs.GetList(), midSamples);
      // Now we're ready!
    }
    else
    {
      mResamplerUp = nullptr;
      mResamplerDown = nullptr;
      mLatency = 0;
      ClearBuffers();  // Takes care of that junk
    }
    totalFramesProcessed=0;
  }

  /** Resample an input block with a per-block function (up sample input -> process with function -> down sample)
   * @param inputs Two-dimensional array containing the non-interleaved input buffers of audio samples for all channels
   * @param outputs Two-dimensional array for audio output (non-interleaved).
   * @param nFrames The block size for this block: number of samples per channel.
   * @param func The function that processes the audio sample at the higher sampling rate. NOTE: std::function can call malloc if you pass in captures */
  void ProcessBlock(T** inputs, T** outputs, int nFrames, BlockProcessFunc func)
  {
    totalFramesProcessed += nFrames;
    switch (mResamplingMode)
    {
      case ESRCMode::kLinearInterpolation:
      {
        const auto nNewFrames = LinearInterpolate(inputs, mScratchPtrs.GetList(), nFrames, mUpRatio, mMaxBlockSize);
        func(mScratchPtrs.GetList(), mScratchPtrs.GetList(), nNewFrames);
        LinearInterpolate(mScratchPtrs.GetList(), outputs, nNewFrames, mDownRatio, nFrames);
        break;
      }
      case ESRCMode::kCubicInterpolation:
      {
        const auto nNewFrames = CubicInterpolate(inputs, mScratchPtrs.GetList(), nFrames, mUpRatio, mMaxBlockSize);
        func(mScratchPtrs.GetList(), mScratchPtrs.GetList(), nNewFrames);
        CubicInterpolate(mScratchPtrs.GetList(), outputs, nNewFrames, mDownRatio, nFrames);
        break;
      }
      case ESRCMode::kLancsoz:
      {
        // Push into the up-resampler. This will give it what it needs to pop `outputLen` samples, and
        // `GetNumSamplesRequriedFor(outputLen)` will be zero, meaning that that it has what it needs.
        mResamplerUp->PushBlock(inputs, nFrames);
        
        // This is the most samples you might get. Sometimes you'll get fewer.
        const auto maxOutputLen = static_cast<int>(std::ceil(static_cast<double>(nFrames) / mUpRatio));
        size_t populatedUp = 0;

        // Why _is_ is a while-loop? If we _don't_ execute the following code only once, then are things
        // executing correctly?
        //
        // Since the new buffer has been pushed into the up-resampler, it should have what's needed (needs zero more
        // samples) and this while will enter.
        while (mResamplerUp->GetNumSamplesRequiredFor(1) == 0)
        {
          // This will deplete the up-resampler and it will need more samples to do more, so the while loop should
          // exit.
          populatedUp += mResamplerUp->PopBlock(mScratchPtrs.GetList(), maxOutputLen);
          if (populatedUp > maxOutputLen) {
            throw std::runtime_error("Got too many samples!");
          }
          func(mScratchPtrs.GetList(), mScratchPtrs.GetList(), (int)populatedUp);
          // This *should* give the down-resampler what it needs to pop the required output `nFrames` output buffer.
          mResamplerDown->PushBlock(mScratchPtrs.GetList(), populatedUp);
        }
        
        // Should there be a while loop here to pop as much as is needed? We want to keep a *little* wiggle room in
        // the down-resampler so that we never don't have *enough* to fill `outputs`.
        // Maybe a ring buffer?
        const auto populatedDown = mResamplerDown->PopBlock(outputs, nFrames);
        if (populatedDown < nFrames) {
          std::cerr << "Needed " << nFrames << "frames, but only popped " << populatedDown << "!" << std::endl;
        }
        // Is it valid to renormalize phases here?
        mResamplerUp->RenormalizePhases();
        mResamplerDown->RenormalizePhases();
        break;
      }
      default:
        break;
    }
  }
  
  int GetLatency() const { return mLatency; }

private:
  static inline int LinearInterpolate(T** inputs, T** outputs, int inputLen, double ratio, int maxOutputLen)
  {
    const auto outputLen =
      std::min(static_cast<int>(std::ceil(static_cast<double>(inputLen) / ratio)), maxOutputLen);

    for (auto writePos = 0; writePos < outputLen; writePos++)
    {
      const auto readPos = ratio * static_cast<double>(writePos);
      const auto readPostionTrunc = std::floor(readPos);
      const auto readPosInt = static_cast<int>(readPostionTrunc);

      if (readPosInt < inputLen)
      {
        const auto y = readPos - readPostionTrunc;

        for (auto chan=0; chan<NCHANS; chan++)
        {
          const auto x0 = inputs[chan][readPosInt];
          const auto x1 = ((readPosInt + 1) < inputLen) ? inputs[chan][readPosInt + 1] : inputs[chan][readPosInt-1];
          outputs[chan][writePos] = (1.0 - y) * x0 + y * x1;
        }
      }
    }

    return outputLen;
  }
  
  static inline int CubicInterpolate(T** inputs, T** outputs, int inputLen, double ratio, int maxOutputLen)
  {
    const auto outputLen =
      std::min(static_cast<int>(std::ceil(static_cast<double>(inputLen) / ratio)), maxOutputLen);

    for (auto writePos = 0; writePos < outputLen; writePos++)
    {
      const auto readPos = ratio * static_cast<double>(writePos);
      const auto readPostionTrunc = std::floor(readPos);
      const auto readPosInt = static_cast<int>(readPostionTrunc);

      if (readPosInt < inputLen)
      {
        const auto y = readPos - readPostionTrunc;
        
        for (auto chan=0; chan<NCHANS; chan++)
        {
          const auto xm1 = ((readPosInt - 1) > 0) ? inputs[chan][readPosInt - 1] : 0.0f;
          const auto x0 = ((readPosInt) < inputLen) ? inputs[chan][readPosInt] : inputs[chan][readPosInt-1];
          const auto x1 = ((readPosInt + 1) < inputLen) ? inputs[chan][readPosInt + 1] : inputs[chan][readPosInt-1];
          const auto x2 = ((readPosInt + 2) < inputLen) ? inputs[chan][readPosInt + 2] : inputs[chan][readPosInt-1];
          
          const auto  c = (x1 - xm1) * 0.5;
          const auto  v = x0 - x1;
          const auto  w = c + v;
          const auto  a = w + v + (x2 - x0) * 0.5;
          const auto  b = w + a;
          
          outputs[chan][writePos] = ((((a * y) -b) * y + c) * y + x0);
        }
      }
    }
    
    return outputLen;
  }
  
  void ClearBuffers()
  {
    memset(mResampledData.Get(), 0.0f, mMaxBlockSize * NCHANS * sizeof(T));
    
    if (mResamplingMode == ESRCMode::kLancsoz)
    {
      if (mResamplerUp != nullptr) {
        mResamplerUp->ClearBuffer();
      }
      if (mResamplerDown != nullptr) {
        mResamplerDown->ClearBuffer();
      }
    }
  }

  WDL_TypedBuf<T> mResampledData;
  WDL_PtrList<T> mScratchPtrs;
  double mUpRatio = 0.0, mDownRatio = 0.0;
  double mInputSampleRate = 0.0;
  int mMaxBlockSize = 0;
  int mLatency = 0;
  const double mRenderingSampleRate;
  ESRCMode mResamplingMode;
  long totalFramesProcessed=0;
  
  std::unique_ptr<LanczosResampler> mResamplerUp, mResamplerDown;
};

END_IPLUG_NAMESPACE
