// ==================================================================================
// Copyright (c) 2017 HiFi-LoFi
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is furnished
// to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
// FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
// COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
// IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
// WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
// ==================================================================================

#include "AudioFFT.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>

#if ATHANOR_USE_HIGHWAY
  #include "hwy/highway.h"
#endif

#if defined(AUDIOFFT_APPLE_ACCELERATE)
  #define AUDIOFFT_APPLE_ACCELERATE_USED
  #include <Accelerate/Accelerate.h>
#else
  #define AUDIOFFT_MUFFT_USED
  #include "muFFT/fft.h"
#endif

namespace audiofft
{

  namespace detail
  {

    class AudioFFTImpl
    {
    public:
      AudioFFTImpl() = default;
      AudioFFTImpl(const AudioFFTImpl&) = delete;
      AudioFFTImpl& operator=(const AudioFFTImpl&) = delete;
      virtual ~AudioFFTImpl() = default;
      virtual void init(size_t size) = 0;
      virtual void fft(const float* data, float* re, float* im) = 0;
      virtual void ifft(float* data, const float* re, const float* im) = 0;
    };

    constexpr bool IsPowerOf2(size_t val)
    {
      return (val == 1 || (val & (val - 1)) == 0);
    }

    template<typename TypeDest, typename TypeSrc, typename TypeFactor>
    void ScaleBuffer(TypeDest* dest, const TypeSrc* src, const TypeFactor factor, size_t len)
    {
      for (size_t i = 0; i < len; ++i)
        dest[i] = static_cast<TypeDest>(static_cast<TypeFactor>(src[i]) * factor);
    }

  } // namespace detail

  // ================================================================

#ifdef AUDIOFFT_MUFFT_USED

  class MufftFFT : public detail::AudioFFTImpl
  {
  public:
    MufftFFT() = default;
    MufftFFT(const MufftFFT&) = delete;
    MufftFFT& operator=(const MufftFFT&) = delete;

    ~MufftFFT() override
    {
      reset();
    }

    void init(size_t size) override
    {
      if (_size == size)
        return;

      reset();
      if (size == 0)
        return;
      if (size < 4)
        throw std::invalid_argument("muFFT requires sizes of at least 4");
      if (size > static_cast<size_t>(std::numeric_limits<unsigned>::max()))
        throw std::invalid_argument("muFFT only supports sizes up to UINT_MAX");

      _size = size;
      _complexSize = AudioFFT::ComplexSize(size);

      _forwardPlan = mufft_create_plan_1d_r2c(static_cast<unsigned>(_size), 0);
      _inversePlan = mufft_create_plan_1d_c2r(static_cast<unsigned>(_size), 0);
      if (!_forwardPlan || !_inversePlan)
      {
        reset();
        throw std::runtime_error("Failed to initialize muFFT plans");
      }

      _alignedInput = static_cast<float *>(mufft_calloc(_size * sizeof(float)));
      _alignedOutput = static_cast<float *>(mufft_calloc(_size * sizeof(float)));
      _spectrum = static_cast<Complex *>(mufft_calloc(_complexSize * sizeof(Complex)));
      if (!_alignedInput || !_alignedOutput || !_spectrum)
      {
        reset();
        throw std::bad_alloc();
      }
    }

    void fft(const float* data, float* re, float* im) override
    {
      if (_size == 0)
        return;

      ::memcpy(_alignedInput, data, _size * sizeof(float));
      mufft_execute_plan_1d(_forwardPlan, _spectrum, _alignedInput);
      for (size_t k = 0; k < _complexSize; ++k)
      {
        re[k] = _spectrum[k].real;
        im[k] = _spectrum[k].imag;
      }
    }

    void ifft(float* data, const float* re, const float* im) override
    {
      if (_size == 0)
        return;

      for (size_t k = 0; k < _complexSize; ++k)
      {
        _spectrum[k].real = re[k];
        _spectrum[k].imag = im[k];
      }
      _spectrum[0].imag = 0.0f;
      if (_complexSize > 1)
        _spectrum[_complexSize - 1].imag = 0.0f;

      mufft_execute_plan_1d(_inversePlan, _alignedOutput, _spectrum);
      const float scale = 1.0f / static_cast<float>(_size);

#if ATHANOR_USE_HIGHWAY
      namespace hn = hwy::HWY_NAMESPACE;
      const hn::ScalableTag<float> df;
      const size_t lanes = hn::Lanes(df);
      if (lanes > 1)
      {
        const auto scaleVec = hn::Set(df, scale);
        size_t n = 0;
        for (; n + lanes <= _size; n += lanes)
        {
          const auto values = hn::LoadU(df, _alignedOutput + n) * scaleVec;
          hn::StoreU(values, df, data + n);
        }
        for (; n < _size; ++n)
          data[n] = _alignedOutput[n] * scale;
        return;
      }
#endif

      detail::ScaleBuffer(data, _alignedOutput, scale, _size);
    }

  private:
    struct Complex
    {
      float real;
      float imag;
    };

    void reset()
    {
      if (_forwardPlan)
      {
        mufft_free_plan_1d(_forwardPlan);
        _forwardPlan = nullptr;
      }
      if (_inversePlan)
      {
        mufft_free_plan_1d(_inversePlan);
        _inversePlan = nullptr;
      }
      if (_spectrum)
      {
        mufft_free(_spectrum);
        _spectrum = nullptr;
      }
      if (_alignedOutput)
      {
        mufft_free(_alignedOutput);
        _alignedOutput = nullptr;
      }
      if (_alignedInput)
      {
        mufft_free(_alignedInput);
        _alignedInput = nullptr;
      }
      _size = 0;
      _complexSize = 0;
    }

    size_t _size = 0;
    size_t _complexSize = 0;
    mufft_plan_1d *_forwardPlan = nullptr;
    mufft_plan_1d *_inversePlan = nullptr;
    float *_alignedInput = nullptr;
    float *_alignedOutput = nullptr;
    Complex *_spectrum = nullptr;
  };

  using AudioFFTImplementation = MufftFFT;

#endif // AUDIOFFT_MUFFT_USED

  // ================================================================

#ifdef AUDIOFFT_APPLE_ACCELERATE_USED

  class AppleAccelerateFFT : public detail::AudioFFTImpl
  {
  public:
    AppleAccelerateFFT() :
      _size(0),
      _powerOf2(0),
      _fftSetup(0)
    {
    }

    AppleAccelerateFFT(const AppleAccelerateFFT&) = delete;
    AppleAccelerateFFT& operator=(const AppleAccelerateFFT&) = delete;

    ~AppleAccelerateFFT() override
    {
      reset();
    }

    void init(size_t size) override
    {
      if (_size == size)
        return;

      reset();
      if (size == 0)
        return;

      _size = size;
      _powerOf2 = 0;
      while ((static_cast<size_t>(1) << _powerOf2) < _size)
        ++_powerOf2;
      _fftSetup = vDSP_create_fftsetup(_powerOf2, FFT_RADIX2);
      _re.resize(_size / 2);
      _im.resize(_size / 2);
    }

    void fft(const float* data, float* re, float* im) override
    {
      const size_t size2 = _size / 2;
      DSPSplitComplex splitComplex{ re, im };
      vDSP_ctoz(reinterpret_cast<const COMPLEX*>(data), 2, &splitComplex, 1, size2);
      vDSP_fft_zrip(_fftSetup, &splitComplex, 1, _powerOf2, FFT_FORWARD);
      const float factor = 0.5f;
      vDSP_vsmul(re, 1, &factor, re, 1, size2);
      vDSP_vsmul(im, 1, &factor, im, 1, size2);
      re[size2] = im[0];
      im[0] = 0.0f;
      im[size2] = 0.0f;
    }

    void ifft(float* data, const float* re, const float* im) override
    {
      const size_t size2 = _size / 2;
      ::memcpy(_re.data(), re, size2 * sizeof(float));
      ::memcpy(_im.data(), im, size2 * sizeof(float));
      _im[0] = re[size2];
      DSPSplitComplex splitComplex{ _re.data(), _im.data() };
      vDSP_fft_zrip(_fftSetup, &splitComplex, 1, _powerOf2, FFT_INVERSE);
      vDSP_ztoc(&splitComplex, 1, reinterpret_cast<COMPLEX*>(data), 2, size2);
      const float factor = 1.0f / static_cast<float>(_size);
      vDSP_vsmul(data, 1, &factor, data, 1, _size);
    }

  private:
    void reset()
    {
      if (_fftSetup)
      {
        vDSP_destroy_fftsetup(_fftSetup);
        _fftSetup = 0;
      }
      _size = 0;
      _powerOf2 = 0;
      _re.clear();
      _im.clear();
    }

    size_t _size;
    size_t _powerOf2;
    FFTSetup _fftSetup;
    std::vector<float> _re;
    std::vector<float> _im;
  };

  using AudioFFTImplementation = AppleAccelerateFFT;

#endif // AUDIOFFT_APPLE_ACCELERATE_USED

  // =============================================================

  AudioFFT::AudioFFT() :
    _impl(new AudioFFTImplementation())
  {
  }

  AudioFFT::~AudioFFT() = default;

  void AudioFFT::init(size_t size)
  {
    assert(size > 0 && detail::IsPowerOf2(size));
    _impl->init(size);
  }

  void AudioFFT::fft(const float* data, float* re, float* im)
  {
    _impl->fft(data, re, im);
  }

  void AudioFFT::ifft(float* data, const float* re, const float* im)
  {
    _impl->ifft(data, re, im);
  }

  size_t AudioFFT::ComplexSize(size_t size)
  {
    return (size / 2) + 1;
  }

} // namespace audiofft
