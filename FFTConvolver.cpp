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

#include "FFTConvolver.h"

#include <cassert>
#include <cmath>
#include <memory>

#if defined (FFTCONVOLVER_USE_SSE)
  #include <xmmintrin.h>
#endif

#ifdef AUDIOFFT_MUFFT
  #include "muFFT/fft.h"
#endif


namespace fftconvolver
{  

#ifdef AUDIOFFT_MUFFT

struct FFTConvolver::MufftBackend
{
  explicit MufftBackend(size_t requestedBlockSize) :
    blockSize(NextPowerOf2(requestedBlockSize)),
    segSize(2 * blockSize),
    complexSize(audiofft::AudioFFT::ComplexSize(segSize)),
    normalization(1.0f / static_cast<float>(segSize)),
    inputBuffer(blockSize),
    overlap(blockSize),
    current(0),
    inputBufferFill(0)
  {
  }

  ~MufftBackend()
  {
    clear();
  }

  MufftBackend(const MufftBackend&) = delete;
  MufftBackend& operator=(const MufftBackend&) = delete;

  bool init(const Sample *ir, size_t irLen)
  {
    if (irLen == 0)
      return true;
    if (segSize < 4)
      return false;

    segCount = static_cast<size_t>(::ceil(static_cast<float>(irLen) / static_cast<float>(blockSize)));

    const unsigned convMethod =
      MUFFT_CONV_METHOD_FLAG_MONO_MONO |
      MUFFT_CONV_METHOD_FLAG_ZERO_PAD_UPPER_HALF_FIRST |
      MUFFT_CONV_METHOD_FLAG_ZERO_PAD_UPPER_HALF_SECOND;
    constexpr unsigned fftFlags = 0;

    auto *probeConv = mufft_create_plan_conv(static_cast<unsigned>(segSize), fftFlags, convMethod);
    if (!probeConv)
      return false;

    transformedBytes = mufft_conv_get_transformed_block_size(probeConv);
    mufft_free_plan_conv(probeConv);

    forwardPlan = mufft_create_plan_1d_r2c(static_cast<unsigned>(segSize), MUFFT_FLAG_ZERO_PAD_UPPER_HALF | fftFlags);
    inversePlan = mufft_create_plan_1d_c2r(static_cast<unsigned>(segSize), fftFlags);
    convolveFunc = mufft_get_convolve_func(fftFlags);
    alignedInput = static_cast<Sample *>(mufft_calloc(blockSize * sizeof(Sample)));
    ifftOutput = static_cast<Sample *>(mufft_calloc(segSize * sizeof(Sample)));
    preMultiplied = mufft_calloc(transformedBytes);
    convAccum = mufft_calloc(transformedBytes);
    multiplyScratch = mufft_calloc(transformedBytes);

    if (!forwardPlan || !inversePlan || !convolveFunc || !alignedInput || !ifftOutput || !preMultiplied || !convAccum || !multiplyScratch)
      return false;

    segments.resize(segCount, nullptr);
    segmentsIR.resize(segCount, nullptr);
    for (size_t i = 0; i < segCount; ++i)
    {
      segments[i] = mufft_calloc(transformedBytes);
      segmentsIR[i] = mufft_calloc(transformedBytes);
      if (!segments[i] || !segmentsIR[i])
        return false;
    }

    for (size_t i = 0; i < segCount; ++i)
    {
      const size_t remaining = irLen - (i * blockSize);
      const size_t sizeCopy = (remaining >= blockSize) ? blockSize : remaining;
      ::memcpy(alignedInput, &ir[i * blockSize], sizeCopy * sizeof(Sample));
      if (sizeCopy < blockSize)
        ::memset(alignedInput + sizeCopy, 0, (blockSize - sizeCopy) * sizeof(Sample));
      mufft_execute_plan_1d(forwardPlan, segmentsIR[i], alignedInput);
    }

    resetInput();
    return true;
  }

  void resetInput()
  {
    inputBuffer.setZero();
    overlap.setZero();
    inputBufferFill = 0;
    current = 0;
    zeroBlock(preMultiplied);
    zeroBlock(convAccum);
    for (auto *segment : segments)
      zeroBlock(segment);
  }

  void process(const Sample *input, Sample *output, size_t len)
  {
    if (segCount == 0)
    {
      ::memset(output, 0, len * sizeof(Sample));
      return;
    }

    size_t processed = 0;
    while (processed < len)
    {
    const bool inputBufferWasEmpty = (inputBufferFill == 0);
    const size_t processing = std::min(len - processed, blockSize - inputBufferFill);
    const size_t inputBufferPos = inputBufferFill;
    if (inputBufferWasEmpty && processing == blockSize)
    {
      ::memcpy(alignedInput, input + processed, blockSize * sizeof(Sample));
    }
    else
    {
      ::memcpy(inputBuffer.data() + inputBufferPos, input + processed, processing * sizeof(Sample));
      ::memcpy(alignedInput, inputBuffer.data(), blockSize * sizeof(Sample));
    }
    mufft_execute_plan_1d(forwardPlan, segments[current], alignedInput);

      if (inputBufferWasEmpty)
      {
        zeroBlock(preMultiplied);
        for (size_t i = 1; i < segCount; ++i)
        {
          const size_t indexIr = i;
          const size_t indexAudio = (current + i) % segCount;
          accumulateProduct(preMultiplied, segmentsIR[indexIr], segments[indexAudio]);
        }
      }

      ::memcpy(convAccum, preMultiplied, transformedBytes);
      accumulateProduct(convAccum, segments[current], segmentsIR[0]);
      mufft_execute_plan_1d(inversePlan, ifftOutput, convAccum);

      Sum(output + processed, ifftOutput + inputBufferPos, overlap.data() + inputBufferPos, processing);

      inputBufferFill += processing;
      if (inputBufferFill == blockSize)
      {
        inputBuffer.setZero();
        inputBufferFill = 0;
        ::memcpy(overlap.data(), ifftOutput + blockSize, blockSize * sizeof(Sample));
        current = (current > 0) ? (current - 1) : (segCount - 1);
      }

      processed += processing;
    }
  }

  void clear()
  {
    for (auto *segment : segments)
      mufft_free(segment);
    for (auto *segment : segmentsIR)
      mufft_free(segment);
    segments.clear();
    segmentsIR.clear();

    if (multiplyScratch)
      mufft_free(multiplyScratch);
    if (convAccum)
      mufft_free(convAccum);
    if (preMultiplied)
      mufft_free(preMultiplied);
    if (ifftOutput)
      mufft_free(ifftOutput);
    if (alignedInput)
      mufft_free(alignedInput);
    if (inversePlan)
      mufft_free_plan_1d(inversePlan);
    if (forwardPlan)
      mufft_free_plan_1d(forwardPlan);

    multiplyScratch = nullptr;
    convAccum = nullptr;
    preMultiplied = nullptr;
    ifftOutput = nullptr;
    alignedInput = nullptr;
    inversePlan = nullptr;
    forwardPlan = nullptr;
    transformedBytes = 0;
    segCount = 0;
    inputBuffer.clear();
    overlap.clear();
    inputBufferFill = 0;
    current = 0;
  }

  void zeroBlock(void *block) const
  {
    if (block && transformedBytes > 0)
      ::memset(block, 0, transformedBytes);
  }

  void accumulateProduct(void *dst, const void *a, const void *b) const
  {
    convolveFunc(multiplyScratch, a, b, normalization, static_cast<unsigned>(complexSize));

    auto *dstFloats = static_cast<float *>(dst);
    const auto *srcFloats = static_cast<const float *>(multiplyScratch);
    const size_t floatCount = transformedBytes / sizeof(float);
    for (size_t i = 0; i < floatCount; ++i)
      dstFloats[i] += srcFloats[i];
  }

  size_t blockSize = 0;
  size_t segSize = 0;
  size_t segCount = 0;
  size_t complexSize = 0;
  size_t transformedBytes = 0;
  float normalization = 0.0f;
  mufft_plan_1d *forwardPlan = nullptr;
  mufft_plan_1d *inversePlan = nullptr;
  mufft_convolve_func convolveFunc = nullptr;
  Sample *alignedInput = nullptr;
  Sample *ifftOutput = nullptr;
  void *preMultiplied = nullptr;
  void *convAccum = nullptr;
  void *multiplyScratch = nullptr;
  std::vector<void *> segments;
  std::vector<void *> segmentsIR;
  SampleBuffer inputBuffer;
  SampleBuffer overlap;
  size_t current = 0;
  size_t inputBufferFill = 0;
};

#endif

FFTConvolver::FFTConvolver() :
  _blockSize(0),
  _segSize(0),
  _segCount(0),
  _fftComplexSize(0),
  _segments(),
  _segmentsIR(),
  _fftBuffer(),
  _fft(),
  _preMultiplied(),
  _conv(),
  _overlap(),
  _current(0),
  _inputBuffer(),
  _inputBufferFill(0)
#ifdef AUDIOFFT_MUFFT
  , _mufftBackend()
#endif
{
}

  
FFTConvolver::~FFTConvolver()
{
  reset();
}

  
void FFTConvolver::reset()
{  
#ifdef AUDIOFFT_MUFFT
  _mufftBackend.reset();
#endif
  for (auto* segment : _segments)
    delete segment;
  for (auto* segment : _segmentsIR)
    delete segment;
  
  _blockSize = 0;
  _segSize = 0;
  _segCount = 0;
  _fftComplexSize = 0;
  _segments.clear();
  _segmentsIR.clear();
  _fftBuffer.clear();
  _fft.init(0);
  _preMultiplied.clear();
  _conv.clear();
  _overlap.clear();
  _current = 0;
  _inputBuffer.clear();
  _inputBufferFill = 0;
}

void FFTConvolver::resetInput()
{
#ifdef AUDIOFFT_MUFFT
  if (_mufftBackend)
  {
    _mufftBackend->resetInput();
    return;
  }
#endif

  _inputBuffer.setZero();
  _inputBufferFill = 0;
  _current = 0;
  _conv.setZero();
  _preMultiplied.setZero();
  _overlap.setZero();

  for (auto *segment : _segments)
    segment->setZero();
}

  
bool FFTConvolver::init(size_t blockSize, const Sample* ir, size_t irLen)
{
  reset();

  if (blockSize == 0)
  {
    return false;
  }
  
  // Ignore zeros at the end of the impulse response because they only waste computation time
  while (irLen > 0 && ::fabs(ir[irLen-1]) < 0.000001f)
  {
    --irLen;
  }

  if (irLen == 0)
  {
    return true;
  }

#ifdef AUDIOFFT_MUFFT
  if (NextPowerOf2(blockSize) > 1)
  {
    auto backend = std::make_unique<MufftBackend>(blockSize);
    if (!backend->init(ir, irLen))
    {
      reset();
      return false;
    }
    _blockSize = backend->blockSize;
    _segSize = backend->segSize;
    _segCount = backend->segCount;
    _fftComplexSize = backend->complexSize;
    _current = backend->current;
    _inputBufferFill = backend->inputBufferFill;
    _mufftBackend = std::move(backend);
    return true;
  }
#endif
  
  _blockSize = NextPowerOf2(blockSize);
  _segSize = 2 * _blockSize;
  _segCount = static_cast<size_t>(::ceil(static_cast<float>(irLen) / static_cast<float>(_blockSize)));
  _fftComplexSize = audiofft::AudioFFT::ComplexSize(_segSize);

  try
  {
    // FFT
    _fft.init(_segSize);
    _fftBuffer.resize(_segSize);

    // Prepare segments
    for (size_t i=0; i<_segCount; ++i)
    {
      auto segment = std::make_unique<SplitComplex>(_fftComplexSize);
      _segments.push_back(segment.get());
      segment.release();
    }

    // Prepare IR
    for (size_t i=0; i<_segCount; ++i)
    {
      auto segment = std::make_unique<SplitComplex>(_fftComplexSize);
      const size_t remaining = irLen - (i * _blockSize);
      const size_t sizeCopy = (remaining >= _blockSize) ? _blockSize : remaining;
      CopyAndPad(_fftBuffer, &ir[i*_blockSize], sizeCopy);
      _fft.fft(_fftBuffer.data(), segment->re(), segment->im());
      _segmentsIR.push_back(segment.get());
      segment.release();
    }

    // Prepare convolution buffers
    _preMultiplied.resize(_fftComplexSize);
    _conv.resize(_fftComplexSize);
    _overlap.resize(_blockSize);

    // Prepare input buffer
    _inputBuffer.resize(_blockSize);
    _inputBufferFill = 0;

    // Reset current position
    _current = 0;

    return true;
  }
  catch (...)
  {
    reset();
    return false;
  }
}


void FFTConvolver::process(const Sample* input, Sample* output, size_t len)
{
#ifdef AUDIOFFT_MUFFT
  if (_mufftBackend)
  {
    _mufftBackend->process(input, output, len);
    return;
  }
#endif

  if (_segCount == 0)
  {
    ::memset(output, 0, len * sizeof(Sample));
    return;
  }

  size_t processed = 0;
  while (processed < len)
  {
    const bool inputBufferWasEmpty = (_inputBufferFill == 0);
    const size_t processing = std::min(len-processed, _blockSize-_inputBufferFill);
    const size_t inputBufferPos = _inputBufferFill;
    ::memcpy(_inputBuffer.data()+inputBufferPos, input+processed, processing * sizeof(Sample));

    // Forward FFT
    CopyAndPad(_fftBuffer, &_inputBuffer[0], _blockSize); 
    _fft.fft(_fftBuffer.data(), _segments[_current]->re(), _segments[_current]->im());

    // Complex multiplication
    if (inputBufferWasEmpty)
    {
      _preMultiplied.setZero();
      for (size_t i=1; i<_segCount; ++i)
      {
        const size_t indexIr = i;
        const size_t indexAudio = (_current + i) % _segCount;
        ComplexMultiplyAccumulate(_preMultiplied, *_segmentsIR[indexIr], *_segments[indexAudio]);
      }
    }
    _conv.copyFrom(_preMultiplied);
    ComplexMultiplyAccumulate(_conv, *_segments[_current], *_segmentsIR[0]);

    // Backward FFT
    _fft.ifft(_fftBuffer.data(), _conv.re(), _conv.im());

    // Add overlap
    Sum(output+processed, _fftBuffer.data()+inputBufferPos, _overlap.data()+inputBufferPos, processing);

    // Input buffer full => Next block
    _inputBufferFill += processing;
    if (_inputBufferFill == _blockSize)
    {
      // Input buffer is empty again now
      _inputBuffer.setZero();
      _inputBufferFill = 0;

      // Save the overlap
      ::memcpy(_overlap.data(), _fftBuffer.data()+_blockSize, _blockSize * sizeof(Sample));

      // Update current segment
      _current = (_current > 0) ? (_current - 1) : (_segCount - 1);
    }

    processed += processing;
  }
}
  
} // End of namespace fftconvolver
