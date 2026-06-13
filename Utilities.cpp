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

#include "Utilities.h"

#if ATHANOR_USE_HIGHWAY
  #undef HWY_TARGET_INCLUDE
  #define HWY_TARGET_INCLUDE "FFTConvolver/Utilities.cpp"
  #include "hwy/foreach_target.h"
  #include "hwy/highway.h"

HWY_BEFORE_NAMESPACE();
namespace fftconvolver
{
namespace HWY_NAMESPACE
{
namespace
{
  namespace hn = hwy::HWY_NAMESPACE;

  HWY_ATTR void ComplexMultiplyAccumulateHWY(Sample* FFTCONVOLVER_RESTRICT re,
                                             Sample* FFTCONVOLVER_RESTRICT im,
                                             const Sample* FFTCONVOLVER_RESTRICT reA,
                                             const Sample* FFTCONVOLVER_RESTRICT imA,
                                             const Sample* FFTCONVOLVER_RESTRICT reB,
                                             const Sample* FFTCONVOLVER_RESTRICT imB,
                                             const size_t len)
  {
    const hn::ScalableTag<float> df;
    const size_t lanes = hn::Lanes(df);
    size_t i = 0;
    for (; i + lanes <= len; i += lanes)
    {
      const auto vra = hn::LoadU(df, reA + i);
      const auto via = hn::LoadU(df, imA + i);
      const auto vrb = hn::LoadU(df, reB + i);
      const auto vib = hn::LoadU(df, imB + i);
      const auto vrr = hn::LoadU(df, re + i);
      const auto vri = hn::LoadU(df, im + i);

      const auto real = (vrr + vra * vrb) - via * vib;
      const auto imag = (vri + vra * vib) + via * vrb;

      hn::StoreU(real, df, re + i);
      hn::StoreU(imag, df, im + i);
    }

    for (; i < len; ++i)
    {
      re[i] += reA[i] * reB[i] - imA[i] * imB[i];
      im[i] += reA[i] * imB[i] + imA[i] * reB[i];
    }
  }
} // namespace
} // namespace HWY_NAMESPACE
} // namespace fftconvolver
HWY_AFTER_NAMESPACE();
#endif


namespace fftconvolver
{

#if !ATHANOR_USE_HIGHWAY || HWY_ONCE

#if ATHANOR_USE_HIGHWAY
HWY_EXPORT(ComplexMultiplyAccumulateHWY);
#endif

bool SSEEnabled()
{
#if defined(FFTCONVOLVER_USE_SSE)
  return true;
#else
  return false;
#endif
}


void Sum(Sample* FFTCONVOLVER_RESTRICT result,
         const Sample* FFTCONVOLVER_RESTRICT a,
         const Sample* FFTCONVOLVER_RESTRICT b,
         size_t len)
{
  const size_t end4 = 4 * (len / 4);
  for (size_t i=0; i<end4; i+=4)
  {
    result[i+0] = a[i+0] + b[i+0];
    result[i+1] = a[i+1] + b[i+1];
    result[i+2] = a[i+2] + b[i+2];
    result[i+3] = a[i+3] + b[i+3];
  }
  for (size_t i=end4; i<len; ++i)
  {
    result[i] = a[i] + b[i];
  }
}


void ComplexMultiplyAccumulate(SplitComplex& result, const SplitComplex& a, const SplitComplex& b)
{
  assert(result.size() == a.size());
  assert(result.size() == b.size());
  ComplexMultiplyAccumulate(result.re(), result.im(), a.re(), a.im(), b.re(), b.im(), result.size());
}


void ComplexMultiplyAccumulate(Sample* FFTCONVOLVER_RESTRICT re, 
                               Sample* FFTCONVOLVER_RESTRICT im,
                               const Sample* FFTCONVOLVER_RESTRICT reA,
                               const Sample* FFTCONVOLVER_RESTRICT imA,
                               const Sample* FFTCONVOLVER_RESTRICT reB,
                               const Sample* FFTCONVOLVER_RESTRICT imB,
                               const size_t len)
{
#if ATHANOR_USE_HIGHWAY
  HWY_DYNAMIC_DISPATCH(ComplexMultiplyAccumulateHWY)(re, im, reA, imA, reB, imB, len);
#elif defined(FFTCONVOLVER_USE_SSE)
  const size_t end4 = 4 * (len / 4);
  for (size_t i=0; i<end4; i+=4)
  {
    const __m128 ra = _mm_load_ps(&reA[i]);
    const __m128 rb = _mm_load_ps(&reB[i]);
    const __m128 ia = _mm_load_ps(&imA[i]);
    const __m128 ib = _mm_load_ps(&imB[i]);
    __m128 real = _mm_load_ps(&re[i]);
    __m128 imag = _mm_load_ps(&im[i]);
    real = _mm_add_ps(real, _mm_mul_ps(ra, rb));
    real = _mm_sub_ps(real, _mm_mul_ps(ia, ib));
    _mm_store_ps(&re[i], real);
    imag = _mm_add_ps(imag, _mm_mul_ps(ra, ib));
    imag = _mm_add_ps(imag, _mm_mul_ps(ia, rb));
    _mm_store_ps(&im[i], imag);
  }
  for (size_t i=end4; i<len; ++i)
  {
    re[i] += reA[i] * reB[i] - imA[i] * imB[i];
    im[i] += reA[i] * imB[i] + imA[i] * reB[i];
  }
#else
  const size_t end4 = 4 * (len / 4);
  for (size_t i=0; i<end4; i+=4)
  {
    re[i+0] += reA[i+0] * reB[i+0] - imA[i+0] * imB[i+0];
    re[i+1] += reA[i+1] * reB[i+1] - imA[i+1] * imB[i+1];
    re[i+2] += reA[i+2] * reB[i+2] - imA[i+2] * imB[i+2];
    re[i+3] += reA[i+3] * reB[i+3] - imA[i+3] * imB[i+3];
    im[i+0] += reA[i+0] * imB[i+0] + imA[i+0] * reB[i+0];
    im[i+1] += reA[i+1] * imB[i+1] + imA[i+1] * reB[i+1];
    im[i+2] += reA[i+2] * imB[i+2] + imA[i+2] * reB[i+2];
    im[i+3] += reA[i+3] * imB[i+3] + imA[i+3] * reB[i+3];
  }
  for (size_t i=end4; i<len; ++i)
  {
    re[i] += reA[i] * reB[i] - imA[i] * imB[i];
    im[i] += reA[i] * imB[i] + imA[i] * reB[i];
  }
#endif
}

#endif // !ATHANOR_USE_HIGHWAY || HWY_ONCE

} // End of namespace fftconvolver
