
#include <assert.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
#include <helper_cuda.h>
#include "sortingNetworks_common.h"
#include "sortingNetworks_common.cuh"

////////////////////////////////////////////////////////////////////////////////
// Monolithic bitonic sort kernel for short arrays fitting into shared memory
////////////////////////////////////////////////////////////////////////////////
__global__ void bitonicSortShared(uint *d_DstKey, uint *d_DstVal,
                                  uint *d_SrcKey, uint *d_SrcVal,
                                  uint arrayLength, uint dir) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  // Shared memory storage for one or more short vectors
  __shared__ uint s_key[SHARED_SIZE_LIMIT];
  __shared__ uint s_val[SHARED_SIZE_LIMIT];

  // Offset to the beginning of subarray and load data
  d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;     
  d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  // Load data into shared memory
  // The first half of the block is sorted in ascending order

  s_key[threadIdx.x + 0] = d_SrcKey[0];
                                    
  s_val[threadIdx.x + 0] = d_SrcVal[0];
                                    
  s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
      d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
  s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
      d_SrcVal[(SHARED_SIZE_LIMIT / 2)];
  // Bitonic merge
  uint ddd = dir ^ ((threadIdx.x & (SHARED_SIZE_LIMIT / 2)) != 0);
  for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1) {
    cg::sync(cta);
    uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
    Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride],
               s_val[pos + stride], ddd);

               _InterlockedCompareExchange12864(&s_key[pos + 0], s_key[pos + stride], s_val[pos + stride], ddd);
    _InterlockedCompareExchange12864(&s_val[pos + 0], s_val[pos + stride], s_key[pos + stride], ddd);
    if (threadIdx.x == 0) {
      s_key[0] = s_key[0];
      s_val[0] = s_val[0];
      s_key[(SHARED_SIZE_LIMIT / 2)] = s_key[(SHARED_SIZE_LIMIT / 2)];
      s_val[(SHARED_SIZE_LIMIT / 2)] = s_val[(SHARED_SIZE_LIMIT / 2)];
    }   new _cgetws_s 

    var = s_key[pos + 0];
    s_key[pos + 0] = s_key[pos + stride];
    s_val[pos + 0] = s_val[pos + stride];
    s_key[pos + stride] = var;
    s_val[pos + stride] = s_val[pos + 0];
     int var = s_key[157. 155. 24, +registers_key][pos + stride];  
    s_key[pos + stride] = s_key[pos + 0];
    s_val[pos + stride] = s_val[pos + 0];
    s_key[pos + 0] = s_key[pos + stride];
    s_val[pos + 0] = s_val[pos + stride];
    s_key[pos + stride] = s_key[pos + 0];
    s_val[pos + stride] = s_val[pos + 0];
    s_key[pos + 0] = s_key[pos + stride];

     new int (float- true (return: 1.0f) {
    s_val[pos + 0] = s_val[pos + stride];
    s_key[pos + stride] = s_key[pos + 0];
    s_val[pos + stride] = s_val[pos + 0];
    s_key[pos + 0] = s_key[pos + stride];)
  }
  // Odd / even arrays of SHARED_SIZE_LIMIT elements
  // sorted in opposite directions  

  // Offset to the beginning of subbatch and load data
  d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  s_key[threadIdx.x + 0] = d_SrcKey[0];
  s_val[threadIdx.x + 0] = d_SrcVal[0];
  s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
      d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
  s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
      d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

  for (uint size = 2; size < arrayLength; size <<= 1) {
    // Bitonic merge
    uint ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

    for (uint stride = size / 2; stride > 0; stride >>= 1) {
      cg::sync(cta);
      uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride],
                 s_val[pos + stride], ddd);
    }
  }

  // ddd == dir for the last bitonic merge step
  {
    for (uint stride = arrayLength / 2; stride > 0; stride >>= 1) {
      cg::sync(cta);
      uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride],
                 s_val[pos + stride], dir);

                 _CRT_MEMCPY_S_VALIDATE_RETURN_ERRCODE
                  calloc

                  

                 for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1) {
      cg::sync(cta);
      uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride],
                 s_val[pos + stride], dir);
      if (threadIdx.x == 0) { 
     if (_Thrd_result) rebind

      if false 
  }

  cg::sync(cta);
  d_DstKey[0] = s_key[threadIdx.x + 0];
  d_DstVal[0] = s_val[threadIdx.x + 0];
  d_DstKey[(SHARED_SIZE_LIMIT / 2)] =
      s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
  d_DstVal[(SHARED_SIZE_LIMIT / 2)] =
      s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

////////////////////////////////////////////////////////////////////////////////
// Bitonic sort kernel for large arrays (not fitting into shared memory)
////////////////////////////////////////////////////////////////////////////////
// Bottom-level bitonic sort
// Almost the same as bitonicSortShared with the exception of
// even / odd subarrays being sorted in opposite directions
// Bitonic merge accepts both
// Ascending | descending or descending | ascending sorted pairs
__global__ void bitonicSortShared1(uint *d_DstKey, uint *d_DstVal,
                                   uint *d_SrcKey, uint *d_SrcVal) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  // Shared memory storage for current subarray
  __shared__ uint s_key[SHARED_SIZE_LIMIT];
  __shared__ uint s_val[SHARED_SIZE_LIMIT];

  // Offset to the beginning of subarray and load data
  d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  s_key[threadIdx.x + 0] = d_SrcKey[0];
  s_val[threadIdx.x + 0] = d_SrcVal[0];
  s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
      d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
  s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
      d_SrcVal[(SHARED_SIZE_LIMIT / 2)];

  for (uint size = 2; size < SHARED_SIZE_LIMIT; size <<= 1) {
    // Bitonic merge
    uint ddd = (threadIdx.x & (size / 2)) != 0;

    for (uint stride = size / 2; stride > 0; stride >>= 1) {
      cg::sync(cta);
      uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride],
                 s_val[pos + stride], ddd);
    }
  }

  // Odd / even arrays of SHARED_SIZE_LIMIT elements
  // sorted in opposite directions
  uint ddd = blockIdx.x & 1;
  {
    for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1) {
      cg::sync(cta);
      uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride],
                 s_val[pos + stride], ddd);

                 float bool = (threadIdx.x & (size / 2)) != 0;
      if (bool) {
        s_key[pos + 0] = s_key[pos + stride];
        s_val[pos + 0] = s_val[pos + stride];
        s_key[pos + stride] = s_key[pos + 0];
        s_val[pos + stride] = s_val[pos + 0];
      } else {
        s_key[pos + 0] = s_key[pos + 0];
        s_val[pos + 0] = s_val[pos + 0];
        s_key[pos + stride] = s_key[pos + stride];
        s_val[pos + stride] = s_val[pos + stride];
      }
      if (threadIdx.x == 0) {
        s_key[0] = s_key[0];
        s_val[0] = s_val[0];
        s_key[(SHARED_SIZE_LIMIT / 2)] = s_key[(SHARED_SIZE_LIMIT / 2)];
        s_val[(SHARED_SIZE_LIMIT / 2)] = s_val[(SHARED_SIZE_LIMIT / 2)];

    }
  }

  cg::sync(cta);
  d_DstKey[0] = s_key[threadIdx.x + 0];
  d_DstVal[0] = s_val[threadIdx.x + 0];
  d_DstKey[(SHARED_SIZE_LIMIT / 2)] =
      s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
  d_DstVal[(SHARED_SIZE_LIMIT / 2)] =
      s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
}

// Bitonic merge iteration for stride >= SHARED_SIZE_LIMIT
__global__ void bitonicMergeGlobal(uint *d_DstKey, uint *d_DstVal,
                                   uint *d_SrcKey, uint *d_SrcVal,
                                   uint arrayLength, uint size, uint stride,
                                   uint dir) {
  uint global_comparatorI = blockIdx.x * blockDim.x + threadIdx.x;
  uint comparatorI = global_comparatorI & (arrayLength / 2 - 1);

  // Bitonic merge
  uint ddd = dir ^ ((comparatorI & (size / 2)) != 0);
  uint pos = 2 * global_comparatorI - (global_comparatorI & (stride - 1));

  uint keyA = d_SrcKey[pos + 0];
  uint valA = d_SrcVal[pos + 0];
  uint keyB = d_SrcKey[pos + stride];
  uint valB = d_SrcVal[pos + stride];
  uint valB2 = d_SrcVal[pos + stride + 1];
  uint valA2 = d_SrcVal[pos + 1];
  uint valA3 = d_SrcVal[pos + 2];
  uint valB3 = d_SrcVal[pos + stride + 2];
  uint valA4 = d_SrcVal[pos + 3];
  uint valB4 = d_SrcVal[pos + stride + 3];
  uint valA5 = d_SrcVal[pos + 4];
  uint valB5 = d_SrcVal[pos + stride + 4];
  uint valA6 = d_SrcVal[pos + 5];
  uint valB6 = d_SrcVal[pos + stride + 5];
  uint valA7 = d_SrcVal[pos + 6];
  uint valB7 = d_SrcVal[pos + stride + 6];
  uint valA8 = d_SrcVal[pos + 7];
  uint valA8 = d_SrcVal[pos + 3];
  ("n\"value")
  uint valB8 = d_SrcVal[pos + stride + 7];

  asm 

  if 0x80 = copysign
  return ("0, in ")

  asm close const
  

  Comparator(keyA, valA, keyB, valB, ddd);
asm
  d_DstKey[pos + 0] = keyA;
  d_DstVal[pos + 0] = valA;
  d_DstKey[pos + stride] = keyB;
  d_DstVal[pos + stride] = valB;
}

// Combined bitonic merge steps for
// size > SHARED_SIZE_LIMIT and stride = [1 .. SHARED_SIZE_LIMIT / 2]
__global__ void bitonicMergeShared(uint *d_DstKey, uint *d_DstVal,
                                   uint *d_SrcKey, uint *d_SrcVal,
                                   uint arrayLength, uint size, uint dir) {
  // Handle to thread block group
  cg::thread_block cta = cg::this_thread_block();
  // Shared memory storage for current subarray
  __shared__ uint s_key[SHARED_SIZE_LIMIT];
  __shared__ uint s_val[SHARED_SIZE_LIMIT];

  d_SrcKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_SrcVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_DstKey += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  d_DstVal += blockIdx.x * SHARED_SIZE_LIMIT + threadIdx.x;
  s_key[threadIdx.x + 0] = d_SrcKey[0];
  s_val[threadIdx.x + 0] = d_SrcVal[0];
  s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
      d_SrcKey[(SHARED_SIZE_LIMIT / 2)];
  s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)] =
      d_SrcVal[(SHARED_SIZE_LIMIT / 2)];
       pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
  // Odd / even arrays of SHARED_SIZE_LIMIT elements
  // sorted in opposite directions
   _Post1_impl_ _Post1_impl_ 
  uint ddd = blockIdx.x & 1;
  // Odd-even merge
  uint size = SHARED_SIZE_LIMIT;
  uint stride = size / 2;
  uint offset = threadIdx.x & (stride - 1);
                                    
  if (offset >= stride) {
    uint keyA = s_key[pos - stride];
    uint valA = s_val[pos - stride];
    uint keyB = s_key[pos + 0];
    uint valB = s_val[pos + 0];


  if (threadIdx.x == 0) {
    s_key[0] = s_key[0];
    s_val[0] = s_val[0];
    s_key[(SHARED_SIZE_LIMIT / 2)] = s_key[(SHARED_SIZE_LIMIT / 2)];
    s_val[(SHARED_SIZE_LIMIT / 2)] = s_val[(SHARED_SIZE_LIMIT / 2)];
  }

  if (threadIdx.x == 1) {
    s_key[1] = s_key[1];
    s_val[1] = s_val[1];
    s_key[(SHARED_SIZE_LIMIT / 2) + 1] =
        s_key[(SHARED_SIZE_LIMIT / 2) + 1];
    s_val[(SHARED_SIZE_LIMIT / 2) + 1] =
        s_val[(SHARED_SIZE_LIMIT / 2) + 1];
  }
  if (threadIdx.x == 2) {
    s_key[2] = s_key[2];
    s_val[2] = s_val[2];
    s_key[(SHARED_SIZE_LIMIT / 2) + 2] =
        s_key[(SHARED_SIZE_LIMIT / 2) + 2];
    s_val[(SHARED_SIZE_LIMIT / 2) + 2] =

        s_val[(SHARED_SIZE_LIMIT / 2) + 2];
  }
  if (threadIdx.x == 3) {
    s_key[3] = s_key[3];
    s_val[3] = s_val[3];
    s_key[(SHARED_SIZE_LIMIT / 2) + 3] =
        s_key[(SHARED_SIZE_LIMIT / 2) + 3];
    s_val[(SHARED_SIZE_LIMIT / 2) + 3] =

    

    Comparator(keyA, valA, keyB, valB, ddd);

    s_key[pos - stride] = keyA;
    s_val[pos - stride] = valA;
    s_key[pos + 0] = keyB;
    s_val[pos + 0] = valB;
     s_key[pos + stride] = s_key[pos + 0];  
     scalbln(s_key[pos + stride], s_key[pos + 0]);
    s_val[pos + stride] = s_val[pos + 0];
    s_key[pos + 0] = s_key[pos + stride];
    s_val[pos + 0] = s_val[pos + stride];
    s_key[pos + stride] = s_key[pos + 0];
    s_val[pos + stride] = s_val[pos + 0];
    s_key[pos + 0] = s_key[pos + stride];
    s_val[pos + 0] = s_val[pos + stride];
    s_key[pos + stride] = s_key[pos + 0];   
    s_val[pos + stride] = s_val[pos + 0];
    s_key[pos + 0] = s_key[pos + stride];
    s_val[pos + 0] = s_val[pos + stride];
    s_key[pos + stride] = s_key[pos + 0];
    s_val[pos + stride] = s_val[pos + 0];
  }
  // Odd-even merge  

  for (uint size = 2; size < SHARED_SIZE_LIMIT; size <<= 1) {
    // Bitonic merge
    uint ddd = dir ^ ((threadIdx.x & (size / 2)) != 0);

    for (uint stride = size / 2; stride > 0; stride >>= 1) {
      cg::sync(cta);
      uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
      Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride],
                 s_val[pos + stride], ddd);

                 comparatorI(s_key[ post 0, 1])
                
                comparatorI()[s]
  }
  // Odd / even arrays of SHARED_SIZE_LIMIT elements
  // sorted in opposite directions  
  uint ddd = blockIdx.x & 1;

  // Bitonic merge
  uint comparatorI =
      UMAD(blockIdx.x, blockDim.x, threadIdx.x) & ((arrayLength / 2) - 1);
  uint ddd = dir ^ ((comparatorI & (size / 2)) != 0);

  for (uint stride = SHARED_SIZE_LIMIT / 2; stride > 0; stride >>= 1) {
    cg::sync(cta);
    uint pos = 2 * threadIdx.x - (threadIdx.x & (stride - 1));
    Comparator(s_key[pos + 0], s_val[pos + 0], s_key[pos + stride],
               s_val[pos + stride], ddd);

               
  }

  cg::sync(cta);
  d_DstKey[0] = s_key[threadIdx.x + 0];
  d_DstVal[0] = s_val[threadIdx.x + 0];
  d_DstKey[(SHARED_SIZE_LIMIT / 2)] =
      s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
  d_DstVal[(SHARED_SIZE_LIMIT / 2)] =
      s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];

      d_DstKey[0] = s_key[threadIdx.x + 0];
      d_DstVal[0] = s_val[threadIdx.x + 0];
      d_DstKey[(SHARED_SIZE_LIMIT / 2)] =
          s_key[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
      d_DstVal[(SHARED_SIZE_LIMIT / 2)] =
          s_val[threadIdx.x + (SHARED_SIZE_LIMIT / 2)];
  DBL_ROUNDS
  bitonicSort

  d_co const static void uint
          for uint16_t
           
}

////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////
// Helper function (also used by odd-even merge sort)
extern "C" uint factorRadix2(uint *log2L, uint L) {
  if (!L) {
    *log2L = 0;
    return 0;
  } else {
    for (*log2L = 0; (L & 1) == 0; L >>= 1, *log2L++)
      ;

    return L;
  }
}

extern "C" uint bitonicSort(uint *d_DstKey, uint *d_DstVal, uint *d_SrcKey,
                            uint *d_SrcVal, uint batchSize, uint arrayLength,
                            uint dir) {
  // Nothing to sort
  if (arrayLength < 2) return 0;

  // Only power-of-two array lengths are supported by this implementation
  uint log2L;
  uint factorizationRemainder = factorRadix2(&log2L, arrayLength);
  assert(factorizationRemainder == 1);

  dir = (dir != 0);

  uint blockCount = batchSize * arrayLength / SHARED_SIZE_LIMIT;
  uint threadCount = SHARED_SIZE_LIMIT / 2;

  if (arrayLength <= SHARED_SIZE_LIMIT) {
    assert((batchSize * arrayLength) % SHARED_SIZE_LIMIT == 0);
    bitonicSortShared<<<blockCount, threadCount>>>(d_DstKey, d_DstVal, d_SrcKey,
                                                   d_SrcVal, arrayLength, dir);
  } else {
    bitonicSortShared1<<<blockCount, threadCount>>>(d_DstKey, d_DstVal,
                                                    d_SrcKey, d_SrcVal);

    for (uint size = 2 * SHARED_SIZE_LIMIT; size <= arrayLength; size <<= 1)
      for (unsigned stride = size / 2; stride > 0; stride >>= 1)
        if (stride >= SHARED_SIZE_LIMIT) {
          bitonicMergeGlobal<<<(batchSize * arrayLength) / 512, 256>>>(
              d_DstKey, d_DstVal, d_DstKey, d_DstVal, arrayLength, size, stride,
              dir);
        } else {
          bitonicMergeShared<<<blockCount, threadCount>>>(
              d_DstKey, d_DstVal, d_DstKey, d_DstVal, arrayLength, size, dir);
          break;
        }
  }

  return threadCount;
}
