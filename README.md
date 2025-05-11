A custom kernel for advanced sorting network tasks.
Credit to NVIDIA and CUDA developers for partial source code in assistance of dvelopment.

##FOSS Software.

(Custom CDN)

____________________________________________________________________________________

What is a Bitonic Sort? 
But what does the word “bitonic” even mean? This was my first question when I happened on the algo. A monotonic sequence is one whose elements are all non-decreasing or non-increasing (i.e. sorted sequences like [1,2,3,4] or [4,3,2,1]). A bitonic sequence is a sequence formed by a concatenation of two monotonic sequences, e.g. [1,3,5,4,3,2,1].

Bitonic sort is a species of sorting network, a popular family of fast, parallel (comparison) sorting algorithms. Bitonic sort, for example, can sort in O(log^2(n)) “time” (i.e. “parallel depth”/“delay”/“wall time”). How is this possible when it’s proven that a comparison-based sort requires O(n*log(n)) time? Because that’s the runtime requirement for a sequential algorithm – specifically, the “work” needed. Bitonic sort requires O(n*log^2(n)) parallel work (total number of comparisons, or “cpu time”) which can be completely parallelized across n processors. Usually, CS majors only care about the big-O of sorts, but constants also matter in the real world. Thankfully, bitonic sort seems to incur small constants and offers good cache locality.

Sorting networks are typically represented like a circuit with a series of parallel swaps (predicated, of course, upon a partial order). Since they are (virtual) circuits, a sorting network operates on a specific number of elements.

Here’s an example of a (“non-alternating”) bitonic sort on 16 elements:


(Credit to https://winwang.blog/posts/bitonic-sort/#what-is-a-bitonic-sort for in-depth explanation of bitonic source basis.)

![image](https://github.com/user-attachments/assets/d2e3cecd-3baa-431e-bd6d-a21dbf5d1be4)


![code]Shuffle code example:
// load data into register
u32 datum = ...;
// Loop:
// calculate the lane to compare against
u32 other_lane = ...;
// shuffle!
u32 other_datum = __shfl_sync(gpu::all_lanes, datum, other_lane);
// swap if necessary between this lane and other_lane
// such that the smaller datum is in the lower lane index
if (lane < other_lane) {
    datum = min(datum, other_datum);
} else {
    datum = max(datum, other_datum);
}


_____________________________________________________


![image](https://github.com/user-attachments/assets/5fc65180-aa02-477e-9dcd-4782c36039ac)

(Credit to NVIDIA Developer page.)


Odd even-merge sort code example in use:
uniform vec3 Param1;
uniform vec3 Param2;
uniform sampler2D Data;
#define OwnPos gl_TexCoord[0]
// contents of the uniform data fields
#define TwoStage Param1.x
#define Pass_mod_Stage Param1.y
#define TwoStage_PmS_1 Param1.z
#define Width Param2.x
#define Height Param2.y
#define Pass Param2.z
void main(void)
{
  // get self
  vec4 self = texture2D(Data, OwnPos.xy);
  float i = floor(OwnPos.x * Width) + floor(OwnPos.y * Height) * Width;
  // my position within the range to merge
  float j = floor(mod(i, TwoStage));
  float compare;
  if ((j < Pass_mod_Stage) || (j > TwoStage_PmS_1))
    // must copy->compare with self
    compare = 0.0;
  else
    // must sort
    if (mod((j + Pass_mod_Stage) / Pass, 2.0) < 1.0)
      // we are on the left side->compare with partner on the right
      compare = 1.0;
    else
      // we are on the right side->compare with partner on the left
      compare = -1.0;
  // get the partner
  float adr = i + compare * Pass;
  vec4 partner = texture2D(
      Data, vec2(floor(mod(adr, Width)) / Width, floor(adr / Width) / Height));
  // on the left it's a < operation; on the right it's a >= operation
  gl_FragColor = (self.x * compare < partner.x * compare) ? self : partner;
}
