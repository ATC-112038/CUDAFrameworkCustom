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
