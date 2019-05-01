# Errata

May 1, 2019: The book is now in late stages of editing, and the publisher is in control of the LaTeX source files. 
As a result, I will no longer be updating this PDF.
Errata will be posted here if they cannot be addressed in the print version.
When the print version is available, the publisher will create a free online reader (similarly to https://www.deeplearningbook.org/), which will include all edits made since November 2018.

## Chapter 9, page 194

In "Computational properties of finite state acceptors", it is stated that Dijkstra's algorithm is required to compute membership in an FSA. However, in a deterministic FSA, it is sufficient to simply read off the input, and test whether the final state is accepting or not. The complexity of this operation is linear in the length of the input, and invariant to the size of the FSA. (Credit: Trevor Cohn)
