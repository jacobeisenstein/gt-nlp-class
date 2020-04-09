# Errata

## Chapter 9, page 194

In "Computational properties of finite state acceptors", it is stated that Dijkstra's algorithm is required to compute membership in an FSA. However, in a deterministic FSA, it is sufficient to simply read off the input, and test whether the final state is accepting or not. The complexity of this operation is linear in the length of the input, and invariant to the size of the FSA. (Credit: Trevor Cohn)

## Chapter 4, page 93

In line (6) of Algorithm 8, the denominator should include a summation $\sum_{i=1}^N$. (Credit: Janina Sarol)
