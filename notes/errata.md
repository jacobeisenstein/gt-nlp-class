# Errata

## Exercise 2.5

The statement to be proved is not correct. Take the additional assumption that there is only a single scalar feature, and not a vector of features. (Credit: Leo Du)

## Chapter 9, page 194

In "Computational properties of finite state acceptors", it is stated that Dijkstra's algorithm is required to compute membership in an FSA. However, in a deterministic FSA, it is sufficient to simply read off the input, and test whether the final state is accepting or not. The complexity of this operation is linear in the length of the input, and invariant to the size of the FSA. (Credit: Trevor Cohn)

## Chapter 4, page 93

In line (6) of Algorithm 8, the denominator should include a summation $\sum_{i=1}^N$. (Credit: Janina Sarol)

## Chapter 12, page 273 of the printed version

In the paragraph "functions", the example should be $[\textsc{capital-of}(\textsc{georgia})] = [\textsc{atlanta}]$, not $[\textsc{capital-of}][(\textsc{georgia})] = [\textsc{atlanta}]$. (Credit: Varol Akman)

## Exercise 6.4

This problem was intended to have a short and fairly simple solution, but the derivation that I had in mind is not correct. The fully correct derivation is much longer, but leads to very nearly the same answer for practical value of the parameters in the problem. (Credit: Sofia Serrano, Jungo Kasai, Lianhui Qin, Sewon Min, Harrison Bay, Pemi Nguyen, and Noah Smith)
