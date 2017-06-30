package wordembed

import "github.com/unixpickle/essentials"

// TokenCounts keeps track of how many times different
// tokens occurr in some corpus.
type TokenCounts map[string]int

// CountTokens counts the tokens from a stream.
func CountTokens(stream <-chan string) TokenCounts {
	counts := TokenCounts{}
	for tok := range stream {
		counts[tok]++
	}
	return counts
}

// MostCommon produces the n tokens with the most
// occurrences.
// If there are less than n total tokens, then all tokens
// are returned.
func (t TokenCounts) MostCommon(n int) []string {
	var counts []int
	var tokens []string
	for tok, num := range t {
		tokens = append(tokens, tok)
		counts = append(counts, num)
	}

	if len(tokens) <= n {
		return tokens
	}

	essentials.VoodooSort(counts, func(i, j int) bool {
		return counts[i] > counts[j]
	}, tokens)
	return tokens[:n]
}
