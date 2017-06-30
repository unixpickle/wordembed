package wordembed

import (
	"sort"

	"github.com/unixpickle/essentials"
)

// TokenCounts keeps track of how many times different
// tokens occurr in some corpus.
type TokenCounts map[string]int

// Add adds another occurrence of the token.
func (t TokenCounts) Add(token string) {
	t[token]++
}

// AddAll adds each token from the stream.
func (t TokenCounts) AddAll(stream <-chan string) {
	for tok := range stream {
		t.Add(tok)
	}
}

// MostCommon produces the n tokens with the most
// occurrences.
// If there are less than n total tokens, then all tokens
// are returned.
func (t TokenCounts) MostCommon(n int) TokenSet {
	var counts []int
	var tokens []string
	for tok, num := range t {
		tokens = append(tokens, tok)
		counts = append(counts, num)
	}

	if len(tokens) <= n {
		sort.Strings(tokens)
		return tokens
	}

	essentials.VoodooSort(counts, func(i, j int) bool {
		return counts[i] > counts[j]
	}, tokens)

	// Don't waste lots of memory.
	tokens = append([]string{}, tokens[:n]...)

	sort.Strings(tokens)
	return tokens
}
