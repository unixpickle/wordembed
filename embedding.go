package wordembed

import "github.com/unixpickle/anyvec"

// Embedding is a generic word embedding.
type Embedding interface {
	// Dim returns the dimensionality of the embedding.
	Dim() int

	// Embed returns the embedding for the token.
	Embed(token string) anyvec.Vector

	// EmbedID returns the embedding for the token ID.
	EmbedID(id int) anyvec.Vector

	// Lookup finds the n nearest token IDs.
	//
	// If n is greater than the total number of words,
	// there will be fewer than n results.
	Lookup(vec anyvec.Vector, n int) ([]int, []anyvec.Numeric)

	// Token looks up the token for the token ID.
	Token(id int) string
}
