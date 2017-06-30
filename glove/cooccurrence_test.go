package glove

import (
	"math/rand"
	"sort"
	"strconv"
	"testing"

	"github.com/unixpickle/wordembed"
)

func BenchmarkAddCooccurrences(b *testing.B) {
	tokens := wordembed.TokenSet{}
	for i := 0; i < 10000; i++ {
		tokens = append(tokens, strconv.Itoa(rand.Int()))
	}
	sort.Strings(tokens)

	documents := make([][]string, 20000)
	for i := range documents {
		for j := 0; j < rand.Intn(20); j++ {
			documents[i] = append(documents[i], tokens[rand.Intn(len(tokens))])
		}
	}

	counter := &CooccurCounter{
		Tokens: tokens,
		Matrix: NewSparseMatrix(tokens.NumIDs(), tokens.NumIDs()),
	}

	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		for _, doc := range documents {
			counter.Add(doc)
		}
	}
}
