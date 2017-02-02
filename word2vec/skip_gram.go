package word2vec

import (
	"math/rand"
	"sort"

	"github.com/unixpickle/anyvec"
)

const defaultMinDist = 1
const defaultMaxDist = 5

// SkipGram can train skip-gram models.
type SkipGram struct {
	Net       *Net
	Hierarchy Hierarchy
	Samples   []*Sample

	// StepSize should be negative for gradient descent.
	StepSize anyvec.Numeric

	// The minimum and maximum number of neighbors to use as
	// context during training.
	//
	// If these are 0, the defaults from the word2vec paper
	// are used.
	MinDist int
	MaxDist int

	// StatusFunc, if non-nil, is called after every training
	// iteration with the cost from that iteration.
	StatusFunc func(lastCost anyvec.Numeric)
}

// Train trains the skip-gram model until the done channel
// is closed.
func (s *SkipGram) Train(done <-chan struct{}) {
	w2i := s.wordToIndex()
	creator := s.Net.Encoder.Vector.Creator()
	one := creator.MakeNumeric(1)

	for {
		select {
		case <-done:
			return
		default:
		}

		sample := s.Samples[rand.Intn(len(s.Samples))]
		radius := rand.Intn(s.MaxDist-s.MinDist+1) + s.MinDist
		sample = sample.Trim(radius)

		in := map[int]anyvec.Numeric{w2i[sample.Word]: one}
		allWords := append(append([]string{}, sample.Left...), sample.Right...)
		paths := s.Hierarchy.Paths(allWords)

		cost := s.Net.Step(in, paths, s.StepSize)
		if s.StatusFunc != nil {
			s.StatusFunc(cost)
		}
	}
}

func (s *SkipGram) wordToIndex() map[string]int {
	var words []string
	for x := range s.Hierarchy {
		words = append(words, x)
	}
	sort.Strings(words)

	res := map[string]int{}
	for index, word := range words {
		res[word] = index
	}
	return res
}
