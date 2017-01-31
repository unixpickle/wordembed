package word2vec

// A Sample contains a word and its surrounding context.
type Sample struct {
	// Left contains the words leading up to the word.
	Left []string

	// Word is the word itself.
	Word string

	// Right contains the words following the word.
	Right []string
}

// AllSamples creates one Sample per word in a slice of
// words.
// The samples will all include sub-slices of the given
// slice, so the slice should not be modified after the
// call.
func AllSamples(words []string) []*Sample {
	res := make([]*Sample, len(words))
	for i, x := range words {
		res[i] = &Sample{
			Left:  words[:i],
			Word:  x,
			Right: words[i+1:],
		}
	}
	return res
}

// Trim trims the sample to have no more than the given
// number of neighboring words on each size.
func (s *Sample) Trim(max int) *Sample {
	res := *s
	if len(res.Left) > max {
		res.Left = res.Left[len(res.Left)-max:]
	}
	if len(res.Right) > max {
		res.Right = res.Right[:max]
	}
	return &res
}
