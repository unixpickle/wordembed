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
