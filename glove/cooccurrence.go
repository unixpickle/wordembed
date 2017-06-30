package glove

import (
	"runtime"
	"sync"

	"github.com/unixpickle/wordembed"
)

// A CooccurCounter tallies the co-occurrences of tokens
// in a stream of tokenized documents.
type CooccurCounter struct {
	// Tokens is the set of tokens to tally.
	// IDs from Tokens are used as indices in the matrix.
	Tokens wordembed.TokenSet

	// The matrix in which tallying is done.
	//
	// Each row is a context.
	// Entry (i, j) counts the word j in the context of
	// word i.
	//
	// Currently, the matrix will always be symmetrical.
	// However, this will not necessarily be true in the
	// future.
	//
	// Note that this should have size (len(Tokens)+1)^2,
	// since that is the number of token IDs.
	Matrix *SparseMatrix

	// Window is the maximum distance a word must be from
	// another word for the co-occurrence to be counted.
	// If this is 0, all words from the same document are
	// counted as co-occurring.
	Window int

	// WeightWords, if true, indicates that more distant
	// co-occurrences should be counted less than closer
	// ones.
	WeightWords bool
}

// Add adds all the co-occurrences from the tokenized
// document.
func (c *CooccurCounter) Add(document []string) {
	c.addWithIDs(nil, c.Tokens.IDs(document))
}

// AddAll adds the co-occurrences from each document.
//
// Unlike Add, AddAll can utilize more than one thread.
func (c *CooccurCounter) AddAll(documents <-chan []string) {
	rowLocks := make([]*sync.Mutex, len(c.Matrix.Rows))
	for i := range rowLocks {
		rowLocks[i] = &sync.Mutex{}
	}

	var wg sync.WaitGroup
	for i := 0; i < runtime.GOMAXPROCS(0); i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for input := range documents {
				ids := c.Tokens.IDs(input)
				c.addWithIDs(rowLocks, ids)
			}
		}()
	}
	wg.Wait()
}

func (c *CooccurCounter) addWithIDs(rowLocks []*sync.Mutex, ids []int) {
	for i := range ids {
		for j := i - 1; j >= 0 && (j >= i-c.Window || c.Window == 0); j-- {
			weight := float32(1)
			if c.WeightWords {
				weight = 1 / float32(i-j)
			}
			id1, id2 := ids[i], ids[j]

			if rowLocks != nil {
				rowLocks[id1].Lock()
				c.Matrix.Add(id1, id2, weight)
				rowLocks[id1].Unlock()
				rowLocks[id2].Lock()
				c.Matrix.Add(id2, id1, weight)
				rowLocks[id2].Unlock()
			} else {
				c.Matrix.Add(id1, id2, weight)
				c.Matrix.Add(id2, id1, weight)
			}
		}
	}
}
