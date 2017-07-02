package glove

import (
	"math/rand"
	"sort"
)

// A randomEntryPicker selects random non-zero entries
// from a matrix.
// While a randomEntryPicker is being used, the matrix
// should not be modified.
type randomEntryPicker struct {
	matrix        *SparseMatrix
	offsetsPerRow []int
	numEntries    int
	gen           *rand.Rand
}

func newRandomEntryPicker(s *SparseMatrix) *randomEntryPicker {
	r := &randomEntryPicker{
		matrix: s,
		gen:    rand.New(rand.NewSource(rand.Int63())),
	}
	for _, row := range s.Rows {
		r.numEntries += len(row.Indices)
		r.offsetsPerRow = append(r.offsetsPerRow, r.numEntries)
	}
	return r
}

func (r *randomEntryPicker) Pick() (row, col int) {
	offset := r.gen.Intn(r.numEntries)
	row = r.rowForOffset(offset)
	rowObj := r.matrix.Rows[row]
	rowStart := r.offsetsPerRow[row] - len(rowObj.Indices)
	col = int(rowObj.Indices[offset-rowStart])
	return
}

func (r *randomEntryPicker) rowForOffset(off int) int {
	return sort.Search(len(r.offsetsPerRow), func(x int) bool {
		return r.offsetsPerRow[x] > off
	})
}
