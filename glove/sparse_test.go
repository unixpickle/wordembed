package glove

import (
	"math/rand"
	"testing"
)

func TestSparseMatrix(t *testing.T) {
	dense := make([]float32, 3*4)
	sparse := NewSparseMatrix(3, 4)

	for i := 0; i < 100; i++ {
		row := rand.Intn(3)
		col := rand.Intn(4)
		val := rand.Float32()
		switch rand.Intn(2) {
		case 0:
			sparse.Add(row, col, val)
			dense[row*4+col] += val
		case 1:
			sparse.Set(row, col, val)
			dense[row*4+col] = val
		}
		for row := 0; row < 3; row++ {
			for col := 0; col < 4; col++ {
				actual := sparse.Get(row, col)
				expected := dense[row*4+col]
				if actual != expected {
					t.Fatal("matrices are out of sync")
				}
			}
		}
	}
}
