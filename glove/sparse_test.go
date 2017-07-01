package glove

import (
	"fmt"
	"math/rand"
	"testing"
)

func TestSparseMatrix(t *testing.T) {
	dense := make([]float32, 3*4)
	sparse := NewSparseMatrix(3, 4)

	entriesUsed := map[string]bool{}

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
		entriesUsed[fmt.Sprintf("%d,%d", row, col)] = true
		if len(entriesUsed) != sparse.NumEntries() {
			t.Errorf("entry count should be %d but got %d", len(entriesUsed),
				sparse.NumEntries())
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

func BenchmarkMatrixRandomEntry(b *testing.B) {
	matrix := NewSparseMatrix(100000, 100000)
	for i := 0; i < 100000; i++ {
		matrix.Set(rand.Intn(100000), rand.Intn(100000), rand.Float32())
	}
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		matrix.RandomEntry()
	}
}
