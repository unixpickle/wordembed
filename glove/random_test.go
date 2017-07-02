package glove

import (
	"math"
	"math/rand"
	"testing"
)

func TestRandomEntry(t *testing.T) {
	const numIters = 1000000

	matrix := NewSparseMatrix(3, 4)
	matrix.Set(2, 3, 15)
	matrix.Set(1, 3, 4)
	matrix.Set(2, 0, 9)

	picker := newRandomEntryPicker(matrix)

	var sum float64
	for i := 0; i < numIters; i++ {
		row, col := picker.Pick()
		sum += float64(matrix.Get(row, col))
	}
	mean := sum / numIters
	if math.Abs(mean-28.0/3) > 1e-2 {
		t.Errorf("mean should be %v but got %v", 28.0/3, mean)
	}
}

func BenchmarkRandomEntry(b *testing.B) {
	matrix := NewSparseMatrix(100000, 100000)
	for i := 0; i < 100000; i++ {
		matrix.Set(rand.Intn(100000), rand.Intn(100000), rand.Float32())
	}
	picker := newRandomEntryPicker(matrix)
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		picker.Pick()
	}
}
