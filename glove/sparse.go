package glove

import (
	"fmt"
	"sort"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	serializer.RegisterTypedDeserializer((&SparseMatrix{}).SerializerType(),
		DeserializeSparseMatrix)
	serializer.RegisterTypedDeserializer((&SparseVector{}).SerializerType(),
		DeserializeSparseVector)
}

// A SparseMatrix is a sparse matrix for storing word
// co-occurrences.
//
// Concurrent reads/writes to/from separate rows in the
// matrix are safe.
type SparseMatrix struct {
	Rows []*SparseVector
}

// DeserializeSparseMatrix deserializes a SparseMatrix.
func DeserializeSparseMatrix(d []byte) (mat *SparseMatrix, err error) {
	defer essentials.AddCtxTo("deserialize SparseMatrix", &err)

	rows, err := serializer.DeserializeSlice(d)
	if err != nil {
		return nil, err
	}

	var res SparseMatrix
	for _, row := range rows {
		if obj, ok := row.(*SparseVector); ok {
			res.Rows = append(res.Rows, obj)
		} else {
			return nil, fmt.Errorf("unexpected type: %T", row)
		}
	}

	return &res, nil
}

// NewSparseMatrix creates a zero matrix.
func NewSparseMatrix(rows, cols int) *SparseMatrix {
	res := &SparseMatrix{
		Rows: make([]*SparseVector, rows),
	}
	for i := range res.Rows {
		res.Rows[i] = &SparseVector{Len: cols}
	}
	return res
}

// Get reads an entry in the matrix.
func (s *SparseMatrix) Get(row, col int) float32 {
	return s.Rows[row].Get(col)
}

// Set sets an entry in the matrix.
func (s *SparseMatrix) Set(row, col int, val float32) {
	s.Rows[row].Set(col, val)
}

// Add adds a value to the entry.
//
// This is faster than using a Get followed by a Set.
func (s *SparseMatrix) Add(row, col int, val float32) {
	s.Rows[row].Add(col, val)
}

// NumEntries returns the number of entries that have been
// set with Set.
func (s *SparseMatrix) NumEntries() int {
	var res int
	for _, row := range s.Rows {
		res += len(row.Indices)
	}
	return res
}

// SerializerType returns the unique ID used to serialize
// a SparseMatrix with the serializer package.
func (s *SparseMatrix) SerializerType() string {
	return "github.com/unixpickle/wordembed/glove.SparseMatrix"
}

// Serialize serializes the SparseMatrix.
func (s *SparseMatrix) Serialize() ([]byte, error) {
	var res []serializer.Serializer
	for _, obj := range s.Rows {
		res = append(res, obj)
	}
	return serializer.SerializeSlice(res)
}

// SparseVector is a list with potentially many zero
// entries.
type SparseVector struct {
	// Len is the total number of elements (including
	// zeros) in the represented vector.
	Len int

	// Indices stores the index of each non-zero value.
	// Each index corresponds to an entry in Values.
	Indices []int32

	Values []float32
}

// DeserializeSparseVector deserializes a SparseVector.
func DeserializeSparseVector(d []byte) (*SparseVector, error) {
	var res SparseVector
	err := serializer.DeserializeAny(d, &res.Len, &res.Indices, &res.Values)
	if err != nil {
		return nil, essentials.AddCtx("deserialize SparseVector", err)
	}
	if len(res.Indices) == 0 {
		// Save memory and make deep equality hold.
		res.Indices = nil
		res.Values = nil
	}
	return &res, nil
}

// Get reads the entry at the index.
func (s *SparseVector) Get(i int) float32 {
	idx := s.searchIndices(i)
	if idx == len(s.Indices) || int(s.Indices[idx]) != i {
		return 0
	}
	return s.Values[idx]
}

// Set sets the entry at the index.
func (s *SparseVector) Set(i int, val float32) {
	idx := s.searchIndices(i)
	if idx == len(s.Indices) {
		s.Indices = append(s.Indices, int32(i))
		s.Values = append(s.Values, val)
	} else if int(s.Indices[idx]) != i {
		s.Indices = append(s.Indices, 0)
		s.Values = append(s.Values, 0)
		copy(s.Indices[idx+1:], s.Indices[idx:])
		copy(s.Values[idx+1:], s.Values[idx:])
		s.Indices[idx] = int32(i)
		s.Values[idx] = val
	} else {
		s.Values[idx] = val
	}
}

// Add adds to the entry at the index.
//
// If a new entry was allocated, true is returned.
func (s *SparseVector) Add(i int, val float32) {
	idx := s.searchIndices(i)
	if idx == len(s.Indices) {
		s.Indices = append(s.Indices, int32(i))
		s.Values = append(s.Values, val)
	} else if int(s.Indices[idx]) != i {
		s.Indices = append(s.Indices, 0)
		s.Values = append(s.Values, 0)
		copy(s.Indices[idx+1:], s.Indices[idx:])
		copy(s.Values[idx+1:], s.Values[idx:])
		s.Indices[idx] = int32(i)
		s.Values[idx] = val
	} else {
		s.Values[idx] += val
	}
}

// SerializerType returns the unique ID used to serialize
// a SparseVector with the serializer package.
func (s *SparseVector) SerializerType() string {
	return "github.com/unixpickle/wordembed/glove.SparseVector"
}

// Serialize serializes the SparseVector.
func (s *SparseVector) Serialize() ([]byte, error) {
	return serializer.SerializeAny(s.Len, s.Indices, s.Values)
}

func (s *SparseVector) searchIndices(index int) int {
	return sort.Search(len(s.Indices), func(arg2 int) bool {
		return int(s.Indices[arg2]) >= index
	})
}
