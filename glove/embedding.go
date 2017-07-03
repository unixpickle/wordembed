package glove

import (
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/wordembed"
)

func init() {
	serializer.RegisterTypedDeserializer((&Embedding{}).SerializerType(),
		DeserializeEmbedding)
}

// Embedding is a trained word embedding.
type Embedding struct {
	// Tokens is the list of available words.
	Tokens wordembed.TokenSet

	// Vectors contains one row per token ID.
	Vectors *anyvec.Matrix
}

// DeserializeEmbedding deserializes an Embedding.
func DeserializeEmbedding(d []byte) (*Embedding, error) {
	var res Embedding
	var rows, cols int
	var data *anyvecsave.S
	if err := serializer.DeserializeAny(d, &res.Tokens, &rows, &cols, &data); err != nil {
		return nil, essentials.AddCtx("deserialize Embedding", err)
	}
	res.Vectors = &anyvec.Matrix{
		Data: data.Vector,
		Rows: rows,
		Cols: cols,
	}
	return &res, nil
}

// Nomalize makes all the vectors have the same magnitude.
// This may improve performance on certain tasks.
func (e *Embedding) Normalize() {
	c := e.Vectors.Data.Creator()
	squares := e.Vectors.Data.Copy()
	anyvec.Pow(squares, c.MakeNumeric(2))
	normalizers := anyvec.SumCols(squares, e.Vectors.Rows)
	anyvec.Pow(normalizers, c.MakeNumeric(-0.5))
	anyvec.ScaleChunks(e.Vectors.Data, normalizers)
}

// Embed returns the embedding for the token.
func (e *Embedding) Embed(token string) anyvec.Vector {
	return e.EmbedID(e.Tokens.ID(token))
}

// EmbedID returns the embedding for the token ID.
func (e *Embedding) EmbedID(id int) anyvec.Vector {
	return extractRow(e.Vectors, id).Copy()
}

// Lookup finds the n closest token IDs to the given
// vector, using cosine similarity.
// For each ID, it also returns the similarity.
//
// If n is greater than the number of IDs, then there will
// be fewer than n results.
func (e *Embedding) Lookup(vec anyvec.Vector, n int) ([]int, []anyvec.Numeric) {
	if vec.Len() != e.Vectors.Cols {
		panic("incorrect vector length")
	}

	c := e.Vectors.Data.Creator()
	squares := e.Vectors.Data.Copy()
	anyvec.Pow(squares, c.MakeNumeric(2))
	normalizers := anyvec.SumCols(squares, e.Vectors.Rows)
	anyvec.Pow(normalizers, c.MakeNumeric(-0.5))

	masked := e.Vectors.Data.Copy()
	anyvec.ScaleChunks(masked, normalizers)
	normVec := vec.Copy()
	normVec.Scale(c.NumOps().Div(c.MakeNumeric(1), anyvec.Norm(vec)))
	anyvec.ScaleRepeated(masked, normVec)

	dots := anyvec.SumCols(masked, e.Vectors.Rows)

	var ids []int
	var dists []anyvec.Numeric
	for i := 0; i < n && i < dots.Len(); i++ {
		idx := anyvec.MaxIndex(dots)
		ids = append(ids, idx)

		dist := anyvec.Sum(dots.Slice(idx, idx+1))
		dists = append(dists, dist)

		// Make sure we don't get this ID again.
		dots.Slice(idx, idx+1).AddScalar(c.MakeNumeric(-3))
	}
	return ids, dists
}

// SerializerType returns the unique ID used to serialize
// an Embedding with the serializer package.
func (e *Embedding) SerializerType() string {
	return "github.com/unixpickle/wordembed/glove.Embedding"
}

// Serialize serializes the Embedding.
func (e *Embedding) Serialize() ([]byte, error) {
	return serializer.SerializeAny(
		e.Tokens,
		e.Vectors.Rows,
		e.Vectors.Cols,
		&anyvecsave.S{Vector: e.Vectors.Data},
	)
}
