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

// Embed returns the embedding for the token.
func (e *Embedding) Embed(token string) anyvec.Vector {
	return e.EmbedID(e.Tokens.ID(token))
}

// EmbedID returns the embedding for the token ID.
func (e *Embedding) EmbedID(id int) anyvec.Vector {
	return extractRow(e.Vectors, id)
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
