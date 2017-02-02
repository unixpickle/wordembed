package word2vec

import (
	"encoding/json"
	"errors"
	"sort"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/serializer"
)

func init() {
	var e Embed
	serializer.RegisterTypedDeserializer(e.SerializerType(), DeserializeEmbed)
}

// Embed produces embeddings for words.
type Embed struct {
	Matrix      *anydiff.Var
	WordToIndex map[string]int
}

// DeserializeEmbed deserializes an Embed.
func DeserializeEmbed(d []byte) (*Embed, error) {
	var vec *anyvecsave.S
	var wordList serializer.Bytes
	if err := serializer.DeserializeAny(d, &vec, &wordList); err != nil {
		return nil, errors.New("deserialize Embed: " + err.Error())
	}
	var m map[string]int
	if err := json.Unmarshal(wordList, &m); err != nil {
		return nil, errors.New("deserialize Embed: " + err.Error())
	}
	return &Embed{Matrix: anydiff.NewVar(vec.Vector), WordToIndex: m}, nil
}

// NewEmbed creates an Embed from the vectors in an
// encoder matrix and the words from a hierarchy.
//
// It is assumed that the rows in the matrix correspond to
// the words in a sorted order.
// If you used SkipGram to train the model, then NewEmbed
// will just work.
func NewEmbed(mat *anydiff.Var, h Hierarchy) *Embed {
	var words []string
	for w := range h {
		words = append(words, w)
	}
	sort.Strings(words)

	m := map[string]int{}
	for i, w := range words {
		m[w] = i
	}

	return &Embed{
		Matrix:      mat,
		WordToIndex: m,
	}
}

// Embed produces an embedding for the word.
// If no embedding was found, the provided default word is
// used.
// If the default word is also not found, nil is returned.
func (e *Embed) Embed(word, defaultWord string) anydiff.Res {
	idx, ok := e.WordToIndex[word]
	if !ok {
		idx, ok = e.WordToIndex[defaultWord]
		if !ok {
			return nil
		}
	}
	cols := e.Matrix.Vector.Len() / len(e.WordToIndex)
	start := cols * idx
	vec := e.Matrix.Vector.Creator().MakeVector(cols)
	vec.SetSlice(-start, e.Matrix.Vector)
	return &embedRes{
		Mat:    e.Matrix,
		OutVec: vec,
		Index:  start,
	}
}

// SerializerType returns the unique ID used to serialize
// an Embed with the serializer package.
func (e *Embed) SerializerType() string {
	return "github.com/unixpickle/wordembed/word2vec.Embed"
}

// Serialize serializes the Embed.
func (e *Embed) Serialize() ([]byte, error) {
	mapData, _ := json.Marshal(e.WordToIndex)
	return serializer.SerializeAny(
		&anyvecsave.S{Vector: e.Matrix.Vector},
		serializer.Bytes(mapData),
	)
}

type embedRes struct {
	Mat    *anydiff.Var
	OutVec anyvec.Vector
	Index  int
}

func (e *embedRes) Output() anyvec.Vector {
	return e.OutVec
}

func (e *embedRes) Vars() anydiff.VarSet {
	res := anydiff.VarSet{}
	res.Add(e.Mat)
	return res
}

func (e *embedRes) Propagate(u anyvec.Vector, g anydiff.Grad) {
	if v, ok := g[e.Mat]; ok {
		tempSlice := u.Creator().MakeVector(u.Len())
		tempSlice.SetSlice(-e.Index, v)
		tempSlice.Add(u)
		v.SetSlice(e.Index, tempSlice)
	}
}
