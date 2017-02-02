package word2vec

import (
	"reflect"
	"testing"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anydiff/anydifftest"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/serializer"
)

func TestEmbedProp(t *testing.T) {
	vec := anyvec32.MakeVector(10)
	anyvec.Rand(vec, anyvec.Normal, nil)
	e := &Embed{
		Matrix:      anydiff.NewVar(vec),
		WordToIndex: map[string]int{"a": 0, "b": 1, "c": 2, "d": 3, "e": 4},
	}
	checker := anydifftest.ResChecker{
		F: func() anydiff.Res {
			return e.Embed("foo", "c")
		},
		V:     []*anydiff.Var{e.Matrix},
		Delta: 1e-2,
		Prec:  1e-3,
	}
	checker.FullCheck(t)
}

func TestEmbedSortSimilar(t *testing.T) {
	vec := anyvec32.MakeVectorData([]float32{
		1, 1,
		0, 1,
		1, 0,
		0, -1,
	})
	e := &Embed{
		Matrix:      anydiff.NewVar(vec),
		WordToIndex: map[string]int{"a": 0, "b": 1, "c": 2, "d": 3},
	}
	aVec := e.Embed("d", "").Output()
	actual := e.SortSimilar(aVec)
	expected := []string{"d", "c", "a", "b"}
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected %v got %v", expected, actual)
	}
}

func TestEmbedSerialize(t *testing.T) {
	vec := anyvec32.MakeVector(10)
	anyvec.Rand(vec, anyvec.Normal, nil)
	e := &Embed{
		Matrix:      anydiff.NewVar(vec),
		WordToIndex: map[string]int{"a": 0, "b": 1, "c": 2, "d": 3, "e": 4},
	}
	data, err := serializer.SerializeAny(e)
	if err != nil {
		t.Fatal(err)
	}
	var e1 *Embed
	if err := serializer.DeserializeAny(data, &e1); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(e, e1) {
		t.Error("invalid result")
	}
}
