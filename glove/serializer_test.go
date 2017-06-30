package glove

import (
	"reflect"
	"testing"

	"github.com/unixpickle/anyvec/anyvec32"
	"github.com/unixpickle/serializer"
)

func TestMatrixSerialize(t *testing.T) {
	testSerialize(t, exampleCooccurrenceMatrix())
}

func TestTrainerSerialize(t *testing.T) {
	c := anyvec32.DefaultCreator{}
	trainer := NewTrainer(c, 15, exampleCooccurrenceMatrix())
	trainer.Weighter = &StandardWeighter{
		Power: 0.78,
		Max:   1337,
	}
	trainer.Rate = 0.9
	trainer.NumUpdates = 666
	testSerialize(t, trainer)
}

func testSerialize(t *testing.T, obj interface{}) {
	var newObj interface{}
	data, err := serializer.SerializeAny(obj)
	if err != nil {
		t.Error(err)
		return
	}
	if err := serializer.DeserializeAny(data, &newObj); err != nil {
		t.Error(err)
		return
	}
	if !reflect.DeepEqual(newObj, obj) {
		t.Errorf("expected %#v but got %#v", obj, newObj)
	}
}

func exampleCooccurrenceMatrix() *SparseMatrix {
	res := NewSparseMatrix(4, 4)
	res.Set(3, 1, 0.5)
	res.Set(1, 3, 0.2)
	res.Set(0, 3, 0.3)
	return res
}
