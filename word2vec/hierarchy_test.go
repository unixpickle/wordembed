package word2vec

import (
	"reflect"
	"testing"
)

func TestHierarchy(t *testing.T) {
	actual := BuildHierarchy(map[string]float64{
		"a": 0.501,
		"b": 0.25,
		"c": 0.125,
		"d": 0.124,
	})
	expected := Hierarchy{
		"a": []int{-1},
		"b": []int{1, -2},
		"c": []int{1, 2, -3},
		"d": []int{1, 2, 3},
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Errorf("expected %v but got %v", expected, actual)
	}
}
