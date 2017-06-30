package wordembed

import (
	"reflect"
	"sort"
	"testing"
)

func TestMostCommon(t *testing.T) {
	counts := TokenCounts{}
	for _, tok := range []string{"c", "a", "a", "b", "b", "a", "c", "d", "c"} {
		counts.Add(tok)
	}
	common := counts.MostCommon(2)
	sort.Strings(common)
	if !reflect.DeepEqual(common, []string{"a", "c"}) {
		t.Error("expected [a c] but got", common)
	}
}
