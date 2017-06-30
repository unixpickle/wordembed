package wordembed

import (
	"reflect"
	"sort"
	"testing"
)

func TestMostCommon(t *testing.T) {
	stream := make(chan string, 9)
	for _, tok := range []string{"c", "a", "a", "b", "b", "a", "c", "d", "c"} {
		stream <- tok
	}
	close(stream)
	counts := CountTokens(stream)
	common := counts.MostCommon(2)
	sort.Strings(common)
	if !reflect.DeepEqual(common, []string{"a", "c"}) {
		t.Error("expected [a c] but got", common)
	}
}
