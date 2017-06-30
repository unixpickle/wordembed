package wordembed

import (
	"encoding/json"
	"sort"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	serializer.RegisterTypedDeserializer(TokenSet{}.SerializerType(), DeserializeTokenSet)
}

// A TokenSet is a set of tokens which can translate
// between token IDs and tokens.
//
// A TokenSet is represented as a sorted list of tokens.
// Each token's index corresponds to that token's ID.
//
// The token identified as len(TokenSet) is used as a
// placeholder for tokens not in the set.
type TokenSet []string

// DeserializeTokenSet deserializes a TokenSet.
func DeserializeTokenSet(d []byte) (TokenSet, error) {
	var res TokenSet
	if err := json.Unmarshal(d, &res); err != nil {
		return nil, essentials.AddCtx("deserialize TokenSet", err)
	}
	return res, nil
}

// ID gets an ID for the token.
func (t TokenSet) ID(token string) int {
	idx := sort.SearchStrings(t, token)
	if idx < len(t) && t[idx] != token {
		return len(t)
	}
	return idx
}

// IDs computes the ID for each token.
func (t TokenSet) IDs(tokens []string) []int {
	res := make([]int, len(tokens))
	for i, tok := range tokens {
		res[i] = t.ID(tok)
	}
	return res
}

// Token gets the token for the given ID.
//
// If the token ID corresponds to an absent token, then ""
// is returned.
func (t TokenSet) Token(id int) string {
	if id >= len(t) {
		return ""
	}
	return t[id]
}

// SerializerType returns the unique ID used to serialize
// a TokenSet with the serializer package.
func (t TokenSet) SerializerType() string {
	return "github.com/unixpickle/wordembed.TokenSet"
}

// Serialize serializes the TokenSet.
func (t TokenSet) Serialize() ([]byte, error) {
	return json.Marshal(t)
}
