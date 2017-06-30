package glove

import (
	"math"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
)

func init() {
	serializer.RegisterTypedDeserializer((&StandardWeighter{}).SerializerType(),
		DeserializeStandardWeighter)
}

// A Weighter assigns weight to co-occurrences.
type Weighter interface {
	serializer.Serializer
	Weight(x float64) float64
}

// A StandardWeighter is a Weighter that uses the the
// piecewise function defined in equation 9 of this
// paper: https://nlp.stanford.edu/pubs/glove.pdf.
type StandardWeighter struct {
	// Power is the "a" parameter in the paper.
	//
	// If 0, 3/4 is used.
	Power float64

	// Max is the x_max parameter in the paper.
	//
	// If 0, 100 is used.
	Max float64
}

// DeserializeStandardWeighter deserializes a
// StandardWeighter.
func DeserializeStandardWeighter(d []byte) (*StandardWeighter, error) {
	var res StandardWeighter
	err := serializer.DeserializeAny(d, &res.Power, &res.Max)
	if err != nil {
		return nil, essentials.AddCtx("deserialize StandardWeighter", err)
	}
	return &res, nil
}

// Weight applies the weighting function.
func (s *StandardWeighter) Weight(x float64) float64 {
	a := s.Power
	if a == 0 {
		a = 3.0 / 4
	}
	max := s.Max
	if max == 0 {
		max = 100
	}
	if x > max {
		return 1
	}
	return math.Pow(x/max, a)
}

// SerializerType returns the unique ID used to serialize
// a StandardWeighter with the serializer package.
func (s *StandardWeighter) SerializerType() string {
	return "github.com/unixpickle/wordembed/glove.StandardWeighter"
}

// Serialize serializes a StandardWeighter.
func (s *StandardWeighter) Serialize() ([]byte, error) {
	return serializer.SerializeAny(s.Power, s.Max)
}
