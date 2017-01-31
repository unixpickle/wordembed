// Package word2vec implements word2vec word embeddings.
package word2vec

import (
	"errors"
	"math"
	"sort"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/serializer"
)

func init() {
	var n Net
	serializer.RegisterTypedDeserializer(n.SerializerType(), DeserializeNet)
}

// A Net is the encoder/decoder network used to train a
// word2vec model.
type Net struct {
	In     int
	Hidden int
	Out    int

	// Encoder is the column-major input matrix.
	Encoder *anydiff.Var

	// Decoder is the row-major output matrix.
	Decoder *anydiff.Var
}

// DeserializeNet deserializes a Net.
func DeserializeNet(d []byte) (*Net, error) {
	var in, hidden, out serializer.Int
	var encoder, decoder *anyvecsave.S
	err := serializer.DeserializeAny(d, &in, &hidden, &out, &encoder, &decoder)
	if err != nil {
		return nil, errors.New("deserialize net: " + err.Error())
	}
	return &Net{
		In:      int(in),
		Hidden:  int(hidden),
		Out:     int(out),
		Encoder: anydiff.NewVar(encoder.Vector),
		Decoder: anydiff.NewVar(decoder.Vector),
	}, nil
}

// NewNet creates a new, randomized network with the given
// dimensions.
func NewNet(c anyvec.Creator, in, hidden, out int) *Net {
	res := &Net{
		In:      in,
		Hidden:  hidden,
		Out:     out,
		Encoder: anydiff.NewVar(c.MakeVector(in * hidden)),
		Decoder: anydiff.NewVar(c.MakeVector(hidden * out)),
	}
	anyvec.Rand(res.Encoder.Vector, anyvec.Normal, nil)
	anyvec.Rand(res.Decoder.Vector, anyvec.Normal, nil)
	scaler := c.MakeNumeric(math.Sqrt(1 / float64(hidden)))
	res.Decoder.Vector.Scale(scaler)
	return res
}

// Step performs a step of gradient descent for the sparse
// input and desired output.
//
// It returns the cost before the step was taken.
//
// For gradient descent, the provided step size should be
// negative.
func (n *Net) Step(in, desired map[int]anyvec.Numeric, step anyvec.Numeric) anyvec.Numeric {
	if len(in) == 0 {
		panic("cannot have empty input")
	}
	if len(desired) == 0 {
		panic("cannot have empty desired output")
	}

	hidden, out := n.forward(in, desired)
	actualRes := anydiff.NewVar(out)
	desiredRes := sparseVectorRes(out.Creator(), desired)

	cost := anynet.SigmoidCE{}.Cost(desiredRes, actualRes, 1)
	upstream := out.Creator().MakeVector(1)
	upstream.AddScaler(out.Creator().MakeNumeric(1))
	grad := anydiff.NewGrad(actualRes)
	cost.Propagate(upstream, grad)

	outGrad := grad[actualRes]

	n.backward(in, hidden, out, outGrad, sortedKeys(desired), step)

	return anyvec.Sum(cost.Output())
}

// SerializerType returns the unique ID used to serialize
// a Net with the serializer package.
func (n *Net) SerializerType() string {
	return "github.com/unixpickle/wordembed/word2vec.Net"
}

// Serialize serializes the Net.
func (n *Net) Serialize() ([]byte, error) {
	return serializer.SerializeAny(
		serializer.Int(n.In),
		serializer.Int(n.Hidden),
		serializer.Int(n.Out),
		&anyvecsave.S{Vector: n.Encoder.Vector},
		&anyvecsave.S{Vector: n.Decoder.Vector},
	)
}

func (n *Net) forward(in, desired map[int]anyvec.Numeric) (hidden, out anyvec.Vector) {
	temp := n.Encoder.Vector.Creator().MakeVector(n.Hidden)
	for i, v := range in {
		temp.SetSlice(-i*n.Hidden, n.Encoder.Vector)
		temp.Scale(v)
		if hidden == nil {
			hidden = temp.Copy()
		} else {
			hidden.Add(temp)
		}
	}
	var outNums []anyvec.Vector
	for _, i := range sortedKeys(desired) {
		temp.SetSlice(-i*n.Hidden, n.Decoder.Vector)
		dot := temp.Dot(hidden)
		num := temp.Creator().MakeVector(1)
		num.AddScaler(dot)
		outNums = append(outNums, num)
	}
	out = hidden.Creator().Concat(outNums...)
	return
}

func (n *Net) backward(in map[int]anyvec.Numeric, hidden, out, outGrad anyvec.Vector,
	outIndices []int, stepSize anyvec.Numeric) {
	var hiddenGrad anyvec.Vector

	tempGrad := out.Creator().MakeVector(n.Hidden)
	tempOldRow := out.Creator().MakeVector(n.Hidden)

	// Propagate through the decoder weights.
	for i, outIndex := range outIndices {
		rowStart := outIndex * n.Hidden
		tempOldRow.SetSlice(-rowStart, n.Decoder.Vector)
		upstreamScaler := anyvec.Sum(outGrad.Slice(i, i+1))

		if hiddenGrad == nil {
			hiddenGrad = tempOldRow.Copy()
			hiddenGrad.Scale(upstreamScaler)
		} else {
			tempGrad.Set(tempOldRow)
			tempGrad.Scale(upstreamScaler)
			hiddenGrad.Add(tempGrad)
		}

		tempGrad.Set(hidden)
		tempGrad.Scale(upstreamScaler)
		tempGrad.Scale(stepSize)

		tempOldRow.Add(tempGrad)
		n.Decoder.Vector.SetSlice(rowStart, tempOldRow)
	}

	// Propagate through the hidden layer.
	for inIndex, scaler := range in {
		rowStart := inIndex * n.Hidden
		tempOldRow.SetSlice(-rowStart, n.Encoder.Vector)

		tempGrad.Set(hiddenGrad)
		tempGrad.Scale(scaler)
		tempGrad.Scale(stepSize)

		tempOldRow.Add(tempGrad)
		n.Encoder.Vector.SetSlice(rowStart, tempOldRow)
	}
}

func sparseVectorRes(c anyvec.Creator, m map[int]anyvec.Numeric) anydiff.Res {
	var resValues []anyvec.Vector
	for _, idx := range sortedKeys(m) {
		vec := c.MakeVector(1)
		vec.AddScaler(m[idx])
		resValues = append(resValues, vec)
	}
	return anydiff.NewConst(c.Concat(resValues...))
}

func sortedKeys(m map[int]anyvec.Numeric) []int {
	var res []int
	for x := range m {
		res = append(res, x)
	}
	sort.Ints(res)
	return res
}
