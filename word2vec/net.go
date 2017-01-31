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

// Step performs a step of gradient descent for the batch
// of sparse inputs and the desired sparse outputs.
//
// It returns the average cost before the step was taken.
//
// For gradient descent, the provided step size should be
// negative.
func (n *Net) Step(ins, desireds []map[int]anyvec.Numeric, step anyvec.Numeric) anyvec.Numeric {
	if len(ins) == 0 {
		panic("batch size cannot be 0")
	}
	if len(ins) != len(desireds) {
		panic("input and desired lengths must match")
	}

	var totalCost anyvec.Vector
	batchGrad := newGradient()
	for i, in := range ins {
		desired := desireds[i]
		if len(in) == 0 {
			panic("cannot have empty input")
		}
		if len(desired) == 0 {
			panic("cannot have empty desired output")
		}

		hidden, out := n.forward(in, desired)
		actualRes := anydiff.NewVar(out)
		desiredRes := sparseVectorRes(out.Creator(), desired)

		cost := anynet.SigmoidCE{}.Cost(actualRes, desiredRes, 1)
		upstream := out.Creator().MakeVector(1)
		upstream.AddScaler(out.Creator().MakeNumeric(1 / float64(len(ins))))
		grad := anydiff.NewGrad(actualRes)
		cost.Propagate(upstream, grad)

		outGrad := grad[actualRes]

		n.backward(in, hidden, out, outGrad, sortedKeys(desired), batchGrad)

		if totalCost == nil {
			totalCost = cost.Output().Copy()
		} else {
			totalCost.Add(cost.Output())
		}
	}

	batchGrad.Scale(step)
	n.addGradient(batchGrad)

	scaler := totalCost.Creator().MakeNumeric(1 / float64(len(ins)))
	totalCost.Scale(scaler)
	return anyvec.Sum(totalCost)
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
	for i, v := range in {
		slice := n.Encoder.Vector.Slice(i*n.Hidden, (i+1)*n.Hidden)
		slice.Scale(v)
		if hidden == nil {
			hidden = slice
		} else {
			hidden.Add(slice)
		}
	}
	var outNums []anyvec.Vector
	for _, i := range sortedKeys(desired) {
		outRow := n.Decoder.Vector.Slice(i*n.Hidden, (i+1)*n.Hidden)
		dot := outRow.Dot(hidden)
		num := outRow.Creator().MakeVector(1)
		num.AddScaler(dot)
		outNums = append(outNums, num)
	}
	out = hidden.Creator().Concat(outNums...)
	return
}

func (n *Net) backward(in map[int]anyvec.Numeric, hidden, out, outGrad anyvec.Vector,
	outIndices []int, output *gradient) {
	var hiddenGrad anyvec.Vector

	// Propagate through the decoder weights.
	for i, outIndex := range outIndices {
		rowStart := outIndex * n.Hidden
		rowEnd := (outIndex + 1) * n.Hidden
		row := n.Decoder.Vector.Slice(rowStart, rowEnd)
		upstreamComp := outGrad.Slice(i, i+1)

		if hiddenGrad == nil {
			hiddenGrad = row.Copy()
			anyvec.ScaleRepeated(hiddenGrad, upstreamComp)
		} else {
			rc := row.Copy()
			anyvec.ScaleRepeated(rc, upstreamComp)
			hiddenGrad.Add(rc)
		}

		rowGrad := hidden.Copy()
		anyvec.ScaleRepeated(rowGrad, upstreamComp)

		if vec, ok := output.OutGrads[outIndex]; ok {
			vec.Add(rowGrad)
		} else {
			output.OutGrads[outIndex] = rowGrad
		}
	}

	// Propagate through the hidden layer.
	for inIndex, scaler := range in {
		scaledU := hiddenGrad.Copy()
		scaledU.Scale(scaler)
		if vec, ok := output.InGrads[inIndex]; ok {
			vec.Add(scaledU)
		} else {
			output.InGrads[inIndex] = scaledU
		}
	}
}

func (n *Net) addGradient(g *gradient) {
	for outIndex, rowGrad := range g.OutGrads {
		rowStart := outIndex * n.Hidden
		rowEnd := (outIndex + 1) * n.Hidden
		row := n.Decoder.Vector.Slice(rowStart, rowEnd)
		row.Add(rowGrad)
		n.Decoder.Vector.SetSlice(rowStart, row)
	}
	for inIndex, rowGrad := range g.InGrads {
		rowStart := inIndex * n.Hidden
		rowEnd := (inIndex + 1) * n.Hidden
		row := n.Encoder.Vector.Slice(rowStart, rowEnd)
		row.Add(rowGrad)
		n.Encoder.Vector.SetSlice(rowStart, row)
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

type gradient struct {
	InGrads  map[int]anyvec.Vector
	OutGrads map[int]anyvec.Vector
}

func newGradient() *gradient {
	return &gradient{
		InGrads:  map[int]anyvec.Vector{},
		OutGrads: map[int]anyvec.Vector{},
	}
}

func (g *gradient) Scale(s anyvec.Numeric) {
	for _, v := range g.InGrads {
		v.Scale(s)
	}
	for _, v := range g.OutGrads {
		v.Scale(s)
	}
}
