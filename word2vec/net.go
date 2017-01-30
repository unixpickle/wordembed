// Package word2vec implements word2vec word embeddings.
package word2vec

import (
	"math"
	"sort"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
	"github.com/unixpickle/anyvec"
)

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
// input and the desired sparse output.
//
// It returns the cost before the step was taken.
//
// For gradient descent, the provided step size should be
// negative.
func (n *Net) Step(in, desired map[int]anyvec.Numeric, stepSize anyvec.Numeric) anyvec.Numeric {
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
	upstream.AddScaler(out.Creator().MakeNumeric(1))
	grad := anydiff.NewGrad(actualRes)
	cost.Propagate(upstream, grad)

	outGrad := grad[actualRes]

	n.backward(in, hidden, out, outGrad, sortedKeys(desired), stepSize)
	return anyvec.Sum(cost.Output())
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
	outIndices []int, stepSize anyvec.Numeric) {
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
		rowGrad.Scale(stepSize)
		row.Add(rowGrad)
		n.Decoder.Vector.SetSlice(rowStart, row)
	}

	// Propagate through the hidden layer.
	for inIndex, scaler := range in {
		rowStart := inIndex * n.Hidden
		rowEnd := (inIndex + 1) * n.Hidden
		row := n.Encoder.Vector.Slice(rowStart, rowEnd)

		scaledU := hiddenGrad.Copy()
		scaledU.Scale(scaler)
		scaledU.Scale(stepSize)
		row.Add(scaledU)
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
