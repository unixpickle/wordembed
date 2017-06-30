package glove

import (
	"math"
	"sync"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/anyvec/anyvecsave"
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/serializer"
	"github.com/unixpickle/wordembed"
)

func init() {
	serializer.RegisterTypedDeserializer((&Trainer{}).SerializerType(), DeserializeTrainer)
}

// Default learning rate from the GloVe paper.
const DefaultRate = 0.05

// A Trainer trains a GloVe model using the AdaGrad
// variant of stochastic gradient descent.
//
// A Trainer can be serialized and deserialized to pause
// and resume training.
type Trainer struct {
	// Cooccur is the co-occurrence matrix.
	Cooccur *SparseMatrix

	// Weighter is used to weight co-occurrences.
	Weighter Weighter

	// Rate is the learing rate for training.
	//
	// Note that, even with a constant learning rate,
	// AdaGrad will reduce the effective learning rate
	// automatically.
	Rate float64

	// Matrices for word and context word vectors, where
	// each row corresponds to a vector.
	Vectors    *anyvec.Matrix
	CtxVectors *anyvec.Matrix

	// Biases for words and context words.
	Biases    anyvec.Vector
	CtxBiases anyvec.Vector

	// AdaGrad traces for all of the parameters.
	// These are changed automatically by Update.
	AdaVectors    *anyvec.Matrix
	AdaCtxVectors *anyvec.Matrix
	AdaBiases     anyvec.Vector
	AdaCtxBiases  anyvec.Vector

	// NumUpdates counts the total number of updates.
	// It is changed automatically by Update.
	NumUpdates int
}

// DeserializeTrainer deserializes a Trainer.
func DeserializeTrainer(d []byte) (*Trainer, error) {
	var res Trainer
	var rows, cols int
	var vectors, ctxVectors, adaVectors, adaCtxVectors *anyvecsave.S
	var biases, ctxBiases, adaBiases, adaCtxBiases *anyvecsave.S
	err := serializer.DeserializeAny(
		d,
		&res.Cooccur,
		&res.Weighter,
		&res.Rate,
		&rows,
		&cols,
		&vectors,
		&ctxVectors,
		&adaVectors,
		&adaCtxVectors,
		&biases,
		&ctxBiases,
		&adaBiases,
		&adaCtxBiases,
		&res.NumUpdates,
	)
	if err != nil {
		return nil, essentials.AddCtx("deserialize Trainer", err)
	}
	matrices := []**anyvec.Matrix{&res.Vectors, &res.CtxVectors, &res.AdaVectors,
		&res.AdaCtxVectors}
	savedMats := []*anyvecsave.S{vectors, ctxVectors, adaVectors, adaCtxVectors}
	for i, saved := range savedMats {
		*matrices[i] = &anyvec.Matrix{
			Data: saved.Vector,
			Rows: rows,
			Cols: cols,
		}
	}
	res.Biases = biases.Vector
	res.CtxBiases = ctxBiases.Vector
	res.AdaBiases = adaBiases.Vector
	res.AdaCtxBiases = adaCtxBiases.Vector
	return &res, nil
}

// NewTrainer creates a new Trainer with randomized
// initial parameters.
//
// The resulting Trainer will use a StandardWeighter and a
// learning rate of DefaultRate.
func NewTrainer(c anyvec.Creator, vecSize int, cooccur *SparseMatrix) *Trainer {
	res := &Trainer{
		Cooccur:  cooccur,
		Weighter: &StandardWeighter{},
		Rate:     DefaultRate,
	}
	n := len(cooccur.Rows)
	matrices := []**anyvec.Matrix{&res.Vectors, &res.CtxVectors, &res.AdaVectors,
		&res.AdaCtxVectors}
	for i, mat := range matrices {
		*mat = &anyvec.Matrix{
			Data: c.MakeVector(n * vecSize),
			Rows: n,
			Cols: vecSize,
		}
		if i < 2 {
			anyvec.Rand((*mat).Data, anyvec.Normal, nil)
		}
	}
	biases := []*anyvec.Vector{&res.Biases, &res.CtxBiases, &res.AdaBiases,
		&res.AdaCtxBiases}
	for _, vec := range biases {
		*vec = c.MakeVector(n)
	}
	return res
}

// Update applies n updates to the model in parallel.
// It returns the average cost for the mini-batch.
func (t *Trainer) Update(n int) anyvec.Numeric {
	var wg sync.WaitGroup
	results := make(chan *trainerResult, n)

	for i := 0; i < n; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()

			ctxID, wordID := t.Cooccur.RandomEntry()

			cooccur := t.Cooccur.Get(ctxID, wordID)
			weighting := t.Weighter.Weight(float64(cooccur))

			wordVec := anydiff.NewVar(extractRow(t.Vectors, wordID))
			ctxVec := anydiff.NewVar(extractRow(t.CtxVectors, ctxID))
			wordBias := anydiff.NewVar(t.Biases.Slice(wordID, wordID+1))
			ctxBias := anydiff.NewVar(t.CtxBiases.Slice(ctxID, ctxID+1))

			creator := wordVec.Vector.Creator()

			cost := anydiff.Scale(
				anydiff.Square(
					anydiff.Add(
						anydiff.Add(
							anydiff.Dot(wordVec, ctxVec),
							wordBias,
						),
						anydiff.AddScalar(
							ctxBias,
							creator.MakeNumeric(-math.Log(float64(cooccur))),
						),
					),
				),
				creator.MakeNumeric(weighting),
			)

			gradScale := creator.MakeVectorData(
				creator.MakeNumericList([]float64{t.Rate}),
			)
			grad := anydiff.NewGrad(cost.Vars().Slice()...)
			cost.Propagate(gradScale, grad)

			results <- &trainerResult{
				WordID:      wordID,
				CtxID:       ctxID,
				Cost:        cost.Output().Copy(),
				Grad:        grad[wordVec].Copy(),
				CtxGrad:     grad[ctxVec].Copy(),
				BiasGrad:    grad[wordBias].Copy(),
				CtxBiasGrad: grad[ctxBias].Copy(),
			}
		}()
	}

	wg.Wait()
	close(results)

	var totalCost anyvec.Vector
	for result := range results {
		t.applyUpdate(result)
		if totalCost == nil {
			totalCost = result.Cost
		} else {
			totalCost.Add(result.Cost)
		}
	}
	t.NumUpdates += n

	totalCost.Scale(totalCost.Creator().MakeNumeric(1 / float64(n)))
	return anyvec.Sum(totalCost)
}

// Embedding creates an embedding from the parameters.
//
// If avg is true, then the word vectors and context
// vectors are averaged to create the embedding.
//
// The parameters are copied, so t may be modified after
// the embeddings are created.
func (t *Trainer) Embedding(tokens wordembed.TokenSet, avg bool) *Embedding {
	data := t.Vectors.Data.Copy()
	if avg {
		data.Add(t.CtxVectors.Data)
		data.Scale(data.Creator().MakeNumeric(1.0 / 2))
	}
	return &Embedding{
		Tokens: tokens,
		Vectors: &anyvec.Matrix{
			Data: data,
			Rows: t.Vectors.Rows,
			Cols: t.Vectors.Cols,
		},
	}
}

// SerializerType returns the unique ID used to serialize
// a Trainer with the serializer package.
func (t *Trainer) SerializerType() string {
	return "github.com/unixpickle/wordembed/glove.Trainer"
}

// Serialize serializes the Trainer.
func (t *Trainer) Serialize() ([]byte, error) {
	return serializer.SerializeAny(
		t.Cooccur,
		t.Weighter,
		t.Rate,
		t.Vectors.Rows,
		t.Vectors.Cols,
		&anyvecsave.S{Vector: t.Vectors.Data},
		&anyvecsave.S{Vector: t.CtxVectors.Data},
		&anyvecsave.S{Vector: t.AdaVectors.Data},
		&anyvecsave.S{Vector: t.AdaCtxVectors.Data},
		&anyvecsave.S{Vector: t.Biases},
		&anyvecsave.S{Vector: t.CtxBiases},
		&anyvecsave.S{Vector: t.AdaBiases},
		&anyvecsave.S{Vector: t.AdaCtxBiases},
		t.NumUpdates,
	)
}

func (t *Trainer) applyUpdate(r *trainerResult) {
	adaVecs := []anyvec.Vector{
		extractRow(t.AdaVectors, r.WordID),
		extractRow(t.AdaCtxVectors, r.CtxID),
		t.AdaBiases.Slice(r.WordID, r.WordID+1),
		t.AdaCtxBiases.Slice(r.CtxID, r.CtxID+1),
	}
	targetVecs := []anyvec.Vector{
		extractRow(t.Vectors, r.WordID),
		extractRow(t.CtxVectors, r.CtxID),
		t.Biases.Slice(r.WordID, r.WordID+1),
		t.CtxBiases.Slice(r.CtxID, r.CtxID+1),
	}
	gradVecs := []anyvec.Vector{r.Grad, r.CtxGrad, r.BiasGrad, r.CtxBiasGrad}
	for i, grad := range gradVecs {
		ada := adaVecs[i]
		sqGrad := grad.Copy()
		sqGrad.Mul(sqGrad)
		ada.Add(sqGrad)

		adaScale := ada.Copy()
		anyvec.Pow(adaScale, ada.Creator().MakeNumeric(-1.0/2))
		grad.Mul(adaScale)

		target := targetVecs[i]
		target.Sub(grad)
	}
}

type trainerResult struct {
	WordID int
	CtxID  int

	Cost anyvec.Vector

	Grad        anyvec.Vector
	CtxGrad     anyvec.Vector
	BiasGrad    anyvec.Vector
	CtxBiasGrad anyvec.Vector
}

func extractRow(mat *anyvec.Matrix, row int) anyvec.Vector {
	idx := mat.Cols * row
	return mat.Data.Slice(idx, idx+mat.Cols)
}
