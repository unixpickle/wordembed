package word2vec

import (
	"sort"

	"github.com/unixpickle/anydiff"
	"github.com/unixpickle/anynet"
)

func sortedNodesInPaths(p [][]int) []int {
	seen := map[int]bool{}
	var all []int
	for _, path := range p {
		for _, x := range path {
			idx := pathElementNodeIndex(x)
			if !seen[idx] {
				seen[idx] = true
				all = append(all, idx)
			}
		}
	}
	sort.Ints(all)
	return all
}

func costForPath(all, path []int, actual anydiff.Res) anydiff.Res {
	mask := make([]float64, len(all))
	desired := make([]float64, len(all))
	for _, x := range path {
		nodeIdx := pathElementNodeIndex(x)
		elementIdx := sort.SearchInts(all, nodeIdx)
		mask[elementIdx] = 1
		if x < 0 {
			desired[elementIdx] = 0
		} else {
			desired[elementIdx] = 1
		}
	}
	c := actual.Output().Creator()
	desiredRes := anydiff.NewConst(c.MakeVectorData(c.MakeNumericList(desired)))
	maskRes := anydiff.NewConst(c.MakeVectorData(c.MakeNumericList(desired)))
	cost := anynet.SigmoidCE{}.Cost(desiredRes, actual, len(all))
	return anydiff.Sum(anydiff.Mul(cost, maskRes))
}

func pathElementNodeIndex(pathElement int) int {
	if pathElement > 0 {
		return pathElement - 1
	} else {
		return -pathElement - 1
	}
}
