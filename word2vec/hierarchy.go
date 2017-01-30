package word2vec

import (
	"github.com/unixpickle/anyvec"
	"github.com/unixpickle/splaytree"
)

// Hierarchy is used to encode words for a hierarchical
// softmax layer.
//
// Each word is mapped to a list of nodes indices which
// start at 1.
// If the node index is positive, then the layer should
// opt to go right (positive) at the node.
// If the node index is negative, then the layer should
// opt to go left (negative) at the node.
type Hierarchy map[string][]int

// BuildHierarchy builds a hierarchy using Huffman coding.
// The word list is represented as a map from words to
// their probabilities (frequencies).
func BuildHierarchy(words map[string]float64) Hierarchy {
	node := buildHuffman(words)
	res := Hierarchy{}
	nodeIdx := 1
	node.PutInHierarchy(res, nil, &nodeIdx)
	return res
}

// DesiredOuts takes a bag of words and produces a sparse
// vector of target probabilities for the hierarchical
// softmax.
func (h Hierarchy) DesiredOuts(c anyvec.Creator, words []string) map[int]anyvec.Numeric {
	numVisits := map[int]int{}
	numRights := map[int]int{}
	for _, w := range words {
		for _, node := range h[w] {
			if node < 0 {
				numVisits[-node]++
			} else {
				numVisits[node]++
				numRights[node]++
			}
		}
	}
	res := map[int]anyvec.Numeric{}
	for k, v := range numVisits {
		rightProb := float64(numRights[k]) / float64(v)
		res[k] = c.MakeNumeric(rightProb)
	}
	return res
}

func buildHuffman(words map[string]float64) *huffmanNode {
	tree := &splaytree.Tree{}
	for word, prob := range words {
		tree.Insert(treeValue{Node: &huffmanNode{Word: word}, Prob: prob})
	}
	for {
		min1, ok := popMinProb(tree)
		if !ok {
			panic("no nodes")
		}
		min2, ok := popMinProb(tree)
		if !ok {
			return min1.Node
		}
		node := &huffmanNode{
			Left:  min2.Node,
			Right: min1.Node,
		}
		prob := min1.Prob + min2.Prob
		tree.Insert(treeValue{Node: node, Prob: prob})
	}
}

func popMinProb(t *splaytree.Tree) (treeValue, bool) {
	if t.Root == nil {
		return treeValue{}, false
	}
	var n *splaytree.Node
	n = t.Root
	for n.Left != nil {
		n = n.Left
	}
	t.Delete(n.Value)
	return n.Value.(treeValue), true
}

type treeValue struct {
	Node *huffmanNode
	Prob float64
}

func (t treeValue) Compare(v2 splaytree.Value) int {
	p1 := t.Prob
	p2 := v2.(treeValue).Prob
	if p1 < p2 {
		return -1
	} else if p1 == p2 {
		return 0
	} else {
		return 1
	}
}

type huffmanNode struct {
	Word  string
	Left  *huffmanNode
	Right *huffmanNode
}

func (h *huffmanNode) PutInHierarchy(hier Hierarchy, path []int, nodeIdx *int) {
	if h.Left == nil {
		hier[h.Word] = append([]int{}, path...)
		return
	}

	id := *nodeIdx
	(*nodeIdx)++

	path = append(path, -id)
	h.Left.PutInHierarchy(hier, path, nodeIdx)
	path[len(path)-1] = id
	h.Right.PutInHierarchy(hier, path, nodeIdx)
}
