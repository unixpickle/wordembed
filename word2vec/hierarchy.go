package word2vec

import (
	"encoding/json"
	"errors"

	"github.com/unixpickle/serializer"
	"github.com/unixpickle/splaytree"
)

func init() {
	var h Hierarchy
	serializer.RegisterTypedDeserializer(h.SerializerType(), DeserializeHierarchy)
}

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

// DeserializeHierarchy deserializes a Hierarchy.
func DeserializeHierarchy(d []byte) (Hierarchy, error) {
	var res Hierarchy
	if err := json.Unmarshal(d, &res); err != nil {
		return nil, errors.New("deserialize hierarchy: " + err.Error())
	}
	return res, nil
}

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

// Paths gets the path for each word in a bag and collects
// the paths in a slice.
func (h Hierarchy) Paths(words []string) [][]int {
	var res [][]int
	for _, w := range words {
		res = append(res, h[w])
	}
	return res
}

// NumNodes computes the total number of node IDs.
func (h Hierarchy) NumNodes() int {
	var max int
	for _, x := range h {
		for _, k := range x {
			if k < 0 {
				k *= -1
			}
			if k > max {
				max = k
			}
		}
	}
	return max
}

// SerializerType returns the unique ID used to serialize
// a Hierarchy.
func (h Hierarchy) SerializerType() string {
	return "github.com/unixpickle/wordembed/word2vec.Hierarchy"
}

// Serialize serializes the Hierarchy.
func (h Hierarchy) Serialize() ([]byte, error) {
	return json.Marshal(h)
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
	if t == v2 {
		return 0
	}

	p1 := t.Prob
	p2 := v2.(treeValue).Prob
	if p1 < p2 {
		return -1
	} else if p1 > p2 {
		return 1
	}

	// Tie breaker
	leftNode := t.Node
	for leftNode.Left != nil {
		leftNode = leftNode.Left
	}
	leftNode1 := v2.(treeValue).Node
	for leftNode1.Left != nil {
		leftNode1 = leftNode1.Left
	}
	if leftNode.Word < leftNode1.Word {
		return -1
	} else if leftNode.Word > leftNode1.Word {
		return 1
	} else {
		panic("should not equal")
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
