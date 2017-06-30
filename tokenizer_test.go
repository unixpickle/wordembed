package wordembed

import "fmt"

func ExampleTokenizer() {
	t := &Tokenizer{}
	fmt.Println(t.Tokenize("Hello, said Alex's friend."))

	// Output: [hello , said alex ' s friend .]
}
