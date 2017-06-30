package wordembed

import (
	"strings"
	"unicode"
)

// PunctuationMode is a way to deal with punctuation and
// other symbols when tokenizing strings.
type PunctuationMode int

const (
	// Treat each piece of punctuation as its own token.
	SeparatePunctuation PunctuationMode = iota

	// Remove all punctuation.
	DropPunctuation

	// Treat punctuation as just another character.
	IncludePunctuation
)

// A Tokenizer separates strings into word tokens.
//
// By default, a Tokenizer converts all tokens to
// lowercase and treats punctuation as its own token.
type Tokenizer struct {
	// PunctuationMode is used to decide how to treat
	// punctuation.
	PunctuationMode PunctuationMode

	// PreserveCase, if true, indicates that fields should
	// not automatically be converted to lowercase.
	PreserveCase bool
}

// Tokenize produces tokens for the string.
func (t *Tokenizer) Tokenize(s string) []string {
	var res []string
	for _, field := range strings.Fields(s) {
		if !t.PreserveCase {
			field = strings.ToLower(field)
		}
		res = append(res, handlePunctuation(t.PunctuationMode, field)...)
	}
	return res
}

func handlePunctuation(m PunctuationMode, field string) []string {
	switch m {
	case SeparatePunctuation:
		var res []string
		var cur string
		for _, ch := range field {
			if unicode.IsPunct(ch) {
				if cur != "" {
					res = append(res, cur)
				}
				res = append(res, string(ch))
				cur = ""
			} else {
				cur += string(ch)
			}
		}
		if cur != "" {
			res = append(res, cur)
		}
		return res
	case DropPunctuation:
		var res string
		for _, ch := range field {
			if !unicode.IsPunct(ch) {
				res += string(ch)
			}
		}
		return []string{res}
	case IncludePunctuation:
		return []string{field}
	}
	panic("unknown punctuation mode")
}
