package parser

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func TestParse(t *testing.T) {
	type testCase struct {
		expect []Command
		err    error
	}

	testCases := map[string]*testCase{
		"Modelfile":            {expect: []Command{{Name: "from", Args: "llama2"}}},
		"Modelfile.parameters": {expect: []Command{{Name: "from", Args: "llama2"}, {Name: "temperature", Args: "2"}, {Name: "top_k", Args: "35"}}},
		"Modelfile.stop":       {expect: []Command{{Name: "from", Args: "llama2"}, {Name: "stop", Args: "### User:"}, {Name: "stop", Args: "### Assistant:"}}},
		"Modelfile.template":   {expect: []Command{{Name: "from", Args: "llama2"}, {Name: "template", Args: "[INST] <<SYS>>{{ .System }}<</SYS>>\n\n{{ .Prompt }} [/INST]"}}},
	}

	for k, v := range testCases {
		t.Run(k, func(t *testing.T) {
			testData, err := os.Open(filepath.Join("testdata", k))
			if err != nil {
				t.Fatal(err)
			}

			commands, err := Parse(testData)
			if v.err != err {
				t.Fatalf("expected %q, got %q", v.err, err)
			}

			if len(v.expect) != len(commands) {
				t.Fatalf("expected %d commands, got %d", len(v.expect), len(commands))
			}

			for i := range commands {
				expect := v.expect[i]
				actual := commands[i]
				if expect.Name != actual.Name {
					t.Errorf("expected %q, got %q", expect.Name, actual.Name)
				}

				if expect.Args != actual.Args {
					t.Errorf("expected %q, got %q", expect.Args, actual.Args)
				}
			}
		})
	}
}

func TestFormat(t *testing.T) {
	type testCase struct {
		commands []Command
	}

	testCases := map[string]*testCase{
		"Modelfile":            {commands: []Command{{Name: "from", Args: "llama2"}}},
		"Modelfile.parameters": {commands: []Command{{Name: "from", Args: "llama2"}, {Name: "temperature", Args: "2"}, {Name: "top_k", Args: "35"}}},
		"Modelfile.stop":       {commands: []Command{{Name: "from", Args: "llama2"}, {Name: "stop", Args: "### User:"}, {Name: "stop", Args: "### Assistant:"}}},
		"Modelfile.template":   {commands: []Command{{Name: "from", Args: "llama2"}, {Name: "template", Args: "[INST] <<SYS>>{{ .System }}<</SYS>>\n\n{{ .Prompt }} [/INST]"}}},
	}

	for k, v := range testCases {
		t.Run(k, func(t *testing.T) {
			var b bytes.Buffer
			if err := Format(&b, v.commands); err != nil {
				t.Fatal(err)
			}

			testData, err := os.ReadFile(filepath.Join("testdata", k))
			if err != nil {
				t.Fatal(err)
			}

			if !bytes.Equal(b.Bytes(), testData) {
				t.Fatalf("expected %q, got %q", b.String(), testData)
			}
		})
	}
}
