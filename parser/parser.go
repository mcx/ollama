package parser

import (
	"bufio"
	"bytes"
	"errors"
	"fmt"
	"io"
	"log"
	"strings"
)

const (
	CommandFrom      = "from"
	CommandLicense   = "license"
	CommandTemplate  = "template"
	CommandSystem    = "system"
	CommandPrompt    = "prompt"
	CommandAdapter   = "adapter"
	CommandParameter = "parameter"
)

type Command struct {
	Name string `json:"name"`
	Args string `json:"args"`
}

func (c *Command) Reset() {
	c.Name = ""
	c.Args = ""
}

func Parse(reader io.Reader) ([]Command, error) {
	var commands []Command
	var command, modelCommand Command

	scanner := bufio.NewScanner(reader)
	scanner.Buffer(make([]byte, 0, bufio.MaxScanTokenSize), bufio.MaxScanTokenSize)
	scanner.Split(scanModelfile)
	for scanner.Scan() {
		line := scanner.Bytes()

		fields := bytes.SplitN(line, []byte(" "), 2)
		if len(fields) == 0 || len(fields[0]) == 0 {
			continue
		}

		switch name := bytes.ToLower(fields[0]); string(name) {
		case CommandFrom:
			command.Name = string(name)
			command.Args = string(fields[1])
			// copy command for validation
			modelCommand = command
		case CommandLicense, CommandTemplate, CommandSystem, CommandPrompt, CommandAdapter:
			command.Name = string(name)
			command.Args = string(fields[1])
		case CommandParameter:
			fields = bytes.SplitN(fields[1], []byte(" "), 2)
			if len(fields) < 2 {
				return nil, fmt.Errorf("missing value for %s", fields)
			}

			command.Name = string(fields[0])
			command.Args = string(fields[1])
		case "EMBED":
			return nil, fmt.Errorf("deprecated command: EMBED is no longer supported, use the /embed API endpoint instead")
		default:
			if !bytes.HasPrefix(fields[0], []byte("#")) {
				// log a warning for unknown commands
				log.Printf("WARNING: Unknown command: %s", fields[0])
			}
			continue
		}

		commands = append(commands, command)
		command.Reset()
	}

	if modelCommand.Args == "" {
		return nil, errors.New("no FROM line for the model was specified")
	}

	return commands, scanner.Err()
}

func Format(w io.Writer, commands []Command) error {
	for _, command := range commands {
		multiline := ""
		if strings.Contains(command.Args, "\n") {
			multiline = `"""`
		}

		switch command.Name {
		case CommandFrom, CommandAdapter, CommandLicense, CommandTemplate, CommandSystem:
			fmt.Fprintf(w, "%s %s%s%s\n", strings.ToUpper(command.Name), multiline, command.Args, multiline)
		default:
			fmt.Fprintf(w, "PARAMETER %s %s%s%s\n", command.Name, multiline, command.Args, multiline)
		}
	}

	return nil
}

func scanModelfile(data []byte, atEOF bool) (advance int, token []byte, err error) {
	advance, token, err = scan([]byte(`"""`), []byte(`"""`), data, atEOF)
	if err != nil {
		return 0, nil, err
	}

	if advance > 0 && token != nil {
		return advance, token, nil
	}

	advance, token, err = scan([]byte(`"`), []byte(`"`), data, atEOF)
	if err != nil {
		return 0, nil, err
	}

	if advance > 0 && token != nil {
		return advance, token, nil
	}

	return bufio.ScanLines(data, atEOF)
}

func scan(openBytes, closeBytes, data []byte, atEOF bool) (advance int, token []byte, err error) {
	newline := bytes.IndexByte(data, '\n')

	if start := bytes.Index(data, openBytes); start >= 0 && start < newline {
		end := bytes.Index(data[start+len(openBytes):], closeBytes)
		if end < 0 {
			if atEOF {
				return 0, nil, fmt.Errorf("unterminated %s: expecting %s", openBytes, closeBytes)
			} else {
				return 0, nil, nil
			}
		}

		n := start + len(openBytes) + end + len(closeBytes)

		newData := data[:start]
		newData = append(newData, data[start+len(openBytes):n-len(closeBytes)]...)
		return n, newData, nil
	}

	return 0, nil, nil
}
