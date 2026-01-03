# LaTeX Project Makefile
# Usage: make [target]

.PHONY: all clean compile view help

# Default target
all: compile

# Clean all generated files
clean:
	@echo "Cleaning up LaTeX outputs..."
	rm -f *.aux *.log *.out *.toc *.pdf
	@echo "Cleanup completed!"

# Compile the document
compile:
	@echo "Compiling LaTeX document..."
	pdflatex -interaction=nonstopmode main.tex || true
	pdflatex -interaction=nonstopmode main.tex || true
	@if [ -f main.pdf ]; then \
		echo "Compilation completed! Output: main.pdf"; \
	else \
		echo "Compilation failed. Check main.log for errors."; \
		exit 1; \
	fi

# Quick compile (faster for testing)
quick:
	@echo "Quick compilation..."
	pdflatex main.tex
	@echo "Quick compilation completed!"

# View the PDF
view: compile
	@echo "Opening PDF viewer..."
	@if command -v xdg-open > /dev/null; then \
		xdg-open main.pdf; \
	elif command -v evince > /dev/null; then \
		evince main.pdf; \
	elif command -v okular > /dev/null; then \
		okular main.pdf; \
	else \
		echo "No PDF viewer found."; \
	fi

# Show help
help:
	@echo "Available targets:"
	@echo "  all         - Compile the document (default)"
	@echo "  compile     - Compile with latexmk"
	@echo "  quick       - Quick compile without bibliography"
	@echo "  clean       - Remove all generated files"
	@echo "  view        - Compile and open PDF"
	@echo "  help        - Show this help message"

