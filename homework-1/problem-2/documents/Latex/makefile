# LaTeX Project Makefile for FedPilot Paper
# Usage: make [target]

.PHONY: all clean compile view install-deps help

# Default target
all: compile

# Clean all generated files
clean:
	@echo "Cleaning up LaTeX outputs..."
	latexmk -C
	@echo "Cleanup completed!"

# Compile the document with bibliography
compile:
	@echo "Compiling LaTeX document with bibliography..."
	latexmk -pdf -bibtex main.tex
	@echo "Compilation completed! Output: main.pdf"

# Quick compile without bibliography (faster for testing)
quick:
	@echo "Quick compilation (without bibliography)..."
	pdflatex main.tex
	@echo "Quick compilation completed!"

# View the PDF (requires a PDF viewer)
view: compile
	@echo "Opening PDF viewer..."
	@if command -v xdg-open > /dev/null; then \
		xdg-open main.pdf; \
	elif command -v evince > /dev/null; then \
		evince main.pdf; \
	elif command -v okular > /dev/null; then \
		okular main.pdf; \
	else \
		echo "No PDF viewer found. Please install evince, okular, or another PDF viewer."; \
	fi

# Install required LaTeX packages on Arch Linux
install-deps:
	@echo "Installing LaTeX packages on Arch Linux..."
	@echo "This requires sudo privileges. Please run:"
	@echo "sudo pacman -S texlive-core texlive-latexextra texlive-bibtexextra texlive-science texlive-publishers"
	@echo ""
	@echo "For a complete LaTeX installation:"
	@echo "sudo pacman -S texlive-most"

# Show help
help:
	@echo "Available targets:"
	@echo "  all         - Compile the document (default)"
	@echo "  compile     - Compile with bibliography"
	@echo "  quick       - Quick compile without bibliography"
	@echo "  clean       - Remove all generated files"
	@echo "  view        - Compile and open PDF"
	@echo "  install-deps- Show installation instructions"
	@echo "  help        - Show this help message"
