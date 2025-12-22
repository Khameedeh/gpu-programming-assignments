TEX_FILE = main.tex
PDF_FILE = analysis_report.pdf

all: $(PDF_FILE)

$(PDF_FILE): $(TEX_FILE)
	pdflatex -interaction=nonstopmode $(TEX_FILE)
	pdflatex -interaction=nonstopmode $(TEX_FILE)  # Run twice for references

clean:
	rm -f *.aux *.log *.out *.pdf *.fls *.fdb_latexmk *.synctex.gz

.PHONY: all clean
