MAIN := main

PUML_FILES := $(wildcard figures/*.puml)
PDF_FILES  := $(PUML_FILES:.puml=.pdf)

all: $(PDF_FILES) $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(PDF_FILES) gen_feature_csv.py
	latexmk -pdf -shell-escape -interaction=nonstopmode $(MAIN)

%.pdf: %.puml
	plantuml -tpdf $<

clean:
	latexmk -C
	rm -f *.bbl *.run.xml $(PDF_FILES)

fmt:
	tex-fmt *.tex --recursive

.PHONY: all clean fmt
