MAIN := main

PUML_FILES    := $(wildcard figures/*.puml)
PDF_FILES     := $(PUML_FILES:.puml=.pdf)
SVG_FIGS      := correlation_heatmap feature_difference_anova \
                 final_model_comparison individual_confusion_matrices svm_k_sweep
SVG_PDF_FILES := $(addprefix figures/,$(addsuffix .pdf,$(SVG_FIGS)))

all: $(PDF_FILES) $(SVG_PDF_FILES) $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex $(PDF_FILES) $(SVG_PDF_FILES) gen_feature_csv.py ../../Feature_Comparison.xlsx
	latexmk -pdf -shell-escape -interaction=nonstopmode $(MAIN)

%.pdf: %.puml
	plantuml -tpdf $<

figures/%.pdf: ../../analysis/figures/%.svg
	rsvg-convert -f pdf -o $@ $<

clean:
	latexmk -C
	rm -f *.bbl *.run.xml $(PDF_FILES) $(SVG_PDF_FILES)

fmt:
	tex-fmt *.tex --recursive

.PHONY: all clean fmt
