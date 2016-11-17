.SUFFIXES:	# Delete the default suffixes
.PHONY:		pdf clean pdfnup

LATEX_FILES	= $(wildcard *.tex)
PDF_FILES	= $(LATEX_FILES:%.tex=%.pdf)
AUX_FILES	= $(LATEX_FILES:%.tex=%.aux)
PDFNUP_FILES	= $(LATEX_FILES:%.tex=%-nup.pdf)
PNG_FILES	= $(wildcard *.png)
BIB_FILES	= $(wildcard *.bib)
BBL_FILES	= $(LATEX_FILES:%.tex=%.bbl)

FORMAT_FILES	= 

default: pdf

all: pdf

# Remember to set the "svn:keywords Id" property on the key files:
# svn propset svn:keywords Id *.tex Makefile

pdf: $(PDF_FILES)

 $(BBL_FILES): %.bbl: %.aux $(BIB_FILES) 
	biber $*

$(AUX_FILES): %.aux: %.tex Makefile $(FORMAT_FILES)
	-pdflatex $*

$(PDF_FILES): %.pdf: %.tex $(BBL_FILES) $(PNG_FILES) Makefile $(FORMAT_FILES)
	-pdflatex $*
	@while grep -i 'Rerun' $*.log; \
		do pdflatex $*; done

pdfnup: $(PDFNUP_FILES)

$(PDFNUP_FILES): %-nup.pdf: %.pdf
	pdfnup $*.pdf

clean:
	-rm *.log *.dvi *.aux *.synctex.gz  $(PDFNUP_FILES)
	-rm *.blg *.bcf *.blg *.run.xml *.rel $(BBL_FILES)
	-rm *~ \#*#
	-chmod a-x *
