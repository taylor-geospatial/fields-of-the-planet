.PHONY: install build watch clean

# CTAN packages required by main.tex / wacv.sty that are not vendored in this
# folder and may be absent from a minimal TeX install (e.g. TinyTeX). Style
# files like wacv.sty, algorithm(ic).sty, eso-pic.sty, fancyhdr.sty are checked
# into this dir, so they are not listed here. Idempotent: tlmgr skips installed.
TEXPKGS = silence lineno enumitem makecell siunitx cleveref grfext newtx \
	printlen soul multirow colortbl microtype caption booktabs \
	xcolor amsmath etoolbox hyperref natbib float xkeyval l3packages tools \
	times preprint forloop

install:
	tlmgr install $(TEXPKGS)

build:
	latexmk -pdf -interaction=nonstopmode main.tex

watch:
	latexmk -pdf -pvc -interaction=nonstopmode main.tex

clean:
	latexmk -C
	rm -f *.bbl *.run.xml
