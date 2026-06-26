.PHONY: install build build-main build-supp watch clean

# CTAN packages required by main.tex / wacv.sty that are not vendored in this
# folder and may be absent from a minimal TeX install (e.g. TinyTeX). wacv.sty
# is checked into this dir, so it is not listed here. Idempotent: tlmgr skips
# already-installed packages.
TEXPKGS = silence lineno enumitem makecell siunitx cleveref grfext newtx \
	printlen soul multirow colortbl microtype caption booktabs \
	xcolor amsmath etoolbox hyperref natbib float xkeyval l3packages tools \
	times preprint forloop

# tlmgr needs write access to the TeX Live tree. System-wide installs
# (e.g. /usr/local/texlive) are root-owned, so fall back to sudo when the
# tree is not writable by the current user.
install:
	@if [ -w "$$(kpsewhich -var-value TEXMFROOT)" ]; then \
		tlmgr install $(TEXPKGS); \
	else \
		echo "TeX Live tree not user-writable; retrying with sudo..."; \
		sudo tlmgr install $(TEXPKGS); \
	fi

# main.tex and supplementary.tex cross-reference each other via xr-hyper, so
# each needs the other's up-to-date .aux. Build main -> supp -> main -> supp so
# the mutual \Cref numbers (Table/Appendix references) converge.
build:
	latexmk -pdf -interaction=nonstopmode main.tex
	latexmk -pdf -interaction=nonstopmode supplementary.tex
	latexmk -pdf -interaction=nonstopmode main.tex
	latexmk -pdf -interaction=nonstopmode supplementary.tex

build-main:
	latexmk -pdf -interaction=nonstopmode main.tex

build-supp:
	latexmk -pdf -interaction=nonstopmode supplementary.tex

watch:
	latexmk -pdf -pvc -interaction=nonstopmode main.tex

clean:
	latexmk -C main.tex supplementary.tex
	rm -f *.bbl *.run.xml
