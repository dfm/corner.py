default: dirhtml

SPHINXOPTS    = -W --keep-going -n
SPHINXBUILD   = sphinx-build
BUILDDIR      = _build
ALLSPHINXOPTS   = -d $(BUILDDIR)/doctrees $(SPHINXOPTS) .

.PHONY: clean
clean:
	rm -rf $(BUILDDIR)/*

.PHONY: dirhtml
dirhtml:
	$(SPHINXBUILD) -b dirhtml $(ALLSPHINXOPTS) $(BUILDDIR)/dirhtml
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/dirhtml."
