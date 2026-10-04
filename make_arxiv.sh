#!/bin/bash
# Build the arXiv source bundle: tex + sections + figures + style + precompiled .bbl
set -e
cd "$(dirname "$0")/paper"
tectonic -X compile --keep-intermediates main.tex > /dev/null
rm -rf ../arxiv && mkdir -p ../arxiv/sections
cp main.tex main.bbl acl.sty acl_natbib.bst refs.bib ../arxiv/
cp sections/*.tex ../arxiv/sections/
for f in $(grep -ho "includegraphics\[[^]]*\]{[^}]*}" main.tex sections/*.tex | sed 's/.*{\(.*\)}/\1/' | sort -u); do cp "$f" ../arxiv/; done
cp main.pdf ../arxiv_preview.pdf
cd ../arxiv && tar czf ../arxiv_submission.tar.gz . && cd ..
echo "bundle: arxiv_submission.tar.gz ($(du -h arxiv_submission.tar.gz | cut -f1)); files: $(ls arxiv | wc -l)"
grep -L "red" arxiv/sections/*.tex > /dev/null; grep -n "textcolor{red}{\\[" arxiv/sections/*.tex && echo "WARNING: red placeholders remain" || echo "no placeholders"
