#!/bin/bash
# Build the arXiv source bundle from the IEEE-format version (main_arxiv.tex), with precompiled .bbl.
set -e
cd "$(dirname "$0")/paper"
tectonic -X compile --keep-intermediates main_arxiv.tex > /dev/null 2>&1
tectonic -X compile main.tex > /dev/null 2>&1
tectonic -X compile main_anon.tex > /dev/null 2>&1
rm -rf ../arxiv && mkdir -p ../arxiv/sections
cp main_arxiv.tex ../arxiv/main.tex
cp main_arxiv.bbl ../arxiv/main.bbl
cp refs.bib ../arxiv/
CLS=$(find ~/Library/Caches/Tectonic -name IEEEtran.cls 2>/dev/null | head -1); [ -n "$CLS" ] && cp "$CLS" ../arxiv/
cp sections/*.tex ../arxiv/sections/
for f in $(grep -ho "includegraphics\[[^]]*\]{[^}]*}" main_arxiv.tex sections/*.tex | sed 's/.*{\(.*\)}/\1/' | sort -u); do cp "$f" ../arxiv/; done
cp main_arxiv.pdf ../arxiv_preview.pdf
cd ../arxiv && tar czf ../arxiv_submission.tar.gz . && cd ..
echo "bundle: arxiv_submission.tar.gz ($(du -h arxiv_submission.tar.gz | cut -f1)); files: $(ls arxiv | wc -l | tr -d ' ')"
