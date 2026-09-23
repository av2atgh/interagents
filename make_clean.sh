#!/bin/sh
# Build the clean copy: revision markup off, deletions removed.
set -e
sed -e 's|^\\long\\def\\rev#1{{\\color{blue}#1}}$|\\long\\def\\rev#1{#1}|' \
    -e 's|^\\long\\def\\del#1{{\\color{blue}\\sout{#1}}}$|\\long\\def\\del#1{}|' \
    manuscript.tex > manuscript_clean.tex
pdflatex -interaction=nonstopmode manuscript_clean.tex > /dev/null
pdflatex -interaction=nonstopmode manuscript_clean.tex > /dev/null
echo "wrote manuscript_clean.pdf"
