Latex style for dissertations at FEUP
======================================

a) Use "pdflatex" and not "latex".

b) The main file is "thesis.tex" ("tese.tex"). Read the instructions therein.

c) The style is defined in "feupteses.sty".

d) To create the example edit the Makefile and run

  make

or else:

  pdflatex thesis
  bibtex thesis
  pdflatex thesis
  pdflatex thesis

   The file "thesis.pdf" ("tese.pdf") contains the result.

e) The file "plainnat-pt.bst" is required to process references in the format (author, date) in Portuguese. In English, use "plainnat.bst".

The file "alpha-pt.bst" is needed to employ references alphabetically.

f) The character set used in the editing tool must be reported through the appropriate option:

  \usepackage[latin1]{inputenc}

OR

  \usepackage[utf8]{inputenc}

For MAC (native encoding, not UTF-8)

  \usepackage[applemac]{inputenc}

g) In the MiKTeX distribution, to have correct hyphenation, you must select the Portuguese language in "Start-> Programs-> MiKTeX 2-> MiKTeX Option".

h) Figures:

h1) pdf(la)tex supports png, jpeg, tiff and pdf formats.

Files produced by Metapost can also be used (which are written in a simplified version of Postscript).

To convert pictures from eps (Encapsulated Postscript) to pdf you can use the epstopdf program.

To remove space around pdf figures, you can use pdfcrop.

h2) To use the same figures (in eps) with pdflatex and latex as well as use pictures created with the "pstricks" package, use the package "pst-pdf.sty" (Justified in very special cases!)

i) The automation of calls to latex, bibtex, etc. can be done through the latexmk helper, available in MikTeX and TeXlive. (http://texcatalogue.sarovar.org/entries/latexmk.html)


JCL & JCF, 2011-07-31
