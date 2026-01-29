#import "@preview/charged-ieee:0.1.4": ieee
#set text(lang: "de")
#show: ieee.with(
  title: [MM],
  abstract: [TRAINING],
  authors: (
    (
      name: "Frederik Schwarz",
      department: [Informatik],
      organization: [Hof-University],
    ),
    (
      name: "Eugen",
      department: [Informatik],
      organization: [Hof-University],
    ),
  ),
  bibliography: bibliography("refs.bib"),
)
#set text(lang: "de")
#include "struc.typ"
#include "train.typ"
#include "eval.typ"
