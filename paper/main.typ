#import "@preview/charged-ieee:0.1.4": ieee
#set text(lang: "de")
#show: ieee.with(
  title: [Multimodal Artificial Intelligence: Post-training Evaluation Results],
  abstract: [Diese Studienarbeit etabliert eine Pre-Training-
  Baseline für das Vision-Language-Modell Qwen3-VL-2B-Instruct
  im Kontext des Brettspiels Gomoku. Evaluiert werden visuelle
  Wahrnehmungsaufgaben und strategische Fragestellungen über
  definierte Fragefoki hinweg. Zur Reduktion formatbedingter
  Fehlklassifikationen erfolgt die Auswertung mittels LLM-as-a-
  Judge (LISA); die Ergebnisse dienen als Referenz für das
  anschließende Fine-Tuning.],
  authors: (
    (
      name: "Frederik Schwarz",
      department: [Informatik (Bsc.)],
      organization: [Hochschule Hof, Fakultät Informatik],
      location: [Matrikelnummer: 00515923],
      email: "frederik.schwarz@hof-university.de"
    ),
    (
      name: "Eugen Ganscha",
      department: [Informatik (Bsc.)],
      organization: [Hochschule Hof, Fakultät Informatik],
      location: [Matrikelnummer: 00514322],
      email: "eganscha@hof-university.de"
    ),
  ),
  bibliography: bibliography("refs.bib"),
)
#set text(lang: "de")
#include "struc.typ"
#include "train.typ"
#include "eval.typ"
