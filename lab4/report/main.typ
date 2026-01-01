#set page(
  paper: "a4",
  margin: 1cm,
  numbering: "1",
)

#set text(
  font: "Source Han Serif SC",
  size: 12pt,
  lang: "zh",
)

#set par(
  first-line-indent: (amount: 2em, all: true),
  justify: true,
  leading: 1em,
)

#import "@preview/codly:1.3.0": *
#import "@preview/codly-languages:0.1.1": *
#show: codly-init.with()

#set heading(numbering: "1.1")
#show heading.where(level: 1): it => {
  pagebreak(weak: true)
  set text(size: 18pt, weight: "bold")
  v(1em)
  it
  v(0.8em)
}
#show heading.where(level: 2): it => {
  set text(size: 15pt, weight: "bold")
  v(0.7em)
  it
  v(0.5em)
}
#show heading.where(level: 3): it => {
  set text(size: 13pt, weight: "bold")
  v(0.5em)
  it
  v(0.3em)
}

#show math.equation: set text(font: ("New Computer Modern Math", "New Computer Modern"))
#show raw: set text(font: (
  (name: "DejaVu Sans Mono", covers: "latin-in-cjk"),
  "Source Han Serif SC",
))


#show figure.caption: set text(size: 10pt)

// 封面
#align(center)[
  #v(3cm)
  #text(size: 24pt, weight: "bold")[机器学习与数据挖掘]
  #v(1cm)
  #text(size: 20pt)[期末课程报告]
  #v(3cm)
  #text(size: 14pt)[
    数据删除
  ]
  #v(1cm)
]

#pagebreak()

#include "purpose.typ"
#include "principle.typ"
#include "process.typ"
#include "result.typ"