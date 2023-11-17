# dataquality
This repository contains the accompanying code for "Practical Data Quality for Modern Data and Modern Uses, With Applications to America's COVID-19 Data" by Kerstin Frailey, a dissertation in the Field of Statistics at Cornell University, in partial fulfilment towards her doctorate.

## Appendix

The Appendix directory organizes the relevant files in accordance with the appendix in the dissertation.

## Examples

The Examples directory contains use cases that demonstrate use of the code base, including the directory structure that is assumed for local imports.

### Git Commit History

This assumes the commit history of the [JHU CSSE COVID-19 Repo](https://github.com/CSSEGISandData/COVID-19) was gathered via
```
git log master.. --oneline --all --graph --decorate --date=iso-local --pretty=format:'commit::%h---time::%cd'
    $(git reflog | awk '{print $1}') > ../Acquire/JHUCommitHistory.txt
```
