#!/usr/bin/env  bash

for f in data/*.dot
do
  echo $f
  dot -Tpng $f -o ${f}.png
done
