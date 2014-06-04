#!/usr/bin/env  bash

for f in *.dot
do
  echo $f
  dot -Tpng $f -o ${f}.png
done
