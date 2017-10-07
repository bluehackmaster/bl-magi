#!/usr/bin/env bash

rsync -avzh --rsh='ssh -p5664' bluehack@magi:/dataset/deepfashion .