# ay bada bing bada boom gwen.lily@proton.me

# ltbl
let there be light, my personal philips hue light controller and palette manager.

# command line options

```
optional arguments:
  -h, --help            show this help message and exit
  -name N, -n N         Provide a name for the palette to be saved, the default is the utc timestamp at program start.
  -input IN, -i IN      Provide a file or directory containing files, the default is "input"
  -colors C, -c C       Choose the number of colors to pick per image, the default is 10.
  -radius R, -r R       Choose the gaussian blur radius applied to the image before clustering, the default is 4
  -output OUT, -o OUT   Provide the output directory, default is "output"
  --save, --s           Enable to save palette information in the specified output directory
  --filter-grey, --fg   Enable to filter out sufficiently grey colors
  --display, --d        Enable to display palette images during operation
  -transition T, -t T   Specify transition speed: fast / slow
  -brightness B, -b B   Specify brightness value: bright / mid / dim
  -time-limit TL, -tl TL
                        Specify the time limit in seconds
```

## example usage
`-name foo -i example-input -o example-output --save --display --filter-grey`
