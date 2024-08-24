# Eqlm

Simple CLI tool to spatially equalize image luminance

## Examples

`eqlm images/macaron.jpg -n 3 2 -t 0.8 -c`

| Input                              | Result                                   |
| ---------------------------------- | ---------------------------------------- |
| ![Input Image](images/macaron.jpg) | ![Output Image](images/macaron-eqlm.png) |

## Install

```sh
pip3 install eqlm
```

## Usage

The main program can be invoked either through the `eqlm` command or through the Python main module option `python3 -m eqlm`.

```txt
usage: eqlm [-h] [-v] [-m {luminance,brightness,saturation,lightness}]
            [-n M N] [-t RATE] [-c] [-e] [-u] [-g [GAMMA]] [-d {8,16}] [-x]
            IN_FILE [OUT_FILE]
```

### Options

```txt
positional arguments:
  IN_FILE               input image file path (use '-' for stdin)
  OUT_FILE              output PNG image file path (use '-' for stdout)
                        (default: Auto)

options:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -m {luminance,brightness,saturation,lightness}, --mode {luminance,brightness,saturation,lightness}
                        processing mode (default: luminance)
  -n M N, --divide M N  divide image into MxN (Horizontal x Vertical) blocks
                        for aggregation (default: (2, 2))
  -t RATE, --target RATE
                        set the target rate for the output level, ranging from
                        0.0 (minimum) to 1.0 (maximum) (default: Average)
  -c, --clamp           clamp the level values in extrapolated boundaries
                        (default: False)
  -e, --median          aggregate each block using median instead of mean
                        (default: False)
  -u, --unweighted      disable weighting based on the alpha channel
                        (default: False)
  -g [GAMMA], --gamma [GAMMA]
                        apply inverse gamma correction before process
                        [GAMMA=2.2] (default: None)
  -d {8,16}, --depth {8,16}
                        bit depth of the output PNG image (default: 8)
  -x, --no-orientation  ignore the Exif orientation metadata (default: False)
```

## License

GNU Affero General Public License v3.0
