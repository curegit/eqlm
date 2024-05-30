# Eqlm

Simple CLI tool to spatially equalize image luminance

## Example

## Install

```sh
pip3 install eqlm
```

## Usage

The main program can be invoked either through the `eqlm` command or through the Python main module option `python3 -m eqlm`.

```txt
usage: eqlm [-h] [-v] [-m {luminance,brightness,saturation,lightness}]
            [-n M N] [-t RATE] [-e] [-u] [-g [GAMMA]] [-d {8,16}]
            IN_FILE [OUT_FILE]

Simple CLI tool to spatially equalize image luminance

positional arguments:
  IN_FILE               input image file path (use '-' for stdin)
  OUT_FILE              output PNG image file path (use '-' for stdout)
                        (default: Auto)

options:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -m {luminance,brightness,saturation,lightness}, --mode {luminance,brightness,saturation,lightness}
                        processing mode (default: luminance)
  -n M N, --divide M N  divide image into MxN blocks for aggregation
                        (note that it doesn't respect Exif orientation) (default: (2, 2))
  -t RATE, --target RATE
                        output level target rate, 0.0 (min) to 1.0 (max) (default: Average)
  -e, --median          aggregate each block using median (default: False)
  -u, --unweighted      disable alpha channel weighting (default: False)
  -g [GAMMA], --gamma [GAMMA]
                        apply inverse gamma correction before process [GAMMA=2.2] (default: None)
  -d {8,16}, --depth {8,16}
                        bit depth of the output PNG image (default: 8)
```

## License

GNU Affero General Public License v3.0
