# Eqlm

A CLI tool to manipulate images in various ways

## Installation

```sh
pip install eqlm
```

### With Clipboard Support

```sh
pip install eqlm[clipboard]
```

Note that the clipboard functionality depends heavily on the OS you are using.

## Examples

### `$ eqlm eq images/macaron.jpg -n 3 2 -t 0.8 -c`

| Source                             | Result                                   |
| ---------------------------------- | ---------------------------------------- |
| ![Input Image](images/macaron.jpg) | ![Output Image](images/macaron-eqlm.png) |

### `$ eqlm eq images/mayuno.jpg -n 16 2 -t 0.7 -c -i makima`

| Source                            | Result                                  |
| --------------------------------- | --------------------------------------- |
| ![Input Image](images/mayuno.jpg) | ![Output Image](images/mayuno-eqlm.jpg) |

### `$ eqlm eq images/yaesu-wall.jpg -m lightness -t 0.9 -n 9 6`

| Source                                | Result                                      |
| ------------------------------------- | ------------------------------------------- |
| ![Input Image](images/yaesu-wall.jpg) | ![Output Image](images/yaesu-wall-eqlm.jpg) |

### `$ eqlm eq images/hakone.jpg -m saturation -t 0.2 --clamp`

| Source                            | Result                                  |
| --------------------------------- | --------------------------------------- |
| ![Input Image](images/hakone.jpg) | ![Output Image](images/hakone-eqlm.jpg) |

## Commands

The main program can be invoked either through the `eqlm` command or through the Python main module option `python3 -m eqlm`.
Each operation is implemented as a subcommand shown below.

### Eq

Spatially equalize image lightness, saturation, or brightness

```txt
<?= shell_exec("python3 -m eqlm eq --help") ?>
```

### Match

Match histogram of source image to reference image

```txt
<?= shell_exec("python3 -m eqlm match --help") ?>
```

### Laps

Sharpen an image using a Laplacian variant kernel

```txt
<?= shell_exec("python3 -m eqlm laps --help") ?>
```

## License

GNU Affero General Public License v3.0 or later

Copyright (C) 2025 curegit

This program is free software: you can redistribute it and/or modify it under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
See the GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License along with this program.
If not, see <https://www.gnu.org/licenses/>.
