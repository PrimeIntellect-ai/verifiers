# Otsu thresholding on MMMU images

## Question

Does adaptive Otsu thresholding materially change Textify output compared with the fixed
`0.5` threshold, or is it unnecessary configuration?

## Setup

- Images: the same 20 MMMU dev images selected for the model-scale experiment.
- Width: 80 columns.
- Modes: ASCII and Braille.
- Shared settings: gamma `1.0`, auto inversion, derived height.
- Comparison: fixed threshold `0.5` versus `threshold = "otsu"`.

This is a rendering-level analysis; no model calls or rewards were used.

## Results

| Metric across 20 images | Result |
|---|---:|
| Mean Otsu threshold | 0.323 |
| Mean absolute distance from 0.5 | 0.177 |
| Images more than 0.05 from 0.5 | 17/20 |
| Mean ASCII positions changed | 53.3% |
| Mean Braille cells changed | 23.7% |
| Fixed ASCII non-space density | 53.8% |
| Otsu ASCII non-space density | 27.1% |
| Fixed ASCII luminance levels | 9.05 |
| Otsu ASCII luminance levels | 2.00 |
| Fixed Braille dot density | 15.9% |
| Otsu Braille dot density | 27.9% |

Otsu is therefore not a minor perturbation. It usually chooses a substantially lower
threshold than `0.5`. In ASCII mode it deliberately produces a binary foreground/background
render instead of the ordinary grayscale ramp. On many scanned tables and diagrams this
removes weak gray background texture and makes dark structure sparse and explicit.

## Representative excerpts

Trailing spaces are omitted below. These are the first rows of actual 80-column renderings.

### Accounting table (`dev_Accounting_4`)

Fixed `0.5`:

```text
 ...............................................................................
 ...:.............................:...-...................................+.=...
 ..........+....................................................................
 .......................................................................*.:..#..
 ..:...#.#.**...=.+::.#=.#.+::-++.-#....................................#.:+::..
 ...............................................................................
 .....#:.=:-....=.+##......:::..+*.....#*..#..+-.........................+..=...
```

Otsu:

```text
                                                                          @ @
           @
                                                                        @    @
       @ @ @@   @ @   @@ @ @   @@  @                                    @  @

      @  @      @ @@@           @@     @@  @  @                          @  @
```

The fixed rendering fills nearly every background position with a low-density glyph. Otsu
retains a much smaller foreground corresponding to table content.

### Pharmacy diagram (`dev_Pharmacy_2`)

Fixed `0.5`:

```text
................................................................................
.....................................=..........................................
................................................................................
................................................................................
................................................................................
..............=*:-==........=+........-+........:--*.+..=.....:-+.+.............
.............*.....:+.........*......+........+.:..*.+.:..:.:=..-.+.=...........
```

Otsu:

```text
                                     @



              @@  @@        @@         @           @ @  @      @@ @
             @      @         @      @        @    @ @       @  @ @ @
```

### Physics force diagram (`dev_Physics_1`)

Fixed `0.5`:

```text
                           @@@::.....:*@@
                      @-:                   .@
                   @:                           =.           --:
                @.                                 @         #  *
              @.         .+@@+                       @       %@*
             @.       @                                :    .
```

Otsu:

```text
                           @@@        @@@
                      @                      @
                   @                            @
                @                                  @         @  @
              @           @@@@                       @       @@@
             @        @
```

Here both representations preserve the main outline; Otsu removes intermediate-gray texture.

## Interpretation

Otsu should remain available. It is useful when the signal is closer to dark foreground on a
light background—tables, diagrams, scans, and line art. Fixed-threshold grayscale ASCII keeps
more tonal information and may be preferable for photographs. These are different
representations rather than one universally dominating setting.

A reward-level A/B between fixed ASCII and Otsu ASCII is still needed. This report establishes
only that the option has a large, interpretable rendering effect.
