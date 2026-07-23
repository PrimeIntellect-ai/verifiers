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

## Reward-level follow-up

A subsequent full MMMU validation experiment tested whether the large rendering difference
improved model behavior. It used Qwen3.5-9B on all 847 usable multiple-choice prompts with 10
rollouts per prompt and representation (`8,470` clean rollouts per arm). ASCII used width 160;
Braille used width 80, which samples the same 160 horizontal source pixels because each Braille
cell spans two pixels.

| Arm | Accuracy | Mixed prompts | Mean within-prompt sample variance | Cost |
|---|---:|---:|---:|---:|
| Vision | 72.05% | 256/847 | 0.0545 | $8.58 |
| ASCII, fixed | 52.59% | 452/847 | 0.1029 | $11.87 |
| ASCII, Otsu | 52.55% | 454/847 | 0.1025 | $12.18 |
| Braille, fixed | 52.05% | 462/847 | 0.1060 | $13.14 |
| Braille, Otsu | 51.94% | 474/847 | 0.1062 | $13.31 |

Otsu minus fixed accuracy was `-0.04` percentage points for ASCII (task-bootstrap 95% CI
`[-1.11, +1.06]`) and `-0.12` points for Braille (`[-1.16, +0.92]`). Otsu changed which
individual prompts succeeded, but neither interval supports an aggregate reward improvement.
Its variance changes were also negligible.

## Decision

The rendering-level effect is real, but it did not translate into better accuracy or materially
better RL signal over 16,940 paired rollouts per mode. PR #2034 therefore removes adaptive Otsu
thresholding: the extra public configuration state, histogram implementation, validation, tests,
and binary ASCII branch are not justified by the measured behavior. ASCII remains grayscale;
Braille retains its numeric fixed dot cutoff, defaulting to `0.5`.
