"""
Shared chart tokens, so every figure this project produces reads as one set.

Colours are from a palette validated for colour-vision deficiency and for
contrast against the chart surface; identity is never carried by colour alone
(marker shape and direct labels do that work too), which matters in print and
for the ~8% of male readers with a CVD.
"""

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES_1 = "#2a78d6"   # blue   -- the model / primary series
SERIES_2 = "#eb6834"   # orange -- the comparison series
CRITICAL = "#d03b3b"   # status -- this point is not trustworthy


def style_axes(ax):
    """Recessive grid and axes, so the data carries the emphasis."""
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(1.0)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def style_legend(ax, **kwargs):
    kwargs.setdefault("frameon", False)
    kwargs.setdefault("fontsize", 9)
    leg = ax.legend(**kwargs)
    for text in leg.get_texts():
        text.set_color(MUTED)
    return leg


def title(ax, text):
    ax.set_title(text, color=INK, fontsize=12, loc="left")
