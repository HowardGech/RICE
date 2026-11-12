"""Annotation row for the dependent-assignment figure: the realised correlation between
the perturbation indicator and the latent state, printed under each severity.
"""
# ---- realised-correlation row, as annotations ----------------------------------------
CORR_FONTSIZE = 9        # smaller than the tick labels so the values do not collide
LEVELS = sorted(df['sel'].unique())
POS = {c: i for i, c in enumerate(LEVELS)}

# Bind from the reader cell rather than retyping, so a digit cannot slip between the two:
#   CORR = agg['mean'].round(2).to_dict()
CORR = {0.0: 0.01, 0.25: 0.18, 0.5: 0.31, 1.0: 0.44, 2.0: 0.52}

# pointplot has already made x categorical at 0..n-1, so LEVELS and the tick positions must
# line up; if a generation element is missing, df loses that setting and the curves would
# silently close over the gap, so check rather than assume.
assert len(axes[1, 0].get_xticks()) == len(LEVELS), "x positions do not match LEVELS"

for ax in axes.flatten():
    # positions, not values: one unit per setting, with 0.3 of a step of padding at each end
    ax.set_xlim(-0.3, len(LEVELS) - 1 + 0.3)
    ax.set_xticks(range(len(LEVELS)))
    ax.set_xticklabels([str(c) for c in LEVELS])

# Drawn as an annotation rather than a second line of the tick label so the font size can be
# set independently; at tick size the values run into each other. Only the bottom row carries
# x tick labels, so only that row is annotated. A setting with no correlation on record is
# marked (--) rather than silently leaving a gap.
for ax in axes[-1, :]:
    for c in LEVELS:
        txt = f"({CORR[c]:.2f})" if c in CORR else "(--)"
        ax.annotate(txt, xy=(POS[c], .03), xycoords=('data', 'axes fraction'),
                    xytext=(0, -30), textcoords='offset points',
                    ha='center', va='top', fontsize=CORR_FONTSIZE)
axes[-1, 0].annotate("corr. with\nlatent state", xy=(-0.30, .03),
                     xycoords=('data', 'axes fraction'), xytext=(-8, -30),
                     textcoords='offset points', ha='right', va='top',
                     fontsize=CORR_FONTSIZE, color="0.35")

fig.supxlabel(r"Selection strength $\kappa$", fontsize=11, y=0.035)
plt.tight_layout(rect=(0.02, 0.09, 1, 1))   # tight_layout does not reserve space for the
                                            # annotations, so the strip must be reserved here
