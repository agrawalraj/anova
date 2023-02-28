
import matplotlib.pyplot as plt
import matplotlib
import warnings
import numpy as np
import re
from matplotlib.colors import LinearSegmentedColormap
from colors import lab2rgb, lch2lab


def lch2rgb(x):
	return lab2rgb(lch2lab([[x]]))[0][0]

# define our colors using Lch
# note that we intentionally vary the lightness during interpolation so as to better
# enable the eye to see patterns (since patterns are most easily recognized through
# lightness variability)
blue_lch = [54., 70., 4.6588]
l_mid = 40.
red_lch = [54., 90., 0.35470565 + 2* np.pi]
gray_lch = [55., 0., 0.]
blue_rgb = lch2rgb(blue_lch)
red_rgb = lch2rgb(red_lch)
gray_rgb = lch2rgb(gray_lch)
white_rgb = np.array([1.,1.,1.])


light_blue_rgb = np.array([127., 196, 252])/255
light_red_rgb = np.array([255., 127, 167])/255

# define a perceptually uniform color scale using the Lch color space
reds = []
greens = []
blues = []
alphas = []
nsteps = 100
l_vals = list(np.linspace(blue_lch[0], l_mid, nsteps//2)) + list(np.linspace(l_mid, red_lch[0], nsteps//2))
c_vals = np.linspace(blue_lch[1], red_lch[1], nsteps)
h_vals = np.linspace(blue_lch[2], red_lch[2], nsteps)
for pos,l,c,h in zip(np.linspace(0, 1, nsteps), l_vals, c_vals, h_vals):
    lch = [l, c, h]
    rgb = lch2rgb(lch)
    reds.append((pos, rgb[0], rgb[0]))
    greens.append((pos, rgb[1], rgb[1]))
    blues.append((pos, rgb[2], rgb[2]))
    alphas.append((pos, 1.0, 1.0))

red_blue = LinearSegmentedColormap('red_blue', {
    "red": reds,
    "green": greens,
    "blue": blues,
    "alpha": alphas
})
red_blue.set_bad(gray_rgb, 1.0)
red_blue.set_over(gray_rgb, 1.0)
red_blue.set_under(gray_rgb, 1.0) # "under" is incorrectly used instead of "bad" in the scatter plot

red_blue_no_bounds = LinearSegmentedColormap('red_blue_no_bounds', {
    "red": reds,
    "green": greens,
    "blue": blues,
    "alpha": alphas
})

red_blue_transparent = LinearSegmentedColormap('red_blue_no_bounds', {
    "red": reds,
    "green": greens,
    "blue": blues,
    "alpha": [(a[0], 0.5, 0.5) for a in alphas]
})

# define a circular version of the color scale for categorical coloring
reds = []
greens = []
blues = []
alphas = []
nsteps = 100
c_vals = np.linspace(blue_lch[1], red_lch[1], nsteps)
h_vals = np.linspace(blue_lch[2], red_lch[2], nsteps)
for pos,c,h in zip(np.linspace(0, 0.5, nsteps), c_vals, h_vals):
    lch = [blue_lch[0], c, h]
    rgb = lch2rgb(lch)
    reds.append((pos, rgb[0], rgb[0]))
    greens.append((pos, rgb[1], rgb[1]))
    blues.append((pos, rgb[2], rgb[2]))
    alphas.append((pos, 1.0, 1.0))
c_vals = np.linspace(red_lch[1], blue_lch[1], nsteps)
h_vals = np.linspace(red_lch[2] - 2 * np.pi, blue_lch[2], nsteps)
for pos,c,h in zip(np.linspace(0.5, 1, nsteps), c_vals, h_vals):
    lch = [blue_lch[0], c, h]
    rgb = lch2rgb(lch)
    reds.append((pos, rgb[0], rgb[0]))
    greens.append((pos, rgb[1], rgb[1]))
    blues.append((pos, rgb[2], rgb[2]))
    alphas.append((pos, 1.0, 1.0))

red_blue_circle = LinearSegmentedColormap('red_blue_circle', {
    "red": reds,
    "green": greens,
    "blue": blues,
    "alpha": alphas
})

colors = []
for l in np.linspace(1, 0, 100):
    colors.append((30./255, 136./255, 229./255,l))
for l in np.linspace(0, 1, 100):
    colors.append((255./255, 13./255, 87./255,l))
red_transparent_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)

colors = []
for l in np.linspace(0, 1, 100):
    colors.append((30./255, 136./255, 229./255,l))
transparent_blue = LinearSegmentedColormap.from_list("transparent_blue", colors)

colors = []
for l in np.linspace(0, 1, 100):
    colors.append((255./255, 13./255, 87./255,l))
transparent_red = LinearSegmentedColormap.from_list("transparent_red", colors)

old_blue_rgb = np.array([30, 136, 229]) / 255
old_red_rgb = np.array([255, 13, 87]) / 255

colors = []
for alpha in np.linspace(1, 0, 100):
    c = blue_rgb * alpha + (1 - alpha) * white_rgb
    colors.append(c)
for alpha in np.linspace(0, 1, 100):
    c = red_rgb * alpha + (1 - alpha) * white_rgb
    colors.append(c)
red_white_blue = LinearSegmentedColormap.from_list("red_transparent_blue", colors)


import matplotlib.pyplot as plt
import matplotlib
import warnings

# from ._labels import labels
# from ..utils import safe_isinstance, format_value
# from . import colors


def format_value(s, format_str):
    """ Strips trailing zeros and uses a unicode minus sign.
    """

    if not issubclass(type(s), str):
        s = format_str % s
    s = re.sub(r'\.?0+$', '', s)
    if s[0] == "-":
        s = u"\u2212" + s[1:]
    return s


def waterfall_legacy(expected_value, shap_values=None, features=None, feature_names=None, max_display=10, show=True):
    """ Plots an explanation of a single prediction as a waterfall plot.
    The SHAP value of a feature represents the impact of the evidence provided by that feature on the model's
    output. The waterfall plot is designed to visually display how the SHAP values (evidence) of each feature
    move the model output from our prior expectation under the background data distribution, to the final model
    prediction given the evidence of all the features. Features are sorted by the magnitude of their SHAP values
    with the smallest magnitude features grouped together at the bottom of the plot when the number of features
    in the models exceeds the max_display parameter.
    Parameters
    ----------
    expected_value : float
        This is the reference value that the feature contributions start from. For SHAP values it should
        be the value of explainer.expected_value.
    shap_values : numpy.array
        One dimensional array of SHAP values.
    features : numpy.array
        One dimensional array of feature values. This provides the values of all the
        features, and should be the same shape as the shap_values argument.
    feature_names : list
        List of feature names (# features).
    max_display : str
        The maximum number of features to plot.
    show : bool
        Whether matplotlib.pyplot.show() is called before returning. Setting this to False allows the plot
        to be customized further after it has been created.
    """

    # Turn off interactive plot when not calling plt.show
    if show is False:
        plt.ioff()

    # support passing an explanation object
    upper_bounds = None
    lower_bounds = None
    if str(type(expected_value)).endswith("Explanation'>"):
        shap_exp = expected_value
        expected_value = shap_exp.expected_value
        shap_values = shap_exp.values
        features = shap_exp.data
        feature_names = shap_exp.feature_names
        lower_bounds = getattr(shap_exp, "lower_bounds", None)
        upper_bounds = getattr(shap_exp, "upper_bounds", None)

    # make sure we only have a single output to explain
    if (type(expected_value) == np.ndarray and len(expected_value) > 0) or type(expected_value) == list:
        raise Exception("waterfall_plot requires a scalar expected_value of the model output as the first "
                        "parameter, but you have passed an array as the first parameter! "
                        "Try shap.waterfall_plot(explainer.expected_value[0], shap_values[0], X[0]) or "
                        "for multi-output models try "
                        "shap.waterfall_plot(explainer.expected_value[0], shap_values[0][0], X[0]).")

    # make sure we only have a single explanation to plot
    if len(shap_values.shape) == 2:
        raise Exception(
            "The waterfall_plot can currently only plot a single explanation but a matrix of explanations was passed!")

    # fallback feature names
    if feature_names is None:
        feature_names = np.array([labels['FEATURE'] % str(i) for i in range(len(shap_values))])

    # init variables we use for tracking the plot locations
    num_features = min(max_display, len(shap_values))
    row_height = 0.5
    rng = range(num_features - 1, -1, -1)
    order = np.argsort(-np.abs(shap_values))
    pos_lefts = []
    pos_inds = []
    pos_widths = []
    pos_low = []
    pos_high = []
    neg_lefts = []
    neg_inds = []
    neg_widths = []
    neg_low = []
    neg_high = []
    loc = expected_value + shap_values.sum()
    yticklabels = ["" for i in range(num_features + 1)]

    # size the plot based on how many features we are plotting
    plt.gcf().set_size_inches(8, num_features * row_height + 1.5)

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features == len(shap_values):
        num_individual = num_features
    else:
        num_individual = num_features - 1

    # compute the locations of the individual features and plot the dashed connecting lines
    for i in range(num_individual):
        sval = shap_values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            if lower_bounds is not None:
                pos_low.append(lower_bounds[order[i]])
                pos_high.append(upper_bounds[order[i]])
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            if lower_bounds is not None:
                neg_low.append(lower_bounds[order[i]])
                neg_high.append(upper_bounds[order[i]])
            neg_lefts.append(loc)
        if num_individual != num_features or i + 4 < num_individual:
            plt.plot([loc, loc], [rng[i] - 1 - 0.4, rng[i] + 0.4],
                     color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
        if features is None:
            yticklabels[rng[i]] = feature_names[order[i]]
        else:
            yticklabels[rng[i]] = format_value(features[order[i]], "%0.03f") + " = " + feature_names[order[i]]

    # add a last grouped feature to represent the impact of all the features we didn't show
    if num_features < len(shap_values):
        yticklabels[0] = "%d other features" % (len(shap_values) - num_features + 1)
        remaining_impact = expected_value - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
            c = red_rgb
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)
            c = blue_rgb

    points = pos_lefts + list(np.array(pos_lefts) + np.array(pos_widths)) + neg_lefts + \
        list(np.array(neg_lefts) + np.array(neg_widths))
    dataw = np.max(points) - np.min(points)

    # draw invisible bars just for sizing the axes
    label_padding = np.array([0.1*dataw if w < 1 else 0 for w in pos_widths])
    plt.barh(pos_inds, np.array(pos_widths) + label_padding + 0.02*dataw,
             left=np.array(pos_lefts) - 0.01*dataw, color=red_rgb, alpha=0)
    label_padding = np.array([-0.1*dataw if -w < 1 else 0 for w in neg_widths])
    plt.barh(neg_inds, np.array(neg_widths) + label_padding - 0.02*dataw,
             left=np.array(neg_lefts) + 0.01*dataw, color=blue_rgb, alpha=0)

    # define variable we need for plotting the arrows
    head_length = 0.08
    bar_width = 0.8
    xlen = plt.xlim()[1] - plt.xlim()[0]
    fig = plt.gcf()
    ax = plt.gca()
    xticks = ax.get_xticks()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    bbox_to_xscale = xlen/width
    hl_scaled = bbox_to_xscale * head_length
    renderer = fig.canvas.get_renderer()

    # draw the positive arrows
    for i in range(len(pos_inds)):
        dist = pos_widths[i]
        arrow_obj = plt.arrow(
            pos_lefts[i], pos_inds[i], max(dist-hl_scaled, 0.000001), 0,
            head_length=min(dist, hl_scaled),
            color=red_rgb, width=bar_width,
            head_width=bar_width
        )

        if pos_low is not None and i < len(pos_low):
            plt.errorbar(
                pos_lefts[i] + pos_widths[i], pos_inds[i],
                xerr=np.array([[pos_widths[i] - pos_low[i]], [pos_high[i] - pos_widths[i]]]),
                ecolor=light_red_rgb
            )

        txt_obj = plt.text(
            pos_lefts[i] + 0.5*dist, pos_inds[i], format_value(pos_widths[i], '%+0.02f'),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=12
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = plt.text(
                pos_lefts[i] + (5/72)*bbox_to_xscale + dist, pos_inds[i], format_value(pos_widths[i], '%+0.02f'),
                horizontalalignment='left', verticalalignment='center', color=red_rgb,
                fontsize=12
            )

    # draw the negative arrows
    for i in range(len(neg_inds)):
        dist = neg_widths[i]

        arrow_obj = plt.arrow(
            neg_lefts[i], neg_inds[i], -max(-dist-hl_scaled, 0.000001), 0,
            head_length=min(-dist, hl_scaled),
            color=blue_rgb, width=bar_width,
            head_width=bar_width
        )

        if neg_low is not None and i < len(neg_low):
            plt.errorbar(
                neg_lefts[i] + neg_widths[i], neg_inds[i],
                xerr=np.array([[neg_widths[i] - neg_low[i]], [neg_high[i] - neg_widths[i]]]),
                ecolor=light_blue_rgb
            )

        txt_obj = plt.text(
            neg_lefts[i] + 0.5*dist, neg_inds[i], format_value(neg_widths[i], '%+0.02f'),
            horizontalalignment='center', verticalalignment='center', color="white",
            fontsize=12
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = plt.text(
                neg_lefts[i] - (5/72)*bbox_to_xscale + dist, neg_inds[i], format_value(neg_widths[i], '%+0.02f'),
                horizontalalignment='right', verticalalignment='center', color=blue_rgb,
                fontsize=12
            )

    # draw the y-ticks twice, once in gray and then again with just the feature names in black
    plt.yticks(list(range(num_features))*2, yticklabels[:-1] +
               [l.split('=')[-1] for l in yticklabels[:-1]], fontsize=13)

    # put horizontal lines for each feature row
    for i in range(num_features):
        plt.axhline(i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

    # mark the prior expected value and the model prediction
    plt.axvline(expected_value, 0, 1/num_features, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
    fx = expected_value + shap_values.sum()
    plt.axvline(fx, 0, 1, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)

    # clean up the main axis
    plt.gca().xaxis.set_ticks_position('bottom')
    plt.gca().yaxis.set_ticks_position('none')
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    ax.tick_params(labelsize=13)
    #plt.xlabel("\nModel output", fontsize=12)

    # draw the E[f(X)] tick mark
    xmin, xmax = ax.get_xlim()
    ax2 = ax.twiny()
    ax2.set_xlim(xmin, xmax)
    ax2.set_xticks([expected_value, expected_value])
    ax2.set_xticklabels(["\n$E[f(X)]$", "\n$ = "+format_value(expected_value, "%0.03f")+"$"], fontsize=12, ha="left")
    ax2.spines['right'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['left'].set_visible(False)

    # draw the f(x) tick mark
    ax3 = ax2.twiny()
    ax3.set_xlim(xmin, xmax)
    ax3.set_xticks([expected_value + shap_values.sum()] * 2)
    ax3.set_xticklabels(["$f(x)$", "$ = "+format_value(fx, "%0.03f")+"$"], fontsize=12, ha="left")
    tick_labels = ax3.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform(
    ) + matplotlib.transforms.ScaledTranslation(-10/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform(
    ) + matplotlib.transforms.ScaledTranslation(12/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_color("#999999")
    ax3.spines['right'].set_visible(False)
    ax3.spines['top'].set_visible(False)
    ax3.spines['left'].set_visible(False)

    # adjust the position of the E[f(X)] = x.xx label
    tick_labels = ax2.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(tick_labels[0].get_transform(
    ) + matplotlib.transforms.ScaledTranslation(-20/72., 0, fig.dpi_scale_trans))
    tick_labels[1].set_transform(tick_labels[1].get_transform(
    ) + matplotlib.transforms.ScaledTranslation(22/72., -1/72., fig.dpi_scale_trans))
    tick_labels[1].set_color("#999999")

    # color the y tick labels that have the feature values as gray
    # (these fall behind the black ones with just the feature name)
    tick_labels = ax.yaxis.get_majorticklabels()
    for i in range(num_features):
        tick_labels[i].set_color("#999999")

    if show:
        plt.show()
    else:
        return plt.gcf()

if __name__ == "__main__":
	waterfall_legacy(2., np.array([1, 2, -3]), np.array([1, 2, 3]), ['A', 'B', 'C'])
