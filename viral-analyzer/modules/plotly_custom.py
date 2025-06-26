from plotly import optional_imports

# Require that numpy exists for figure_factory
np = optional_imports.get_module("numpy")
if np is None:
    raise ImportError("The figure factory module requires the numpy package")

scipy_interp = optional_imports.get_module("scipy.interpolate")
if scipy_interp is None:
    raise ImportError("The create_ternary_contour figure factory requires the scipy package")

sk_measure = optional_imports.get_module("skimage")
if sk_measure is None:
    raise ImportError("The create_ternary_contour figure factory requires the scikit-image package")

import plotly.figure_factory._ternary_contour as fftc

# -------------------- Figure Factory for ternary contour -------------


def create_ternary_contour(
    coordinates,
    values,
    pole_labels=["a", "b", "c"],
    width=500,
    height=500,
    ncontours=None,
    showscale=False,
    coloring=None,
    colorscale="Bluered",
    linecolor=None,
    title=None,
    interp_mode="ilr",
    showmarkers=False,
    v_min=None,
    v_max=None,
):
    """
    Ternary contour plot.

    Parameters
    ----------

    coordinates : list or ndarray
        Barycentric coordinates of shape (2, N) or (3, N) where N is the
        number of data points. The sum of the 3 coordinates is expected
        to be 1 for all data points.
    values : array-like
        Data points of field to be represented as contours.
    pole_labels : str, default ['a', 'b', 'c']
        Names of the three poles of the triangle.
    width : int
        Figure width.
    height : int
        Figure height.
    ncontours : int or None
        Number of contours to display (determined automatically if None).
    showscale : bool, default False
        If True, a colorbar showing the color scale is displayed.
    coloring : None or 'lines'
        How to display contour. Filled contours if None, lines if ``lines``.
    colorscale : None or str (Plotly colormap)
        colorscale of the contours.
    linecolor : None or rgb color
        Color used for lines. ``colorscale`` has to be set to None, otherwise
        line colors are determined from ``colorscale``.
    title : str or None
        Title of ternary plot
    interp_mode : 'ilr' (default) or 'cartesian'
        Defines how data are interpolated to compute contours. If 'irl',
        ILR (Isometric Log-Ratio) of compositional data is performed. If
        'cartesian', contours are determined in Cartesian space.
    showmarkers : bool, default False
        If True, markers corresponding to input compositional points are
        superimposed on contours, using the same colorscale.

    Examples
    ========

    Example 1: ternary contour plot with filled contours

    >>> import plotly.figure_factory as ff
    >>> import numpy as np
    >>> # Define coordinates
    >>> a, b = np.mgrid[0:1:20j, 0:1:20j]
    >>> mask = a + b <= 1
    >>> a = a[mask].ravel()
    >>> b = b[mask].ravel()
    >>> c = 1 - a - b
    >>> # Values to be displayed as contours
    >>> z = a * b * c
    >>> fig = ff.create_ternary_contour(np.stack((a, b, c)), z)
    >>> fig.show()

    It is also possible to give only two barycentric coordinates for each
    point, since the sum of the three coordinates is one:

    >>> fig = ff.create_ternary_contour(np.stack((a, b)), z)


    Example 2: ternary contour plot with line contours

    >>> fig = ff.create_ternary_contour(np.stack((a, b, c)), z, coloring='lines')

    Example 3: customize number of contours

    >>> fig = ff.create_ternary_contour(np.stack((a, b, c)), z, ncontours=8)

    Example 4: superimpose contour plot and original data as markers

    >>> fig = ff.create_ternary_contour(np.stack((a, b, c)), z, coloring='lines',
    ...                                 showmarkers=True)

    Example 5: customize title and pole labels

    >>> fig = ff.create_ternary_contour(np.stack((a, b, c)), z,
    ...                                 title='Ternary plot',
    ...                                 pole_labels=['clay', 'quartz', 'fledspar'])
    """

    if colorscale is None:
        showscale = False
    if ncontours is None:
        ncontours = 5
    coordinates = fftc._prepare_barycentric_coord(coordinates)
    if v_min is None:
        v_min = values.min()
    if v_max is None:
        v_max = values.max()
    grid_z, gr_x, gr_y = fftc._compute_grid(coordinates, values, interp_mode=interp_mode)

    layout = fftc._ternary_layout(pole_labels=pole_labels, width=width, height=height, title=title)

    contour_trace, _ = fftc._contour_trace(
        gr_x,
        gr_y,
        grid_z,
        ncontours=ncontours,
        colorscale=colorscale,
        linecolor=linecolor,
        interp_mode=interp_mode,
        coloring=coloring,
        v_min=v_min,
        v_max=v_max,
    )

    fig = fftc.go.Figure(data=contour_trace, layout=layout)

    opacity = 1 if showmarkers else 0
    a, b, c = coordinates
    hovertemplate = (
        pole_labels[0] + ": %{a:.3f}<br>" + pole_labels[1] + ": %{b:.3f}<br>" + pole_labels[2] + ": %{c:.3f}<br>"
        "z: %{marker.color:.3f}<extra></extra>"
    )

    fig.add_scatterternary(
        a=a,
        b=b,
        c=c,
        mode="markers",
        marker={
            "color": values,
            "colorscale": colorscale,
            "line": {"color": "rgb(120, 120, 120)", "width": int(coloring != "lines")},
        },
        opacity=opacity,
        hovertemplate=hovertemplate,
    )

    if showscale:
        # if not showmarkers:
        #     colorscale = discrete_cm
        colorbar = dict(
            {
                "type": "scatterternary",
                "a": [None],
                "b": [None],
                "c": [None],
                "marker": {
                    "cmin": v_min,
                    "cmax": v_max,
                    "colorscale": colorscale,
                    "showscale": True,
                },
                "mode": "markers",
            }
        )
        fig.add_trace(colorbar)

    return fig


__all__ = [
    "create_ternary_contour",
]
