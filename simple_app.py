from flask import Flask, render_template

import random
from bokeh.models import (HoverTool, FactorRange, Plot, LinearAxis, Grid,
                          Range1d)
from bokeh.models.glyphs import VBar
from bokeh.plotting import figure
# from bokeh.charts import Bar
from bokeh.embed import components
from bokeh.models.sources import ColumnDataSource
from flask import Flask, render_template

from bokeh.resources import CDN
from bokeh.embed import file_html

from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8

app = Flask(__name__)


@app.route("/")
def chart():

    plot = create_line_chart()
    script, div = components(plot)

    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    html = render_template("chart.html",
                           div=div,
                           script=script,
                           js_resources=js_resources,
                           css_resources=css_resources)
    return encode_utf8(html)


def create_line_chart():
    # create a new plot with default tools, using figure
    p = figure(plot_width=400, plot_height=400)

    # add a circle renderer with a size, color, and alpha
    p.circle([1, 2, 3, 4, 5], [6, 7, 2, 4, 5], size=15, line_color="navy", fill_color="orange", fill_alpha=0.5)

    return p


if __name__ == "__main__":
    app.run(debug=True)