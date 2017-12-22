'''This example demonstrates embedding a standalone Bokeh document
into a simple Flask application, with a basic HTML web form.

To view the example, run:

    python simple.py

in this directory, and navigate to:

    http://localhost:5000

'''
from __future__ import print_function

from flask import Flask, render_template, redirect, url_for, request


from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import INLINE
from bokeh.util.string import encode_utf8
from PIM_product import main_chart

app = Flask(__name__)


def getitem(obj, item, default):
    if item not in obj:
        return default
    else:
        return obj[item]


@app.route("/tracker", methods=['GET', 'POST'])
def tracker():
    """ Very simple embedding of a polynomial chart

    """

    possible_products = ['Anaesthetic', 'Antivirals', 'Surgical Needles', 'Casts']

    # Grab the inputs arguments from the URL
    form = request.form
    print(form)
    # d = args.to_dict()
    # print(d)
    product = getitem(form, 'product', 'Anaesthetic')
    other_products = [item for item in possible_products if item not in [product]]
    products = [product] + other_products
    print(product)
    print(other_products)
    fig = main_chart(product)

    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    script, div = components(fig)
    html = render_template(
        'embed_simple.html',
        plot_script=script,
        plot_div=div,
        js_resources=js_resources,
        css_resources=css_resources,
        products=products,
    )
    return encode_utf8(html)


@app.route("/", methods=['GET', 'POST'])
def login():

    error = None
    if request.method == 'POST':
        if request.form['username'] != 'demo' or request.form['password'] != 'demo':
            error = 'Invalid Credentials. Please try again.'
        else:
            return redirect(url_for('tracker'))
    return encode_utf8(render_template('login.html', error=error))


if __name__ == "__main__":
    print(__doc__)
    app.run()
