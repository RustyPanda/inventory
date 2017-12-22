from __future__ import print_function

import os

from flask import Flask, render_template, redirect, url_for, request
from bokeh.embed import components
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

    possible_products = ['Anaesthetic', 'Antivirals', 'Surgical Needles']
    possible_hospitals = ['Western General', 'Royal Edinburgh Hospital']

    # Grab the inputs arguments from the URL
    form = request.form

    product = getitem(form, 'product', 'Anaesthetic').rstrip(' ') # mysterious right space appears
    other_products = [item for item in possible_products if item != product]
    products = [product] + other_products

    hospital = getitem(form, 'hospital', 'Western General').rstrip(' ') # mysterious right space appears
    other_hospitals = [item for item in possible_hospitals if item != hospital]
    hospitals = [hospital] + other_hospitals

    fig, future_orders = main_chart(product, hospital)

    print(future_orders)
    for order in future_orders:
        print(order['week'])
        print(order['hospital'])
        print(order['product'])
        print(order['mean_future'])
        print(order['cost'])

    js_resources = INLINE.render_js()
    css_resources = INLINE.render_css()

    script, div = components(fig)
    html = render_template(
        'embed_simple.html',
        plot_script=script,
        plot_div=div,
        js_resources=js_resources,
        css_resources=css_resources,
        hospitals=hospitals,
        products=products,
        future_orders=future_orders,
        selected_hospital=hospital,
        selected_product=product
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
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
