import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
plt.style.use('fivethirtyeight')

from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import HoverTool, Range1d
from bokeh.models import TapTool, CustomJS, ColumnDataSource
from bokeh.layouts import column
from bokeh.layouts import widgetbox
from bokeh.models.widgets import Slider

from scipy.stats import bayes_mvs

# Product Forge Future Health demand estimation - Mike Walmsley and Team PIM

days = 720

def generate_synthetic_data():

    # define feature sets
    # each feature represents an underlying known demand driver e.g. operations, admits, season
    # mathematically, this is the known independent (but potentially correlated) variables in the regression

    feat1_w = 0.01
    feat1_scale = 0.3
    feat1_phi = -90
    feat1 = feat1_scale *( np.sin(feat1_w * t + feat1_phi)*np.sin(feat1_w * t + feat1_phi) )

    feat2_base = 0.5
    feat2_factor = 0.2
    feat2_phi = -90
    feat2_w = 0.02
    feat2_scale = 0.4
    feat2 = feat2_scale * ( feat2_base + feat2_factor*np.cos(feat2_w * t + feat2_phi) )

    feat3 = np.ones_like(feat1) * 0.3

    # true_demand is true demand, from combining underlying known demand trends
    true_demand = feat1 + feat2 + feat3
    true_demand = true_demand + true_demand.min() # always positive
    # measured_demand (capital) is measured demand, subject to high noise
    measured_demand = true_demand + np.random.normal(0,0.4,np.shape(true_demand)) * 0.8

    # Model will estimate the true demand true_demand from the known demand drivers feat1, feat2, feat3, by day.
    features = np.stack((feat1, feat2, feat3), axis=1)

    return features, true_demand, measured_demand

# noise represents random variation by day
# mathematically, measured_demand is the dependent variable in the regression. Past measured_demand is known, future measured_demand is not.


def estimate_true_demand(t, features, true_demand, measured_demand, present_timestep):

    past_features = features[:present_timestep]

    past_measured_demand = measured_demand[:present_timestep]

    demand_est_flat = np.ones_like(true_demand) * np.sum(measured_demand) / days # demand_est_previous is routine delivery model i.e. flat estimated demand - status quo

    reg = linear_model.BayesianRidge()
    reg.fit(past_features, past_measured_demand)
    demand_est_pim = reg.predict(features) # demand_est_pim is our model's estimation of true demand by day, inferred from underlying trends

    return demand_est_flat, demand_est_pim


def simulate_hospital_stock(demand_est_previous, demand_est_pim, true_demand, measured_demand):
    # if NHS restocked every 30 days according to each demand level, what would the stock levels look like?

    month_len = 30
    months = int(days/month_len)

    stock_sq_actual = np.zeros(month_len * months)
    stock_est_actual = np.zeros(month_len * months)
    for month in range(months):
        stock_sq = np.sum(demand_est_previous[month*month_len:(month+1)*month_len])
        stock_est = np.sum(demand_est_pim[month*month_len:(month+1)*month_len])
        for day in range(month_len):
            total_day = month*month_len + day
            running_demand = np.sum( measured_demand[month*month_len:month*month_len+day] )
            stock_sq_actual[total_day] = stock_sq - running_demand
            stock_est_actual[total_day] = stock_est - running_demand

    return stock_sq_actual, stock_est_actual


def visualise_hospital_stock(stock_sq_actual, stock_est_actual, demand_est_previous):


    source = ColumnDataSource(
        data=dict(
            stock_sq_actual=stock_sq_actual,
            stock_est_actual=stock_est_actual,
            demand_est_previous=demand_est_previous,
            t=np.arange(0, len(stock_est_actual))
        )
    )

    hover = HoverTool(
        tooltips=[
            ("time", "@t"),
            # ("true demand", "@true_demand"),
            # ("measured demand", "@measured_demand"),
        ]
    )

    # create a new plot (with a title) using figure
    p = figure(plot_width=400, plot_height=400, tools=[hover], title="Stock Level Comparison")

    # add a line renderer
    p.line('t', 'stock_sq_actual', legend='Stock with NHS estimate', line_color='orange', line_width=3, source=source)
    p.line('t', 'stock_est_actual', legend='Stock with PIM estimate', line_width=3, source=source)
    # TODO styling vertical red line at 0

    p.yaxis.axis_label = "Products Required"
    p.xaxis.axis_label = "Day"

    show(p)


def get_end_of_month_stock(stock_sq_actual, stock_est_actual):

    # TODO temporary fix
    month_len = 30
    months = int(days/month_len)

    final_stock_sq = -1 * np.ones(months)
    final_stock_est =  -1 * np.ones(months)
    months = np.arange(4)

    for month in months:
        final_stock_sq[month] = stock_sq_actual[(month+1)*month_len-1]
        final_stock_est[month] = stock_est_actual[(month+1)*month_len-1]

    return final_stock_sq, final_stock_est


def visualise_end_of_month_stock(final_stock_sq, final_stock_est):

    month_len = 30
    months = int(days/month_len)

    # width = 0.35       # the width of the bars
    fig, ax = plt.subplots()
    ax.bar(months, final_stock_sq, width,color='b')
    ax.bar(months+width, final_stock_est, width,color='r')
    # ax.set_xticks([months + width for n in range(months)])
    ax.set_xticklabels(('Jan', 'Feb', 'Mar', 'Apr'))
    plt.title('End of Month Stock')
    plt.ylabel('Products in Stock')
    plt.legend(['Stock with NHS estimate', 'Stock with PIM estiamte'],loc=0)
    plt.show()




def visualise_synthetic_data(t, features, true_demand, measured_demand):

    source = ColumnDataSource(
        data=dict(
            true_demand=true_demand,
            base_measured_demand=measured_demand,
            measured_demand=measured_demand,  # will be adjusted by slider
            t=t,
        )
    )

    hover = HoverTool(
        tooltips=[
            ("time", "@t"),
            ("true demand", "@true_demand"),
            ("measured demand", "@measured_demand"),
        ]
    )

    # create a new plot (with a title) using figure
    p = figure(plot_width=400, plot_height=400, tools=[hover], title="My Line Plot")

    # add a line renderer
    p.line('t', 'measured_demand', legend='measured demand', line_color='orange', line_width=3, source=source)
    p.line('t', 'true_demand', legend='true demand', line_width=3, source=source)

    p.y_range = Range1d(0, 2)

    slider = Slider(start=0., end=1., value=0.5, step=.1, title="power")

    update_curve = CustomJS(args=dict(source=source, slider=slider), code="""
        var data = source.get('data');
        var f = slider.value;
        measured_demand = data['measured_demand']
        base_measured_demand = data['base_measured_demand']
        for (i = 0; i < measured_demand.length; i++) {
            measured_demand[i] = base_measured_demand[i] * f * 2
        }
        source.change.emit();
    """)
    slider.js_on_change('value', update_curve)


    show(column(slider, p))  # show the results




def visualise_demand_estimate(t, demand_est_previous, demand_est_pim, measured_demand, true_demand):

    # visualise status quo estimation of true demand vs. our estimation of true demand
    # plt.subplot(211)



    source = ColumnDataSource(
        data=dict(
            demand_est_previous=demand_est_previous,
            demand_est_pim=demand_est_pim,
            measured_demand=measured_demand,
            true_demand=true_demand,
            t=t,
        )
    )

    hover = HoverTool(
        tooltips=[
            ("time", "@t"),
            ("true demand", "@true_demand"),
            ("measured demand", "@measured_demand"),
        ]
    )

    # create a new plot (with a title) using figure
    p = figure(plot_width=400, plot_height=400, tools=[hover], title="Measured and Estimated Demand")

    # add a line renderer
    p.line('t', 'demand_est_previous', legend='NHS estimate', line_color='orange', line_width=3, source=source)
    p.line('t', 'demand_est_pim', legend='PIM estimate', line_width=3, source=source)
    p.line('t', 'measured_demand', legend='Measured demand', line_width=3, source=source)
    p.line('t', 'true_demand', legend='True demand', line_width=3, source=source)

    p.yaxis.axis_label = "Products Required"
    p.xaxis.axis_label = "Day"

    show(p)


def visualise_product_display(measured_demand, features, t, present_timestep):

    t_past = t[:present_timestep]
    t_future = t[present_timestep:]
    measured_demand_past = measured_demand[:present_timestep]
    features_past = features[:present_timestep]
    features_future = features[present_timestep:]

    bootstrap_n = 20
    bootstrap_keep_fraction = 0.7
    demand_est_pim_samples = np.zeros((bootstrap_n, len(t_future)))
    all_errors = []
    for sample_n in range(bootstrap_n):
        keep_value = np.random.rand(len(t_past)) < bootstrap_keep_fraction
        sample_features = features_past[keep_value]
        sample_measured_demand = measured_demand_past[keep_value]
        reg = linear_model.BayesianRidge()
        reg.fit(sample_features, sample_measured_demand)
        demand_est_pim_samples[sample_n, :] = reg.predict(features_future)

        abs_errors = np.absolute(reg.predict(features_past[~keep_value])-measured_demand_past[~keep_value])
        all_errors.append(abs_errors)


    mean_error = np.mean(np.concatenate(all_errors))

    confidence_interval = 0.8

    # bayes_mvs only supports 1 dimension


    mean_future = np.zeros_like(t_future, dtype=float)
    # lower_future = np.zeros_like(t_future, dtype=float)
    # upper_future = np.zeros_like(t_future, dtype=float)
    for time_n in range(len(t_future)):
        single_result = bayes_mvs(demand_est_pim_samples[:, time_n], confidence_interval)[0]
        mean_future[time_n] = single_result[0]
        # lower_future[time_n] = single_result[1][0]
        # upper_future[time_n] = single_result[1][1]

    # TODO wrap into neat function, then make a few different confidence levels for pro-effect
    lower_future = mean_future - mean_error
    upper_future = mean_future + mean_error
    blank_past = np.array([np.nan for n in range(len(t_past))])
    mean = np.concatenate((blank_past, mean_future))
    lower = np.concatenate((blank_past, lower_future))
    upper = np.concatenate((blank_past, upper_future))

    upper_future_reversed = np.flip(upper_future, axis=0)
    patches = np.concatenate((lower_future, upper_future_reversed))
    patch_length_correction = np.array([np.nan for n in range(len(t) - len(patches))])
    patches = np.concatenate((patch_length_correction, patches))

    t_future_reversed = np.flip(t_future, axis=0)
    patches_t = np.concatenate((patch_length_correction, t_future, t_future_reversed))

    blank_future = np.array([np.nan for n in range(len(t_future))])
    measured_demand_known = np.concatenate((measured_demand_past, blank_future))

    source = ColumnDataSource(
        data=dict(
            mean=mean,
            lower=lower,
            upper=upper,
            t=t,
            patches_t=patches_t,
            measured_demand_known=measured_demand_known,
            patches=patches
        )
    )

    hover = HoverTool(
        tooltips=[
            ('time', '@t'),
            ("measured demand", "@measured_demand_known"),
            ("predicted demand", "@mean"),
            ("max prediction", "@upper"),
            ("min prediction", "@lower"),
        ]
    )

    # create a new plot (with a title) using figure
    p = figure(plot_width=800, plot_height=400, tools=[hover, 'box_zoom', 'pan', 'reset', 'save'], title="Measured and Estimated Demand")

    p.scatter('t', 'measured_demand_known', source=source)
    p.line('t', 'mean', legend='PIM estimate', line_color='orange', line_width=3, source=source)
    # p.line('t', 'lower', legend='PIM lower', line_color='green', line_width=1, source=source)
    # p.line('t', 'upper', legend='PIM upper', line_color='green', line_width=1, source=source)

    p.patch('patches_t', 'patches', legend='PIM uncertainty', alpha=0.2, line_width=2, source=source)

    p.toolbar.logo = None
    # add a line renderer
    # p.multi_line(
    #     xs=[t_future, t_future, t_future, t_past],
    #     ys=[mean_future, lower_future, upper_future, measured_demand_past],
        # color=['blue', 'green', 'green', 'orange'],
        # legend=['PIM estimate', 'PIM lower', 'PIM upper', 'Past demand']
    # )

    # p.multi_line(
    #     xs=['t_future', 't_future', 't_future', 't_past'],
    #     ys=['mean_future', 'lower_future', 'upper_future', 'measured_demand_past'],
    #     source=source)

    # # add a line renderer
    # p.line('t_future', 'mean_future', legend='PIM estimate', line_color='orange', line_width=3, source=source)
    # p.line('t_future', 'lower_future', legend='PIM lower', line_color='green', line_width=1, source=source)
    # p.line('t_future', 'upper_future', legend='PIM upper', line_color='green', line_width=1, source=source)
    # p.line('t_past', 'measured_demand_past', legend='Past demand', line_width=3, source=source)

    p.yaxis.axis_label = "Products Required"
    p.xaxis.axis_label = "Day"

    show(p)


if __name__ == '__main__':

    t = np.arange(0, days)  # t in day units
    features, true_demand, measured_demand = generate_synthetic_data()

    present_timestep = int(3.*len(t)/4.)

    visualise_product_display(measured_demand, features, t, present_timestep)

    # demand_est_flat, demand_est_pim = estimate_true_demand(t, features, true_demand, measured_demand, present_timestep)
    # stock_sq_actual, stock_est_actual = simulate_hospital_stock(demand_est_flat, demand_est_pim, true_demand, measured_demand)
    # final_stock_sq, final_stock_est = get_end_of_month_stock(stock_sq_actual, stock_est_actual)

    # visualise_synthetic_data(t, features, true_demand, measured_demand)
    # visualise_demand_estimate(t, demand_est_flat, demand_est_pim, measured_demand, true_demand)
    # visualise_hospital_stock(stock_sq_actual, stock_est_actual, demand_est_previous)
    # TODO
    # visualise_end_of_month_stock(final_stock_sq, final_stock_est)
