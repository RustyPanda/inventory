import numpy as np
from sklearn import linear_model

from bokeh.io import output_notebook, show
from bokeh.plotting import figure
from bokeh.models import HoverTool, Range1d
from bokeh.models import TapTool, CustomJS, ColumnDataSource

from scipy.stats import bayes_mvs


weeks = 60

def generate_synthetic_data(t):

    # define feature sets
    # each feature represents an underlying known demand driver e.g. operations, admits, season
    # mathematically, this is the known independent (but potentially correlated) variables in the regression

    feat1_w = 0.1
    feat1_scale = 0.3
    feat1_phi = -90
    feat1 = feat1_scale *( np.sin(feat1_w * t + feat1_phi)*np.sin(feat1_w * t + feat1_phi) )

    feat2_base = 0.5
    feat2_factor = 0.2
    feat2_phi = -90
    feat2_w = 0.2
    feat2_scale = 0.4
    feat2 = feat2_scale * ( feat2_base + feat2_factor*np.cos(feat2_w * t + feat2_phi) )

    feat3 = np.arange(0, weeks, dtype=float) **1.5 * 0.001

    # true_demand is true demand, from combining underlying known demand trends
    true_demand = feat1 + feat2 + feat3
    true_demand = true_demand + true_demand.min() # always positive
    true_demand = true_demand / true_demand.mean()
    # measured_demand (capital) is measured demand, subject to high noise
    measured_demand = true_demand + np.random.normal(0,0.4,np.shape(true_demand)) * 0.6
    measured_demand = measured_demand * 100.
    measured_demand = measured_demand.astype(int)

    # Model will estimate the true demand true_demand from the known demand drivers feat1, feat2, feat3, by day.
    features = np.stack((feat1, feat2, feat3), axis=1)

    return features, true_demand, measured_demand


def visualise_product_display(measured_demand, features, t, present_timestep, product):

    t_past = t[:present_timestep]
    t_future = t[present_timestep:]
    measured_demand_past = measured_demand[:present_timestep]
    features_past = features[:present_timestep]
    features_future = features[present_timestep:]

    bootstrap_n = 20
    bootstrap_keep_fraction = 0.7
    demand_est_pim_samples = np.zeros((bootstrap_n, len(t_future)))
    demand_model_sample = np.zeros((bootstrap_n, len(t_past)))
    all_errors = []
    for sample_n in range(bootstrap_n):
        keep_value = np.random.rand(len(t_past)) < bootstrap_keep_fraction
        sample_features = features_past[keep_value]
        sample_measured_demand = measured_demand_past[keep_value]
        reg = linear_model.BayesianRidge()
        reg.fit(sample_features, sample_measured_demand)
        demand_est_pim_samples[sample_n, :] = reg.predict(features_future)
        demand_model_sample[sample_n, :] = reg.predict(features_past)

        abs_errors = np.absolute(reg.predict(features_past[~keep_value])-measured_demand_past[~keep_value])
        all_errors.append(abs_errors)


    mean_error = np.mean(np.concatenate(all_errors))

    confidence_interval = 0.8

    # bayes_mvs only supports 1 dimension


    mean_future = np.zeros_like(t_future, dtype=float)
    mean_past = np.zeros_like(t_past, dtype=float)
    # lower_future = np.zeros_like(t_future, dtype=float)
    # upper_future = np.zeros_like(t_future, dtype=float)
    for time_n in range(len(t_future)):
        single_result = bayes_mvs(demand_est_pim_samples[:, time_n], confidence_interval)[0]
        mean_future[time_n] = single_result[0]

    for time_n in range(len(t_past)):
        single_result = bayes_mvs(demand_model_sample[:, time_n], confidence_interval)[0]
        mean_past[time_n] = single_result[0]
    mean_past = np.concatenate((mean_past, np.array([mean_future[0]])))

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

    mean_past = np.concatenate((mean_past, blank_future[1:]))

    source = ColumnDataSource(
        data=dict(
            mean=mean,
            mean_past=mean_past,
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
            # ("predicted demand", "@mean"),
            # ("max prediction", "@upper"),
            # ("min prediction", "@lower"),
        ],

        mode='mouse',
    )

    # create a new plot (with a title) using figure
    p = figure(plot_width=800,
               plot_height=600,
               tools=[hover, 'box_zoom', 'pan', 'reset', 'save'],
               title="Demand for {}".format(product),
               x_range=[0, weeks*2],
               y_range=[0, measured_demand.max() + 5])

    p.cross('t', 'measured_demand_known', size=20, source=source)
    p.line('t', 'mean', legend='PIM estimate', line_color='orange', line_width=3, source=source)
    # p.line('t', 'lower', legend='PIM lower', line_color='green', line_width=1, source=source)
    # p.line('t', 'upper', legend='PIM upper', line_color='green', line_width=1, source=source)
    p.line('t', 'mean_past', legend='PIM model', line_color='grey', line_dash='dashed', line_width=3, source=source)

    p.patch('patches_t', 'patches', legend='PIM uncertainty', alpha=0.2, line_width=2, source=source)

    p.toolbar.logo = None

    p.yaxis.axis_label = "Products Required"
    p.xaxis.axis_label = "Week"

    # styling
    p.title.text_font_size = '24pt'
    p.background_fill_color = 'whitesmoke'
    p.background_fill_alpha = 0.5
    # p.border_fill_color = "whitesmoke"
    p.axis.axis_line_width = 3
    p.axis.axis_label_text_font_size = '16pt'
    p.legend.label_text_font_size = '16pt'
    p.axis.major_label_text_font_size = '16pt'

    p.legend.location = "bottom_right"

    return p


def main_chart(product):
    t = np.arange(0, weeks).astype(int) * 2 # t in day units
    features, true_demand, measured_demand = generate_synthetic_data(t)

    present_timestep = int(3. * len(t) / 4.)

    p = visualise_product_display(measured_demand, features, t, present_timestep, product)

    return p


if __name__ == '__main__':
    p = main_chart()
    show(p)

# https://stackoverflow.com/questions/39403529/how-to-show-a-pandas-dataframe-as-a-flask-boostrap-table
# https://v4-alpha.getbootstrap.com/content/tables/