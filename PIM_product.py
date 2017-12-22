import numpy as np
from sklearn import linear_model

from bokeh.io import show
from bokeh.plotting import figure
from bokeh.models import HoverTool, Legend
from bokeh.models import ColumnDataSource

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

    # feat2_base = 0.5
    # feat2_factor = 0.2
    # feat2_phi = -90
    # feat2_w = 0.2
    # feat2_scale = 0.4
    # feat2 = feat2_scale * ( feat2_base + feat2_factor*np.cos(feat2_w * t + feat2_phi) )
    steps = 2 * np.random.randint(0, 2, (3, len(t))) - 1
    positions = np.cumsum(steps, axis=1)
    feat2 = positions[1, :] / 5.

    feat3 = np.arange(0, weeks, dtype=float) **1.5 * 0.001

    # true_demand is true demand, from combining underlying known demand trends
    true_demand = feat1 + feat2 + feat3
    true_demand = true_demand + np.abs(true_demand.min() + 0.2) # always positive
    true_demand = true_demand / true_demand.mean()
    # measured_demand (capital) is measured demand, subject to high noise
    measured_demand = true_demand + np.random.normal(0,0.4,np.shape(true_demand)) * 0.6
    measured_demand = measured_demand * 100.
    measured_demand = measured_demand.astype(int)

    # Model will estimate the true demand true_demand from the known demand drivers feat1, feat2, feat3, by day.
    features = np.stack((feat1, feat2, feat3), axis=1)

    return features, true_demand, measured_demand


def generate_synthetic_data_random_walks(t):

    # define feature sets
    # each feature represents an underlying known demand driver e.g. operations, admits, season
    # mathematically, this is the known independent (but potentially correlated) variables in the regression

    # Steps can be -1 or 1 (note that randint excludes the upper limit)
    steps = 2 * np.random.randint(0, 2, (3, len(t))) - 1
    positions = np.cumsum(steps, axis=1)
    feat1 = positions[0, :]
    feat2 = positions[1, :]
    feat3 = positions[2, :]

    # true_demand is true demand, from combining underlying known demand trends
    true_demand = feat1 + feat2 + feat3
    true_demand = true_demand + np.abs(true_demand.min()) # always positive
    true_demand = true_demand / true_demand.mean()
    # measured_demand (capital) is measured demand, subject to high noise
    measured_demand = true_demand + np.random.normal(0, 0.4, np.shape(true_demand)) * 0.8
    measured_demand = measured_demand * 100.
    measured_demand = measured_demand.astype(int)

    # Model will estimate the true demand true_demand from the known demand drivers feat1, feat2, feat3, by day.
    features = np.stack((feat1, feat2, feat3), axis=1)

    print(true_demand.min(), true_demand.mean())

    return features, true_demand, measured_demand


def visualise_product_display(measured_demand, features, t, present_timestep, product, hospital):

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
            ('Week', '@t'),
            ("Products Delivered", "@measured_demand_known"),
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
               title="Deliveries of {} for {}".format(product, hospital),
               x_range=[0, weeks*2],
               y_range=[0, measured_demand.max() + 5])

    measured = p.cross('t', 'measured_demand_known', fill_color='#41B6E6', size=20, source=source)
    estimate = p.line('t', 'mean', line_color='#005EB8', line_width=3, source=source)
    mean_model = p.line('t', 'mean_past', line_color='grey', line_dash='dashed', line_width=3, source=source)

    uncertainty = p.patch('patches_t', 'patches', color='#41B6E6', alpha=0.2, line_width=2, source=source)

    p.toolbar.logo = None

    p.yaxis.axis_label = "Deliveries"
    p.xaxis.axis_label = "Week"

    # styling
    p.title.text_font_size = '22px'
    # p.title.text_font = 'sans-serif'
    p.background_fill_color = 'whitesmoke'
    p.background_fill_alpha = 0.5
    p.axis.axis_line_width = 3
    p.axis.axis_label_text_font_size = '18px'
    p.axis.axis_label_text_font_style = 'normal'
    p.axis.axis_label_text_font = 'sans-serif'
    p.axis.axis_label_text_font_size = '18px'
    p.axis.major_label_text_font_size = '18px'
    p.axis.major_label_text_font = 'sans-serif'

    legend = Legend(items=[
        ("Deliveries", [measured]),
        ("PIM estimate", [estimate]),
        ("PIM model", [mean_model]),
        ("PIM uncertainty", [uncertainty]),
    ], location=(30, 10))

    p.add_layout(legend, 'below')
    p.legend.orientation = "horizontal"
    p.legend.label_text_font = 'sans-serif'
    p.legend.label_text_font_size = '18px'
    p.legend.spacing = 10
    p.legend.click_policy = "hide"

    future_orders = []
    cost_per_unit = np.random.rand() * 40
    for n in range(len(t_future)):
        future_orders.append({
            'week': '{}'.format(t_future[n]),
            'hospital': hospital,
            'product': product,
            'mean_future': '{}'.format(int(mean_future[n])),
            'cost': '£{:7.2f}'.format(mean_future[n] * cost_per_unit)
        })

    return p, future_orders[:10]


def main_chart(product, hospital):
    t = np.arange(0, weeks).astype(int) * 2 # t in day units

    features, true_demand, measured_demand = generate_synthetic_data_random_walks(t)
    present_timestep = int(3. * len(t) / 4.)

    p = visualise_product_display(measured_demand, features, t, present_timestep, product, hospital)

    return p


if __name__ == '__main__':
    p = main_chart('Anaesthetic', 'Western General')
    show(p)

# https://stackoverflow.com/questions/39403529/how-to-show-a-pandas-dataframe-as-a-flask-boostrap-table
# https://v4-alpha.getbootstrap.com/content/tables/