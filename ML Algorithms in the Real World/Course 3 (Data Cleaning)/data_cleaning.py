#!/usr/bin/env python
# coding: utf-8

# # Preparing Data

# In this notebook, you will perform a number of steps that come up in the process of preparing the data to be used by a machine learning algorithm.
# 
# The main data was gathered from an experimental sensor device oserving an electro-chemical plant. The sensor device detects "interesting" events. These events were labeled by an expert into four different classes `"P"`, `"N"` and `"L"`  as well as `"-"`, which denotes a false detection (not an "interesting" event). The data shows readings for 9 different sensors along with a 'detection mode'&mdash;which affects the class of detected phenomenon&mdash; along with the date annd time when events happened. 
# 
# However, the expert also indicated that multiple factors affect the sensor readings as well. The fecators can be put into two broad categories: weather conditions where the device was used and the calibration of the device itself. For this reason, we gathered two more datasets to be used alongside the main data: 
# 
# One shows daily averages of three different weather indicators: temperature, humidity and wind level in the area the device was opertaing, as the expert indicated these may be the most relevant factors.
# 
# The other dataset is acquired from the logs of the maintenance done on the device: Every week at midnight on Friday, the device was turned off and inspected by a technican in the foloowign day. If the device was found to be near-faulty (indicated from certain measurements taken from device), the device was repaired and recalibrated. Again, the expert indicated three different measurements from the device that might be affecting the sensor readings. The device was turned back on again on midnight on Saturday.
# 
# You want to develop an algorithm that detects the class of the events automatically. You are informed, however, that there is a separate device available that indicates false detections, so there is no need for your algorithm to identify those.
# 
# We have to check the data for any defects in the data as well as align these three pieces on information together so hey can be used by a machine learning algorithm.
# 
# Let's start!

# ## Setting up

# First, we iwll import packages that will be used in our analysis and data preparation. We need NumPy to work with numerical data, pandas for data analysis and display, and plotly for visualization. We will also need `LabelEncoder` from scikit-learn to convert textual labels into numbers as well as `StandadScaler` for standardization. We will need `KNeighborsClassifier` and `LinearRegression` as well to do imputation.
# 
# Also, let's tell pandas to show us all the data in a dataset (since our datasets will not contain more than 10,000 datapoints, setting an upper limit of `10000` is going to do that):

# In[1]:


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from IPython.display import display

pd.set_option('display.max_rows', 10000)


# * **Important note**: Do not import any packages apart from above as that will break the notebook and fail the grader. Please do not add more cells to this notebook and enter your code right after `### START CODE BELOW THIS LINE ###` and before `### END CODE ABOVE THIS LINE ###`. Use the blue 'Submit Assignment' button on the header of the notebook to submit your notebook for grading after you completed the notebook.

# ## Events dataset

# The main dataset is stoed in file `"events.csv"`. Let's read the data:

# In[2]:


events_df = pd.read_csv("events.csv")


# ## Inspection

# Now, let's insepect the data. First up, eyeballing:

# In[3]:


display(events_df)


# Observations:
# - **Mode** looks like an integer value but it is actually a categorical variable;
# - The dataset has missing values (shown with `"?"`s). You are informed that the device had a problem so parts of the readings from **Mode** and **Sensor 3** was lost. This loss is MCAR (Missing Completely At Random). You are also informed that values of **Sensor 9** missing are actually due to a faulty sensor that might have not recorded readings in certain conditions, and expert tells you that the faulty sensor may happen when readings of **Sensor 7** have values less than a certain threshold;
# - All readings from **Sensor 6** are showing the same value;
# - The different feature values come in different ranges, so it's a good idea to normalize them.

# Now to visual inspection! First note that we can visualize all features except for **Date and time** (at index `0`). The **Labels** (last column, which can be indexed by `-1` using a Python convention) column is not a feature but rather the class indicator as well.
# 
# We will also extract all labels and transform labels into number labels.

# In[4]:


visualizable_feature_names = events_df.columns[1: -1]
num_visualizable_features = len(visualizable_feature_names)

point_labels = events_df['Label'].values
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(point_labels)
labels = label_encoder.classes_
num_labels = labels.shape[0]


# Now, let's plot the histograms (be patient with the visualizations in this notebook, they may take some time):

# In[5]:


fig_hist = []
for i, feature_name in enumerate(visualizable_feature_names):
    fig_hist.append(go.Figure())
    for label in labels:
        fig_hist[i].add_trace(go.Histogram(x=events_df[events_df["Label"]==label][feature_name], name=label))
    fig_hist[i].update_layout(height=400, width=800, title_text=feature_name)
    fig_hist[i].update_layout(barmode='overlay')
    fig_hist[i].update_traces(opacity=0.5)
    fig_hist[i].show()


# ...and scatter plots for all pairs of features:

# In[6]:


fig_scatmat = go.Figure(data=go.Splom(
                        dimensions=[dict(label=feature, values=events_df[feature]) \
                                    for feature in visualizable_feature_names], \
                        text=events_df['Label'],
                        marker=dict(color=y, showscale=False, line_color='white', line_width=0.5)))

fig_scatmat.update_layout(title='Pairwise feature scatter plots',                   width=400 * num_visualizable_features,                   height=400 * num_visualizable_features)

fig_scatmat.show()


# From inspection, **Sensor 1** and **Sensor 5** look like they might be shifted-and-scaled versions of each other. So do **Sensor 7** and **Sensor 8**. If so, they are duplicates and we can remove one of them, from each pair. We can check and fix that after normalizing the data.

# ## Fixing Issues

# Let's start fixing the issues, then.

# ### Removing Irrelevant class (datapoints)

# Let's start by removing all datapoints under label `"-"`. This is because they are irrelevant to our problem as we know we have other means of avoiding false detections.
# 
# You can use pandas to filter our dataset based on column values. In this case, we want to remove rows whose value for column named `"Label"` is `"-"`. So first let's find those rows. If `df` was a pandas DataFrame (`DataFrame` object, like `events_df` is), then `condition_series = (df[column_name] == specific_value)` would create a pandas series (somehting like a list), called `condition_series`, of Boolean value showing whether the value of column named `column_name` in `df` was `specific_value` or not (`True` is it is, `False` if it is not). Go ahead and use that call to generate a pandas series which shows whether values in column `"Label"` are equal to `"-"` and put it inside `is_false_detection`:

# In[7]:


# GRADED: find_irrelevant_class, 1 point(s)

# Enter your code below (1 line)
### START CODE BELOW THIS LINE ###
is_false_detection = (events_df["Label"] == "-")
### END CODE ABOVE THIS LINE ###


# Now, some checks: if you did things correctly the output of the next cell should show you:
# >`array([False, False,  True, False, False, False, False, False,  True,
#        False])
# Total number of false detections: 792`

# In[8]:


display(is_false_detection[:10].values)
print("Total number of false detections:", is_false_detection.values.astype('uint').sum())


# Now, in order to remove the rows associated with false detections, we have extract the indices of these rows from the Boolean filter series (`is_false_detection`) we created. For a DataFrame `df` and a pandas Boolean series `s`, we get a series containing the indices where `s` is `True` by using `df[s].index`. Now, extract the indices for the rows we have a false detection and put it in `false_detection_row_indices`:

# In[9]:


# GRADED: find_irrelevant_class_indices, 1 point(s)

# Enter your code below (1 line)
### START CODE BELOW THIS LINE ###
false_detection_row_indices = events_df[is_false_detection].index
### END CODE ABOVE THIS LINE ###


# If that was done correctly, the output from the next cell should match:
# >`array([ 2,  8, 12, 24, 25, 29, 32, 37, 43, 61])`

# In[10]:


display(false_detection_row_indices[:10].values)


# We are ready to drop the false detection rows now. If row indices are stored in an index series `i`, `df_b = df_a.drop(index=i)` produces a DataFrame `df_b` in which row indices in `i` are dropped from DataFrame `df_a`. Using that drop rows in false detection indices `false_detection_row_indices` from `events_df` and store the resulting DataFrame in `events_df_2`:

# In[11]:


# GRADED: drop_irrelevant_class, 1 point(s)

# Enter your code below (1 line)
### START CODE BELOW THIS LINE ###
events_df_2 = events_df.drop(index=false_detection_row_indices)
### END CODE ABOVE THIS LINE ###


# You can see the first 10 labels, which should not include any `"-"`s from the output of the next cell. It should read:
# >`array(['P', 'P', 'N', 'N', 'N', 'N', 'P', 'L', 'P', 'P'], dtype=object)`

# In[12]:


display(events_df_2["Label"].values[:10])


# ### Removing single-value feature

# Now, we also know that `"Sensor 6"` has a single value for all datapoints and hence it will not provide any discrimination in any ML algorithm. We can drop that too. To drop a column named `"Name"` from a DataFrame `df_a` and store the result in `df_b`, we can do `df_b = df_a.drop(columns="Name")`. Now, your turn! Drop the `"Sensor 6"` column from `events_df_2` and store it in `events_df_3`:

# In[13]:


# GRADED: drop_irrelevant_feature, 1 point(s)

# Enter your code below (1 line)
### START CODE BELOW THIS LINE ###
events_df_3 = events_df_2.drop(columns="Sensor 6")
### END CODE ABOVE THIS LINE ###


# You can see the columns of `events_df_3` DataFrame from the output of the next cell. It should read:
# >`Index(['Date and time', 'Mode', 'Sensor 1', 'Sensor 2', 'Sensor 3', 'Sensor 4',
#        'Sensor 5', 'Sensor 7', 'Sensor 8', 'Sensor 9', 'Label'],
#       dtype='object')`

# In[14]:


display(events_df_3.columns)


# ### Handling numerical feature with (likely) MCAR missing values

# Now, we can start imputing missing values. We start with `"Sensor 3"`. If we are not completely sure if missing values in a feature like `"Sensor 3"` are MCAR, then there might be some information hidden in the missing-ness itself. So, it would be a good idea to add a feature to the DataFrame which indicates whether the value for a feature (`"Sensor 3"` here) was missing prior to impting the missing values for that feature.
# 
# To do that we first need to create a pandas Boolean series which indicates whether the values in that feature are missing. Such a series would be `True` where there was a missing value and `False` where there was not. You did a similar thing with finding which rows had a `"Label"` value of `"-"`. Now we want to find rows which have a `"?"` value for `"Sensor 6`", in `events_df_3`. Go ahead and do that now and store the values under the variable name `is_sensor_3_missing`: 

# In[15]:


# GRADED: generate_sensor_3_missing, 1 point(s)

# Enter your code below (1 line)
### START CODE BELOW THIS LINE ###
is_sensor_3_missing = (events_df_3["Sensor 3"] == "?")
### END CODE ABOVE THIS LINE ###


# You should see this index series as the output of the next cell:
# >`Int64Index([   0,    1,    3,    4,    5,    6,    7,    9,   10,   11,
#             ...
#             4672, 4673, 4674, 4675, 4676, 4678, 4679, 4680, 4681, 4682],
#            dtype='int64', length=3891)`

# In[16]:


display(is_sensor_3_missing.index)


# Let's do a mean imputation for missing values of `"Sensor 3"` (we could have done median, mode, etc., but let's stick with mean now). Since we have no specific info on `"Sensor 3"` not being MCAR (although we may be unsure whether it is in fact MCAR), a mean of all values that are not missing is a good idea. For that we can filter `events_df_3` to find places where the value is **not** missing. We already have a Boolean filter series `is_sensor_3_missing` which shows whether the values **are** missing. So, the non-missing values are simply the logical _not_ of that. We can apply the logical _not_ opertaion on a NumPy array (or pandas Boolean series) `a` and store the results under name `b` simply by doing `b = np.logical_not(a)`. Then, if we want to filter a DataFrame `df` by that result and get the approprita erow, we can do `c = df[b]`. And if we want to get the value for column named `"Name"` of that DataFrame, we can do `d = c["Name"]`. Finally we want to calculate the mean of those values and for that we have to convert the values to floating-point numbers since the values are stored string-valued in the DataFrame. For that we can do `e = d.astype("float")`. And now that we have the floating-point valued series, we can get the mean simply by `f = e.mean()`. We can chain all these operations together into a one-liner, doing `f = df[np.logical_not(a)]["Name"].astype("str").mean()`. Now, calculate the mean of the non-missing values of `"Sensor 3"` in `events_df_3` and put it under the name `sensor_3_mean`:

# In[17]:


# GRADED: sensor_3_mean, 1 point(s)

# Enter your code below (1–5 lines)
### START CODE BELOW THIS LINE ###
sensor_3_mean = events_df_3[np.logical_not(is_sensor_3_missing)]["Sensor 3"].astype("float").mean()
### END CODE ABOVE THIS LINE ###


# You can see your calculated mean from the output of the next cell:

# In[18]:


display(sensor_3_mean)


# We will be changing things in our DataFrame. For record-keeping purposes, let's make a copy of our DataFrame and make further changes to that copy:

# In[19]:


events_df_4 = events_df_3.copy()


# Now, let's add the sensor 3 missing-ness faeture to our DataFrame. However, adding a Boolean-valued column may not be the best way. In the end, we want all our values to be numbers. A better column formatting would be one where the value is $1$ if the Boolean value is $true$ and $0$ if it is $false$. We can convert `True`s and `False`s in a Boolean series `a` to `1`s and `0`s simpy by converting it into an integer: `b = a.astype("int")`. And then if we want to add this new integer-valued series to a DataFrame `df` under column name `"Name"`, we can do `df["Name"] = b`. We can chain these operations as well and do `df["Name"] = a.astype("int")`. Now, add a nre feature named `"Sensor 3 Missing"` to `events_df_4` by converting `is_sensor_3_missing` to integers:

# In[20]:


# GRADED: add_sensor_3_missing, 1 point(s)

# Enter your code below (1 line)
### START CODE BELOW THIS LINE ###
events_df_4["Sensor 3 Missing"] = is_sensor_3_missing.astype("int")
### END CODE ABOVE THIS LINE ###


# You can see the first 20 rows of that feature from the output of the next cell:

# In[21]:


display(events_df_4["Sensor 3 Missing"].head(20).values)


# Let's make another incremental DataFrame copy (for the purpose of keeping track of changes):

# In[22]:


events_df_5 = events_df_4.copy()


# Now, we can update the values and do the imputation for `"Sensor 3"` values now. To get a portion of a DataFrame `df` indexed by a Boolean series `s` filtering rows and column name `"Name"` you can do `df.loc[s, "Name"]`. You can assign a value (or a series of values) `v` by `df.loc[s, "Name"] = v`. Now, update the values in `events_df_5` for the portion in the intesection of `is_sensor_3_missing` Boolean filter and `"Sensor 3"` to `sensor_3_mean` you calculated before:

# In[23]:


# GRADED: sensor_3_imputation, 1 point(s)

# Enter your code below (1 line)
### START CODE BELOW THIS LINE ###
events_df_5.loc[is_sensor_3_missing, "Sensor 3"] = sensor_3_mean
### END CODE ABOVE THIS LINE ###


# You imputed and addressed the missing values problem with `"Sensor 3"` now. You can see the first 20 rows of that feature from the next output. It should read:
# >`array(['0.130714', '0.1662583', '0.1615738', '0.0820463', '0.0307044',
#        '0.0667005', 0.0972326645914397, '0.0074015', '0.1532542',
#        '0.210529', '0.1613579', '0.1741849', '0.0560326', '0.1392853',
#        '0.0425448', 0.0972326645914397, '0.1235271', '0.1515701',
#        '0.1331701', '0.0743863'], dtype=object)`

# In[24]:


display(events_df_5["Sensor 3"].head(20).values)


# ### One-hot encoding and handling categorical feature with (likely) MCAR missing value

# Now, let's handle the missing values for `"Mode"`. This is, however, a categorical feature and we cannot do a mean, median, mode, ... imputation. What we can do is impute with things like the majority category (there are other options as well, depending on the situation). However, we may not need do that since we are encoding the categorical variable into one-hot encoding. If we add another feature, treating the missing-ness as a category in itself, we can have a dataset that has all its values in numbers and we don't have to provide an imputed number. Actually, imputed numbers are inaccurate information and will not help but sometimes hurt our discrimination power. In the case of numerical features, we have no way but to provide a number but this is not the case with categorical feature. The binary missing-ness feature we add is sufficient.
# 
# To do the one-hot encoding, we first need to extract all the possible categories for `"Mode"`. This is nothing but a list of unique values of that feature. To extract unique value out of a NumPy array (or pandas series), we can use the `np.unique` function from NumPy. `b = np.unique(a)` gives us the unique values in `a`, storing it in variable `b`. In this case we want the unique values of the `"Mode"` feature of `events_df_5`. Get that and put it under variable name `mode_categories`:

# In[25]:


# GRADED: mode_categories, 1 point(s)

# Enter your code below (1 line)
### START CODE BELOW THIS LINE ###
mode_categories = np.unique(events_df_5["Mode"])
### END CODE ABOVE THIS LINE ###


# You should have these unique values (as the output of the next cell):
# >`array(['1', '2', '3', '?'], dtype=object)`

# In[26]:


display(mode_categories)


# Another incremental copy...

# In[27]:


events_df_6 = events_df_5.copy()


# Now, we can use a for loop to go through the unique values and do operations on them, one by one:
# 
# `for item in list:
#     action_1(item)
#     action_2(item)
#     `<br>
# &emsp;&emsp;&emsp;&emsp;$\vdots$
#     
# goes through the items in `list` setting `item` to be each one of them, one at a time and then does `action_1` and `action_2` which can use `item` in their computations. You can have any number of actions you can do (you have to indent the block to delineate where does the for loop end). Use a for loop going over `mode_categories` assigning each item to varaible `category` and do the following for each one:
# - Find where the value of `"Mode"` is that `category` in `events_df_6` and store it in `category_series`;
# - convert `category_series` to integers, making a one-hot vector and put it under name `category_feature`;
# - Add a column named `"Mode " + category` (this will be `"Mode 1"`, `"Mode 2"`, `"Mode 3"` and `"Mode ?"` in different executions of the loop since the unique values of `"Mode"` are `"1"`, `"2"`, `"3"` and `"?"`) to `events_df_6` whose value is `category_feature`.
# 
# This will end up adding the one-hot encoded features. Do that now:

# In[28]:


# GRADED: mode_imputation, 1 point(s)

# Enter your code below (4 lines)
### START CODE BELOW THIS LINE ###
for category in mode_categories:
        category_series = (events_df_6["Mode"] == 'mode_categories')
        category_feature = category_series.astype("int")
        events_df_6["Mode" + category] = category_feature
### END CODE ABOVE THIS LINE ###


# You can see the first 10 datapoints below (you should be able to see your new **Mode 1**, **Mode 2**, **Mode 3** and **Mode ?** features in it):

# In[29]:


display(events_df_6.head(10))


# Another incremental copy:

# In[30]:


events_df_7 = events_df_6.copy()


# Now, you can drop the `"mode"` column from `events_df_7` as well, since it is not needed anymore:

# In[31]:


# GRADED: drop_mode, 1 point(s)

# Enter your code below (1 line)
### START CODE BELOW THIS LINE ###
events_df_7 = events_df_7.drop(columns="Mode")
### END CODE ABOVE THIS LINE ###


# You can see the column names below:

# In[32]:


display(events_df_7.columns)


# ### Handling numerical feature with MAR missing values depending on another feature

# Now, to impute missing values for `"Sensor 9"`: We know that they are MAR (not MCAR) and it looks like they depend on values for `"Sensor 7"` and `"Sensor 8"`. However, `"Sensor 7"` and `"Sensor 8"` are shifted and scaled versions of each other as we will find out later in this notebook. So, let's impute based on the values of `"Sensor 7"`. You know, from the expert, that missing values for `"Sensor 9"` happened due to a faulty sensor which fails (probabilistically) if the reading on `"Sensor 7"` is below a threshold. Let's find that threshold now.
# You have to first find the row in `events_df_7` whose value is `"?"` (missing). Put that in `is_sensor_9_missing`.
# Then, use that filter `events_df_7` and store the values of `"Sensor 7"` from that in `sensor_9_missing_sensor_7_values`. Then, use `.max()` to get the maximum value which would be the threshold (if `a` is a pandas series, `b = a.max()` stores the maximum value of `a` in `b`. Put the maximum of `sensor_9_missing_sensor_7_values` in `sensor_7_threshold`:

# In[33]:


# GRADED: get_threshold, 1 point(s)

# Enter your code below (3 lines)
### START CODE BELOW THIS LINE ###
is_sensor_9_missing = (events_df_7["Sensor 9"] == "?")
sensor_9_missing_sensor_7_values = events_df_7.loc[is_sensor_9_missing, "Sensor 7"]
sensor_7_threshold = sensor_9_missing_sensor_7_values.max()
### END CODE ABOVE THIS LINE ###


# This is the threshold you found:

# In[34]:


display(sensor_7_threshold)


# Now, create a Boolean series for when the readings of `"Sensor 7"` is less than that `sensor_7_threshold` and put it in `sensor_7_below_threshold` (you can find values of series `a` which are less than `v` and put it in `b` by `b = (a <= v)`. You want to find the mean for places where values of sensor 9 are not missing **in conjuction** with the readings sensor 7 are below the threshold. Use `c = np.logical_and(a, b)` to perform the logical _and_ (conjunction) opertaion with `sensor_7_below_threshold` and the logical negation of `is_sensor_9_missing` Boolean series. The result of this is a Boolean series that filters our DataFrame and provides us with values we can take the mean of to do imputation. Put this Boolean series under the name `sensor_9_impuatation_mean_filter`:

# In[35]:


# GRADED: threshold_filter, 1 point(s)

# Enter your code below (2 lines)
### START CODE BELOW THIS LINE ###
sensor_7_below_threshold = (events_df_7["Sensor 7"] <= sensor_7_threshold)
sensor_9_impuatation_mean_filter = np.logical_and(np.logical_not(is_sensor_9_missing), sensor_7_below_threshold)
### END CODE ABOVE THIS LINE ###


# Your filter should have `71` places where you would take the mean from:

# In[36]:


display(sensor_9_impuatation_mean_filter.values.sum())


# Yet another incremental copy:

# In[37]:


events_df_8 = events_df_7.copy()


# Now, compute the mean for `"Sensor 9"` in `events_df_8` filtered by `sensor_9_impuatation_mean_filter`, convert it to floating-point numbers and take the mean, putting the result in `sensor_9_imputation_mean`. Finally, use `.loc` to update the values in `events_df_8` with rows being specified by Boolean filter `is_sensor_9_missing` and column being `"Sensor 9"` to `sensor_9_imputation_mean`:

# In[38]:


# GRADED: sensor_9_imputation, 1 point(s)

# Enter your code below (2 lines)
### START CODE BELOW THIS LINE ###
sensor_9_imputation_mean = events_df_8[sensor_9_impuatation_mean_filter]["Sensor 9"].astype("float").mean()
events_df_8.loc[is_sensor_9_missing, "Sensor 9"] = sensor_9_imputation_mean
### END CODE ABOVE THIS LINE ###


# If everything went well, then below should match the output of the next cell:
# >`array([  78,   98,  133,  255,  289,  438,  551,  552,  571,  670,  697,
#         748,  836,  874,  912,  913,  988, 1058, 1092, 1153, 1237, 1286,
#        1300, 1378, 1459, 1482, 1491, 1581, 1591, 1599, 1631, 1692, 1831,
#        1847, 1877, 1897, 1991, 2021, 2195, 2203, 2262, 2274, 2295, 2324,
#        2445, 2469, 2500, 2522, 2548, 2568, 2610, 2637, 2701, 2711, 2753,
#        2785, 2793, 2871, 2874, 2887, 2891, 2924, 2981, 3013, 3092, 3104,
#        3244, 3252, 3265, 3313, 3498, 3542, 3543, 3547, 3619, 3625, 3637,
#        3662, 3728, 3749, 3792, 3831, 3849, 3883])
# array([-0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183,
#        -0.8123184366197183, -0.8123184366197183, -0.8123184366197183],
#       dtype=object)
# Imputation mean:  -0.8123184366197183`

# In[39]:


display(np.where(is_sensor_9_missing)[0])
display(events_df_8[is_sensor_9_missing]["Sensor 9"].values)
print("Imputation mean: ", sensor_9_imputation_mean)


# ### Standardization (or normalization)

# Let's take a look at the DataFrame at this point:

# In[40]:


display(events_df_8)


# The features come in very different ranges, so we do want to normalize or standardize features. We can do this with `sklearn.preprocessing.MinMaxScaler` and `sklearn.preprocessing.StandardScaler` functions of scikit-learn. However, here, we will do them by hand. And we opt for standardization (making every feature have a mean of $0$ and a standard deviation of $1$) instead of normalization(making every feature to range in $[0,1]$, or, sometimes, $[-1,+1]$). We can only standardize numerical features, do we will make a list containing the names of the features we want to standardize: 

# In[41]:


standarization_numerical_features = ["Sensor 1", "Sensor 2", "Sensor 3", "Sensor 4", "Sensor 5", "Sensor 7", "Sensor 8", "Sensor 9"]


# Let's make an incremental copy:

# In[42]:


events_df_9 = events_df_8.copy()


# Now, we are going to standardize the features. Write a for loop which goes over `standarization_numerical_features` just defined, storing each item in `feature`. Inside:
# - Get a series for values of `feature` in `events_df_8` converting them to floating-point numbers and storing that in varibale named `column`;
# - assigning the value of `feature` column of `events_df_9` to be:
# $$\frac{\mathbf{c}-\mu_\mathbf{c}}{\sigma_\mathbf{c}}$$
# where $\mathbf{c}$ is `column`, $\mu_\mathbf{c}$ is the mean of `column` (remember you can get the mean of a series `a` by `a.mean()`) and $\sigma_\mathbf{c}$ is the standard deviation of `column` (you can get the standard deviation of a series `a` by `a.std()`).

# In[43]:


# GRADED: readings_standardization, 1 point(s)

# Enter your code below (3–6 lines)
### START CODE BELOW THIS LINE ###
for feature in standarization_numerical_features:
    column = events_df_8[feature].astype("float")
    events_df_9[feature] = (column - column.mean())/(column.std())
### END CODE ABOVE THIS LINE ###


# Then, you can see the standardized DataFrame below:

# In[44]:


display(events_df_9)


# ### Removing duplicate and uninformative features

# Now, you can see that `"Sensor 1"` and `"Sensor 5"` and `"Sensor 7"` and `"Sensor 8"` were actual shifted and scaled copies of each other:

# In[45]:


print("\"Sensor 5\" is identical to \"Sensor 1\":", np.allclose(events_df_9["Sensor 1"], events_df_9["Sensor 5"], rtol=1e-6, atol=1e-6))
print("\"Sensor 8\" is identical to \"Sensor 7\":", np.allclose(events_df_9["Sensor 7"], events_df_9["Sensor 8"], rtol=1e-6, atol=1e-6))


# So, we can remove `"Sensor 5"` and `"Sensor 8"` features. Also, the expert tells us that `"Sensor 2"` is surely irrelevant to the problem we are trying to solve. He also tells you that `"Sensor 9"` is likely irrelevant to the problem. However, as we are uncertain, let's just keep that feature. So, go ahead and remove `"Sensor 2"`, `"Sensor 5"` and `"Sensor 8"` features. Remember you can do a `.drop(columns=`$\ldots$`)` not only with single feature names, but also with a list of feature names, e.g., `["Sensor 2", "Sensor 5", "Sensor 8"]`:

# In[46]:


# GRADED: drop_sensor_2_and_5_and_8, 1 point(s)

# Enter your code below (1 line)
### START CODE BELOW THIS LINE ###
events_df_10 = events_df_9.drop(columns=["Sensor 2", "Sensor 5", "Sensor 8"])
### END CODE ABOVE THIS LINE ###


# You can see the final column names below. It should not include the features we just removed:

# In[47]:


display(events_df_10.columns)


# ### Rearranging columns

# Now, we can move the `"Label"` column to be the last one:

# In[48]:


column_order = list(events_df_10.columns)
column_order.remove("Label")
column_order.append("Label")
events_df_11 = events_df_10[column_order]


# ## End-result

# The final DataFrame is displayed below:

# In[49]:


display(events_df_11)


# # Average daily weather dataset

# Now, let's tend to the average daily weather dataset. This is located in file `"daily_weather.csv"`:

# In[50]:


weather_df = pd.read_csv("daily_weather.csv")


# ## Inspection

# Let's see this dataset first:

# In[51]:


display(weather_df)


# Observations:
# - **Index** is the index of the data and is uninformative and does not provide any discrimination power;
# - **Wind level** looks like a numerical feature but it is actually a ordinal feature, so a unary encoding might be the best bet;
# - **Wind level** has missing value. An expert has told you that these missing values are MCAR;
# - Values of features **Temperatue** and **Humidity** come in different ranges, so it's a good idea to normalize them.

# Next up is visualization. To do that, we have to specify which features are suitable to be visualized. The features `"Index"` (column with index `0`) and `"Date"` (column with index `1`) are unsuitable:

# In[52]:


visualizable_feature_names_weather = weather_df.columns[2:]
num_visualizable_features_weather = len(visualizable_feature_names_weather)


# Now we can plot a histogram plot of different features:

# In[53]:


fig_hist_weather = []
for i, feature_name in enumerate(visualizable_feature_names_weather):
    fig_hist_weather.append(go.Figure(go.Histogram(x=weather_df[feature_name])))
    fig_hist_weather[i].update_layout(height=400, width=800, title_text=feature_name)
    fig_hist_weather[i].show()


# We can visualize pairwise scatter plots as well:

# In[54]:


fig_scatmat_weather = go.Figure(data=go.Splom(
                        dimensions=[dict(label=feature, values=weather_df[feature]) \
                                    for feature in visualizable_feature_names_weather],
                        marker=dict(showscale=False, line_color='white', line_width=0.5)))

fig_scatmat_weather.update_layout(title='Pairwise feature scatter plots',                                   width=400 * num_visualizable_features_weather,                                   height=400 * num_visualizable_features_weather)

fig_scatmat_weather.show()


# Everything seems fine. One interesting observation is that `"Temperature"` and `"Humidity"` seem to have a high correlation with each other.

# ## Fixing problems

# We will start fixing the problems now:

# ### Dropping Uninformative feature

# First, let's drop the uninformative feature `"Index"`. Drop that feature from `weather_df`, putting the result under name `weather_df_2`:

# In[55]:


# GRADED: drop_index, 1 point(s)

# Enter your code below (1 line)
### START CODE BELOW THIS LINE ###
weather_df_2 = weather_df.drop(columns = "Index")
### END CODE ABOVE THIS LINE ###


# You can see the column names below. It should not include `"Index"` anymore:

# In[56]:


display(weather_df_2.columns)


# ### Handling an ordinal feature with MCAR missing values

# On to adressing missing values in `"Wind level"` now. First off, we know almost surely this is missing MCAR from what the expert told us, so there is no need to add a new feature that shows whether the value on `"Wind level"` was missing in the original data, since this would almost surely just noise. Second, this is an ordinal feature and thus, imputing values using neighbourhoods (found using proximity among other features, specifically `"Temperature"` and `"Humidity"`) makes more sense compared to some other strategy like replacing it with the majority value.
# 
# To do that, first create a Boolean series indicating where the value of `"Wind level"` in `weather_df_2"` is missing, i.e., is `"?"`, and put it in `is_wind_level_missing`:

# In[57]:


# GRADED: wind_level_missing, 1 point(s)

# Enter your code below (1 line)
### START CODE BELOW THIS LINE ###
is_wind_level_missing = weather_df_2["Wind level"] == "?"
### END CODE ABOVE THIS LINE ###


# You should get the below output from the next cell (which shows the indices of rows where the `"Wind level"` feature is missing:
# >`array([  0,   8, 100, 101, 104, 105, 167, 225, 230, 237, 267, 279, 346,
#        348, 357, 380, 394, 395, 440, 481, 496, 505, 520, 531, 547, 576,
#        584, 598, 613, 616, 622, 624, 632, 643, 658, 670, 720, 756, 767])`

# In[58]:


display(np.where(is_wind_level_missing)[0])


# Now, to impute based on nearest neighbours, we can actually use the $k$-NN algorithm. There is of course the question of choosing the correct $k$, but let's assume $k=3$ is a good choice for this problem.
# 
# So, go ahead and define an object of class `KNeighborsClassifier` (from scikit-learn) with parameter `n_neighbors=3` and give the name `knn_imputor` to it. We have to 'fit' this classifier with training data and labels as well. Training data is this case is nothing but the values of `"Temperature"` and `"Humidity"` for rows where the `"Wind level"` is not missing, so `"Wind level"` can be used as the label. So, filter `weather_df_2` using a Boolean series showing the rows where `"Wind level"` is not missing (remeber, you can use `np.logical_not` to do the logical _not_, or complement, operation), extract the columns `"Temperature"` and `"Humidity"` (using a list that contains these two names and using this list as the indexer) and store the result in `X_train_knn`. Get the same data for column `"Wind level"` (instead of the `["Temperature", "Humidity"]` list) and put that in `y_train_knn`. Then, call the `.fit` method of `knn_imputor` inputting the training data and labels:

# In[59]:


# GRADED: knn_imputation, 1 point(s)

# Enter your code below (4 lines)
### START CODE BELOW THIS LINE ###
knn_imputor = KNeighborsClassifier(n_neighbors=3)
X_train_knn = weather_df_2[np.logical_not(is_wind_level_missing)][["Temperature", "Humidity"]]
y_train_knn = weather_df_2[np.logical_not(is_wind_level_missing)]["Wind level"]
knn_imputor.fit(X_train_knn, y_train_knn)
### END CODE ABOVE THIS LINE ###


# You should see:
# >`KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
#                      metric_params=None, n_jobs=None, n_neighbors=3, p=2,
#                      weights='uniform')`
#                      
# from the output of the cell above. Now use the same call as the one above used to get training data, this time with `is_wind_level_missing` itself (instead of its logical complement) as the filter to get the production data `X_production_knn` (where you want to predict the imputed labels for). Then, the imputed labels, `y_production_knn`, can be predicted by using the `.predict` method of `knn_imputor` feeding in `X_production_knn`:

# In[60]:


# GRADED: knn_imputation, 1 point(s)

# Enter your code below (2 lines)
### START CODE BELOW THIS LINE ###
X_production_knn = weather_df_2[is_wind_level_missing][["Temperature", "Humidity"]]
y_production_knn = knn_imputor.predict(X_production_knn)
### END CODE ABOVE THIS LINE ###


# You should see the predicted labels below:

# In[61]:


display(y_production_knn)


# Now, let's make an incrmental copy before updating with imputed labels:

# In[62]:


weather_df_3 = weather_df_2.copy()


# Now, go ahead and replace the portion of `weather_df_3` at the intersection of `is_wind_level_missing` and `"Wind level"` (remember you have to use `.loc`) with the imputed labels, `y_production_knn`:

# In[63]:


# GRADED: wind_level_imputation, 1 point(s)

# Enter your code below (1 line)
### START CODE BELOW THIS LINE ###
weather_df_3.loc[is_wind_level_missing, "Wind level"] = y_production_knn
### END CODE ABOVE THIS LINE ###


# You can see the values previously missing below. They are exactly the output you got above:

# In[64]:


display(weather_df_3.loc[is_wind_level_missing, "Wind level"].values)


# Now, let's encode the ordinal feature into unary encoding. This one will be provided:

# ### Converting an ordinal feature into unary encoding

# Ordinal features are "categorical" features with an ordering defined between "categories" (so a "category" is bigger than some "categories" in that ordering, equal to itself in that ordering, and smaller than the rest of the "categories"). This means that integers are not a suitable encoding, since in integers the distance between $0$ and $1$ is the same as the distance between $1$ and $2$ and that **is** important to numerical algorithms since they will assign a fixed weight which would multiply this number as a part of how they work (at least with linear models). One-hot encoding is not the best encoding for them neither, since it is too loose. That will assign different weights to each encoding which may break the ordering of the "categories". So, we should resort to something where weights assigned by numerical algorithms can work cumulatively: unary encoding. In unary encoding for integers between $0$ and $n$, each encoding is of length $n$. The encoding of $0$ is $n$ zeros, $1$ is $n-1$ zeros followed by a single one, $k$ ($0 \leq k \leq n$) is $n-k$ zeros followed by $k$ ones and so on and so forth. So you can see the distances between increasing "categories" are the sum of different weights and it is cumulative.
# 
# Now, to encode `"Wind level"` into unary, we extarct the number of rows in the DataFrame (which will be used to specify the number of rows of the matrix holding the encoding, `wind_level_encoded`), convert the `"Wind level"` feature of `weather_df_3` into integers, find the "categories" by using `np.unique` and sort them, and calculate the $n$ for unary encoding, which is named `max_wind_levels` here. Then, we can construct the matrix that will hold the unary encoding. Next, we will go through the `"Wind level"` feature for different points, using `enumerate` in the for loop to only go item by item but also keep the index `i` of whether this is the first item, second item, ...). Then, for `i`th item, we can set the $1$st, $2$nd, $3$rd, ..., up to $k$th column to be `0`, where $k$ is the `"Wind level"` for item `i`. All next indices will be `0` since they are intialized to be `0`s in the beginning. Then, we go through all different columns $1, 2, 3, \ldots, n$ (which is iteratively generated by `range` function of Python) and create a new feature that adds that column of unary encoding. Finally, we remove the original `"Wind level"` feature:

# In[65]:


num_days = weather_df_3.shape[0]
wind_level_int = weather_df_3["Wind level"].astype("int")
wind_level_uniques = np.sort(np.unique(wind_level_int))
max_wind_levels = wind_level_uniques.max()
wind_level_encoded = np.zeros((num_days, max_wind_levels), dtype="int")
weather_df_4 = weather_df_3.copy()

for (i, day_wind_level) in enumerate(wind_level_int):
    wind_level_encoded[i, :day_wind_level] = 1   
for level in range(max_wind_levels):
    weather_df_4["Wind level > " + str(level)] = wind_level_encoded[:, level]
weather_df_5 = weather_df_4.drop(columns="Wind level")


# You can see the DataFrame with the new encoding below:

# In[66]:


display(weather_df_5)


# Now, we can get to standardization:

# ### Standardization

# Let's make an incremental copy first:

# In[67]:


weather_df_6 = weather_df_5.copy()


# Now, you should use `StandardScaler` from scikit-learn to do standardization. First extract a series containing the `"Temperature"` and `"Humidity"` columns of `weather_df_6` by indexing it with the list `["Temperature", "Humidity"]`. Then, you can create an object of the `StandardScaler` class, naming it `weather_scaler` (remember, you can create an object of class `C` and name it `a` by doing `a = C()`. then, use the `.fit_transform` method of that `weather_scaler` object with `weather_numerical` as input to get the standardized data, calling it `weather_numerical_standardized`. Finally, you can set the `"Temperature"` and `"Humidity"` columns of `weather_df_6` (again by using them in a list and indexing using that list) to be `weather_numerical_standardized`. Do that and you are done with this dataset for now:

# In[70]:


# GRADED: weather_standarization, 1 point(s)

# Enter your code below (4 lines)
### START CODE BELOW THIS LINE ###
weather_numerical = weather_df_6[["Temperature", "Humidity"]]
weather_scaler = StandardScaler()
weather_numerical_standardized = weather_scaler.fit_transform(weather_numerical)
weather_df_6[["Temperature", "Humidity"]] = weather_numerical_standardized
### END CODE ABOVE THIS LINE ###


# ## End-result

# You can see the end result below. The features **Temperature** and **Humidity** should be standardized of course:

# In[71]:


display(weather_df_6)


# # Maintenance logs dataset

# Now, we get to the third dataset, which comtains maintenance logs:

# In[72]:


maintenance_df = pd.read_csv("maintenance_logs.csv")


# ## Inspection

# Let's see the dataset as a table first:

# In[73]:


display(maintenance_df)


# Observations:
# - **Maintenance date** is in three different formats. After inquiry, you are informed that there were three different technicians that wrote this log and each put the date in a different style;
# - **Repair?** indicates whether there was a repair and recalibration at that maintenance date or not. This one has different spellings and typos as well `"repair"`, `"Repair"` and `"repiar"` all mean there was a repair and recalibration at that maintenance date and `"no repair"` and `"No repair"` both indicate a lack of repair and recalibration;
# - If there was a repair done, then the columns **M1 (after)**, **M2 (after)** and **M3 (after)** have values, which are measurements $M_1$, $M_2$ and $M_3$ taken after recalibration, otherwise they are `NaN` ("Not-a-Number");
# - Some values of $M_2$ are missing. Upon close examination, it is visible that happens when there is a certain format to the **Maintenance date** column and that means that one of the technicians forgot to measure $M_2$. Also, after inquiry you find out that the technicians each wrote a separate log and this dataset is their logs merged together and one of the technicians did not record $M_2$ measurements in their maintenance log;
# - You are also informed that in the event of a repair, $M_1$ and $M_2$ after recalibration, i.e., **M1 (after)** and **M2 (after)**, should just be noisy readings of a fixed value. The expert also tells you that principally, $M_3$ should not be affected by repair, i.e., **M3 (before)** should be equal (with some noise) to **M3 (after)**, if there was a repair. That seems to hold;
# - The expert also tells you that there is a strong correlation between values of $M_1$ and $M_2$;
# - The different feature values come in different ranges, so it's a good idea to normalize them;
# - There are duplicate rows. The duplications are artefacts of the improper merging of the logs written by different technicians;

# Every feature except for **Maintenance date** (at column index `0`) seems to be visualizable. So we can from column 1 on:

# In[74]:


visualizable_feature_names_maintenance = maintenance_df.columns[1:]
num_visualizable_features_maintenance = len(visualizable_feature_names_maintenance)


# Let's plot the histograms as well as pairwise scatter plots:

# In[75]:


fig_hist_maintenance = []
for i, feature_name in enumerate(visualizable_feature_names_maintenance):
    fig_hist_maintenance.append(go.Figure(go.Histogram(x=maintenance_df[feature_name])))
    fig_hist_maintenance[i].update_layout(height=400, width=800, title_text=feature_name)
    fig_hist_maintenance[i].show()
    
fig_scatmat_maintenance = go.Figure(data=go.Splom(
                              dimensions=[dict(label=feature, values=maintenance_df[feature]) \
                                  for feature in visualizable_feature_names_maintenance],
                              marker=dict(showscale=False, line_color='white', line_width=0.5)))

fig_scatmat_maintenance.update_layout(title='Pairwise feature scatter plots',                                       width=400 * num_visualizable_features_maintenance,                                       height=400 * num_visualizable_features_maintenance)

fig_scatmat_maintenance.show()


# ## Fixing problems

# Let's fix the problems in the dataset then.

# ### Putting timestamps in the same format

# The first thing we have to do is to put dates in the same format. This will help us when we align the three different datasets in near future and also, every column should be in a uniform format, as a principle:

# In[76]:


maintenance_dates = maintenance_df["Maintenance date"].values.astype("str")
splitted_dates = np.char.split(maintenance_dates, " ")
triletterorder = {"Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06", "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12"}
maintenance_dates_formatted = []
for date in splitted_dates:
    if len(date) == 1:
        maintenance_dates_formatted.append(date[0])
    elif date[0].isnumeric():
        maintenance_dates_formatted.append(date[2] + "-" + triletterorder[date[1]] + "-" + date[0].zfill(2))
    else:
        maintenance_dates_formatted.append(date[2] + "-" + triletterorder[date[0][:3]] + "-" + date[1][:-3].zfill(2))
maintenance_df_2 = maintenance_df.copy()
maintenance_df_2["Maintenance date"] = maintenance_dates_formatted


# This is the result:

# In[77]:


display(maintenance_df_2)


# ### Sorting entries by timestamp

# Now, we can sort the timestamps. This will help us later when we try to align the different datasets as well:

# In[78]:


maintenance_df_3 = maintenance_df_2.sort_values("Maintenance date")
display(maintenance_df_3)


# Now, the duplicate rows are more easy to observe.

# ### Fixing misspellings and alternative formattings

# Let's fix the misspellings and alternative spellings and capitalizations for the values of `"Repair?"` column.
# 
# This is simple: first, find a Boolean series filter that shows rows which have `"Repair"` as their value for the `"Repair?"` column of `maintenance_df_3` and store them in `repair_column_Repair`, a Boolean series filter that shows rows which have `"repiar"` as their value for the same column of that DataFrame, storing them in `repair_column_repiar` and a Boolean series filter of rows which have `No repair"` as their value for the `"Repair?"` column of `maintenance_df_3`, putting them in `repair_column_No_repair`:

# In[79]:


# GRADED: fix_repair_misspell_caps, 1 point(s)

# Enter your code below (3 lines)
### START CODE BELOW THIS LINE ###
repair_column_Repair = (maintenance_df_3["Repair?"] == "Repair")
repair_column_repiar = (maintenance_df_3["Repair?"] == "repiar")
repair_column_No_repair = (maintenance_df_3["Repair?"] == "No repair")
### END CODE ABOVE THIS LINE ###


# Let's make an incremental copy:

# In[80]:


maintenance_df_4 = maintenance_df_3.copy()


# Now, set the value of the portion of `maintenance_df_4` at the intersection of `repair_column_Repair` Boolean series filter and `"Repair?"` column to `"repair"`, the portion of that DataFrame at the intersection of `repair_column_repiar` and `"Repair?"` to `"repair"` as well and set the value of the portion of `maintenance_df_4` at the intersection of `repair_column_No_repair` and `"Repair?"` to `"no repair"`:

# In[81]:


# GRADED: fix_repair_misspell_caps, 1 point(s)

# Enter your code below (3 lines)
### START CODE BELOW THIS LINE ###
maintenance_df_4.loc[repair_column_Repair, "Repair?"] = "repair"
maintenance_df_4.loc[repair_column_repiar, "Repair?"] = "repair"
maintenance_df_4.loc[repair_column_No_repair, "Repair?"] = "no repair"
### END CODE ABOVE THIS LINE ###


# This way we have two values for `"Repair?"` column: `"no repair"` and `"repair"`. You can verify that from the output of the next cell, which should only show those two:

# In[82]:


display(np.unique(maintenance_df_4["Repair?"]))


# ### Handling missing values

# Next up, let's fix the missing values on `"M_2 (before)"` and `"M_2 (after)"`. Although these values happen depending on the value of another column, namely the format of the `"Maintenance date"` from the first DataFrame (which shows which technician did the repair), these values do not have a causal relationship as the technician who examines the device, does not affect the readings of the device. Then, we have missing values which are MCAR. We can use mean (or median or mode or ...) to impute values for $M_2$, however, this might not be the best way. Remember we were informed that there is a strong corrletaion between values of $M_1$ and $M_2$. If we can find a relationship between these values, then maybe we can impute the missing values of $M_2$ from $M_1$ using that relationship. For that, let's plot the two measurements against each other in a plot (from `"M_1 (before)"` and `"M_2 (before)"` columns):

# In[83]:


is_m2_before_missing = (maintenance_df_4["M2 (before)"] == "?")
is_m2_before_available = np.logical_not(is_m2_before_missing)
relationship_data = maintenance_df_4[is_m2_before_available][["M1 (before)", "M2 (before)"]].sort_values("M1 (before)").values.astype("float")
x_relationship, y_relationship = relationship_data[:, 0], relationship_data[:, 1]
fig_relationship = go.Figure(data=go.Scatter(x=x_relationship, y=y_relationship))
fig_relationship.update_layout(title="Relationship between M1 and M2", xaxis_title="M1", yaxis_title="M2")
fig_relationship.show()


# It looks like $M_1$ might have an logarithmic relationship with $M_2$. Let's try to fit a linear regression between $M_1$ and $\log M_2$ and plot:

# In[84]:


y_relationship_log = np.log(y_relationship + 1)

m1_m2_regressor = LinearRegression()
m1_m2_regressor.fit(x_relationship[:, None], y_relationship_log)

y_relationship_predicted = np.exp(m1_m2_regressor.predict(x_relationship[:, None])) - 1
fig_relationship_2 = go.Figure(data=go.Scatter(x=x_relationship, y=y_relationship, name="Actual data"))
fig_relationship_2.add_trace(go.Scatter(x=x_relationship, y=y_relationship_predicted, name="Predicted"))
fig_relationship_2.update_layout(title="Relationship between M1 and M2", xaxis_title="M1", yaxis_title="M2")
fig_relationship_2.show()


# Look like we had the right guess. Now, we can use this relationship to impute values for missing values of $M_2$ by predicting using this linear regressor and using appropriate transformations:

# In[85]:


m2_missing_m1 = maintenance_df_4[is_m2_before_missing]["M1 (before)"].values.astype("float")
m2_missing_values = (np.exp(m1_m2_regressor.predict(m2_missing_m1[:, None])) - 1).astype("int")
maintenance_df_5 = maintenance_df_4.copy()
maintenance_df_5.loc[is_m2_before_missing, "M2 (before)"] = m2_missing_values


# If `"M2 (after)"` had missing values, we could have also used this relationship to predict the missing values for `"M2 (after)"`, using values of `"M1 (after)"`. However, we know that these "after" measurements are just noisy readings of a fixed value, so we might as well just used the average of when that value is available. This is because we had no missing values on `"M2 (after)"`. Howeevr, apparently, that technician (who did not record the $M_2$ measurements) did not happen to do any of the repair and recalibrations.
# 
# Let's see our DataFrame at this stage, then:

# In[86]:


display(maintenance_df_5)


# ### Removing duplicate datapoints

# Let's remove the duplicate entries now. we can use the `.drop_duplicates()` method of a DataFrame (`df_b = df_a.drop_duplicates()` if we have a DataFrame `df_a`, storing the result in `df_b`) to drop any duplicate rows. Use that to drop duplicate rows from `maintenance_df_5`, storing the result in `maintenance_df_6`:

# In[87]:


# GRADED: drop_duplicates, 1 point(s)

# Enter your code below (1 line)
### START CODE BELOW THIS LINE ###
maintenance_df_6 = maintenance_df_5.drop_duplicates()
### END CODE ABOVE THIS LINE ###


# We can see how many rows we had before and after removing duplicates:

# In[88]:


print("Before:", maintenance_df_5.shape[0], "rows. After: ", maintenance_df_6.shape[0], "rows.")


# ### Standardization

# Finally, let's standardize the maintenance log dataset as well. Note that we can use the mean and standard deviation of `"M1 (before)"`, `"M2 (before)"` and `"M3 (before)"` to standardize the non-`NaN` values of `"M1 (after)"`, `"M2 (after)"` and `"M3 (after)"`, respectively, since they are the same measurements taken at different points. So, we calculate the mean and standard deviation for `"M1 (before)"`, `"M2 (before)"` and `"M3 (before)"` and update the non-`NaN` values of `"M1 (before)"`, `"M2 (before)"` and `"M3 (before)"` using:
# 
# $$\frac{\mathbf{c}-\mu_\mathbf{c}}{\sigma_\mathbf{c}}$$
# 
# like we did before. Then, we use scikit-learn's `StandardScaler` to standardize `"M1 (before)"`, `"M2 (before)"` and `"M3 (before)"` themselves:

# In[89]:


maintenance_numerical = maintenance_df_6[["M1 (before)", "M2 (before)", "M3 (before)"]].values.astype("float")

maintenance_numerical_mean = maintenance_numerical.mean(axis=0)
maintenance_numerical_std = maintenance_numerical.std(axis=0)
is_repair = (maintenance_df_6["Repair?"] == "repair")

maintenance_df_7 =  maintenance_df_6.copy()
maintenance_df_7.loc[is_repair, "M1 (after)"] = (maintenance_df_6.loc[is_repair, "M1 (after)"] - maintenance_numerical_mean[0]) / maintenance_numerical_std[0]
maintenance_df_7.loc[is_repair, "M2 (after)"] = (maintenance_df_6.loc[is_repair, "M2 (after)"] - maintenance_numerical_mean[1]) / maintenance_numerical_std[1]
maintenance_df_7.loc[is_repair, "M3 (after)"] = (maintenance_df_6.loc[is_repair, "M3 (after)"] - maintenance_numerical_mean[2]) / maintenance_numerical_std[2]
maintenance_df_7[["M1 (before)", "M2 (before)", "M3 (before)"]] = StandardScaler().fit_transform(maintenance_numerical)


# ## End-result

# ...and this wil be our end result for this dataset:

# In[90]:


display(maintenance_df_7)


# # Aligning and aggregating datasets

# Now, that we have fixed the problems with all three datasets, we can start aligning and aggregating them into one unified dataset.

# ## Inspection

# We know that measurements from the device repeat once every seven days when the device went for maintenance. We can take those measurements and copy them seven times, creating a measurement for every day. However, we mayt be able to do something smarter. We knoe that these measurements are indications of the calibration and internal status of the device and change in a continuous fashion. If we can verify that, then we can approximate the measurements for every day by interpolating the values between two maintenance days. Let's visualize the measurements in time to see if our assumption will be true:

# In[91]:


repetitions = is_repair.astype("int") + 1
num_maintenance_days = maintenance_df_7.shape[0]
x_m = np.repeat(np.arange(num_maintenance_days), repetitions)
y_m1 = np.repeat(maintenance_df_7["M1 (before)"].values, repetitions)
y_m2 = np.repeat(maintenance_df_7["M2 (before)"].values, repetitions)
y_m3 = np.repeat(maintenance_df_7["M3 (before)"].values, repetitions)
y_r = np.repeat(np.zeros((num_maintenance_days,)), repetitions)
repair_indices = (repetitions.cumsum() - 1)[is_repair].values
y_m1[repair_indices] = maintenance_df_7.loc[is_repair, "M1 (after)"].values
y_m2[repair_indices] = maintenance_df_7.loc[is_repair, "M2 (after)"].values
y_m3[repair_indices] = maintenance_df_7.loc[is_repair, "M3 (after)"].values
y_r[repair_indices] = 1
fig_m1 = go.Figure(data=go.Scatter(x=x_m, y=y_m1, name="M1"))
fig_m1.add_trace(go.Scatter(x=x_m, y=y_r, name="Repair"))
fig_m1.update_layout(title="M1 over maintenance days", xaxis_title="Maintenance day index", yaxis_title="M1")
fig_m1.show()
fig_m2 = go.Figure(data=go.Scatter(x=x_m, y=y_m2, name="M2"))
fig_m2.add_trace(go.Scatter(x=x_m, y=y_r, name="Repair"))
fig_m2.update_layout(title="M2 over maintenance days", xaxis_title="Maintenance day index", yaxis_title="M2")
fig_m2.show()
fig_m3 = go.Figure(data=go.Scatter(x=x_m, y=y_m3, name="M3"))
fig_m3.add_trace(go.Scatter(x=x_m, y=y_r, name="Repair"))
fig_m3.update_layout(title="M3 over maintenance days", xaxis_title="Maintenance day index", yaxis_title="M3")
fig_m3.show()


# Our guess was correct and all three, $M_1$, $M_2$ and $M_3$ seem to be almost continuous with $M_1$ and $M_2$ having reset-type discontinuities at maintenance days when repair and recalibration happened.

# ## Interpolating data and first alignment and aggregation

# Now, we know we can interpolate to approximate measurements for each day. These approximates will be:
# $$\frac{\left[\left(6-(d-1)\right)\,M_\mathrm{before}\right]+\left[(d-1)\,M_\mathrm{after}\right]}{6}$$
# where $d$ is the number of days from last repair and runs from $1$ to $6$ for operation days (it will be $7$ for a maintenance day). $M_\mathrm{before}$ is any of the $M_1$, $M_2$ or $M_3$ measurements (same as the measurement we want to approximate) at the previous maintenance day: it will be `"M1 (before)"`, `"M2 (before)"` and `"M3 (before)"`  at that day, if there was no repair and recalibration and `"M1 (after)"`, `"M2 (after)"` and `"M3 (after)"` at that day if there was a repair and recalibration. $M_\mathrm{before}$ the appropriate measurement at the next maintenance day: it will be `"M1 (before)"`, `"M2 (before)"` and `"M3 (before)"` at that day. In the first few days when there is no previous maintenance, we can use the value from the first maintenance day (this gives us fixed values equal to the measurements at the first maintenance day for those days). Symmetrically, in the last few days when there is no next maintenance day,  we can use the value from the last maintenance day (this will also give us fixed values equal to the measurements at the last maintenance day for those days).
# 
# We camn then add these values to a copy of the average daily weather dataset to unify the average daily weather and maintenance logs datasets:

# In[92]:


m1_prev = maintenance_df_7["M1 (before)"].values.astype("float")
m1_prev[is_repair] = maintenance_df_7.loc[is_repair, "M1 (after)"].values.astype("float")
m1_next = maintenance_df_7["M1 (before)"].values.astype("float")
m2_prev = maintenance_df_7["M2 (before)"].values.astype("float")
m2_prev[is_repair] = maintenance_df_7.loc[is_repair, "M2 (after)"].values.astype("float")
m2_next = maintenance_df_7["M2 (before)"].values.astype("float")
m3_prev = maintenance_df_7["M3 (before)"].values.astype("float")
m3_prev[is_repair] = maintenance_df_7.loc[is_repair, "M3 (after)"].values.astype("float")
m3_next = maintenance_df_7["M3 (before)"].values.astype("float")

weather_maintenance_df = weather_df_6.copy()
num_days = weather_maintenance_df.shape[0]

m1 = np.zeros((num_days,))
m2 = np.zeros((num_days,))
m3 = np.zeros((num_days,))

maintenance_dates = maintenance_df_7["Maintenance date"].values.astype("str")
num_maintenance_days = maintenance_dates.shape[0]

previous_maintenance_index = 0
next_maintenance_index = 0
d = 1
for i, date in enumerate(weather_maintenance_df["Date"].values):
    if np.datetime64(maintenance_dates[next_maintenance_index]) < np.datetime64(date):
        next_maintenance_index += 1
        previous_maintenance_index = next_maintenance_index - 1
        d = 1
    else:
        d += 1
    if next_maintenance_index == num_maintenance_days:
        next_maintenance_index -= 1
    previous_maintenance_date = np.datetime64(maintenance_dates[previous_maintenance_index])
    next_maintenance_date = np.datetime64(maintenance_dates[next_maintenance_index])
    m1[i] = (((7 - d) * m1_prev[previous_maintenance_index]) + ((d - 1) * m1_next[next_maintenance_index])) / 6
    m2[i] = (((7 - d) * m2_prev[previous_maintenance_index]) + ((d - 1) * m2_next[next_maintenance_index])) / 6
    m3[i] = (((7 - d) * m3_prev[previous_maintenance_index]) + ((d - 1) * m3_next[next_maintenance_index])) / 6
weather_maintenance_df["M1"] = m1
weather_maintenance_df["M2"] = m2
weather_maintenance_df["M3"] = m3
display(weather_maintenance_df)


# Now, let's align and aggregate this unfied dataset we got from aligning and aggregating average daily weather and maintenance logs datasets with the events dataset.

# ## Alignment and aggregation

# ### Reformatting timestamps

# The date and time of events is not a useful feature that is relevant to our classification. So, in the end, we will not need it. However, we will need dates to align the two datasets for aggregation. Times (of day), however, are entirely not useful (since they won't be used in aligning datasets) In the first step, we have to reformat the timestamps of the events dataset to contain ony the dates and not times:

# In[93]:


dates = []
for datetime in events_df_11["Date and time"].values.astype("str"):
    dates.append(datetime[:10])
events_df_12 = events_df_11.drop(columns="Date and time")
events_df_12["Date"] = dates
display(events_df_12)


# Let's take a look at the result:

# In[94]:


display(events_df_12)


# ### Aligning and aggregating the data

# We can start aligning and aggregating. We will store the appropriate readings from the combined average daily weather and maintenance logs datasets for each row in the events dataset in lists:

# In[95]:


final_temperature = []
final_humidity = []
final_wind_level_lt_0 = []
final_wind_level_lt_1 = []
final_wind_level_lt_2 = []
final_m1 = []
final_m2 = []
final_m3 = []
final_df = events_df_12.copy()


# Now, you should have:
# - A for loop to go through dates in `final_df` converted to a string-valued NumPy array, stroring each date in variable `date` (if `"Name"` is a column of DataFrame `df` a string-valued NumPy array containing the values of that column can be obtained by `df["Name"].values.astype("str")`). In there:
#     - Extract a Boolean series of that `date` in the `"Date"` column of `weather_maintenance_df` DataFrame. There will be a single `True` value since we have only and exactly one entry for each operation day in that DataFrame;
#     - Append, to `final_temperature` list, the protion of the `weather_maintenance_df` at the intersection of `weather_maintenance_index` Boolean series (which will act as a filter) and `"Temperature"` column (using `.loc`), converted to a NumPy array and the first element extracted (by chaining a `.values[0]` to the end of your call, since there is always a single `True` in `weather_maintenance_index`, that first element is unique and exactly what we want). You can append `a` to a list `l` by doing `l.append(a)`;
#     - Repeat this for `final_humidity`, `final_wind_level_lt_0`, `final_wind_level_lt_1`, `final_wind_level_lt_2`, `final_m1`, `final_m2` and `final_m3` for columns `"Humidity"`, `"Wind level > 0"`, `"Wind level > 1"`, `"Wind level > 2"`, `"M1"`, `"M2"` and `"M3"` of `weather_maintenance_df` (in conjunction with `weather_maintenance_index`).
# - Add a new column `"Temperature"` to `final_df` having values in `final_temperature`, respectively;
# - Repeat the above for `"Humidity"`, `"Wind level > 0"`, `"Wind level > 1"`, `"Wind level > 2"`, `"M1"`, `"M2"` and `"M3"` in `final_df` with values of `final_humidity`, `final_wind_level_lt_0`, `final_wind_level_lt_1`, `final_wind_level_lt_2`, `final_m1`, `final_m2` and `final_m3`, respectively.

# In[ ]:


# GRADED: final_alignment, 1 point(s)

# Enter your code below (18 lines)
### START CODE BELOW THIS LINE ###
final_df["Temperature"] = final_temperature
### END CODE ABOVE THIS LINE ###


# We have successfully aligned and aggregated the dataset. We need two final touches and we are done.

# ## Removing extra columns

# We don't need the `"Date"` feature, so let's drop it:

# In[ ]:


final_df = final_df.drop(columns="Date")


# ## Rearranging columns

# Let's rearrange the columns to have `"Label"` appear as the last column:

# In[ ]:


final_column_order = list(final_df.columns)
final_column_order.remove("Label")
final_column_order.append("Label")
final_df = final_df[final_column_order]


# ## The final aggregated dataset

# Now, we have our final dataset, aligned and aggregated and ready to use:

# In[ ]:


display(final_df)


# # Saving the Result

# Finally, let's save our dataset to a CSV file, so we can use it later on!

# In[ ]:


final_df.to_csv("final_data.csv", index=False)


# We are done! Do not forget to push the blue 'Submit Assignment' on the header of this Jupyter notebbok to submit your assignment!
