
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import mean_absolute_error
from .util import get_mae



def plot_mae_gbr(grid_cv, d, CV=5, y_mean=None, y_scaled=False, y_scaler=None,
                 limit=[-1, 7], size=12, data_type='bandgap', legend_size=22):

    """
    Plot the results from GridSearchCV
    The Score CV is from the mean of CV test scores
    Args:
        grid_cv(sklearn.model_selection.GridSearchCV)
        d(dict): data set of label ['x_Train', 'x_test', 'y_Train', 'y_test']
        CV(int): CV fold, should be the same as used in the grid CV process
        limit(list): indicates the limit of x and y axis
        size(int): size of the figure
        data_type(str):'bandgap','Ef'
        legend_size(int): fontsize of the legend

    Return:
        plt(matplotlib.pyplot.plot object)
        df(pd.DataFrame): the dataframe of input data. For debug use
        types(pd.Series): the labels of data. For debug use


    """
    # LABELS
    xlabel = r'$BandGap^{DFT}\ \mathrm{(eV)}$'
    ylabel = r'$BandGap^{GBR}\ \mathrm{(eV)}$'


    # data prep
    data_labels = ['x_Train', 'x_test', 'y_Train', 'y_test']
    x_train, x_test, y_train, y_test = [d[k] for k in data_labels]
    ## scores

    if y_scaled:
        score_test = get_mae(grid_cv, x_test, y_test, y_scaler)
        score_cv = get_mae(grid_cv, x_train, y_train, y_scaler)
    else:
        score_test = mean_absolute_error(grid_cv.predict(x_test), y_test)
        # This is average score of all validation sets
        score_cv = -np.mean([grid_cv.cv_results_['split{}_test_score'.format(cv)][grid_cv.best_index_] \
                             for cv in range(0, CV)])

    df1 = pd.DataFrame({"pred_test" :[x for x in grid_cv.predict(x_test)],
                        "dft_test": np.array(y_test)})
    #     df2 = pd.DataFrame({"pred_valid":[x for x in gbr.predict(x_valid)],
    #                             "dft_valid":np.array(y_valid)})
    df3 = pd.DataFrame({"pred_train" :[x for x in grid_cv.predict(x_train)],
                        "dft_train" :np.array(y_train)})


    #     pred = df1.pred_test.append(df2.pred_valid).append(df3.pred_train)
    #     dft = df1.dft_test.append(df2.dft_valid).append(df3.dft_train)
    #     types = pd.Series([0]*len(x_test) + [3]*len(x_valid) + [2]*len(x_train),index=pred.index)


    pred = df1.pred_test.append(df3.pred_train)
    dft = df1.dft_test.append(df3.dft_train)
    types = pd.Series([0 ] *len(x_test) + [2 ] *len(x_train) ,index=pred.index)
    df = pd.DataFrame({ylabel :pred, xlabel :dft, 'type' :types})


    colors = sns.color_palette("Set2")
    #     colors = [colors[0], colors[2]]
    cmap = ListedColormap(sns.color_palette([colors[0], colors[2]]).as_hex())

    sns.set_style('white' ,rc={"xlabelsize" :25})


    # Plot data onto seaborn JointGrid
    g = sns.JointGrid(xlabel, ylabel, data=df, ratio=2, size=size)
    g = g.plot_joint(plt.scatter, c=df['type'], edgecolor="black", cmap=cmap)
    # plot the y=x line
    g.ax_joint.plot(np.linspace(-1000 ,1000, 100),
                    np.linspace(-1000 ,1000, 100),
                    'k--' ,linewidth=1.0 ,label=r'$y=x$')

    # #set x y limit
    g.ax_joint.set_xlim(limit[0] ,limit[1])
    g.ax_joint.set_ylim(limit[0] ,limit[1])


    # Loop through unique categories and plot individual kdes
    #     text = {0: 'Test', 3: 'Validation',2: 'Training'}
    text = {0: 'Test', 2: 'Training'}
    for c in df['type'].unique():
        sns.kdeplot(df[xlabel][df['type' ] == c], ax=g.ax_marg_x, vertical=False,
                    color=colors[c], shade=True ,label=text[c], linewidth=3)
        sns.kdeplot(df[ylabel][df['type' ] == c], ax=g.ax_marg_y, vertical=True,
                    color=colors[c], shade=True ,legend = False, linewidth=3)

    # set the size of kdeplot
    # pos = [left, bottom, width, height]
    pos_x = g.ax_marg_x.get_position()
    g.ax_marg_x.set_position([pos_x.x0, pos_x.y0 - 0.05, pos_x.width, pos_x.height /2])
    pos_y = g.ax_marg_y.get_position()
    g.ax_marg_y.set_position([pos_y.x0 -0.05, pos_y.y0, pos_y.width /2, pos_y.height])

    # manually add legend for scatter plot
    simArtist = plt.Line2D((0 ,1) ,(0 ,0), color=colors[0], marker='o', linestyle='')  # Test
    #     simArtist2 = plt.Line2D((0,1),(0,0), color=colors[3], marker='o', linestyle='') # Valid
    simArtist3 = plt.Line2D((0 ,1) ,(0 ,0), color=colors[2], marker='o', linestyle='') # Training
    handles ,labels = g.ax_joint.get_legend_handles_labels() ## orginal handles and labels

    g.ax_joint.legend(handles +[simArtist ,simArtist3],
                      labels +['Test Score: MAE \n%.2f  ' %(score_test ) +'eV',
                              #                               'Validation: MAE \n%.2f '%(score_vali_valid)+'eV',
                              'CV Score: MAE \n%.2f  ' %(score_cv ) +'eV'],
                      loc=2,
                      prop={'family' :'Helvetica' ,'size' :legend_size})
    #                       prop={'family':'sans-serif','size':25})

    # set label ,tick size
    ticksize = int(size * 2.5)
    g.ax_joint.tick_params(labelsize = ticksize ,size=12 ,direction='in')

    g.ax_joint.set_xlabel(xlabel ,fontsize = 40)
    g.ax_joint.set_ylabel(ylabel ,fontsize = 40)
    g.ax_marg_x.legend(fontsize=28, bbox_to_anchor=(0.75, 1), loc=2, borderaxespad=0.)

    g.ax_joint.tick_params('y' ,pad=10)
    g.ax_joint.tick_params('x' ,pad=10)

    return plt, df, types