import glob
import os
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from ipdb import set_trace
from numpy.random import uniform
matplotlib.rcParams.update({'font.size': 12})
# plt.rcParams["font.family"] = "Times New Roman"
RANDOM_COLOR = False
@click.command()
@click.option('--result', type=str, default='test/success_rate', help='epoch,stats_g/mean,stats_g/std,stats_o/mean,stats_o/std,test/episode,test/mean_Q,test/success_rate,train/episode,train/success_rate')
@click.option('--log_folder', type=str, default = '/home/bourne/log_data/her/', help='the log_path you use in baselines.run commamd')
@click.option('--precentile', type=list, default=[25,50,75], help='the precent you want to use to compare')
@click.option('--n_epoch', type=int, default=200, help='how many epoch you want to see from 0~n_epoch')
@click.option('--title', type=str, default='FetchExperiments', help='tile of the table you plotting')
@click.option('--random_color', type=str, default=False, help=' \'--random_color=True\' for use random color,  \'--random_color=False\' for setup color')
@click.option('--n_win', type=int, default=0, help=' \'--n_win=n\' using n width window to smooth the data')

def plot(result, log_folder, precentile, n_epoch, title, random_color, n_win):

    colours = ['g','black','r','b','gold','c', 'm']

    for exp_type_folder in os.listdir(log_folder):
        exp_type_name = exp_type_folder
        exp_type_folder = os.path.join(log_folder,exp_type_folder)
        fig = plt.figure(figsize=(32,18))
        ax = fig.add_subplot(1, 1, 1)

        # Major ticks every 20, minor ticks every 5
        major_ticks = np.arange(0, n_epoch, 20)
        minor_ticks = np.arange(0, n_epoch, 5)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        # And a corresponding grid
        ax.grid(which='both')
        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

        # Sorting the files in the list: 0,1,2,3..., for plotting the label legends in right orders.
        sorting_folder_list = os.listdir(exp_type_folder)
        sorted_folder_list = []
        for i in range(len(sorting_folder_list)):
            for sorting_folder in sorting_folder_list:
                if i == int(sorting_folder[-4:-3]):
                    sorted_folder_list.append(sorting_folder)

        for sub_exp_type_folder in sorted_folder_list:
            sub_exp_tpye_name = sub_exp_type_folder
            sub_exp_type_folder = os.path.join(exp_type_folder,sub_exp_type_folder)
            num_of_this_type_exps = 0
            all_sub_exps = []
            for sub_exp_folder in os.listdir(sub_exp_type_folder): # fetchpickandplace_2exp
                sub_exp_folder  = os.path.join(sub_exp_type_folder, sub_exp_folder)
                csv = pd.read_csv(sub_exp_folder+'/progress.csv', skipinitialspace=True)
                exp = csv[result]._values
                exp = exp[0:n_epoch] # useful for align, bc some experiments didn't finish all epochs
                all_sub_exps.append(exp)
                num_of_this_type_exps +=1
            print('Exp_Name: '+sub_exp_tpye_name+'\nTotal_sub_experiments={}'.format(num_of_this_type_exps))

            value_mean = np.mean(all_sub_exps, axis=0)
            if n_win != 0:
                smoothed_value_mean = np_move_avg(value_mean,n_win)
            value_up = value_mean + np.std(all_sub_exps, axis=0)
            value_down = value_mean - np.std(all_sub_exps, axis=0)
            # value_down, value_media, value_up = np.percentile(all_sub_exps, precentile, axis=0)
            if random_color:
                
                colour = (uniform(0,1),uniform(0,1),uniform(0,1))
            else:
                colour = colours.pop()

            if n_win == 0:
                # do not smooth data
                plt.plot(np.arange(len(value_mean)), value_mean, color=colour , label=sub_exp_tpye_name, linewidth=4)
                plt.fill_between(np.arange(len(value_mean)),value_up,value_down,color=colour,alpha = 0.2)
            else:
                # smooth data
                plt.plot(np.arange(len(smoothed_value_mean)), smoothed_value_mean, color=colour , label=sub_exp_tpye_name, linewidth=4)
                plt.fill_between(np.arange(len(value_mean)),value_up,value_down,color=colour,alpha = 0.2)

        font1 = {
        'weight' : 'normal',
        'size'   : 20,
        }
        
        y_matrix = filter_special_character(result)
        filtered_exp_type_name = filter_special_character(exp_type_name)

        plt.legend(prop=font1, loc = 4)
        plt.xticks(fontsize=30)
        plt.yticks(fontsize=30)
        plt.suptitle(((filtered_exp_type_name+' smooth_'+str(n_win))), fontsize=30)
        ax.set_xlabel('Epoch', fontsize=30)
        ax.set_ylabel(y_matrix, fontsize=30)
        plt.savefig(filtered_exp_type_name+' '+y_matrix+' smooth_'+str(n_win)+'.png')
        plt.show()

def np_move_avg(input_data,n_width,mode="same"):
    smoothed_data = np.convolve(input_data, np.ones((n_width,))/n_width, mode=mode)
    # the last n data will smooth wrongly, so here just to substitude the last n data with original data.
    start_index = len(input_data)-n_width-1
    for n in range(n_width):
        averaged_data = []
        substituded_index = (len(input_data)-n-1)
        for data in input_data[start_index:substituded_index]:
            averaged_data.append(data)
        smoothed_data[substituded_index] = np.mean(averaged_data)
    return smoothed_data

def filter_special_character(str_name):
    str_name = eval((repr(str_name).replace('/', ' ')))
    str_name = eval((repr(str_name).replace('_', ' ')))
    return str_name

if __name__ == '__main__':
    plot()

    # result_class = [
    #     'epoch',
    #     'stats_g/mean',
    #     'stats_g/std',
    #     'stats_o/mean',
    #     'stats_o/std',
    #     'test/episode',
    #     'test/mean_Q',
    #     'test/success_rate',
    #     'train/episode',
    #     'train/success_rate'
    #     ]
