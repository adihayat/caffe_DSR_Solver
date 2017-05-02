#import check_imports
import argparse
import re
from matplotlib import pyplot as plt
import numpy as np


GrebberDict = {       # grepping_phrase                                 # transform     #is_filter  #is_dot
        'iter'              : ('.*Iteration (\d+), lr = \S+.*',                  lambda x: x,    False ,    '-'),
        'loss'              : ('.*Iteration \d+.* loss = (\S+).*',             lambda x : x,   True  ,    '-'),
        'lr'                : ('.*Iteration \d+, lr = (\S+).*',                  lambda x : x ,  False ,    '-'),
        '1-acc_test'        : ('.*Test net output #\d+: Accuracy = (\S+).*',     lambda x : 1-x, True  ,    '-'),
        '1-acc_train'       : ('.*Train net output #\d+: Accuracy = (\S+).*',    lambda x : 1-x, True  ,    '-'),
        '1-map'             : ('.*detection_eval = (\S+).*',                     lambda x : 1-x, False ,    'o'),
        '1-dtest'           : ('.*Test Accuracy (\S+) on.*',                     lambda x : 1-x, False ,    'o'),
        'dloss'             : ('.*, dloss = (\S+), .*',                             lambda x : x  , True ,     '-'),
        'gloss'             : ('.*, gloss = (\S+), .*',                             lambda x : x  , True ,     '-'),
        'net_loss'          : ('.*, Dout = (\S+), .*',                           lambda x : x  , True ,     '-'),
        '1-avg_acc_per_cls' : ('.*avg_accuracy_per_class=(\S+).*',               lambda x : 1-x  , False , 'o'),
        }

def main(arg,initialized_curves=False,fig=None,axarr=None,found_curves=None,block=True,first=True):
    try:
        if arg.plot=="all":
            what_to_plot = [c for c in GrebberDict.keys() if c !='iter']
        else:
            what_to_plot = arg.plot.split(',')


        for log_file in arg.train_log_file:
            with open(log_file,'r') as f:
                log_lines = f.readlines()

            curves = [[] for c in xrange(len(what_to_plot)+1)]

            for line in log_lines:
                matchObj = re.match( GrebberDict['iter'][0] , line, re.M | re.I)
                if matchObj:
                    curves[0].append(int(matchObj.group(1)))


            for c in xrange(len(what_to_plot)):
                for line in log_lines:
                    matchObj = re.match( GrebberDict[what_to_plot[c]][0] , line, re.M | re.I)
                    if matchObj:
                        curves[c+1].append(GrebberDict[what_to_plot[c]][1](float(matchObj.group(1))))

            if not initialized_curves:
                found_curves = [len(c) > 0 for c in curves]

            if len(curves[0])==0:
                print "unable to find measurments for \"iter\" with {}".format(GrebberDict[what_to_plot[0]][0])
                return


            for c in xrange(len(what_to_plot)):
                if not found_curves[c+1]:
                    continue

                if GrebberDict[what_to_plot[c]][2]:
                   curves[c+1] = np.convolve(np.array(curves[c+1]), np.ones((arg.avg_size,), dtype=np.float32)/arg.avg_size, mode='valid')
                else:
                   curves[c+1] = np.array(curves[c+1])


            if not initialized_curves:
                fig , axarr = plt.subplots(reduce(lambda x,y : x + y,found_curves[1::]),sharex=True,figsize=(12,8),frameon=False)
                if len(what_to_plot)==1:
                    axarr = [axarr]

            ax_counter = 0
            for c in xrange(len(what_to_plot)):
                if not found_curves[c+1]:
                    continue

                marker = GrebberDict[what_to_plot[c]][3]
                curv_min = np.min(curves[c+1])
                curv_max = np.max(curves[c+1])

                current_y_middle = (curv_min + curv_max)/2
                delta = 0 if curv_max > curv_min else abs(curv_max)
                current_y_bottom = curv_min - (delta + current_y_middle - curv_min)*0.15
                current_y_top    = curv_max + (delta + curv_max -  current_y_middle)*0.15
                if initialized_curves:
                    yrange = (np.min([current_y_bottom,axarr[ax_counter].get_ylim()[0]]) ,  np.max([current_y_top,axarr[ax_counter].get_ylim()[1]]) )
                else:
                    yrange = ( current_y_bottom,  current_y_top )

                axarr[ax_counter].semilogy(np.linspace(curves[0][0],curves[0][-1],curves[c+1].shape[0]), curves[c+1], marker ,linewidth=1.0, label=log_file )
                axarr[ax_counter].set_xlabel('iters')
                axarr[ax_counter].set_ylabel(what_to_plot[c])
                axarr[ax_counter].set_yscale('log', basey=1.05)
                if first:
                    axarr[ax_counter].set_ylim(yrange)
                axarr[ax_counter].grid(True,which='both')
                axarr[ax_counter].legend(loc="upper right",  ncol=2, shadow=True, title="Legend", fancybox=True, prop={'size':6})
                ax_counter+=1

            initialized_curves = True

        plt.legend(loc="upper right",  ncol=2, shadow=True, title="Legend", fancybox=True, prop={'size':6})

        plt.grid(True)
        if block:
            plt.show()
        else:
            plt.pause(4)
            for ax in axarr:
                ylim = ax.get_ylim()
                xlim = ax.get_xlim()
                ax.clear()
                ax.set_ylim(ylim)
                ax.set_xlim(xlim)

        return fig, axarr , found_curves

    except Exception as E:
        print "ERROR {}".format(E)

def get_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--avg_size",
                    default=25,
                    type=int,
                    help= "moving average size")

    parser.add_argument("--no_refresh",
                    action='store_true',
                    help= "moving average size")

    parser.add_argument("--plot",'-p',
                        default="all",
                        help= "what to plot as defined by {}\t"
                              "\"all\" prints all curves".format(GrebberDict.keys()))

    parser.add_argument("train_log_file",
            help= "logs of training list i.e.:  train1.log train2.log ...",
            nargs="+")


    return parser


if __name__ == "__main__":

    args = get_args().parse_args()

    if not args.no_refresh:
        plt.ion()
        f, axarr , found_curves = main(args,block=False)
        while True:
            main(args,True,f,axarr,found_curves,False,False)
    else :
        main(args)

