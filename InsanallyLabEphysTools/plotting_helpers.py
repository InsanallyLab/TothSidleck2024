import numpy as np
from scipy.stats import gaussian_kde
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

from .utility import getSpikeTimes, getTrialSpikes

TARGET_COLOR = '#ff3654'
NONTARGET_COLOR = '#5776ff'

CR_COLOR = '#808080'
NCR_COLOR = '#FF767C'
CNO_COLOR = '#00BFFF'
NCR_CR_cmap = mpl.colors.LinearSegmentedColormap.from_list('NCR-CR color map', [NCR_COLOR, CR_COLOR], N=1000)

#Make this propagate nans for better utility
def violin(X,Y=None,width = 1,log=False,discrete=False,logzeroval=0.1,minjitter=0):
    width -= minjitter

    if Y is None:
        Y = X
        X = np.zeros_like(Y)

    X = np.array(X,dtype='float')
    Y = np.array(Y)

    if log:
        Y[np.equal(Y,0)]+=logzeroval

    Xs = np.unique(X)
    finite_idxs = np.isfinite(Y)
    for x in Xs:
        equal_idxs = np.equal(X,x)
        idxs = np.logical_and(equal_idxs,finite_idxs)

        if len(np.unique(Y[idxs]))==1:
            X[idxs] = np.linspace(x-width/2,x+width/2,np.sum(idxs))
            continue

        Ys = Y[idxs]
        if len(Ys) == 0:
            continue

        if discrete:
            Ys = np.unique(Ys)
            max_y = np.max([np.sum(np.equal(Y,y)) for y in Ys])
            for y in Ys:
                y_idxs = np.equal(Y,y)
                this_num_y = np.sum(y_idxs)
                if this_num_y == 1:
                    X[y_idxs] = np.zeros(this_num_y)
                else:
                    X[y_idxs] = np.linspace(-width/2*this_num_y/max_y,width/2*this_num_y/max_y,this_num_y)
        else:
            if log:
                KDE = gaussian_kde(np.log10(Ys), bw_method='scott')
                KDE_eval = KDE.evaluate(np.log10(Ys))
            else:
                KDE = gaussian_kde(Ys, bw_method='scott')
                KDE_eval = KDE.evaluate(Ys)

            X_plot = (KDE_eval / np.max(KDE_eval) * 0.8) * width * (np.random.rand(len(Ys))-0.5) + x
            X_plot += (np.random.rand(len(Ys))-0.5)*2 * minjitter
            
            X[idxs] = X_plot

    #Propagate nans
    nonfinite_idxs = np.logical_not(finite_idxs)
    X[nonfinite_idxs] = np.full(np.sum(nonfinite_idxs),np.nan)

    return X,Y

def visualizeCluster(sessionfile,clust):

    starttime = -250
    endtime = 1500
    bw_scott_factor = 3

    buffer = 500
    bufferedstart = starttime - buffer
    bufferedend = endtime + buffer
    startbuffer = (0-bufferedstart)/1000
    endbuffer = (bufferedend-2500)/1000 #NOTE THAT THIS IS CURRENTLY A HARDCODED VALUE!!!

    #########################################################################################

    region = sessionfile.meta.region

    #########################################################################################

    fig = plt.figure(figsize=(4,6))
    gs = fig.add_gridspec(2,1,height_ratios=[2,1],hspace=0.0125)#,wspace=0.5)#0.1)
    ax1 = plt.subplot(gs[0,0])#Raster
    ax2 = plt.subplot(gs[1,0],sharex=ax1)#PSTH
    ax1.spines['bottom'].set_visible(False)
    ax1.spines['left'].set_visible(False)
    ax1.get_xaxis().set_visible(False)
    ax1.get_yaxis().set_visible(False)

    FRmod = sessionfile.responsiveness[clust]['all_trials'].FRmodulation
    FRmod_color = NCR_CR_cmap( (FRmod-1)/5)

    trialsToPlot = sessionfile.trim[clust].trimmed_trials
    clustSpikes = getSpikeTimes(sessionfile,clust=clust)
    trialSpikes = []
    for trial in trialsToPlot:
        this_trial_spikes = getTrialSpikes(sessionfile,trial,cachedtimes=clustSpikes,outunits='ms',startbuffer=startbuffer,endbuffer=endbuffer)
        trialSpikes.append(this_trial_spikes)
    trialSpikes = np.array(trialSpikes,dtype='object')

    #Plot Raster
    for idx,trial in enumerate(trialsToPlot):
        ax1.scatter(trialSpikes[idx],np.ones_like(trialSpikes[idx])*idx,color=FRmod_color,s=1.5,marker='o',lw=0)

    if sessionfile.meta.task == 'switch':
        try:
            switch_point = np.where(np.greater(trialsToPlot,200))[0][0]-0.5
            ax1.axhline(switch_point,linestyle='--',lw=1,zorder=-10,color='k')
        except:
            pass

    #Plot PSTH
    PSTHstart = starttime - 0.1
    PSTHend = endtime + 0.1
    xrange = np.linspace(starttime,endtime,num=1000)
    FR = np.full((10,len(xrange)),np.nan)
    kf = KFold(n_splits=10)
    for idx,(train_index,_) in enumerate(kf.split(trialSpikes)):
        PSTHspikes = np.concatenate(trialSpikes[train_index])
        bw = len(PSTHspikes)**(-1./5) / bw_scott_factor
        KDE = gaussian_kde(PSTHspikes,bw_method=bw)
        FR[idx,:] = KDE.evaluate(xrange)
        FR[idx,:] = FR[idx,:] * len(PSTHspikes) / len(train_index) * 1000 #1000 is conversion to s/s because units are all in ms for this plot
    avg_FR = np.mean(FR,axis=0)
    sem_FR = np.std(FR,axis=0)

    ax2.plot(xrange,avg_FR,lw=1,color=FRmod_color)
    ax2.fill_between(xrange,avg_FR-sem_FR,avg_FR+sem_FR,color=FRmod_color,alpha=0.5,zorder=-10,lw=0)

    max_FR_value = np.max(avg_FR+sem_FR) * 1.1
    sequence_of_possible_axis_limits = np.array([1,2,4,6,8,10,12,16,20,30,40,50,60,80,100,120,140,150,160,180,200])
    idx_of_okay_limits = np.greater_equal(sequence_of_possible_axis_limits,max_FR_value)
    valid_limits = sequence_of_possible_axis_limits[idx_of_okay_limits]
    limit = np.min(valid_limits)
    ax2.set_ylim([0,limit])
    ax2.set_yticks([0,limit/2,limit])
    ax2.set_yticklabels(['0',str(int(limit/2)),str(limit)])
    ax2.set_ylabel('Firing rate (spikes/s)')

    ax2.plot([0,100],[limit,limit],color='k',lw=3,linestyle='-',zorder=10)

    ax2.set_xlim([starttime,endtime])
    ax2.set_xticks([0,500,1000,1500])
    ax2.set_xticklabels(['0','0.5','1','1.5'])
    ax2.set_xlabel('Time (s)')

    return fig,[ax1,ax2]


def getPrePostLabelYval(ylims,log=False,Yfrac=-0.12498763228188259):
    Yfrac
    
    ymin = ylims[0]
    ymax = ylims[1]
    if log:
        ymin = np.log10(ymin)
        ymax = np.log10(ymax)
        
    YvalRelativeToAxes = Yfrac * (ymax-ymin)
    YvalRelativeToUnits = YvalRelativeToAxes + ymin
    
    if log:
        YvalRelativeToUnits = 10**YvalRelativeToUnits
        
    return YvalRelativeToUnits

def add_axis_size(fig, ax_w, ax_h, left, bottom):
    fig_w, fig_h = fig.get_size_inches()
    ax = fig.add_axes([left/fig_w, bottom/fig_h, ax_w/fig_w, ax_h/fig_h])
    return ax

def make_axis_size(ax_w, ax_h, left=.3, bottom=.3, right=0, top=0):
    fig_w = (ax_w + left + right) * 1.05
    fig_h = (ax_h + bottom + top) * 1.05
    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = add_axis_size(fig, ax_w, ax_h, left, bottom)
    return fig, ax