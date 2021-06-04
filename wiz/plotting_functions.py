import numpy as np
import matplotlib.colors as mcolors
import itertools


def get_contours(CT0):
    if CT0==0.95:
        levelsLines=np.sort(np.array([0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99,1.,1.01]))
    else:
        levelsLines=np.sort(np.array([0.7,0.8,0.85,0.9,0.95,0.98,0.99,0.995,1.01]))
    levelsContour = np.sort(np.concatenate((levelsLines,[1.4,0.2])))
    return levelsContour, levelsLines


# ---- ColorMap
def make_colormap(seq,values=None):
    """Return a LinearSegmentedColormap
    seq: RGB-tuples. 
    values: corresponding values (location betwen 0 and 1)
    """
    hasAlpha=len(seq[0])==4
    if hasAlpha:
        nComp=4
    else:
        nComp=3

    n=len(seq)
    if values is None:
        values=np.linspace(0,1,n)

    doubled     = list(itertools.chain.from_iterable(itertools.repeat(s, 2) for s in seq))
    doubled[0]  = (None,)* nComp
    doubled[-1] = (None,)* nComp
    cdict = {'red': [], 'green': [], 'blue': [], 'alpha':[]}
    for i,v in enumerate(values):
        if hasAlpha:
            r1, g1, b1, a1 = doubled[2*i]
            r2, g2, b2, a2 = doubled[2*i + 1]
        else:
            r1, g1, b1 = doubled[2*i]
            r2, g2, b2 = doubled[2*i + 1]
        cdict['red'].append([v, r1, r2])
        cdict['green'].append([v, g1, g2])
        cdict['blue'].append([v, b1, b2])
        if hasAlpha:
            cdict['alpha'].append([v, a1, a2])
    #print(cdict)
    return mcolors.LinearSegmentedColormap('CustomMap', cdict)

def get_cmap(minSpeed,maxSpeed,alpha=1):
    DS=0.001
    seq=[
    (63/255 ,63/255 ,153/255, alpha), # Dark Blue
    (159/255,159/255,204/255, alpha), # Light Blue
    (158/255,204/255,170/255, alpha), # Light Green
    (1,212/255,96/255, alpha),  # Light Orange
    (1,1,1,alpha),  # White
    (1,1,1,alpha),  # White
    (1,1,1,alpha),  # White
    (138/255 ,42/255 ,93/255,alpha), # DarkRed
    ]
    valuesOri=np.array([
    minSpeed,  # Dark Blue
    0.90,
    0.95,
    0.98,
    1.00-DS , # White
    1.00    , # White
    1.00+DS , # White
    maxSpeed         # DarkRed
    ])
    values=(valuesOri-min(valuesOri))/(max(valuesOri)-min(valuesOri))
    valuesOri=np.around(valuesOri[np.where(np.diff(valuesOri)>DS)[0]],2)
    cmap= make_colormap(seq,values=values)
    return cmap,np.concatenate((valuesOri,[maxSpeed]))
