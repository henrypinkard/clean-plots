#convenience code for plot export

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import cmocean
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.gridspec import GridSpec
from matplotlib import cm
    
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm

# plt.rc('axes.spines', **{'bottom':True, 'left':True, 'right':False, 'top':False})
# mpl.rcParams['axes.spines.left'] = True
# mpl.rcParams['axes.spines.bottom'] = True
# mpl.rcParams['axes.spines.top'] = False
# mpl.rcParams['axes.spines.right']  = False

#editable text in fonts
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42

#make text on figures look good
SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 24
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
# plt.rc('font', sansserif='Arial')
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['font.style'] = 'normal'
# plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "sans-serif"

plt.rc('axes', linewidth=2)    
plt.rc('xtick.major', width=2)    
plt.rc('ytick.major', width=2)    

plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    


##### General stylizing of plots #####
def default_format(ax):
    decimal_format_ticks(ax)
    sparse_ticks(ax)
    clear_spines(ax)
    zero_lims(ax)

def clear_spines(ax):
    ax.spines[["top", "right"]].set_visible(False)
    
def decimal_format_ticks(ax):
    def formatter(x, pos):
        if x == int(x):
            return str(int(x))
        return '{:.1f}'.format(x)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    
def sparse_ticks(ax, mode='both'):
    if mode == 'both':
        ax.set_yticks([0, ax.get_yticks()[-1]])
        ax.set_xticks([0, ax.get_xticks()[-1]])
    elif mode == 'x':
        ax.set_xticks([0, ax.get_xticks()[-1]])
    elif mode == 'y':
        ax.set_yticks([0, ax.get_yticks()[-1]])
        
def zero_lims(ax, mode='both'):
    if mode == 'both':
        ax.set_xlim([0, ax.get_xlim()[-1]])
        ax.set_ylim([0, ax.get_ylim()[-1]])
    elif mode == 'x':
        ax.set_xlim([0, ax.get_xlim()[-1]])
    elif mode == 'y':
        ax.set_ylim([0, ax.get_ylim()[-1]])
        
    
###################################
    
AMPLITUDE_CONTRAST_MIN = 0.1
    
def line_plot_with_phase(line_profile, ax, width=2, amplitude_contrast_min=AMPLITUDE_CONTRAST_MIN, orient='horz'):
    
    interp_x = np.linspace(0, line_profile.size, 800)
    interp_y_mag = np.interp(interp_x, np.arange(line_profile.size), np.abs(line_profile))
    interp_y_phase = np.interp(interp_x, np.arange(line_profile.size), np.unwrap(np.angle(line_profile)))
    interp_y = interp_y_mag * np.exp(1j * interp_y_phase)
    if orient == 'horz':
        points = np.abs(np.array([interp_x, interp_y]).T.reshape(-1, 1, 2))
    else:
        points = np.abs(np.array([interp_y, interp_x]).T.reshape(-1, 1, 2))
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    normalized_phase = (np.angle(np.mean(np.stack([interp_y[1:], interp_y[:-1]]), axis=0)) + np.pi) / (2*np.pi)
    normalized_amplitude = np.abs(interp_y) / np.max(( np.abs(interp_y[:-1]) + np.abs(interp_y[1:])) / 2 )
    colors = cmocean.cm.phase_r(normalized_phase)
    colors[:, :3] = colors[:, :3] * ((1 - amplitude_contrast_min) *normalized_amplitude[1:, None] + amplitude_contrast_min)

    lc = LineCollection(segments, colors=colors)


    lc.set_linewidth(width)
    line = ax.add_collection(lc)
    if orient == 'horz':
        ax.set_xlim(0, line_profile.size)
        ax.set_ylim(0, 1.1* np.max(np.abs(line_profile)))
        ax.set_yticks([0, np.max(np.abs(line_profile))])
    else:
        ax.set_ylim(0, line_profile.size)
        ax.set_xlim(0, 1.1* np.max(np.abs(line_profile)))
        ax.set_xticks([0, np.max(np.abs(line_profile))])    
        
def show_colorbar(ax, image=None, contrast_min=None, contrast_max=None, y_axis_amplitude=True,):
    cmap_image = np.stack(100 *[cm.inferno(np.linspace(0,1,100))], axis=0).T
    if not y_axis_amplitude:
        cmap_image = cmap_image.T
    ax.imshow(cmap_image,  origin='lower', aspect='auto')
    if image is not None:
        contrast_min, contrast_max = np.min(image), np.max(image)
    contrast_max_int = int(np.round(contrast_max))
    if y_axis_amplitude:
        ax.set_ylabel('Intensity', labelpad=len(str(contrast_max_int)) * -5)
        ax.set(yticks=[0, cmap_image.shape[0]], xticks=[], yticklabels=[0, contrast_max_int])
    else:
        ax.set_xlabel('Intensity', labelpad=len(str(contrast_max_int)) * -5)
        ax.set(xticks=[0, cmap_image.shape[0]], yticks=[], xticklabels=[0, contrast_max_int])
    
    
def show_phase_colorbar(ax, y_axis_amplitude=True, max_amplitude=1, amplitude_contrast_min=AMPLITUDE_CONTRAST_MIN):
    dimensions= (200, 200)

    high = cmocean.cm.phase_r(np.linspace(0, 1, dimensions[0]))[..., :3]
    shading = np.linspace(amplitude_contrast_min, 1, dimensions[1])
    shading[0] = 0 # always make 0 amplitude black
    shaded_color = shading[:, None, None] * high[None]
    colorbar = np.swapaxes(shaded_color, 0, 1)
    if not y_axis_amplitude:
        colorbar = np.swapaxes(colorbar, 0, 1)
    ax.imshow(colorbar, origin='lower', aspect='auto')
    if not y_axis_amplitude:
        ax.set(yticks=[0, shading.size], yticklabels=[0, max_amplitude], ylabel='Amplitude',
          xticks=[0, high.shape[0]], xticklabels=['0', '2$\pi$'], xlabel='Phase')
    else:
        ax.set(xticks=[0, shading.size], xticklabels=[0, max_amplitude], xlabel='Amplitude',
          yticks=[0, high.shape[0]], yticklabels=['0', '2$\pi$'], ylabel='Phase')


    
def show_complex_image(image, ax=None, amplitude_contrast_min=AMPLITUDE_CONTRAST_MIN, pixel_size_um=None):
    if ax is None:
        fig, ax = plt.subplots()
    amplitude_contrast_max = np.max(np.abs(image))
    alpha = amplitude_contrast_min + (np.abs(image) / amplitude_contrast_max) * (1 - amplitude_contrast_min)
    alpha[np.abs(image) == 0] = 0 #true balck for background
    converted = cmocean.cm.phase_r((np.pi + np.angle(image)) / (np.pi * 2), alpha=alpha)
    black = np.zeros_like(converted)
    black[..., -1] = 1 # non transparent
    ax.imshow(black, interpolation='nearest')
    ax.imshow(converted, interpolation='nearest')
    ax.set(xticks=[], yticks=[], xlim=[0, image.shape[1]], ylim=[0, image.shape[0]]) #fill whole axis
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    if pixel_size_um is not None:
        add_scalebar(ax, converted, pixel_size_um)


def show_image(image, ax=None, contrast_max=None, contrast_min=None, name='', colorbar=True, origin='upper', pixel_size_um=None,
              **kwargs):
    if ax is None:
        fig, ax = plt.subplots()
    im = ax.imshow(image, cmap='inferno', vmin=contrast_min, vmax=contrast_max, origin=origin,
                  interpolation='nearest')
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.set_title(name)
    ax.set(xticks=[], yticks=[])
    ax.set(xlim=[-0.5, image.shape[1] - 0.5], ylim=[-0.5, image.shape[0] - 0.5]) #fill whole axis
    if origin == 'upper':
        ax.set(ylim=[image.shape[0] - 0.5, -0.5]) #fill whole axis
    if colorbar:
        plt.colorbar(im, ax=ax)
    if pixel_size_um is not None:
        add_scalebar(ax, image, pixel_size_um)
    ax.set(**kwargs)
    return im
    
def add_scalebar(ax, im, pixel_size_um, image_fraction=0.3):
    #infer the best roundish number for the scalebar extent
    desired_size_um = im.shape[0] * image_fraction * pixel_size_um
    min_base_10 = np.floor(np.log10(desired_size_um))
    scalebar_size_um = int(np.round(desired_size_um / 10 ** min_base_10) * 10 ** min_base_10)
    scalebar_in_pix = scalebar_size_um / pixel_size_um
    # add scalebar to axes
    scalebar_in_pix = (0.175 * scalebar_in_pix, scalebar_in_pix)                
    scalebar_text = '{} ??m'.format(scalebar_size_um)
    
    scalebar = AnchoredSizeBar(ax.transData,
                           scalebar_in_pix[1], scalebar_text, 'lower right', label_top=True,
                           pad=0.8, color='white', frameon=False, size_vertical=scalebar_in_pix[0],
                           fontproperties=fm.FontProperties(size=14))
    ax.add_artist(scalebar)
        
def show_histogram(ax, data, bins, name):
    im = ax.hist(np.ravel(data), bins, density=True)
    plt.ylabel('Probability')
    ax.set_title(name)
    
    
def plot_line(x, y, ax=None, **kwargs):
    if ax == None:
        fig, ax = plt.subplots()
    ax.plot(x, y)
    default_format(ax)

    ax.set(**kwargs)
    

