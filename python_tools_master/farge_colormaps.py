#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 18 19:42:20 2018

@author: engels
"""
import numpy as np

def farge_colormap_multi(taille=256, limite_faible_fort=0.3, etalement_du_zero=0.02, blackmargin=0.25, type='vorticity', return_vctor=False):
    import numpy as np
    import matplotlib

    if type == 'vorticity':
        color1=[0.0, 0.5, 1.0]  #light blue
        color2=[1.0, 1.0, 1.0]  #white
        color3=[1.0, 0.5, 0.5]  #light red
        zero=[1.0, 222.0/255.0, 17.0/255.0]
    elif type == 'pressure':
        color1=[0.0, 0.5, 1.0];
        color2=[1.0, 1.0, 1.0];
        color3=[1.0, 1.0, 0.0];
        zero  =[1.0, 0.0, 0.0];
    elif type == 'streamfunction':
        color1=[0.5, 0.0, 1.0];
        color2=[1.0, 1.0, 1.0];
        color3=[1.0, 0.8, 0.0];
        zero=[0.0, 1.0, 0.5];
    elif type == 'velocity':
        color1=[1.0, 1.0, 0.0];
        color2=[1.0, 1.0, 1.0];
        color3=[1.0, 0.5, 0.5];
        zero  =[0.5, 1.0, 0.5];

    etalement_du_zero = int( np.ceil(etalement_du_zero * taille) )

    limite_basse = int( np.floor(taille/2.0*(1.0-limite_faible_fort)) )
    limite_haute = int( np.ceil(taille/2.0*(1.0+limite_faible_fort)) )

    zero_moins = int( np.floor((taille-etalement_du_zero)/2.0) )
    zero_plus = int( np.ceil((taille + etalement_du_zero)/2.0) )

    colors = np.zeros([taille,3])

    # I could not figure out how to handle all colors in one go, so I loop over colors
    for i in range(3):
        # concatenate some linear vectors
        y2 = (np.linspace(blackmargin, 1.0 ,limite_basse)*color1[i],
              np.linspace(blackmargin**3.0, 0.5, zero_moins-limite_basse)*color2[i],
              np.squeeze(np.ones([etalement_du_zero, 1])*zero[i]),
              np.linspace(0.5, 1.0-blackmargin**3, limite_haute-zero_plus)*color2[i],
              np.linspace(blackmargin, 1.0, taille-limite_haute)*color3[i])
        colors[:,i] = np.hstack( y2 )
#    farge_cmap = matplotlib.colors.LinearSegmentedColormap(segmentdata=colors, name='farge')
    farge_cmap = matplotlib.colors.ListedColormap(colors, name='farge', N=None)

    if return_vctor:
        return colors
    else:
        return farge_cmap


# this function writes Marie's colormaps to *.dat files, for usage in other tools
def farge_colormaps_to_dat():

    for cmap in ['vorticity','pressure','streamfunction','velocity']:
        colors = farge_colormap_multi(taille=256, limite_faible_fort=0.3, etalement_du_zero=0.02, blackmargin=0.25, type=cmap, return_vctor=True)

        fid = open( 'colors_'+cmap+'.dat', 'w')

        for i in range(colors.shape[0]):
            fid.write('%f %f %f\n' % (colors[i,0],colors[i,1],colors[i,2]) )

# this function writes Marie's colormaps to *.xmf files, for usage in paraview
def farge_colormaps_to_paraview():

    for cmap in ['vorticity','pressure','streamfunction','velocity']:
        colors = farge_colormap_multi(taille=256, limite_faible_fort=0.2, etalement_du_zero=0.02, blackmargin=0.25, type=cmap, return_vctor=True)

        fid = open( 'colors_'+cmap+'.xml', 'w')
        fid.write('<ColorMaps>\n')
        fid.write('  <ColorMap space="RGB" indexedLookup="false" name="marie-%s">\n' % (cmap))

        for i in range(colors.shape[0]):
            fid.write('<Point x="%f" o="1" r="%f" g="%f" b="%f"/>\n' % (i/(colors.shape[0]-1), colors[i,0],colors[i,1],colors[i,2]) )

        fid.write('  </ColorMap>\n')
        fid.write('</ColorMaps>\n')

def random_colormap_for_paraview(n=256):
    colors = np.random.rand(n,3)

    fid = open( 'colors_random.xml', 'w')
    fid.write('<ColorMaps>\n')
    fid.write('  <ColorMap space="RGB" indexedLookup="false" name="%i_random_colors">\n' % (n))

    for i in range(colors.shape[0]):
        fid.write('<Point x="%f" o="1" r="%f" g="%f" b="%f"/>\n' % (i/(colors.shape[0]-1), colors[i,0],colors[i,1],colors[i,2]) )

    fid.write('  </ColorMap>\n')
    fid.write('</ColorMaps>\n')
    
# source: https://stackoverflow.com/questions/14720331/how-to-generate-random-colors-in-matplotlib
def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=False):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap