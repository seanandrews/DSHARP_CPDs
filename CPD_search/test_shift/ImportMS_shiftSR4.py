import os, sys
import numpy as np
sys.path.append('.')
import diskdictionary as disk

def ImportMS_shiftSR4(msfile, modelfile, suffix='model', make_resid=False):

    # parse msfile name
    filename = msfile
    if filename[-3:] != '.ms':
        print("MS name must end in '.ms'")
        return

    # strip off the '.ms'
    MS_filename = filename.replace('.ms', '')

    # copy the data MS into a model MS
    os.system('rm -rf '+MS_filename+'.'+suffix+'.ms')
    os.system('cp -r '+filename+' '+MS_filename+'.'+suffix+'.ms')

    # open the model file and load the data
    tb.open(MS_filename+'.'+suffix+'.ms')
    data = tb.getcol("DATA")
    flag = tb.getcol("FLAG")
    tb.close()

    # identify the unflagged columns (should be all of them!)
    unflagged = np.squeeze(np.any(flag, axis=0) == False)

    # load the model visibilities
    mdl = (np.load(modelfile+'.npz'))['V']

    # shift the emission to the phase center
    vdat = np.load(MS_filename+'.vis.npz')
    u, v = vdat['u'], vdat['v']
    offx = disk.disk['SR4']['dx'] * np.pi / (180 * 3600)
    offy = disk.disk['SR4']['dy'] * np.pi / (180 * 3600)
    mdl *= np.exp(-2 * np.pi * 1.0j * (u*offx + v*offy))

    # replace with the model visibilities (equal in both polarizations)
    data[:, :, unflagged] = mdl

    # re-pack those model visibilities back into the .ms file
    tb.open(MS_filename+'.'+suffix+'.ms', nomodify=False)
    tb.putcol("DATA", data)
    tb.flush()
    tb.close()


    # make a residuals MS if requested
    if make_resid:

        # copy the data MS into a model MS
        os.system('rm -rf '+MS_filename+'.resid.ms')
        os.system('cp -r '+filename+' '+MS_filename+'.resid.ms')

        # open the model file and load the data
        tb.open(MS_filename+'.resid.ms')
        data = tb.getcol("DATA")
        flag = tb.getcol("FLAG")
        tb.close()

        # identify the unflagged columns (should be all of them!)
        unflagged = np.squeeze(np.any(flag, axis=0) == False)

        # replace with the model visibilities (equal in both polarizations)
        data[:, :, unflagged] -= mdl

        # re-pack those model visibilities back into the .ms file
        tb.open(MS_filename+'.resid.ms', nomodify=False)
        tb.putcol("DATA", data)
        tb.flush()
        tb.close()
