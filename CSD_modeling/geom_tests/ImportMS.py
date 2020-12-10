import os
import numpy as np

def ImportMS(msfile, modelfile, suffix='model', make_resid=False):

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
    mdl = (np.load(modelfile+'.npz'))['Vis']

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
        os.system('rm -rf '+MS_filename+'.'+suffix+'.resid.ms')
        os.system('cp -r '+filename+' '+MS_filename+'.'+suffix+'.resid.ms')

        # open the model file and load the data
        tb.open(MS_filename+'.'+suffix+'.resid.ms')
        data = tb.getcol("DATA")
        flag = tb.getcol("FLAG")
        tb.close()

        # identify the unflagged columns (should be all of them!)
        unflagged = np.squeeze(np.any(flag, axis=0) == False)

        # replace with the model visibilities (equal in both polarizations)
        data[:, :, unflagged] -= mdl

        # re-pack those model visibilities back into the .ms file
        tb.open(MS_filename+'.'+suffix+'.resid.ms', nomodify=False)
        tb.putcol("DATA", data)
        tb.flush()
        tb.close()
