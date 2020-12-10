import os
import numpy as np

def ExportMS(msfile, spwavg=False, timebin=None):

    # parse input filename
    filename = msfile
    if filename[-3:] != '.ms':
        print("MS name must end in '.ms'")
        return

    # strip off the '.ms'
    MS_filename = filename.replace('.ms', '')

    # get information about the spectral windows (SPWs)
    tb.open(MS_filename+'.ms/SPECTRAL_WINDOW')
    num_chan = tb.getcol('NUM_CHAN').tolist()
    tb.close()

    # spectral averaging to 1 channel per SPW (with optional time-averaging)
    if timebin is not None:
        suffix = '_spavg_tbin'+timebin
        os.system('rm -rf ' + MS_filename + suffix + '.ms')
        split(vis=MS_filename+'.ms', width=num_chan, datacolumn='data', 
              outputvis=MS_filename+suffix+'.ms', keepflags=False, 
              timebin=timebin)
    else:
        suffix = '_spavg'
        os.system('rm -rf ' + MS_filename + suffix + '.ms')
        split(vis=MS_filename+'.ms', width=num_chan, datacolumn='data',
              outputvis=MS_filename+suffix+'.ms', keepflags=False,
              timebin=timebin)

    # get the data tables out of the MS file
    tb.open(MS_filename+suffix+'.ms')
    data = np.squeeze(tb.getcol("DATA"))
    flag = np.squeeze(tb.getcol("FLAG"))
    uvw = tb.getcol("UVW")
    weight = tb.getcol("WEIGHT")
    spwid = tb.getcol("DATA_DESC_ID")
    times = tb.getcol("TIME")
    tb.close()

    # get frequency information
    tb.open(MS_filename+suffix+'.ms/SPECTRAL_WINDOW')
    freqlist = np.squeeze(tb.getcol("CHAN_FREQ"))
    tb.close()

    # get rid of any lingering flagged columns (but there shouldn't be any!)
    good = np.squeeze(np.any(flag, axis=0) == False)
    data = data[:,good]
    weight = weight[:,good]
    uvw = uvw[:,good]
    spwid = spwid[good]
    times = times[good]

    # average the polarizations
    Re = np.sum(data.real * weight, axis=0) / np.sum(weight, axis=0)
    Im = np.sum(data.imag * weight, axis=0) / np.sum(weight, axis=0)
    vis = Re + 1j*Im
    wgt = np.sum(weight, axis=0)

    # associate each datapoint with a frequency
    get_freq = lambda ispw: freqlist[ispw]
    freqs = get_freq(spwid)

    # retrieve (u,v) positions in meters
    um = uvw[0,:]
    vm = uvw[1,:]
    wm = uvw[2,:]

    # SPW-averaging if requested
    if spwavg:

        # filename indicator
        suffix += '_SPWavg'

        # identify unique timestamps

        # set up the output and tracker lists
        u_avg = []
        v_avg = []
        real_avg = []
        imag_avg = []
        wgt_avg = []
        timestamps_done = []

        # loop over unique timestamps
        for i in range(len(times)):

            # ensure that we do not repeat timestamps
            if i in timestamps_done:
                continue
            else:
                # identify a timestamp
                ts = times[i]

                # collect array indices with that timestamp
                index_ts = np.where(times == ts)[0]

                # collect array values with those indices
                um_ts = um[index_ts]
                vm_ts = vm[index_ts]
                wm_ts = wm[index_ts]
                real_ts = vis.real[index_ts]
                imag_ts = vis.imag[index_ts]
                wgt_ts = wgt[index_ts]
                nu_ts = freqs[index_ts]

                # update the index tracker
                ind_done = []

                # loop through records to find co-located u,v,w points
                for j in range(len(um_ts)):
                    if j in ind_done:
                        continue
                    else:
                        ind_uvw = np.where( (np.abs(um_ts - um_ts[j]) < 1) & 
                                            (np.abs(vm_ts - vm_ts[j]) < 1) & 
                                            (np.abs(wm_ts - wm_ts[j]) < 1) )[0]
                        if (len(ind_uvw) > 4):
                            print("Ack!  There's >4 SPWs in this timestamp...")
                    
                        # spectral averaging
                        sum_real = np.sum(real_ts[ind_uvw] * wgt_ts[ind_uvw])
                        sum_imag = np.sum(imag_ts[ind_uvw] * wgt_ts[ind_uvw])
                        sum_wgt = np.sum(wgt_ts[ind_uvw])
                        sum_um = np.sum(um_ts[ind_uvw] * wgt_ts[ind_uvw])
                        sum_vm = np.sum(vm_ts[ind_uvw] * wgt_ts[ind_uvw])

                        real_avg.append(sum_real / sum_wgt)
                        imag_avg.append(sum_imag / sum_wgt)
                        wgt_avg.append(sum_wgt)
                        nu0 = np.sum(nu_ts[ind_uvw] * wgt_ts[ind_uvw]) / sum_wgt
                        u_avg.append((sum_um / sum_wgt) * nu0 / 2.9979e8)
                        v_avg.append((sum_vm / sum_wgt) * nu0 / 2.9979e8)
                    
                        # store list of processed records
                        ind_done = np.concatenate((ind_done, ind_uvw))

                # store list of processed timestamps
                timestamps_done = np.concatenate((timestamps_done, index_ts))

        # save-able arrays (& visibilities in complex number format)
        vis = np.array(real_avg) + 1.0j*np.array(imag_avg)
        wgt = wgt_avg
        u, v = u_avg, v_avg

    else:
        u, v = um * freqs / 2.9979e8, vm * freqs / 2.9979e8

    # output to npz file 
    os.system('rm -rf '+MS_filename+suffix+'.vis.npz')
    np.savez(MS_filename+suffix+'.vis', u=u, v=v, Vis=vis, Wgt=wgt)
    print("# Measurement set exported to "+MS_filename+suffix+".vis.npz")
