disk = {}

disk['SR4'] =      {'name': 'SR4', 
                    'label': 'SR 4', 
                    'distance': 134.8,
                    'mstar': 0.68,
                    'lstar': 1.17,
                    'incl': 22.0, 
                    'PA': 18.0,
                    'dx': -0.060, 
                    'dy': -0.509,
                    'rgap': [0.079],
                    'wgap': [0.010],
                    'dgap': [30],
                    'rout': 0.25, 
                    'maxTb': 50,
                    'hyp-alpha': 1.3,
                    'hyp-wsmth': 0.1,
                    'hyp-Ncoll': 300,
                    'cmask': 'circle[[16h25m56.16s, -24.20.48.71], 0.7arcsec]',
                    'cscales': [0, 5, 30, 75, 150],
		    'gscales': [0, 5], 
                    'cthresh': '0.05mJy',
                    'gthresh': '0.034mJy',
                    'crobust': -0.5,
                    'ctaper': ['0.035arcsec', '0.01arcsec', '0deg'],
                    'cgain': 0.3,
                    'ccycleniter': 300,
                    'RMS': 17.8,
                    'peakr': [84.], 
                    'peakaz': [104.]
}

disk['RULup'] =    {'name': 'RULup', 
                    'label': 'RU Lup', 
                    'distance': 157.5,
                    'mstar': 0.63,
                    'lstar': 1.45,
                    'incl': 18.9, 
                    'PA': 121.0, 
                    'dx': -0.017,
                    'dy': 0.086, 
                    'rgap': [0.179],
                    'wgap': [0.010],
                    'dgap': [2.0],
                    'rout': 0.42, 
                    'maxTb': 75,
                    'hyp-alpha': 1.3,
                    'hyp-wsmth': 0.1,
                    'hyp-Ncoll': 300,
                    'cmask': 'circle[[15h56m42.29s, -37.49.15.89], 1.2arcsec]',
                    'cscales': [0, 5, 30, 75],
                    'gscales': [0, 5], 
                    'cthresh': '0.08mJy',
                    'gthresh': '0.032mJy',
                    'crobust': -0.5,
                    'ctaper': ['0.022arcsec', '0.01arcsec', '-6deg'],
                    'cgain': 0.3,
                    'ccycleniter': 300,
                    'RMS': 14.2, #16,
                    'peakr': [183.], 
                    'peakaz': [157.]
}

disk['Elias20'] =  {'name': 'Elias20', 
                    'label': 'Elias 20', 
                    'distance': 137.5,
                    'mstar': 0.48,
                    'lstar': 2.24,
                    'incl': 54.0, 
                    'PA': 153.2, 
                    'dx': -0.052, 
                    'dy': -0.490, 
                    'rgap': [0.181],
                    'wgap': [0.011],
                    'dgap': [1.8],
                    'rout': 0.48, 
                    'maxTb': 50,
                    'hyp-alpha': 1.3,
                    'hyp-wsmth': 0.1,
                    'hyp-Ncoll': 300,
                    'cmask': 'ellipse[[16h26m18.87s, -24.28.20.18], ' + \
                             '[0.8arcsec, 0.5arcsec], 154deg]',
                    'cscales': [0, 10, 25, 50, 100],
                    'cthresh': '0.06mJy',
                    'gthresh': '0.02mJy',
                    'crobust': 0.0,
                    'ctaper': [],
                    'cgain': 0.3,
                    'ccycleniter': 300,
                    'RMS': 10.1,
                    'peakr': [175.], 
                    'peakaz': [-24.]
}

disk['Sz129'] =    {'name': 'Sz129', 
                    'label': 'Sz 129', 
                    'distance': 160.1,
                    'mstar': 0.83,
                    'lstar': 0.44,
                    'incl': 31.8,
                    'PA': 153.0, 
                    'dx': 0.005, 
                    'dy': 0.006, 
                    'rgap': [0.243, 0.376],
                    'wgap': [0.020, 0.020],
                    'dgap': [1.3, 1.8],
                    'rout': 0.48, 
                    'maxTb': 30,
                    'hyp-alpha': 1.3,
                    'hyp-wsmth': 0.1,
                    'hyp-Ncoll': 300,
                    'cmask': 'ellipse[[15h59m16.454s, -41.57.10.693631], ' + \
                             '[0.85arcsec, 0.7arcsec], 150deg]',
                    'cscales': [0, 5, 30, 75],
                    'gscales': [0, 5, 10],
                    'cthresh': '0.05mJy',
                    'gthresh': '0.024mJy',
                    'crobust': 0.0,
                    'ctaper': [],
                    'cgain': 0.3,
                    'ccycleniter': 300,
                    'RMS': 11.8, #12,
                    'peakr': [223., 393.],
                    'peakaz': [1., -52.]
}

disk['HD143006'] = {'name': 'HD143006', 
                    'label': 'HD 143006', 
                    'distance': 167.3,
                    'mstar': 1.78,
                    'lstar': 3.80,
                    'incl': 16.2, 
                    'PA': 167.0, 
                    'dx': -0.006, 
                    'dy': 0.023,
                    'rgap': [0.140, 0.315],
                    'wgap': [0.035, 0.026],
                    'dgap': [20.0, 5.0],
                    'rout': 0.53, 
                    'maxTb': 15,
                    'hyp-alpha': 1.3,
                    'hyp-wsmth': 0.1,
                    'hyp-Ncoll': 300,
                    'cmask': 'circle[[15h58m36.9s, -22.57.15.60], 0.8arcsec]',
                    'cscales': [0, 5, 30, 75],
                    'gscales': [0, 5], 
                    'cthresh': '0.05mJy',
                    'gthresh': '0.01mJy',
                    'crobust': 0.0,
                    'ctaper': ['0.042arcsec', '0.02arcsec', '172.1deg'],
                    'cgain': 0.3,
                    'ccycleniter': 300,
                    'RMS': 10.,
                    'peakr': [106., 338.],
                    'peakaz': [-71., 161.]
}

disk['GWLup'] =    {'name': 'GWLup', 
                    'label': 'GW Lup', 
                    'distance': 155.2, 
                    'mstar': 0.46,
                    'lstar': 0.33,
                    'incl': 39.0, 
                    'PA': 37.0, 
                    'dx': 0.0005, 
                    'dy': 0.0005, 
                    'rgap': [0.485],
                    'wgap': [0.014],
                    'dgap': [10.0],
                    'rout': 0.75, 
                    'maxTb': 13,
                    'hyp-alpha': 1.3,
                    'hyp-wsmth': 0.1,
                    'hyp-Ncoll': 300,
                    'cmask': 'ellipse[[15h46m44.709s, -34.30.36.076], ' + \
                             '[1.3arcsec, 1.1arcsec], 42deg]',
                    'cscales': [0, 20, 50, 100, 200],
                    'cthresh': '0.05mJy',
                    'gthresh': '0.015mJy',
                    'crobust': 0.5,
                    'ctaper': ['0.035arcsec', '0.015arcsec', '0deg'],
                    'cgain': 0.3,
                    'ccycleniter': 300,
                    'RMS': 10.,
                    'peakr': [492.],
                    'peakaz': [129.]
}

disk['Elias24'] =  {'name': 'Elias24', 
                    'label': 'Elias 24', 
                    'distance': 139.3,
                    'mstar': 0.78,
                    'lstar': 6.03,
                    'incl': 30.0, 
                    'PA': 45.0, 
                    'dx': 0.107, 
                    'dy': -0.383, 
                    'rgap': [0.420],
                    'wgap': [0.035],
                    'dgap': [40.0],
                    'rout': 1.05, 
                    'maxTb': 25,
                    'hyp-alpha': 1.3,
                    'hyp-wsmth': 0.1,
                    'hyp-Ncoll': 300,
                    'cmask': 'ellipse[[16h26m24.0771s, -24.16.13.883], ' + \
                             '[1.26arcsec, 1.09arcsec], 45deg]',
		    'cscales': [0, 10, 20, 50, 100, 200],
                    'gscales': [0, 10], 
                    'cthresh': '0.08mJy',
                    'gthresh': '0.036mJy',
                    'crobust': 0.0,
                    'ctaper': ['0.035arcsec', '0.01arcsec', '166deg'],
                    'cgain': 0.3,
                    'ccycleniter': 300,
                    'RMS': 12,
                    'peakr': [394.],
                    'peakaz': [-1.]
}

disk['HD163296'] = {'name': 'HD163296', 
                    'label': 'HD 163296', 
                    'distance': 101.0,
                    'mstar': 2.04,
                    'lstar': 17.0,
                    'incl': 46.7,
                    'PA': 133.3,
                    'dx': -0.0035,
                    'dy': 0.004, 
                    'rgap': [0.49, 0.85],
                    'wgap': [0.040, 0.041],
                    'dgap': [50., 12.],
                    'rout': 1.08,
                    'maxTb': 50,
                    'hyp-alpha': 1.3,
                    'hyp-wsmth': 0.1,
                    'hyp-Ncoll': 300,
                    'cmask': 'ellipse[[17h56m21.279s, -21.57.22.556], ' + \
                             '[2.5arcsec, 1.8arcsec], 140deg]',
                    'cscales': [0, 10, 30, 75, 150, 300],
                    'gscales': [0, 5, 10],
                    'cthresh': '0.06mJy',
                    'gthresh': '0.04mJy',
                    'crobust': -0.5,
                    'ctaper': [],
                    'cgain': 0.3,
                    'ccycleniter': 300,
                    'RMS': 18.6,
                    'peakr': [529., 854.],
                    'peakaz': [56., -1.]
}

disk['AS209'] =    {'name': 'AS209',
                    'label': 'AS 209',
                    'distance': 121.2,
                    'mstar': 0.83,
                    'lstar': 1.41,
                    'incl': 34.9,
                    'PA': 85.8, 
                    'dx': 0.002,
                    'dy': -0.003,
                    'rgap': [0.51, 0.80],
                    'wgap': [0.032, 0.062],
                    'dgap': [25., 30.],
                    'rout': 1.20,
                    'maxTb': 15,
                    'hyp-alpha': 1.3,
                    'hyp-wsmth': 0.001,
                    'hyp-Ncoll': 300,
                    'cmask': 'ellipse[[16h49m15.29463s,-14.22.09.048165], ' + \
                             '[1.3arcsec, 1.1arcsec], 86deg]',
                    'cscales': [0, 5, 30, 100, 200],
                    'cthresh': '0.08mJy',
                    'gthresh': '0.022mJy',
                    'crobust': -0.5,
                    'ctaper': ['0.037arcsec', '0.01arcsec', '162deg'],
                    'cgain': 0.3,
                    'ccycleniter': 300,
                    'RMS': 10,
                    'peakr': [520., 814.],
                    'peakaz': [-172., -158.]
}
