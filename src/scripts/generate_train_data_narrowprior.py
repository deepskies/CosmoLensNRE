"""
Script to generate training data for the NRE model from a prior on cosmological parameters (Om0 and w0) using Deeplenstronomy.
"""
import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(1, '/home/jarugula/deeplenstronomy_cosmology/deeplenstronomy')

import deeplenstronomy.deeplenstronomy as dl

# python script.py --name exp012_train_500k --outdir exp012_train_500k --seed 48 --num 100 --Om0 0.3 --w0 -1.0

def main():
    # Create the argument parser
    parser = argparse.ArgumentParser(description='Process some parameters.')

    # Add arguments
    parser.add_argument('--name', type=str, required=True, help='Name of the dataset')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory')
    parser.add_argument('--seed', type=int, required=True, help='Random seed')
    parser.add_argument('--num', type=int, required=True, help='Number of lenses')
    parser.add_argument('--Om_min', type=float, required=True, help='Omega matter (Om0)')
    parser.add_argument('--Om_max', type=float, required=True, help='Omega matter (Om0)')
    parser.add_argument('--w_min', type=float, required=True, help='Equation of state parameter (w0)')
    parser.add_argument('--w_max', type=float, required=True, help='Equation of state parameter (w0)')

    # Parse the arguments
    args = parser.parse_args()

    # Create the configuration dictionary
    config = {
        'DATASET': {
            'NAME': args.name,
            'PARAMETERS': {
                'SIZE': args.num,  
                'OUTDIR': args.outdir,
                'SEED': args.seed
            }
        },
        'COSMOLOGY': {
            'NAME': 'wCDM',  
            'PARAMETERS': {
                'H0': 70.0,  
                'Om0': {
                    'DISTRIBUTION': {
                        'NAME': 'uniform',
                        'PARAMETERS': {
                            'minimum': args.Om_min,
                            'maximum': args.Om_max
                        }
                    }
                },
                'w0':{ 
                    'DISTRIBUTION': {
                        'NAME': 'uniform',
                        'PARAMETERS': {
                            'minimum': args.w_min,
                            'maximum': args.w_max
                        }
                    }
                },
            }
        },
        'IMAGE': {
            'PARAMETERS': {
                'exposure_time': {
                    'DISTRIBUTION': {
                        'NAME': 'des_exposure_time',
                        'PARAMETERS': None
                    }
                },
                'numPix': 32,
                'pixel_scale': 0.263,
                'psf_type': 'GAUSSIAN',
                'read_noise': 7,
                'ccd_gain': {
                    'DISTRIBUTION': {
                        'NAME': 'des_ccd_gain',
                        'PARAMETERS': None
                    }
                }
            }
        },
        'SURVEY': {
            'PARAMETERS': {
                'BANDS': 'g',
                'seeing': 0.9,
                'magnitude_zero_point': 30.0,
                'sky_brightness': 30.0,
                'num_exposures': 10
            }
        },
        'SPECIES': {
            'GALAXY_1': {
                'NAME': 'LENS',
                'LIGHT_PROFILE_1': {
                    'NAME': 'SERSIC_ELLIPSE',
                    'PARAMETERS': {
                        'magnitude': 100,
                        'center_x': 0,
                        'center_y': 0,
                        'R_sersic': 1,
                        'n_sersic': 4,
                        'e1': 0,
                        'e2': 0.5
                    }
                },
                'MASS_PROFILE_1': {
                    'NAME': 'SIE',
                    'PARAMETERS': {
                        'sigma_v': {
                            'DISTRIBUTION': {
                                'NAME': 'uniform',
                                'PARAMETERS': {
                                    'minimum': 225,
                                    'maximum': 300
                                }
                            }
                        },
                        'e1': {
                            'DISTRIBUTION': {
                                'NAME': 'uniform',
                                'PARAMETERS': {
                                    'minimum': -0.1,
                                    'maximum': 0.1
                                }
                            }
                        },
                        'e2': {
                            'DISTRIBUTION': {
                                'NAME': 'uniform',
                                'PARAMETERS': {
                                    'minimum': -0.1,
                                    'maximum': 0.1
                                }
                            }
                        },
                        'center_x': {
                            'DISTRIBUTION':{
                                'NAME': 'normal',
                                'PARAMETERS':{
                                    'mean': 0.0,
                                    'std': 0.8
                                }
                            }
                        },
                        'center_y': {
                            'DISTRIBUTION':{
                                'NAME': 'normal',
                                'PARAMETERS':{
                                    'mean': 0.0,
                                    'std': 0.8
                                }
                            }
                        }
                    }
                }
            },
            'GALAXY_2': {
                'NAME': 'SOURCE',
                'LIGHT_PROFILE_1': {
                    'NAME': 'SERSIC_ELLIPSE',
                    'PARAMETERS': {
                        'magnitude': {
                            'DISTRIBUTION': {
                                'NAME': 'uniform',
                                'PARAMETERS': {
                                    'minimum': 19,
                                    'maximum': 24
                                }
                            }
                        },
                        'center_x': 0,
                        'center_y': 0,
                        'R_sersic': {
                            'DISTRIBUTION': {
                                'NAME': 'uniform',
                                'PARAMETERS': {
                                    'minimum': 0.1,
                                    'maximum': 3
                                }
                            }
                        },
                        'n_sersic': {
                            'DISTRIBUTION': {
                                'NAME': 'uniform',
                                'PARAMETERS': {
                                    'minimum': 0.5,
                                    'maximum': 8
                                }
                            }
                        },
                        'e1': {
                            'DISTRIBUTION': {
                                'NAME': 'uniform',
                                'PARAMETERS': {
                                    'minimum': -0.1,
                                    'maximum': 0.1
                                }
                            }
                        },
                        'e2': {
                            'DISTRIBUTION': {
                                'NAME': 'uniform',
                                'PARAMETERS': {
                                    'minimum': -0.1,
                                    'maximum': 0.1
                                }
                            }
                        }
                    }
                },
                'MASS_PROFILE_1': {
                    'NAME': 'SIE',
                    'PARAMETERS': {
                        'theta_E': 2.0,
                        'e1': 0.1,
                        'e2': -0.1,
                        'center_x': 0.0,
                        'center_y': 0.0
                    }
                }
            }
        },
        'GEOMETRY': {
            'CONFIGURATION_1': {
                'NAME': 'GALAXYGALAXY',
                'FRACTION': 1,
                'PLANE_1': {
                    'OBJECT_1': 'LENS',
                    'PARAMETERS': {
                        'REDSHIFT': {
                            'DISTRIBUTION': {
                                'NAME': 'uniform',
                                'PARAMETERS': {
                                    'minimum': 0.3,
                                    'maximum': 0.8
                                }
                            }
                        }
                    }
                },
                'PLANE_2': {
                    'OBJECT_1': 'SOURCE',
                    'PARAMETERS': {
                        'REDSHIFT': {
                            'DISTRIBUTION': {
                                'NAME': 'uniform',
                                'PARAMETERS': {
                                    'minimum': 1.5,
                                    'maximum': 2.5
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    dl.make_dataset(config, verbose=True, save_to_disk=True, image_file_format='npy')

if __name__ == '__main__':
    main()
