def wikititles(data_dir):
    return {
        'train' : {
            'path': {
                'train': {
                    'data_lbl': f'{data_dir}/LF-Wikipedia-500K/trn_X_Y.txt',
                    'data_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/train.raw.txt',
                    'lbl_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/label.raw.txt',
                },
            },
            'parameters': PARAM,
        },
        'data' : {
            'path': {
                'train': {
                    'data_lbl': f'{data_dir}/LF-Wikipedia-500K/trn_X_Y.txt',
                    'data_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/train.raw.txt',
                    'lbl_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/label.raw.txt',
                },
                'test': {
                    'data_lbl': f'{data_dir}/LF-Wikipedia-500K/tst_X_Y.txt',
                    'data_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/test.raw.txt',
                    'lbl_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/label.raw.txt',
                },
            },
            'parameters': PARAM,
        },
        'train_meta' : {
            'path': {
                'train': {
                    'data_lbl': f'{data_dir}/LF-Wikipedia-500K/trn_X_Y.txt',
                    'data_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/train.raw.txt',
                    'lbl_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/label.raw.txt',
                    'hlk_meta': {
                        'prefix': 'hlk',
                        'data_meta': f'{data_dir}/LF-Wikipedia-500K/hyper_link_trn_X_Y.txt',
                        'lbl_meta': f'{data_dir}/LF-Wikipedia-500K/hyper_link_lbl_X_Y.txt',
                        'meta_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/hyper_link.raw.txt'
                    },
                },
            },
            'parameters': PARAM,
        },
        'data_meta' : {
            'path': {
                'train': {
                    'data_lbl': f'{data_dir}/LF-Wikipedia-500K/trn_X_Y.txt',
                    'data_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/train.raw.txt',
                    'lbl_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/label.raw.txt',
                    'sal_meta': {
                        'prefix': 'sal',
                        'data_meta': f'{data_dir}/LF-Wikipedia-500K/see_also_trn_X_Y.txt',
                        'lbl_meta': f'{data_dir}/LF-Wikipedia-500K/see_also_lbl_X_Y.txt',
                        'meta_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/see_also.raw.txt'
                    },
                },
                'test': {
                    'data_lbl': f'{data_dir}/LF-Wikipedia-500K/tst_X_Y.txt',
                    'data_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/test.raw.txt',
                    'lbl_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/label.raw.txt',
                    'sal_meta': {
                        'prefix': 'sal',
                        'data_meta': f'{data_dir}/LF-Wikipedia-500K/see_also_tst_X_Y.txt',
                        'lbl_meta': f'{data_dir}/LF-Wikipedia-500K/see_also_lbl_X_Y.txt',
                        'meta_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/see_also.raw.txt',
                    },
                },
            },
            'parameters': PARAM,
        },
        'data_metas' : {
            'path': {
                'train': {
                    'data_lbl': f'{data_dir}/LF-Wikipedia-500K/trn_X_Y.txt',
                    'data_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/train.raw.txt',
                    'lbl_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/label.raw.txt',
                    'hlk_meta': {
                        'prefix': 'hlk',
                        'data_meta': f'{data_dir}/LF-Wikipedia-500K/hyper_link_trn_X_Y.txt',
                        'lbl_meta': f'{data_dir}/LF-Wikipedia-500K/hyper_link_lbl_X_Y.txt',
                        'meta_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/hyper_link.raw.txt'
                    },
                    'sal_meta': {
                        'prefix': 'sal',
                        'data_meta': f'{data_dir}/LF-Wikipedia-500K/see_also_trn_X_Y.txt',
                        'lbl_meta': f'{data_dir}/LF-Wikipedia-500K/see_also_lbl_X_Y.txt',
                        'meta_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/see_also.raw.txt'
                    },
                },
                'test': {
                    'data_lbl': f'{data_dir}/LF-Wikipedia-500K/tst_X_Y.txt',
                    'data_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/test.raw.txt',
                    'lbl_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/label.raw.txt',
                    'hlk_meta': {
                        'prefix': 'hlk',
                        'data_meta': f'{data_dir}/LF-Wikipedia-500K/hyper_link_tst_X_Y.txt',
                        'lbl_meta': f'{data_dir}/LF-Wikipedia-500K/hyper_link_lbl_X_Y.txt',
                        'meta_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/hyper_link.raw.txt',
                    },
                    'sal_meta': {
                        'prefix': 'sal',
                        'data_meta': f'{data_dir}/LF-Wikipedia-500K/see_also_tst_X_Y.txt',
                        'lbl_meta': f'{data_dir}/LF-Wikipedia-500K/see_also_lbl_X_Y.txt',
                        'meta_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/see_also.raw.txt',
                    },
                },
            },
            'parameters': PARAM,
        },
        'data_hlklnk' : {
            'path': {
                'train': {
                    'data_lbl': f'{data_dir}/LF-Wikipedia-500K/trn_X_Y.txt',
                    'data_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/train.raw.txt',
                    'lbl_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/label.raw.txt',
                    'hlk_meta': {
                        'prefix': 'hlk',
                        'data_meta': f'{data_dir}/LF-Wikipedia-500K/hyper_link_trn_X_Y.txt',
                        'lbl_meta': f'{data_dir}/LF-Wikipedia-500K/hyper_link_lbl_X_Y.txt',
                        'meta_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/hyper_link.raw.txt'
                    },
                    'lnk_meta': {
                        'prefix': 'lnk',
                        'data_meta': f'{data_dir}/LF-Wikipedia-500K/hyper_link_renee_mean_trn_X_Y.txt',
                        'lbl_meta': f'{data_dir}/LF-Wikipedia-500K/hyper_link_renee_mean_lbl_X_Y.txt',
                        'meta_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/hyper_link.raw.txt'
                    },
                },
                'test': {
                    'data_lbl': f'{data_dir}/LF-Wikipedia-500K/tst_X_Y.txt',
                    'data_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/test.raw.txt',
                    'lbl_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/label.raw.txt',
                    'hlk_meta': {
                        'prefix': 'hlk',
                        'data_meta': f'{data_dir}/LF-Wikipedia-500K/hyper_link_tst_X_Y.txt',
                        'lbl_meta': f'{data_dir}/LF-Wikipedia-500K/hyper_link_lbl_X_Y.txt',
                        'meta_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/hyper_link.raw.txt',
                    },
                    'lnk_meta': {
                        'prefix': 'lnk',
                        'data_meta': f'{data_dir}/LF-Wikipedia-500K/hyper_link_renee_mean_tst_X_Y.txt',
                        'lbl_meta': f'{data_dir}/LF-Wikipedia-500K/hyper_link_renee_mean_lbl_X_Y.txt',
                        'meta_info': f'{data_dir}/LF-Wikipedia-500K/raw_data/hyper_link.raw.txt',
                    },
                },
            },
            'parameters': PARAM,
        },
    }
