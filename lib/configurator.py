import itertools
import json


def generate_data_configurations(data_preprocessing_options):
    result = set()

    for vector_method in data_preprocessing_options['vectorization']['method']:
        for data_method in data_preprocessing_options['data_creation']['data_creation_method']:
            sliding_window_options = generate_sliding_window_options(data_method, data_preprocessing_options['data_creation'])
            for sliding_window_option in sliding_window_options:
                config = {
                    'data_creation': {
                        'data_creation_method': data_method,
                        'sliding_window_size': sliding_window_option['sliding_window_size'],
                        'window_movement': sliding_window_option['window_movement']
                    },
                    'vectorization': {
                        'method': vector_method
                    }
                }
                config_str = json.dumps(config, sort_keys=True)
                result.add(config_str)

    return [json.loads(config_str) for config_str in result]

def generate_resampling_outlier_configurations(resampling_outlier_preprocessing_options):
    result = set()

    for resample_option in generate_resample_options(resampling_outlier_preprocessing_options['resampling']):
        for outlier_option in generate_outlier_options(resampling_outlier_preprocessing_options['outlier_filtering']):
            config = {
                'resampling': resample_option,
                'outlier_filtering': outlier_option
            }
            config_str = json.dumps(config, sort_keys=True)
            result.add(config_str)

    return [json.loads(config_str) for config_str in result]

def generate_resample_options(resampling_options):
    options = []
    for do_resample in resampling_options['do_resample']:
        if do_resample:
            for n_neighbours in resampling_options['n_neighbours']:
                options.append({
                    'do_resample': do_resample,
                    'n_neighbours': n_neighbours
                })
        else:
            options.append({
                'do_resample': do_resample,
                'resample_before_outlier_filtering': False,
                'n_neighbours': 0
            })
    return options

def generate_outlier_options(outlier_options):
    options = []
    for do_filter in outlier_options['do_filter']:
        if do_filter:
            for contamination in outlier_options['contamination']:
                for n_neighbours in outlier_options['n_neighbours']:
                    options.append({
                        'do_filter': do_filter,
                        'contamination': contamination,
                        'contamination': contamination,
                        'n_neighbours': n_neighbours,
                        'n_neighbours': n_neighbours,
                    })
        else:
            options.append({
                'do_filter': do_filter,
                'contamination': 0,
                'contamination': 0,
                'n_neighbours': 0,
                'n_neighbours': 0,
            })
    return options

def generate_sliding_window_options(data_method, data_creation_options):
    if 'sliding_window' in data_method:
        return [{'sliding_window_size': size, 'window_movement': movement} 
                for size in data_creation_options['sliding_window_size'] 
                for movement in data_creation_options['window_movement']]
    else:
        return [{'sliding_window_size': [0], 'window_movement': 'default'}]

def generate_machine_learning_configs(classifier_options, classifier_hyperparameters):
    machineLearningConfigurations = []
    for classifier_method in classifier_options:
        hyperparameters = classifier_hyperparameters.get(classifier_method, {})
        for hyperparameter_combination in itertools.product(*hyperparameters.values()):
            hyperparameter_dict = dict(zip(hyperparameters.keys(), hyperparameter_combination))
            MLConfig = {
                "classifier_method": classifier_method,
                "hyperparameters": hyperparameter_dict,
            }
            machineLearningConfigurations.append(MLConfig)
    return machineLearningConfigurations
