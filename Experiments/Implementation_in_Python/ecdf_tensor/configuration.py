class ECDFDistanceMeasureConfiguration:
    # For the Distance Measure Computation Filtering using p-Values
    filtering_active = True
    p_value_alpha_threshold: float = 0.05
    number_bootstrap_samples = 1000


class InitializationConfiguration:
    # For the Power Analysis (Sample Size Computation)
    # & for the Distance Threshold Computation
    p_value_alpha_threshold: float = 0.05
    power_value: float = 0.8
    number_bootstrap_samples = 1000
    threshold_std_factor_maximum = 30.0
    threshold_std_factor_precision = 0.1
    threshold_std_factor_incorrect_detection_percentage = 0.8
    # For window overlap to compute the threshold factor computation
    window_percent_overlap: float = 0.8
    window_correct_detection_def = 0.8
    window_incorrect_detection_def = 0
    # For the Initialization with only 1 dataset
    k_fold_data_splitting = 10
    # For the Runtime Representation of the Data
    data_approximation_sample_threshold: int = 5000