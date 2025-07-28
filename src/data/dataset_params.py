
class StoredExperiments:
    """
    Stores information about the experiments that have been run.
    """

    def __init__(self):
        """
        Initialize dictionarys.
        """

        # Store information about which files should be processed when importing data for each experiment
        self.nodemap_nums = {
            "S_950_1.6": [str(i) for i in range(100, 304, 2)],  # each 2nd sample it at maximum force, starting at sample 100
            "S_160_4.7": [str(i) for i in range(14, 842, 5)], # each 5th sample it at maximum force, starting at sample 7
            "S_160_2.0": [str(i) for i in range(30, 1430, 5)], # each 5th sample it at maximum force, starting at sample 20
        }

        # Store the sample sizes for each experiment (in mm)
        self.sizes = {
            "S_950_1.6": 450,  # mm, ~90% of max x_undef (~475mm)
            "S_160_4.7": 70, # mm, ~90% of max x_undef (~80mm)
            "S_160_2.0": 70,# mm, ~90% of max x_undef (~80mm)
        }

        # Define which experiments have manual ground truth data
        self.exists_target = {
            "S_950_1.6": False,
            "S_160_4.7": True,
            "S_160_2.0": False,
        }