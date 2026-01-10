class BinnerType(Enum):
    IV = 1
    R2 = 2


class Binner:
    def __init__(self, binner_type=BinnerType.IV, good_mark=0, bad_mark=1):
        """
        Create binner instance
        :param binner_type: Type of binner (target metric: IV or R2)
        :param good_mark: Target variable value for "good" observations
        :param bad_mark: Target variable value for "bad" observations
        """
        if good_mark == bad_mark:
            raise Exception('Classes cant be equal!!!')
        self._good_v = good_mark
        self._bad_v = bad_mark
        self._binner_type = binner_type
        self._fitted_bins = None
        self._target_variable = None
        self._exclude = []

    def __eq__(self, other):
        assert (self._fitted_bins is not None) and (other._fitted_bins is not None), "Binner is not fitted"
        assert len(self._fitted_bins) == len(other._fitted_bins), "Different number of binned features"
        flag = True
        for i in range(len(self._fitted_bins)):
            if (self._fitted_bins[i]._name != other._fitted_bins[i]._name) or \
                    (self._fitted_bins[i]._woes != other._fitted_bins[i]._woes) or \
                    (self._fitted_bins[i]._gaps != other._fitted_bins[i]._gaps):
                flag = False
        return flag

    # Fit binner to the data
    def fit(self, data, target, power=3, binning_settings=(), exclude=[], verbose=False):
        """
        Fit binner to data
        :param data: Source dataframe
        :param target: Target variable name
        :param power: Maximum depth of splitting
        :param binning_settings: Additional binning parameters
        :param exclude: Columns that should not be binned
        :return: List of fitted binning objects
        """
        self._target_variable = target
        self._exclude = exclude

        bin_data = list()
        bin_names = list()
        for column in [x for x in data.columns if x not in [target, ] + self._exclude]:
            if verbose:
                print(column)
            variable_settings = None
            settings_list = [bs for bs in binning_settings if bs._variable_name == column]
            if len(settings_list) == 1:
                variable_settings = settings_list[0]
            x = data[column]
            y = data[target]
            w = Binning(
                x, y, column,
                power=power,
                settings=variable_settings,
                binner_type=self._binner_type,
                good_mark=self._good_v,
                bad_mark=self._bad_v
            )
            if len(w._gaps) > 0:
                bin_data.append(w)
            else:
                print("could not bin feature " + column)
                del w
        self._fitted_bins = bin_data
        return bin_data

    def get_bin_names(self):
        """
        Get names of all binned variables
        :return: List of variable names
        """
        names = []
        for binning_object in self._fitted_bins:
            names.append(binning_object._name)
        return names

    def get_binning(self, name):
        """
        Get binning object by variable name
        :param name: Feature name
        :return: Binning object if found, otherwise NaN
        """
        for binning_object in self._fitted_bins:
            if name == binning_object._name:
                return binning_object
            else:
                continue
        return np.nan

    # Apply bins to data and return WoE-transformed features
    def transform(self, data_in, exclude=[]):
        """
        Transform data using fitted bins
        :param data_in: Input dataframe
        :param exclude: Variables to exclude from transformation
        :return: Dataframe with WoE features
        """
        if self._fitted_bins is None or self._target_variable is None:
            raise Exception("Binner is not fitted!")

        names = self.get_bin_names()
        target = self._target_variable
        data = data_in.copy()
        for binning_obj in self._fitted_bins:
            if binning_obj._name in exclude:
                continue
            data = pd.concat(
                [data, binning_obj.transform(data[binning_obj._name])],
                axis=1
            )

        learn_columns = [col for col in data.columns if ('woe' in col)] + self._exclude
        learn_columns.append(target)
        learn_data = data[learn_columns]
        return learn_data

    # Return fitted binning objects
    def get_fitted_bins(self):
        """
        Get fitted bins
        :return: List of fitted binning objects
        """
        return self._fitted_bins

    # Generate SQL script for WoE transformation
    def to_sql(self):
        """
        Generate SQL script for WoE transformation
        :return: SQL script as string
        """
        sql = ''
        for bining in self._fitted_bins:
            sql_variable = 'case \n'
            for i in range(len(bining._gaps)):
                gap = bining._gaps[i]
                woe = bining._woes[i]
                if gap[0] is None:
                    a = "    when {} is NULL then {} \n".format(bining._name, woe)
                    sql_variable += a
                else:
                    a = (
                        "    when {}>{} and {}<={} then {} \n"
                        .format(bining._name, gap[0], bining._name, gap[1], woe)
                    )
                    sql_variable += a
            sql_variable += ' end, \n'
            sql += sql_variable
        return sql

    # Save binner to file
    def to_file(self, filename='bins.prdb'):
        """
        Save binner object to file
        :param filename: File path
        """
        with open(filename, 'wb') as output:
            pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
