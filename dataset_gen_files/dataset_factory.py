'''
Factory that provides the dataset's logic as python object.
'''

from .arrow_vs_non_arrow import Arrow_vs_NonArrows
from .arrow_data_conf import defaults

class DatasetFactory:

    def factory_call(self, requested_dataset, **kwargs):
        '''
        Returns an object of the desired dataset.
        requested_dataset : dataset's name,
        kwargs : parameters to override the defaults.
        '''



        if requested_dataset == 'arrow_vs_nonarrow':
            parameters = defaults

            for key,value in kwargs.items():
                try:
                    parameters[key] = value
                except KeyError:
                    raise KeyError("The parameter that you have supplied is not valid for this dataset!")

            return Arrow_vs_NonArrows(**parameters) # func(**{'key':'value'}) is equivalent to func(key=value)
        else:
            raise ValueError("The requested dataset does not exist! Try checking your spelling.")
