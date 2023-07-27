#from bin.MLEngine import MLEngine
from bin.MLEngineRefined import MLEngine
from bin.my_priors import CorrelatedGaussianPrior, CombinedPrior, StudentTPrior
from optbnn.bnn.priors import FixedGaussianPrior

# the base code structure of the MLEngine class comes from https://github.com/fbcsptoolbox/fbcsp_code

if __name__ == "__main__":

    '''Example for loading BCI Competition IV Dataset 2a'''
    dataset_details = {
        'data_path': "../BCIC2a",
        'file_to_load': 'A03T.gdf',
        'ntimes': 25,
        'kfold': 3,
        'm_filters': 2,
        'window_details': {'tmin': 0.5, 'tmax': 2.5}
        }

    ML_experiment = MLEngine(**dataset_details)
    subjects = 1

    if subjects == 1:
        #ML_experiment.experiment()
        #ML_experiment.SGHMC_experiment()
        # ML_experiment.SGHMC_experiment()
        #ML_experiment.NeuroNetExperiment('EEGNet', weights_to_sample=120, subject_id=3)
        #ML_experiment.NeuroNetExperiment('ShallowFBCSPNet', weights_to_sample=120, subject_id=3)
        prior_dict = {'default': FixedGaussianPrior(), 'fc': StudentTPrior(), 'time': 1.0, 'spat': 1.0} # time and spat float or Prior() instance
        # Usual for CG+t is 'time':1.0, 'spat':1.0
        # usual burn-in == keep_every == 200 num_chains == 4, num_samples == 40, print_every == 5, n_discarded == 10
        sampler_config = {
            "batch_size": 0,  # Mini-batch size will be filled in during experiment
            "num_samples": 40,                # Total number of samples for each chain
            "n_discarded": 10,                # Number of the first samples to be discared for each chain
            "num_burn_in_steps": 200,         # Number of burn-in steps
            "keep_every": 200,                # Thinning interval
            "lr": 1e-2,                       # Step size
            "num_chains": 4,                  # Number of chains
            "mdecay": 1e-2,                   # Momentum coefficient
            "print_every_n_samples": 5
        }

        subjects = [5, 6]

        for i in subjects:

            #ML_experiment.EnsembleExperiment('EEGNet', n_epochs=350, number_of_nets=120, subject_id=i, experiment_name='LogitEnsembleExperiment')

            #ML_experiment.EnsembleExperiment('ShallowFBCSPNet', n_epochs=100, number_of_nets=120, subject_id=i, experiment_name='LogitEnsembleExperiment')

            #ML_experiment.NetExperiment('EEGNet', prior_dict=prior_dict, sampler_config=sampler_config, subject_id=i,
            #                                experiment_name='LogitsOODdetectionFinal_СG_and_T')

            #ML_experiment.NetExperiment('ShallowFBCSPNet', prior_dict=prior_dict, sampler_config=sampler_config, subject_id=i,
            #                                experiment_name='LogitsOODdetectionPaperFinal_СG_and_T')

        #for i in range(1, 10):

            #ML_experiment.EnsembleExperiment('EEGNet', n_epochs=350, number_of_nets=120, subject_id=i,
            #                                 experiment_name='LogitEnsembleExperimentFinal')

            #ML_experiment.EnsembleExperiment('ShallowFBCSPNet', n_epochs=100, number_of_nets=120, subject_id=i,
            #                                 experiment_name='LogitEnsembleExperimentFinal')

            ML_experiment.FullsetExperiment('EEGNet', prior_dict=prior_dict, sampler_config=sampler_config, subject_id=i,
                                           experiment_name='FullsetExperimentFinal')
            ML_experiment.FullsetExperiment('ShallowFBCSPNet', prior_dict=prior_dict, sampler_config=sampler_config, subject_id=i,
                                            experiment_name='FullsetExperimentFinal')

            ML_experiment.FullsetDetExperiment('EEGNet', n_epochs=350, subject_id=i, experiment_name='FullsetDetExperimentFinal')
            ML_experiment.FullsetDetExperiment('ShallowFBCSPNet', n_epochs=100, subject_id=i, experiment_name='FullsetDetExperimentFinal')

        #lrs = [0.1, 0.001, 3.e-4]

        #for lr in lrs:

        #    sampler_config['lr'] = lr

        #    ML_experiment.NetExperiment('EEGNet', prior_dict=prior_dict, sampler_config=sampler_config, subject_id=5,
        #                                experiment_name='VarLr_CorrAndTPrior')
        #    ML_experiment.NetExperiment('ShallowFBCSPNet', prior_dict=prior_dict, sampler_config=sampler_config, subject_id=5,
        #                                experiment_name='VarLr_CorrAndTPrior')

    if subjects == 2:
        second_subject = 'A02T.gdf'
        #ML_experiment.experiment(subj2_filename=second_subject)
        ML_experiment.SGHMC_experiment(subj2_filename=second_subject)

