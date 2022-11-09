# © MERL 2022
# Created by Efthymios Tzinis
"""
Conditional separation dataset loader using all heterogeneous conditions includin text for the FSD50K dataset.

There is also a hierarchical mixing strategy available for the below pre-defined classes of sounds where the user might
choose between mixing from different super-classes ()
"""

import optimal_condition_training.dataset_loader.abstract_dataset as abstract_dataset
from __config__ import FSD50K_ROOT_PATH, FSD50K_SYNONYMS_P

import warnings
import torch
import os
import numpy as np
import pickle
import glob2
import sys
import librosa
from tqdm import tqdm
from time import time
from scipy.io import wavfile
import csv
import torchaudio
import itertools


HIERARCHICAL_CLASSES_DIC = {
    "Animal": {"id": ('/m/0jbk', 'Animal'),
               "classes": [('/m/068hy', 'Domestic_animals_and_pets'),
                           ('/m/0bt9lr', 'Dog'), ('/m/05tny_', 'Bark'),
                           ('/m/0ghcn6', 'Growling'), ('/m/01yrx', 'Cat'),
                           ('/m/02yds9', 'Purr'), ('/m/07qrkrw', 'Meow'),
                           ('/m/07rjwbb', 'Hiss'), ('/m/0ghcn6', 'Growling'),
                           ('/m/0ch8v', 'Livestock_and_farm_animals_and_working_animals'),
                           ('/m/025rv6n', 'Fowl'), ('/m/09b5t', 'Chicken_and_rooster'),
                           ('/m/01280g', 'Wild_animals'), ('/m/015p6', 'Bird'),
                           ('/m/020bb7', 'Bird_vocalization_and_bird_call_and_bird_song'),
                           ('/m/07pggtn', 'Chirp_and_tweet'), ('/m/04s8yn', 'Crow'),
                           ('/m/01dwxx', 'Gull_and_seagull'), ('/m/03vt0', 'Insect'),
                           ('/m/09xqv', 'Cricket'), ('/m/09ld4', 'Frog')]},
    "Musical_instrument": {
        "id": ('/m/04szw', 'Musical_instrument'),
        "classes": [('/m/0fx80y', 'Plucked_string_instrument'),
                    ('/m/0342h', 'Guitar'), ('/m/02sgy', 'Electric_guitar'),
                    ('/m/018vs', 'Bass_guitar'), ('/m/042v_gx', 'Acoustic_guitar'),
                    ('/m/07s0s5r', 'Strum'), ('/m/05148p4', 'Keyboard_(musical)'),
                    ('/m/05r5c', 'Piano'), ('/m/013y1f', 'Organ'),
                    ('/m/0l14md', 'Percussion'), ('/m/02hnl', 'Drum_kit'),
                    ('/m/026t6', 'Drum'), ('/m/06rvn', 'Snare_drum'),
                    ('/m/0bm02', 'Bass_drum'), ('/m/01p970', 'Tabla'),
                    ('/m/01qbl', 'Cymbal'), ('/m/03qtq', 'Hi-hat'),
                    ('/m/0bm0k', 'Crash_cymbal'),
                    ('/m/07brj', 'Tambourine'), ('/m/05r5wn', 'Rattle_(instrument)'),
                    ('/m/0mbct', 'Gong'), ('/m/0j45pbj', 'Mallet_percussion'),
                    ('/m/0dwsp', 'Marimba_and_xylophone'), ('/m/0dwtp', 'Glockenspiel'),
                    ('/m/01kcd', 'Brass_instrument'), ('/m/07gql', 'Trumpet'),
                    ('/m/0l14_3', 'Bowed_string_instrument'),
                    ('/m/085jw', 'Wind_instrument_and_woodwind_instrument'),
                    ('/m/03m5k', 'Harp'),
                    ('/m/0239kh', 'Cowbell'), ('/m/0f8s22', 'Chime'),
                    ('/m/026fgl', 'Wind_chime'), ('/m/03qjg', 'Harmonica'),
                    ('/m/0mkg', 'Accordion'),
                    ('/m/01hgjl', 'Scratching_(performance_technique)')]},

    "Vehicle": {
        "id": ('/m/07yv9', 'Vehicle'),
        "classes": [('/m/019jd', 'Boat_and_Water_vehicle'),
                    ('/m/012f08', 'Motor_vehicle_(road)'),
                    ('/m/0k4j', 'Car'),
                    ('/m/0912c9', 'Vehicle_horn_and_car_horn_and_honking'),
                    ('/t/dd00134', 'Car_passing_by'),
                    ('/m/0ltv', 'Race_car_and_auto_racing'), ('/m/07r04', 'Truck'),
                    ('/m/01bjv', 'Bus'), ('/m/04_sv', 'Motorcycle'),
                    ('/m/0btp2', 'Traffic_noise_and_roadway_noise'),
                    ('/m/06d_3', 'Rail_transport'), ('/m/07jdr', 'Train'),
                    ('/m/0195fx', 'Subway_and_metro_and_underground'),
                    ('/m/0k5j', 'Aircraft'),
                    ('/m/0cmf2', 'Fixed-wing_aircraft_and_airplane')]},

    "Domestic_sounds_and_home_sounds": {
        "id": ('/t/dd00071', 'Domestic_sounds_and_home_sounds'),
        "classes": [('/m/02dgv', 'Door'), ('/m/03wwcy', 'Doorbell'),
                    ('/m/02y_763', 'Sliding_door'), ('/m/07rjzl8', 'Slam'),
                    ('/m/07r4wb8', 'Knock'), ('/m/07qcpgn', 'Tap'),
                    ('/m/07q6cd_', 'Squeak'), ('/m/0642b4', 'Cupboard_open_or_close'),
                    ('/m/0fqfqc', 'Drawer_open_or_close'),
                    ('/m/04brg2', 'Dishes_and_pots_and_pans'),
                    ('/m/023pjk', 'Cutlery_and_silverware'),
                    ('/m/0dxrf', 'Frying_(food)'), ('/m/0fx9l', 'Microwave_oven'),
                    ('/m/02jz0l', 'Water_tap_and_faucet'),
                    ('/m/0130jx', 'Sink_(filling_or_washing)'),
                    ('/m/03dnzn', 'Bathtub_(filling_or_washing)'),
                    ('/m/01jt3m', 'Toilet_flush'), ('/m/01s0vc', 'Zipper_(clothing)'),
                    ('/m/03v3yw', 'Keys_jangling'), ('/m/0242l', 'Coin_(dropping)'),
                    ('/m/05mxj0q', 'Packing_tape_and_duct_tape'),
                    ('/m/01lsmm', 'Scissors'), ('/m/0316dw', 'Typing'),
                    ('/m/0c2wf', 'Typewriter'), ('/m/01m2v', 'Computer_keyboard'),
                    ('/m/081rb', 'Writing')]},


    "Speech": {
        "id": ('/m/09x0r', 'Speech'),
        "classes": [('/m/05zppz', 'Male_speech_and_man_speaking'),
                    ('/m/02zsn', 'Female_speech_and_woman_speaking'),
                    ('/m/0ytgt', 'Child_speech_and_kid_speaking'),
                    ('/m/01h8n0', 'Conversation'), ('/m/0brhx', 'Speech_synthesizer')]},

    "Water": {
        "id": ('/m/0838f', 'Water'),
        "classes": [('/m/06mb1', 'Rain'), ('/m/07r10fb', 'Raindrop'),
                    ('/m/0j6m2', 'Stream'), ('/m/05kq4', 'Ocean'),
                    ('/m/034srq', 'Waves_and_surf'), ('/m/07swgks', 'Gurgling')]}
}
POSSIBLE_SUPER_CLASSES = list(HIERARCHICAL_CLASSES_DIC.keys())

HARMONICITY_LEVELS = ['harmonic', 'percussive']
NUM_HARMONICITY_LEVELS = len(HARMONICITY_LEVELS)
HARMONICITY_STR_TO_INT_DIC = dict([(x, i) for i, x in enumerate(HARMONICITY_LEVELS)])
LOUDNESS_LEVELS = ['louder', 'whispering']
NUM_LOUDNESS_LEVELS = len(LOUDNESS_LEVELS)
LOUDNESS_LEVELS_STR_TO_INT_DIC = dict([(x, i) for i, x in enumerate(LOUDNESS_LEVELS)])
POSSIBLE_ORDERINGS = ["first", "second"]
NUM_POSSIBLE_ORDERINGS = len(POSSIBLE_ORDERINGS)
ORDERINGS_STR_TO_INT_DIC = dict([(x, i) for i, x in enumerate(POSSIBLE_ORDERINGS)])
VALID_CONDITIONS = ["harmonicity", "energy", "source_order",
                    "one_hot_class", "one_hot_super_class", "multi_hot_class"]


class Dataset(torch.utils.data.Dataset, abstract_dataset.Dataset):
    """ Dataset class for multiple conditions dependent source separation."""
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.kwargs = kwargs

        self.zero_pad = self.get_arg_and_check_validness(
            'zero_pad', known_type=bool)

        self.augment = self.get_arg_and_check_validness(
            'augment', known_type=bool)

        self.split = self.get_arg_and_check_validness(
            'split', known_type=str, choices=['val', 'train', 'test'])

        self.n_samples = self.get_arg_and_check_validness(
            'n_samples', known_type=int, extra_lambda_checks=[lambda x: x >= 0])

        self.type_of_query = self.get_arg_and_check_validness(
            'type_of_query', known_type=str,
            choices=["one_condition", "all_conditions"])

        self.sample_rate = self.get_arg_and_check_validness('sample_rate',
                                                            known_type=int)

        self.timelength = self.get_arg_and_check_validness(
            'timelength', known_type=float)

        self.overlap_ratio = self.get_arg_and_check_validness(
            'overlap_ratio', known_type=float,
            extra_lambda_checks=[lambda x: 1 >= x >= 0])

        self.input_max_snr = self.get_arg_and_check_validness(
            'input_max_snr', known_type=float, extra_lambda_checks=[lambda x: x >= 0])

        self.time_samples = int(self.sample_rate * self.timelength)

        self.vocabulary_dic = self.get_vocabulary_dic()
        self.files_dic = self.get_files_dic_for_split()
        self.classes_to_files_dic = self.get_classes_to_files_dic()
        self.possible_classes = list(self.classes_to_files_dic.keys())
        self.possible_file_paths = list(self.files_dic.keys())

        # Used for text conditions
        self.synonym_str_dic = self.get_synonyms_for_class()

        # Valid distribution of conditional vectors
        self.valid_conditions = self.get_arg_and_check_validness(
            'valid_conditions', known_type=list,
            extra_lambda_checks=[lambda y: all([(x in VALID_CONDITIONS) for x in y])])

        self.queries_priors = self.get_arg_and_check_validness(
            'queries_priors', known_type=list,
            extra_lambda_checks=[lambda y: (sum(y) - 1.) < 1e-5,
                                 lambda y: len(y) == len(self.valid_conditions)])
        self.queries_cdf = np.cumsum(self.queries_priors).tolist()

        # In the case of hierarchical sampling, we also control the distribution of
        # the intra/inter super class sampling
        self.is_hierarchical = self.get_arg_and_check_validness(
            'is_hierarchical', known_type=bool)

        # This is only valid when the hierarchical flag is set
        self.intra_class_prior = self.get_arg_and_check_validness(
            'intra_class_prior', known_type=float,
            extra_lambda_checks=[lambda x: 1 >= x >= 0])

        self.CONDITIONS_OFFSET_DIC = {
            "harmonicity": {"start": 0, "end": 2},
            "energy": {"start": 2, "end": 4},
            "source_order": {"start": 4, "end": 6},
            "one_hot_class": {"start": 6, "end": 6 + len(self.possible_classes)},
            "one_hot_super_class": {"start": 6, "end": 6 + len(self.possible_classes)},
            "multi_hot_class": {"start": 6, "end": 6 + len(self.possible_classes)},
        }
        self.cond_emd_dim = self.get_cond_emb_dim()

        self.could_sample_harmonicity_condition = False
        for valid_cond, cond_prior in zip(self.valid_conditions, self.queries_priors):
            if valid_cond == "harmonicity" and cond_prior > 0:
                self.could_sample_harmonicity_condition = True
                break

    def __len__(self):
        return self.n_samples

    def get_cond_emb_dim(self):
        return self.CONDITIONS_OFFSET_DIC["multi_hot_class"]["end"]

    def get_classes_to_files_dic(self):
        classes_to_files_dic = {}
        for file_path, this_file_classes in self.files_dic.items():
            for class_name in this_file_classes:
                if class_name not in classes_to_files_dic:
                    classes_to_files_dic[class_name] = [file_path]
                else:
                    classes_to_files_dic[class_name].append(file_path)
        return classes_to_files_dic

    def get_synonyms_for_class(self):
        with open(FSD50K_SYNONYMS_P, 'r') as f:
            lines = f.readlines()

        synonym_dic = {}
        for l in lines:
            synonym_dic[l.split(",")[0]] = l.split("\n")[0].split(",")[2:]

        return synonym_dic

    def get_files_dic_for_split(self):
        files_dic = {}
        csv_name = "dev.csv"
        if self.split == "test":
            csv_name = "eval.csv"
        csv_file = os.path.join(FSD50K_ROOT_PATH, 'FSD50K.ground_truth', csv_name)

        if os.path.lexists(csv_file):
            with open(csv_file, 'r') as f:
                lines = f.readlines()

            for unproc_l in lines[1:]:
                l = unproc_l.split("\n")[0]
                if not self.split == "test":
                    if self.split not in l.split(",")[-1]:
                        continue
                    l = ",".join(l.split(",")[:-1])

                try:
                    info = l.split()[0].split('"')
                    classes_in_file = info[-2].split(",")
                except Exception as e:
                    classes_in_file = [l.split(",")[-1]]
                file_name = l.split(",")[0]
                this_set = "eval" if self.split == "test" else "dev"
                file_path = os.path.join(FSD50K_ROOT_PATH, f'FSD50K.{this_set}_audio',
                                         file_name + '.wav')
                if not os.path.lexists(file_path):
                    raise FileExistsError(file_path)
                files_dic[file_path] = classes_in_file
            return files_dic
        else:
            raise IOError(f"File: {csv_file} could not be parsed!")

    def get_n_classes(self):
        return len(self.possible_classes)

    def get_vocabulary_dic(self,):
        vocabulary_path = os.path.join(FSD50K_ROOT_PATH, 'FSD50K.ground_truth',
                                       'vocabulary.csv')
        voc_dic = {}
        if os.path.lexists(vocabulary_path):
            with open(vocabulary_path, 'r') as f:
                lines = f.readlines()

            for l in lines:
                class_id, class_str, class_audioset_id = l.split()[0].split(",")
                voc_dic[class_audioset_id] = {
                    "class_id": int(class_id), "class_str": class_str
                }
            return voc_dic
        else:
            raise IOError(f"File: {vocabulary_path} could not be parsed!")

    def wavread(self, path):
        waveform, fs = torchaudio.load(path)
        # Resample in case of a given sample rate
        if self.sample_rate < fs:
            waveform = torchaudio.functional.resample(
                waveform, fs, self.sample_rate, resampling_method="kaiser_window")
        elif self.sample_rate > fs:
            raise ValueError("Cannot upsample.")

        # Convert to single-channel
        if len(waveform.shape) > 1:
            waveform = waveform.sum(0)

        return (1. * waveform - waveform.mean()) / (waveform.std() + 1e-8)

    def sample_target_sound_class(self):
        if self.is_hierarchical:
            # Sample between the available superclasses
            target_super_class = np.random.choice(POSSIBLE_SUPER_CLASSES)
            target_class = HIERARCHICAL_CLASSES_DIC[target_super_class]["classes"][
                np.random.randint(
                    len(HIERARCHICAL_CLASSES_DIC[target_super_class]["classes"]))][0]
        else:
            target_class = np.random.choice(self.possible_classes)
            target_super_class = None
        return target_class, target_super_class

    def sample_conditional_query(self):
        random_draw = np.random.random()
        selected_query = self.valid_conditions[-1]
        for query_cdf_val, query_str in zip(self.queries_cdf, self.valid_conditions):
            if random_draw <= query_cdf_val:
                selected_query = query_str
                break
        return selected_query

    def sample_two_valid_source_files(self, sampled_conditional_query):
        # Sample the class of sound, the super-class is only used in hierarchical mode
        sampled_target_class, sampled_target_super_class = \
            self.sample_target_sound_class()
        target_source_file_path = np.random.choice(
            self.classes_to_files_dic[sampled_target_class])
        all_target_classes = self.files_dic[target_source_file_path]

        # If it is hierarchical, we need to setup the possible set of interference classes
        if self.is_hierarchical:
            class_sample_mode = np.random.uniform(low=0., high=1.)
            if class_sample_mode > self.intra_class_prior:
                possible_interference_classes = [
                    HIERARCHICAL_CLASSES_DIC[el]["id"][0]
                    for el in POSSIBLE_SUPER_CLASSES
                    if el != sampled_target_super_class]
                if sampled_conditional_query == "one_hot_super_class":
                    final_class_str_used = HIERARCHICAL_CLASSES_DIC[
                        sampled_target_super_class]["id"][0]
                else:
                    final_class_str_used = sampled_target_class
            else:
                # Sample from the same class
                possible_interference_classes = [
                    el[0]
                    for el in
                    HIERARCHICAL_CLASSES_DIC[sampled_target_super_class]["classes"]
                    if el[0] != sampled_target_class]
                final_class_str_used = sampled_target_class
        else:
            # If not, sample regularly from all available classes
            possible_interference_classes = self.possible_classes
            final_class_str_used = sampled_target_class

        # Sample an interference source which is not described by the same class id
        found_valid_interference_file = False
        while not found_valid_interference_file:
            # Sample uniformly from all other classes
            interf_sampled_class = np.random.choice(possible_interference_classes)
            interf_sampled_file_path = np.random.choice(
                self.classes_to_files_dic[interf_sampled_class])

            interference_classes = self.files_dic[interf_sampled_file_path]
            if sampled_target_class in interference_classes:
                continue
            else:
                found_valid_interference_file = True

        return (target_source_file_path, interf_sampled_file_path,
                final_class_str_used, all_target_classes)

    def delay_and_mix_2_sources(self, wav1_tensor, wav2_tensor, input_snr, olp_ratio):
        # Sample a random overlap ratio between [self.overlap_ratio, 1.]
        non_olp_samples = int((1. - olp_ratio) * wav1_tensor.shape[0])

        delayed_wav2_tensor = torch.zeros_like(wav2_tensor)
        delayed_wav2_tensor[non_olp_samples:] = \
            wav2_tensor[:wav2_tensor.shape[0] - non_olp_samples]

        # Mix the two tensors with a specified SNR ratio
        source_1_tensor, source_2_tensor = self.mix_2_with_specified_snr(
            wav1_tensor, delayed_wav2_tensor, input_snr)

        return source_1_tensor, source_2_tensor

    def get_sources_tensors_and_metafeatures(self, wav_path_1, wav_path_2,
                                             extract_harmonic_features=False,
                                             input_snr=None,
                                             olp_ratio=None,
                                             source_order=None):
        waveform_1 = self.wavread(wav_path_1)
        waveform_2 = self.wavread(wav_path_2)

        # Mix with the specified overlap ratio
        wav1_tensor = self.get_padded_tensor(waveform_1)
        wav2_tensor = self.get_padded_tensor(waveform_2)

        # Sample which source is going to be mixed first or second uniformly
        if source_order == 0:
            source_1_tensor, source_2_tensor = self.delay_and_mix_2_sources(
                wav1_tensor, wav2_tensor, input_snr, olp_ratio)
        else:
            source_2_tensor, source_1_tensor = self.delay_and_mix_2_sources(
                wav2_tensor, wav1_tensor, -input_snr, olp_ratio)

        if extract_harmonic_features:
            # Harmonic / Percussive decomposition
            D1 = librosa.stft(source_1_tensor.cpu().numpy())
            D2 = librosa.stft(source_2_tensor.cpu().numpy())
            H1, P1 = librosa.decompose.hpss(D1)
            H2, P2 = librosa.decompose.hpss(D2)
            Harmonic_energy1 = np.sum(np.abs(H1) ** 2)
            Harmonic_energy2 = np.sum(np.abs(H2) ** 2)
            Percussive_energy1 = np.sum(np.abs(P1) ** 2)
            Percussive_energy2 = np.sum(np.abs(P2) ** 2)

            HP_ratio1 = Harmonic_energy1 / (Percussive_energy1 + 1e-8)
            HP_ratio2 = Harmonic_energy2 / (Percussive_energy2 + 1e-8)
            HP_ratio = HP_ratio1 / (HP_ratio2 + 1e-8) + 1e-8
            logHP_ratio = float(10. * np.log(HP_ratio))
        else:
            logHP_ratio = None

        return source_1_tensor, source_2_tensor, logHP_ratio

    def get_multi_hot_encoding(self, target_classes, offset=0):
        """Converts to one hot encoding"""
        return sum(
            [torch.nn.functional.one_hot(
                torch.tensor([self.vocabulary_dic[this_class]["class_id"] + offset]
                             ).to(torch.long),
                num_classes=self.get_n_classes() + offset).to(torch.float32)[0]
             for this_class in target_classes]
        )

    def construct_one_condition_embedding(self, condition, input_snr,
                                          target_source_order, logHP_ratio,
                                          target_class, all_target_classes):
        offset = self.CONDITIONS_OFFSET_DIC[condition]["start"]
        if condition == "harmonicity":
            sub_idx = 0 if logHP_ratio >= 0 else 1
        elif condition == "energy":
            sub_idx = 0 if input_snr >= 0 else 1
        elif condition == "source_order":
            sub_idx = target_source_order
        else:
            sub_idx = None

        if sub_idx is not None:
            conditional_vector = torch.nn.functional.one_hot(
                torch.tensor([offset + sub_idx]).to(torch.long),
                num_classes=self.cond_emd_dim).to(torch.float32)[0]
        else:
            # This means that we have class-based information
            if condition == "multi_hot_class":
                conditional_vector = self.get_multi_hot_encoding(
                    all_target_classes, offset=offset)
            else:
                conditional_vector = self.get_multi_hot_encoding(
                    [target_class], offset=offset)

        return conditional_vector

    def __getitem__(self, idx):
        if self.augment:
            seed = int(np.modf(time())[0] * 100000000)
        else:
            seed = idx
        np.random.seed(seed)

        sampled_conditional_query = self.sample_conditional_query()

        (target_source_file_path, interf_sampled_file_path,
         sampled_target_class, all_target_classes) = self.sample_two_valid_source_files(
            sampled_conditional_query)

        # Sample all the appropriate synthetic meta-values
        # Sample a random overlap ratio between [self.overlap_ratio, 1.]
        sampled_olp_ratio = np.random.uniform(low=self.overlap_ratio, high=1.)
        sampled_input_snr = np.random.uniform(-self.input_max_snr, self.input_max_snr)
        sampled_target_source_order = np.random.choice([0, 1])

        # Mix the source files with some input SNR and delay and extract meta-features
        (target_wav_tensor, interf_wav_tensor, logHP_ratio) = \
            self.get_sources_tensors_and_metafeatures(
            target_source_file_path, interf_sampled_file_path,
            extract_harmonic_features=(
                sampled_conditional_query == "harmonicity" or
                (self.could_sample_harmonicity_condition and
                 self.type_of_query == "all_conditions")
            ),
            input_snr=sampled_input_snr, olp_ratio=sampled_olp_ratio,
            source_order=sampled_target_source_order)

        # Pack the source tensors
        sources_tensor = torch.stack([target_wav_tensor, interf_wav_tensor], axis=0)

        # Construct the conditioning vector based on the sampled conditional query
        binary_conditional_vector = \
            self.construct_one_condition_embedding(
                sampled_conditional_query, sampled_input_snr,
                sampled_target_source_order, logHP_ratio,
                sampled_target_class, all_target_classes)

        # Always return the string to be used by a text embedding mechanism
        text_emb = self.synonym_str_dic[sampled_target_class][0]

        if self.type_of_query == "all_conditions":
            # Return all possible condition binary queries
            # We can construct the all-hot set of vectors based on all available conditions
            all_conditions_vectors = []
            for valid_cond, cond_prior in zip(self.valid_conditions, self.queries_priors):
                if cond_prior:
                    all_conditions_vectors.append(
                        self.construct_one_condition_embedding(
                            valid_cond, sampled_input_snr,
                            sampled_target_source_order, logHP_ratio,
                            sampled_target_class, all_target_classes))

            binary_conditional_vector = torch.stack(all_conditions_vectors, axis=0)

        return sources_tensor, binary_conditional_vector, text_emb


def test_generator():
    def get_snr(tensor_1, tensor_2):
        return 10. * torch.log10((tensor_1**2).sum(-1) / ((tensor_2**2).sum(-1) + 1e-9))

    batch_size = 8
    n_jobs=4
    sample_rate = 8000
    timelength = 5.0
    overlap_ratio = 0.5
    time_samples = int(sample_rate * timelength)
    valid_conditions = ["harmonicity", "energy", "source_order",
                        "one_hot_class", "one_hot_super_class", "multi_hot_class"]
    queries_priors = [0.4] + [0.3] + [0.3] + [0.] * 3
    input_max_snr = 5.
    type_of_query = "all_conditions"
    type_of_query = "one_condition"

    for (is_hierarchical, intra_class_prior) in [(False, 1.), (True, 0.), (True, 1.)]:
        data_loader = Dataset(
            split='test',
            input_max_snr=input_max_snr,
            overlap_ratio=overlap_ratio,
            sample_rate=sample_rate,
            timelength=timelength,
            type_of_query=type_of_query,
            valid_conditions=valid_conditions,
            queries_priors=queries_priors,
            is_hierarchical=is_hierarchical,
            intra_class_prior=intra_class_prior,
            zero_pad=True,
            augment=True,
            n_samples=10)
        generator = data_loader.get_generator(batch_size=batch_size,
                                              num_workers=n_jobs,
                                              pin_memory=False)

        before = time()
        # this_time = time() - before
        # print(this_time)
        for data in generator:
            # source vectors, binary conditional vector, text query
            sources_tensor, cond_vec, q_str = data
            if is_hierarchical:
                print(f"For a hierarchical FSD50K version with probability of sampling from"
                      f" the same super class: {intra_class_prior}")
            else:
                print(f"For randomly sampling between 200 classes of sounds in FSD50K.")
            print(f"Sampled the classes: {q_str}")
            print(f"Sampled the following conditional vectors: {cond_vec}")
            break
        this_time = time() - before
        print(f"It took me: {this_time} secs to fetch the batch")
        print()

if __name__ == "__main__":
    test_generator()
